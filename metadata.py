# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:20:32 2026

@author: vijit
"""

from pathlib import Path
from typing import Any, Iterable
import re

import pandas as pd

## File Details
experimenter = "Moradi Chameh, Homeira"
lab = "Neuron to Brain Lab"

SUBJECT_ID_COLUMNS = (
    "subject_id",
    "Subject ID",
    "SubjectID",
    "Patient ID",
    "PatientID",
    "patient_id",
    "Study ID",
    "StudyID",
)

LAYER_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9])
    (?:layer\s*|l\s*)
    (?P<first>[1-6])
    (?:
        \s*[/&-]\s*(?P<second>[1-6])
        |
        (?P<suffix>[a-c])
    )?
    (?![A-Za-z0-9])
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
    
CELL_ID_PATTERNS = (
    re.compile(
        r"(?<![A-Za-z0-9])C\s*[:#-]?\s*(?P<cell_id>[0-9]{1,4}[A-Za-z]?)(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bcell\s*[:#-]?\s*(?P<cell_id>[A-Za-z0-9._-]+)\b",
        flags=re.IGNORECASE,
    ),
)

def normalize_file_id(value: Any) -> str:
    """Normalize an ABF filename or file ID for comparison."""
    if pd.isna(value):
        return ""

    return Path(str(value).strip()).stem.casefold()

def iso8601_convert(age: Any):
    """Convert an age in years to an ISO 8601 duration."""
    if pd.isna(age) or not str(age).strip():
        return None

    years = int(age)
    return f"P{years:g}Y"

def normalize_sex(sex: Any) -> str:
    """Convert common sex labels to NWB-compatible values."""
    ## Return unknown ##
    if pd.isna(sex):
        return "U"

    normalized = str(sex).strip().casefold()

    mapping = {
        "m": "M",
        "male": "M",
        "f": "F",
        "female": "F",
        "u": "U",
        "unknown": "U",
        "o": "O",
        "other": "O",
    }

    if normalized not in mapping:
        raise ValueError(
            f"Unrecognized sex value: {sex!r}"
        )

    return mapping[normalized]

def get_abf_tag_comments(abf) -> list[str]:
    """Collect and deduplicate tag comments from a pyABF object."""
    comments: list[str] = []

    def add_comments(values: Any) -> None:
        if values is None:
            return

        if isinstance(values, str):
            values = [values]
        else:
            try:
                values = list(values)
            except TypeError:
                values = [values]

        for value in values:
            # Handle values such as (tag_time, tag_comment).
            if isinstance(value, (list, tuple)) and value:
                value = value[-1]

            text = str(value).strip()

            if text:
                comments.append(text)

    # Public pyABF interface.
    add_comments(getattr(abf, "tagComments", None))

    # Fallback for files where comments are only exposed privately.
    tag_section = getattr(abf, "_tagSection", None)
    add_comments(getattr(tag_section, "sComment", None))

    # Optional general file comment.
    add_comments(getattr(abf, "abfFileComment", None))

    # Deduplicate while preserving order.
    return list(dict.fromkeys(comments))


def extract_layer_from_text(value: Any) -> str | None:
    """
    Extract and normalize a cortical layer from text.

    Examples:
        "l2/3"       -> "L2/3"
        "Layer 5"    -> "L5"
        "L2-3"       -> "L2/3"
        "L2&3"       -> "L2/3"
        "L3c"        -> "L3c"

    Returns None when layer information is absent.
    """
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    match = LAYER_PATTERN.search(text)

    if match is None:
        return None

    first = match.group("first")
    second = match.group("second")
    suffix = match.group("suffix")

    if second is not None:
        return f"L{first}/{second}"

    if suffix is not None:
        return f"L{first}{suffix.lower()}"

    return f"L{first}"


def extract_layer_from_comments(comments: list[str]) -> tuple[str | None, str | None]:
    """
    Return the normalized layer and the comment it came from.

    Returns:
        (layer, source_comment)

    If no layer is found:
        (None, None)
    """
    for comment in comments:
        layer = extract_layer_from_text(comment)

        if layer is not None:
            return layer, comment

    return None, None

def extract_cell_id_from_comments(comments: Iterable[str]) -> tuple[str | None, str | None]:
    """Return ``(cell_id, source_comment)`` for the first ABF cell label found."""
    for comment in comments:
        text = str(comment)
        for index, pattern in enumerate(CELL_ID_PATTERNS):
            match = pattern.search(text)
            if match is None:
                continue

            parsed_id = match.group("cell_id").strip()
            cell_id = f"C{parsed_id}" if index == 0 else parsed_id
            return cell_id, text

    return None, None


def extract_cell_id_from_abf_tags(abf: Any) -> tuple[str | None, str | None]:
    """Extract a cell identifier and its source comment from an ABF object."""
    return extract_cell_id_from_comments(get_abf_tag_comments(abf))


def get_file_metadata(abf):
    tag_comments = get_abf_tag_comments(abf)
    layer, layer_source = extract_layer_from_comments(tag_comments)
    cell_id, cell_id_source = extract_cell_id_from_comments(tag_comments)

    
    metadata = {
        "file": {
            "version": abf.abfVersion,
            "versionStr": abf.abfVersionString,
            "protocol": abf.protocol,
            "creator": abf.creator,
            "ID": abf.abfID,
            'file_path': abf.abfFilePath,
            "comment": abf.abfFileComment,
            "guid": abf.fileGUID,
            "datetime": abf.abfDateTime,
            "datetimestr": abf.abfDateTimeString,
        },
        
        "acquisition": {
            "sample_rate": abf.dataRate,
            "sweeps": abf.sweepCount,
            "points_per_sweep": abf.sweepPointCount,
            "sweep_length_s": abf.sweepLengthSec,
        },
        
        "channels": {
            "count": abf.channelCount,
            "names": abf.adcNames,
            "units": abf.adcUnits,
            # "gains": abf.adcGains,
        },
        
        "ephys": {
            "comments": tag_comments,
            "layer": layer,
            "layer_source": layer_source
        },
        
        "clamp": {
            # "mode": abf.clampMode,
            # "mode_str": abf.clampModeString,
        },
    }
    
    return metadata


def get_patient_metadata(file_name: str, patient_data_path: Path, strict_subject_id: bool = False) -> dict[str, Any]:
    if not patient_data_path.exists():
        raise FileNotFoundError(
            f"Patient metadata file was not found: {patient_data_path}"
        )
        
    if patient_data_path is not None and patient_data_path.exists():
        database = pd.read_csv(patient_data_path)
        if "Resection Date" not in database.columns:
            raise ValueError(
                f"Patient metadata file {patient_data_path} is missing the 'Resection Date' column"
            )
            
        target_file_id = normalize_file_id(file_name)

        matches = database.loc[
            database["File ID"].map(normalize_file_id) == target_file_id
        ]
    
        if matches.empty:
            raise LookupError(
                f"No patient metadata was found for {file_name}"
            )
    
        if len(matches) > 1:
            raise ValueError(
                f"Multiple patient metadata rows matched {file_name}"
            )
    
        row = matches.iloc[0]
    
        subject_id_column = next(
            (
                column
                for column in SUBJECT_ID_COLUMNS
                if column in database.columns
            ),
            None,
        )
    
        if subject_id_column is None:
            if strict_subject_id:
                raise ValueError(
                    "No subject-ID column was found in the metadata file."
                )
            subject_id = None
        else:
            subject_id = row[subject_id_column]
    
            if pd.isna(subject_id) or not str(subject_id).strip():
                if strict_subject_id:
                    raise ValueError(
                        f"Subject ID is missing for {file_name}"
                    )
                subject_id = None
            else:
                subject_id = str(subject_id).strip()
    
        return {
            "subject_id": subject_id,
            "age": iso8601_convert(row["Age"]),
            "sex": normalize_sex(row["Sex"]),
            "condition": row["Condition"],
            "resection_date": row["Resection Date"],
            "seizure_duration": row["Seizure Duration"],
            "seizure_type": row["Seizure Type"],
            "tissue_type": row["Tissue Type"],
            "tumour": row["Tumor"],
            "structure": row["Tissue Location"],
            "medication": row["Antiseziure Medications Used"],
        }