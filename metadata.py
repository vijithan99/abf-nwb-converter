# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:20:32 2026

@author: vijit
"""

from pathlib import Path
from typing import Any

import pandas as pd


SUBJECT_ID_COLUMNS = (
    "Patient ID",
    "Patient_ID",
    "patient_id"
    "Subject ID",
    "Subject_ID",
    "subject_id",
)


def normalize_file_id(value: Any) -> str:
    """Normalize an ABF filename or file ID for comparison."""
    if pd.isna(value):
        return ""

    return Path(str(value).strip()).stem.casefold()

def iso8601_convert(age: Any):
    """Convert an age in years to an ISO 8601 duration."""
    if pd.isna(age):
        return None

    age_text = str(age).strip()

    if not age_text:
        return None

    # Preserve ages that are already ISO 8601 formatted.
    if age_text.upper().startswith("P"):
        return age_text.upper()

    try:
        numeric_age = float(age_text)
    except ValueError as error:
        raise ValueError(
            f"Age must be numeric or ISO 8601 formatted, received {age!r}"
        ) from error

    return f"P{numeric_age:g}Y"

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

def get_file_metadata(abf):
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