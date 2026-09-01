# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import annotations

import os
import hashlib
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyabf
import util_functions
import metadata as md

from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBHDF5IO, NWBFile, TimeSeries, validate
from pynwb.file import Subject
from pynwb.icephys import (
    CurrentClampSeries,
    CurrentClampStimulusSeries,
    IZeroClampSeries,
    PatchClampSeries,
    VoltageClampSeries,
    VoltageClampStimulusSeries,
)

# Analyze ABF file
current_dir = os.getcwd()

species = "mice"

input_abf_root = os.path.join(current_dir, "miceabfData")
output_nwb_root = os.path.join(current_dir, "nwbData")
patient_data_path = os.path.join(current_dir, "patientData", "patientData.csv")
mouse_data_path = os.path.join(current_dir, "patientData", "mouseData.csv")

TORONTO_TZ = ZoneInfo("America/Toronto")
PAPER_DOI = "https://doi.org/10.1093/gigascience/giac108"

VOLTAGE_UNITS = {"v", "mv", "uv"}
CURRENT_UNITS = {"a", "ma", "ua", "na", "pa"}

# Cell Data missing Metadata
dates_missing = set()

converted = 0
failures = []

def nwb_conversion_from_unit(unit):
    unit = unit.lower()
    if unit == "mv":
        return 1e-3, "volts"
    if unit == "v":
        return 1.0, "volts"
    if unit == "pa":
        return 1e-12, "amperes"
    if unit == "na":
        return 1e-9, "amperes"
    if unit == "ua":
        return 1e-6, "amperes"
    raise ValueError(f"Unhandled ABF unit: {unit}")

def _clean_optional(value: Any) -> Any | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown", "no info", "n/a"}:
        return None
    return value


def subject_from_metadata(species: str, metadata: dict[str, Any]) -> Subject:
    if species == "human":
        species_name = "Homo sapiens"
        description_parts = ["De-identified human participant"]
        for key, label in (("condition", "condition"), ("tumour", "tumour status")):
            if _clean_optional(metadata.get(key)) is not None:
                description_parts.append(f"{label}: {metadata[key]}")
    else:
        species_name = "Mus musculus"
        description_parts = ["Laboratory mouse"]
        for key in ("condition", "model"):
            if _clean_optional(metadata.get(key)) is not None:
                description_parts.append(f"{key}: {metadata[key]}")
                
    kwargs: dict[str, Any] = {
        "subject_id": str(metadata["subject_id"]),
        "description": "; ".join(description_parts),
        "species": species_name,
        "sex": str(metadata.get("sex") or "U"),
    }
    if metadata.get("age"):
        kwargs["age"] = str(metadata["age"])
    return Subject(**kwargs)

def convert_abf_to_nwb(dirpath, filename, species = "human"):
    '''
    Species = "human" or "mice" -- Metadata comes from different places
    
    '''
    
    print(f"\nConverting {filename}")
    recordings = []
    
    ## Make file directory strings to access ABF files and output folders for NWB files
 
    dirpath = Path(dirpath)
    abf_path = dirpath / filename
    if not abf_path.exists():
        raise FileNotFoundError(
            f"ABF file was not found: {abf_path}"
        )

    species_key = species.strip().casefold()

    if species_key == "mice":
        species_key = "mouse"

    if species_key not in {"human", "mouse"}:
        raise ValueError(
            f"Unsupported species: {species!r}. "
            "Expected 'human', 'mouse', or 'mice'."
        )

    abf = pyabf.ABF(str(abf_path))
    file_metadata = md.get_file_metadata(abf)
    
    # ---------------------------------------------------------
    # Output path
    # ---------------------------------------------------------
    relative_path = os.path.relpath(dirpath, input_abf_root)
    nwb_output_dir = os.path.join(output_nwb_root, relative_path)
    os.makedirs(nwb_output_dir, exist_ok=True)
    
    nwb_path = os.path.join(
        nwb_output_dir,
        os.path.splitext(filename)[0] + ".nwb"
    )
    
    # ---------------------------------------------------------
    # Subject metadata
    # ---------------------------------------------------------    
    if species == "human":
        patient_metadata = md.get_patient_metadata(filename, Path(patient_data_path), False)
    else:
        patient_metadata = md.get_mouse_metadata(filename, Path(mouse_data_path), False)


    # Get comments/conditions
    conditions = abf._tagSection.sComment
    if len(conditions) < 1:
        conditions = "No Description"
    else:
        conditions = conditions[0]
    
    ## if species is human then extract data from the patient_data.csv
    ## for mice data what to do
   
    ## Convert Start Time to Time Zone
    session_start_time = file_metadata["file"]["datetime"]

    if session_start_time.tzinfo is None:
        session_start_time = (
            session_start_time.replace(
                tzinfo=ZoneInfo("America/Toronto")
            )
        )

    tag_comments = file_metadata["ephys"]["comments"]
    tag_summary = "; ".join(tag_comments) if tag_comments else "No ABF tag comments available."

    # Create NWB file
    abf_nwbfile = NWBFile(
        session_description=(
            "Whole-cell intracellular electrophysiology "
            "recording."
        ),
        identifier=str(file_metadata["file"]["ID"]),
        session_start_time=session_start_time,
        experimenter=("Moradi Chameh, Homeira",),
        lab="Neuron to Brain Lab",
        institution="Krembil Research Institute",
        experiment_description=str(file_metadata["file"]["protocol"] or "Unspecified protocol"),
        session_id=os.path.splitext(filename)[0],
        keywords=(
            "whole-cell patch clamp",
            "intracellular electrophysiology",
            species_key,
        ),
        notes=f"ABF tag comments: {tag_summary}",
        pharmacology=(
            str(patient_metadata["medication"])
            if _clean_optional(patient_metadata.get("medication")) is not None
            else None
        ),
        surgery=(
            f"Tissue resection date: {patient_metadata['resection_date']}"
            if _clean_optional(patient_metadata.get("resection_date")) is not None
            else None
        ),
    )
    
    # Add subject info
    subject = subject_from_metadata(species_key, patient_metadata)
    abf_nwbfile.subject = subject

    # Determine the response and recorded-stimulus ADC channels once per ABF.
    channel_map = util_functions.find_file_level_channels(abf)
    response_channel = channel_map["response_channel"]
    stim_channel = channel_map["stim_channel"]

    if response_channel is None:
        raise ValueError(
            f"Could not identify the response channel for {filename}. "
            f"ADC names={abf.adcNames}, units={abf.adcUnits}"
        )

    response_channel = int(response_channel)
    adc_unit = str(abf.adcUnits[response_channel])

    # Determine clamp mode before entering the sweep loop.
    clamp_mode = util_functions.infer_clamp_mode(
        abf,
        response_channel=response_channel,
    )

    if clamp_mode == "unknown":
        raise ValueError(
            f"Could not determine clamp mode for {filename}"
        )

    # Add the amplifier/device with a non-empty description.
    try:
        device_name = str(abf._adcSection.sTelegraphInstrument[response_channel]).strip()
    except Exception:
        device_name = "Patch-clamp amplifier"
    if not device_name or "unknown" in device_name.casefold():
        device_name = "Patch-clamp amplifier"

    abf_device = abf_nwbfile.create_device(
        name=device_name,
        description="Patch-clamp amplifier connected to the selected response ADC channel.",
        manufacturer="Molecular Devices",
    )
    
    # Store the layer and a unique cell ID on the IntracellularElectrode.
    layer = (
        file_metadata["ephys"].get("layer")
        or patient_metadata.get("layer")
    )
    
    cell_id = file_metadata["ephys"].get("cell_id")
    if cell_id is None:
        cell_id = f"cell-{file_metadata['file']['ID']}"
        warnings.warn(
            f"No cell label was found in the ABF tags for {filename}; using {cell_id}. "
            "Add a tag such as C1 or Cell 1 if several ABF files belong to the same cell."
        )

    structure = _clean_optional(patient_metadata.get("structure")) or "unknown structure"
    tissue_type = _clean_optional(patient_metadata.get("tissue_type")) or "unknown tissue type"
    electrode_location = f"{structure}; layer={layer or 'unknown'}"

    abf_electrode_rec = abf_nwbfile.create_icephys_electrode(
        name=f"rec_{abf.adcNames[response_channel]}",
        description=(
            f"Whole-cell patch-clamp recording electrode for cell {cell_id}; "
            f"native response unit={adc_unit}."
        ),
        device=abf_device,
        location=electrode_location,
        slice=str(tissue_type),
        cell_id=str(cell_id),
    )
        
    simultaneous_recordings = []
    
    # Add sweeps. NWB times are relative to session_start_time, so each
    # sweep uses its ABF sweep start plus its within-sweep sample times.
    for i in map(int, abf.sweepList):
        try:
            sweep_start_time = float(abf.sweepTimesSec[i])
        except Exception:
            sweep_start_time = float(i) * float(abf.sweepLengthSec)

        abf.setSweep(
            i,
            channel=response_channel,
            absoluteTime=False,
        )

        relative_time = np.array(abf.sweepX, copy=True)
        response_data = np.array(abf.sweepY, copy=True)
        command_data = np.array(abf.sweepC, copy=True)
        abf_stimulus = None

        response_conversion, response_unit = nwb_conversion_from_unit(abf.sweepUnitsY)
        rate = float(file_metadata["acquisition"]["sample_rate"])
        response_description = (
            f"Recorded {clamp_mode.replace('_', ' ')} response for ABF sweep {i}; "
            f"source ADC channel {response_channel} ({abf.adcNames[response_channel]}), "
            f"native unit {abf.sweepUnitsY}."
        )

        response_common = dict(
            name=f"response_sweep_{i:04d}",
            data=response_data,
            unit=response_unit,
            conversion=response_conversion,
            starting_time=sweep_start_time,
            rate=rate,
            electrode=abf_electrode_rec,
            sweep_number=np.uint64(i),
            description=response_description,
            comments="Samples are stored in the original ABF display unit and converted to SI by conversion.",
        )

        if clamp_mode == "voltage_clamp":
            abf_response = VoltageClampSeries(
                **response_common,
                stimulus_description="Long Square voltage command",
            )
            stimulus_conversion, stimulus_unit = nwb_conversion_from_unit(abf.sweepUnitsC)
            abf_stimulus = VoltageClampStimulusSeries(
                name=f"stimulus_sweep_{i:04d}",
                data=command_data,
                unit=stimulus_unit,
                conversion=stimulus_conversion,
                starting_time=sweep_start_time,
                rate=rate,
                electrode=abf_electrode_rec,
                sweep_number=np.uint64(i),
                stimulus_description="Long Square voltage command",
                description=(
                    f"Ideal command waveform reconstructed by PyABF for sweep {i}; "
                    f"native unit {abf.sweepUnitsC}."
                ),
                comments="Command stimulus associated with the paired voltage-clamp response.",
            )

        elif clamp_mode == "current_clamp":
            abf_response = CurrentClampSeries(
                **response_common,
                stimulus_description="Long Square current command",
            )
            stimulus_conversion, stimulus_unit = nwb_conversion_from_unit(abf.sweepUnitsC)
            abf_stimulus = CurrentClampStimulusSeries(
                name=f"stimulus_sweep_{i:04d}",
                data=command_data,
                unit=stimulus_unit,
                conversion=stimulus_conversion,
                starting_time=sweep_start_time,
                rate=rate,
                electrode=abf_electrode_rec,
                sweep_number=np.uint64(i),
                stimulus_description="Long Square current command",
                description=(
                    f"Ideal command waveform reconstructed by PyABF for sweep {i}; "
                    f"native unit {abf.sweepUnitsC}."
                ),
                comments="Command stimulus associated with the paired current-clamp response.",
            )

        elif clamp_mode == "izero":
            abf_response = IZeroClampSeries(**response_common)

        else:
            raise ValueError(f"Unsupported clamp mode for {filename}: {clamp_mode}")

        abf_nwbfile.add_acquisition(abf_response)
        if abf_stimulus is not None:
            abf_nwbfile.add_stimulus(abf_stimulus)
            rec_idx = abf_nwbfile.add_intracellular_recording(
                electrode=abf_electrode_rec,
                stimulus=abf_stimulus,
                response=abf_response,
            )
        else:
            rec_idx = abf_nwbfile.add_intracellular_recording(
                electrode=abf_electrode_rec,
                response=abf_response,
            )

        recordings.append(rec_idx)
        sim_idx = abf_nwbfile.add_icephys_simultaneous_recording(recordings=[rec_idx])
        simultaneous_recordings.append(sim_idx)

        # Epoch times must be session-relative. Add each sweep's start time so
        # the interval start times are not repeated for every sweep.
        if abf_stimulus is not None:
            stim_start, stim_end, stim_amp, stim_mode = util_functions.stim_range(
                command_data,
                relative_time,
            )
            if stim_start is not None:
                abf_nwbfile.add_epoch(
                    start_time=sweep_start_time + float(stim_start),
                    stop_time=sweep_start_time + float(stim_end),
                    tags=[f"sweep_{i}", "stim", str(stim_mode)],
                    timeseries=[abf_response, abf_stimulus],
                )
            
    # Group recordings   
    if simultaneous_recordings:
        abf_nwbfile.add_icephys_sequential_recording(
            simultaneous_recordings=simultaneous_recordings,
            stimulus_type="Long Square",
        )

    # Preserve the complete normalized metadata and its extraction sources in
    # a machine-readable record. Exclude the absolute local ABF path.
    safe_file_metadata = dict(file_metadata["file"])
    safe_file_metadata.pop("file_path", None)
    conversion_metadata = {
        "source_file": safe_file_metadata,
        "acquisition": file_metadata["acquisition"],
        "channels": file_metadata["channels"],
        "ephys": file_metadata["ephys"],
        "subject": patient_metadata,
        "selected_channels": {
            "response_adc_channel": response_channel,
            "recorded_stimulus_adc_channel": stim_channel,
            "clamp_mode": clamp_mode,
        },
    }
    
    abf_nwbfile.add_scratch(
        json.dumps(conversion_metadata, sort_keys=True, default=str),
        name="conversion_metadata",
        description=(
            "Normalized subject, tissue, ABF tag, cortical-layer, cell-ID, "
            "acquisition, and channel-selection metadata used during conversion."
        ),
    )
    
    # Write file
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(abf_nwbfile)

    print(f"Saved NWB: {nwb_path}")
    return Path(nwb_path)

    
for dirpath, dirnames, filenames in os.walk(input_abf_root):
    for filename in filenames:
        if not filename.lower().endswith(".abf"):
            continue
        
        try:
            convert_abf_to_nwb(dirpath, filename, species)
            converted += 1
            print("Success!")
        except Exception as e:
            failures.append((os.path.join(dirpath, filename), e))
            print(f"Failed to convert {filename}: {e}", file=sys.stderr)
            
        if failures:
            print(f"{len(failures)} conversion(s) failed:", file=sys.stderr)
            for path, exc in failures:
                print(f"- {path}: {exc}", file=sys.stderr)
        