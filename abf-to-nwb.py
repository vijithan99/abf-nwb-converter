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

species = "human"

input_abf_root = os.path.join(current_dir, "abfData")
output_nwb_root = os.path.join(current_dir, "nwbData")
patient_data_path = os.path.join(current_dir, "patientData", "patientDataFormatted.csv")

TORONTO_TZ = ZoneInfo("America/Toronto")
PAPER_DOI = "https://doi.org/10.1093/gigascience/giac108"

VOLTAGE_UNITS = {"v", "mv", "uv"}
CURRENT_UNITS = {"a", "ma", "ua", "na", "pa"}

# Cell Data missing Metadata
dates_missing = set()

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
        for key, label in (("diagnosis", "diagnosis"), ("tumour", "tumour status")):
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
    abf_path = os.path.join(dirpath, filename)
    abf = pyabf.ABF(abf_path)
    
    relative_path = os.path.relpath(dirpath, input_abf_root)
    nwb_output_dir = os.path.join(output_nwb_root, relative_path)
    os.makedirs(nwb_output_dir, exist_ok=True)
    
    nwb_path = os.path.join(
        nwb_output_dir,
        os.path.splitext(filename)[0] + ".nwb"
    )
    
    # Get ABF metadata
    file_metadata = md.get_file_metadata(abf)
    patient_metadata = md.get_patient_metadata(filename, Path(patient_data_path), False)

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
        
    # Create NWB file
    abf_nwbfile = NWBFile(
        session_description=(
            "Whole-cell intracellular electrophysiology "
            "recording."
        ),
        identifier=str(file_metadata["file"]["ID"]),
        session_start_time=session_start_time,
        experimenter="Homeira Moradi Chameh",
        lab="Neuron to Brain Lab",
        institution="Krembil Research Institute",
        experiment_description=str(file_metadata["file"]["protocol"]),
        session_id=os.path.splitext(filename)[0],
    )

    # Add device
    abf_device = abf_nwbfile.create_device(
        name=abf._adcSection.sTelegraphInstrument[0]
    )

    # Add electrodes
    adc_unit = file_metadata["channels"]["units"][0]

    abf_electrode_rec = abf_nwbfile.create_icephys_electrode(
        name="rec_" + file_metadata["channels"]["names"][0],
        description="recording electrode in " + adc_unit,
        device=abf_device,
    )

    dac_unit = None

    if file_metadata["channels"]["count"] > 1:
        dac_unit = file_metadata["channels"]["units"][1]

    # Add subject info
    subject = subject_from_metadata("human", patient_metadata)
    abf_nwbfile.subject = subject

    # Add sweep table
    # clamp_mode = abf._adcSection.nTelegraphMode[0]
    # abf_nwbfile.sweep_table = SweepTable()
    
    # Determine the response and recorded-stimulus ADC channels once per ABF.
    channel_map = util_functions.find_file_level_channels(abf)
    
    response_channel = channel_map["response_channel"]
    stim_channel = channel_map["stim_channel"]
    
    if response_channel is None:
        raise ValueError(
            f"Could not identify the response channel for {filename}. "
            f"ADC names={abf.adcNames}, units={abf.adcUnits}"
        )
    
    # Determine clamp mode before entering the sweep loop.
    clamp_mode = util_functions.infer_clamp_mode(
        abf,
        response_channel=response_channel,
    )

    if clamp_mode == "unknown":
        raise ValueError(
            f"Could not determine clamp mode for {filename}"
        )

    # Add sweeps
    for i in abf.sweepList:
        # This must occur after infer_clamp_mode(), because that function
        # currently changes the active ABF sweep to sweep 0.
        abf.setSweep(
            i,
            channel=response_channel,
            absoluteTime=True,
        )
        
        dataX = np.array(abf.sweepX, copy=True)
        dataY = np.array(abf.sweepY, copy=True)
        dataStim = np.array(abf.sweepC, copy=True)
        
        ## Convert the values to the units
        response_conversion, response_unit = nwb_conversion_from_unit(
            abf.sweepUnitsY
        )
        
        stimulus_conversion, stimulus_unit = nwb_conversion_from_unit(
            abf.sweepUnitsC
        )

        if clamp_mode == "voltage_clamp":
            resp_name = f"Index_0_{i}_0 {adc_unit}"
            
            
            abf_response = VoltageClampSeries(
                name=resp_name,
                data=dataY,
                unit=response_unit,
                conversion=response_conversion,
                starting_time=float(dataX[0]),
                rate=float(file_metadata["acquisition"]["sample_rate"]),
                electrode=abf_electrode_rec,
                gain=abf._dataGain[0],
                sweep_number=np.uint64(i),
                stimulus_description="Long Square",
            )

            abf_nwbfile.add_acquisition(abf_response)

            if file_metadata["channels"]["count"] > 1:
                stim_name = f"Index_0_{i}_1 {dac_unit}"

                abf_stimulus = VoltageClampStimulusSeries(
                    name=stim_name,
                    data=dataStim,
                    unit=stimulus_unit,
                    conversion=stimulus_conversion,
                    starting_time=float(dataX[0]),
                    rate=float(file_metadata["acquisition"]["sample_rate"]),
                    electrode=abf_electrode_rec,
                    gain=abf._dataGain[1],
                    sweep_number=np.uint64(i),
                    stimulus_description="Long Square",
                )

                abf_nwbfile.add_stimulus(abf_stimulus)

        elif clamp_mode == "current_clamp":
            resp_name = f"Index_0_{i}_0 {adc_unit}"

            abf_response = CurrentClampSeries(
                name=resp_name,
                data=dataY,
                unit=response_unit,
                conversion=response_conversion,
                starting_time=float(dataX[0]),
                rate=float(file_metadata["acquisition"]["sample_rate"]),
                electrode=abf_electrode_rec,
                gain=abf._dataGain[0],
                sweep_number=np.uint64(i),
                stimulus_description="Long Square",
            )

            abf_nwbfile.add_acquisition(abf_response)

            if file_metadata["channels"]["count"] > 1:
                stim_name = f"Index_0_{i}_1 {dac_unit}"

                abf_stimulus = CurrentClampStimulusSeries(
                    name=stim_name,
                    data=dataStim,
                    unit=stimulus_unit,
                    conversion=stimulus_conversion,
                    starting_time=float(dataX[0]),
                    rate=float(file_metadata["acquisition"]["sample_rate"]),
                    electrode=abf_electrode_rec,
                    gain=abf._dataGain[1],
                    sweep_number=np.uint64(i),
                    stimulus_description="Long Square",
                )

                abf_nwbfile.add_stimulus(abf_stimulus)

        elif clamp_mode == "izero":
            resp_name = f"Index_0_{i}_0 {adc_unit}"

            abf_response = IZeroClampSeries(
                name=resp_name,
                data=dataY,
                unit=response_unit,
                conversion=response_conversion,
                starting_time=float(dataX[0]),
                rate=float(file_metadata["acquisition"]["sample_rate"]),
                electrode=abf_electrode_rec,
                gain=abf._dataGain[0],
                sweep_number=np.uint64(i),
            )

            abf_nwbfile.add_acquisition(abf_response)
        
        else:
            raise ValueError(
                f"Unsupported clamp mode for {filename}: {clamp_mode}"
            )
            
        # Add intracellular recording
        if abf_stimulus is not None:
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

        # # Add sweep table entries
        # abf_nwbfile.sweep_table.add_entry(abf_response)

        # if abf_stimulus is not None:
        #     abf_nwbfile.sweep_table.add_entry(abf_stimulus)

        # Add stimulus epoch
        stim_start, stim_end, stim_amp, stim_mode = util_functions.stim_range(
            dataStim,
            dataX,
        )

        if stim_start is not None:
            abf_nwbfile.add_epoch(
                start_time=float(stim_start),
                stop_time=float(stim_end),
                tags=[f"sweep_{i}", "stim", stim_mode],
                timeseries=[abf_response],
            )

    # Group recordings
    if len(recordings) > 0:
        abf_nwbfile.add_icephys_simultaneous_recording(recordings=recordings)

    # Write file
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(abf_nwbfile)

    print(f"Saved NWB: {nwb_path}")
    
for dirpath, dirnames, filenames in os.walk(input_abf_root):
    for filename in filenames:
        if not filename.lower().endswith(".abf"):
            continue
        
        try:
            convert_abf_to_nwb(dirpath, filename, species)
            print("Success!")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
        
        
        