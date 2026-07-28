# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# General Helper Functions
import util_functions

# Import python ABF library
import pyabf
import numpy as np
import os
from zoneinfo import ZoneInfo

# Dataframe for opening CSVs
import pandas as pd

# Import I/O class used for reading and writing NWB files
# Import main NWB file class
from pynwb import NWBHDF5IO, NWBFile

# Import additional core datatypes used in the example
from pynwb.file import Subject

# Import icephys TimeSeries types used
from pynwb.icephys import (
    VoltageClampSeries,
    VoltageClampStimulusSeries,
    CurrentClampSeries,
    CurrentClampStimulusSeries,
    IZeroClampSeries,
)

# Analyze ABF file
current_dir = os.getcwd()

species = "human"

input_abf_root = os.path.join(current_dir, "abfData")
output_nwb_root = os.path.join(current_dir, "nwbData")
patient_data_path = os.path.join(current_dir, "patientData", "patientData.csv")

# V_CLAMP_MODE = 0
# I_CLAMP_MODE = 1
# I0_CLAMP_MODE = 2

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

def get_metadata(abf):
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

def convert_abf_to_nwb(dirpath, filename, species = "human"):
    '''
    Species = "human" or "mice" -- Metadata comes from different places
    
    '''
    
    print(f"\nConverting {filename}")
    recordings = []
    metadata = {}
    
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
    metadata = get_metadata(abf)

    # Get comments/conditions
    conditions = abf._tagSection.sComment
    if len(conditions) < 1:
        conditions = "No Description"
    else:
        conditions = conditions[0]
    
    ## if species is human then extract data from the patient_data.csv
    if species == "human":
        patient_date_db, csv_missing, tissue_type = util_functions.patient_data_parse(
            metadata['file']['datetimestr'],
            patient_data_path,
            metadata['file']['protocol']
        )
    
        if csv_missing:
            dates_missing.add(csv_missing)
    
        has_patient_info = not patient_date_db.empty
        structure, sex, age, tissue_type, tumour = util_functions.patient_data_extract(patient_date_db)

        description = (
            f"{conditions}; "
            f"tissue type: {tissue_type}; "
            f"tumour status: {tumour}; "
            f"structure: {structure}"
        )
    ## for mice data what to do
    else:
        mouse_meta = util_functions.parse_mouse_metadata_from_path(dirpath, input_abf_root)

        has_patient_info = True
    
        structure = mouse_meta["structure"]
        condition = mouse_meta["condition"]
        model = mouse_meta["model"]
        cell_type = mouse_meta["cell_type"]
        tissue_type = mouse_meta["tissue_type"]
    
        sex = "unknown"
        age = "unknown"
        
        description = f"{condition}; tissue type: {tissue_type}; structure: {structure}"
        
    ## Convert Start Time to Time Zone
    session_start_time = metadata["file"]["datetime"]

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
        identifier=str(metadata["file"]["ID"]),
        session_start_time=session_start_time,
        experimenter="Homeira Moradi Chameh",
        lab="Neuron to Brain Lab",
        institution="Krembil Research Institute",
        experiment_description=str(metadata["file"]["protocol"]),
        session_id=os.path.splitext(filename)[0],
    )

    # Add device
    abf_device = abf_nwbfile.create_device(
        name=abf._adcSection.sTelegraphInstrument[0]
    )

    # Add electrodes
    adc_unit = metadata["channels"]["units"][0]

    abf_electrode_rec = abf_nwbfile.create_icephys_electrode(
        name="rec_" + metadata["channels"]["names"][0],
        description="recording electrode in " + adc_unit,
        device=abf_device,
    )

    dac_unit = None

    if metadata["channels"]["count"] > 1:
        dac_unit = metadata["channels"]["units"][1]

    # Add subject info
    if has_patient_info:
        if species == "human":
            subject_species = "Homo sapiens"
            subject_description = (
                f"{conditions}; tissue type: {tissue_type}; structure: {structure}"
            )
    
        else:
            subject_species = "Mus musculus"
            subject_description = (
                f"{conditions}; "
                f"structure: {structure}; "
                f"condition: {condition}; "
                f"model: {model}; "
                # f"cell type: {cell_type}; "
                f"tissue type: {tissue_type}"
            )
            
        subject = Subject(
            subject_id=abf.abfDateTimeString,
            age=str(age),
            description=subject_description,
            species=subject_species,
            sex=str(sex),
        )
    
        abf_nwbfile.subject = subject

    # Add sweep table
    # clamp_mode = abf._adcSection.nTelegraphMode[0]
    # abf_nwbfile.sweep_table = SweepTable()

    # Add sweeps
    for i in abf.sweepList:
        rec_idx = None
        abf_stimulus = None

        abf.setSweep(i)
        clamp_mode = util_functions.infer_clamp_mode(abf)
        
        if clamp_mode == "unknown":
            raise ValueError(
                f"Could not determine clamp mode for {filename}: "
                f"Y={abf.sweepUnitsY}, C={abf.sweepUnitsC}"
            )

        dataX = abf.sweepX
        dataY = abf.sweepY
        dataStim = abf.sweepC
        
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
                rate=float(metadata["acquisition"]["sample_rate"]),
                electrode=abf_electrode_rec,
                gain=abf._dataGain[0],
                sweep_number=np.uint64(i),
                stimulus_description="Long Square",
            )

            abf_nwbfile.add_acquisition(abf_response)

            if metadata["channels"]["count"] > 1:
                stim_name = f"Index_0_{i}_1 {dac_unit}"

                abf_stimulus = VoltageClampStimulusSeries(
                    name=stim_name,
                    data=dataStim,
                    unit=stimulus_unit,
                    conversion=stimulus_conversion,
                    starting_time=float(dataX[0]),
                    rate=float(metadata["acquisition"]["sample_rate"]),
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
                rate=float(metadata["acquisition"]["sample_rate"]),
                electrode=abf_electrode_rec,
                gain=abf._dataGain[0],
                sweep_number=np.uint64(i),
                stimulus_description="Long Square",
            )

            abf_nwbfile.add_acquisition(abf_response)

            if metadata["channels"]["count"] > 1:
                stim_name = f"Index_0_{i}_1 {dac_unit}"

                abf_stimulus = CurrentClampStimulusSeries(
                    name=stim_name,
                    data=dataStim,
                    unit=stimulus_unit,
                    conversion=stimulus_conversion,
                    starting_time=float(dataX[0]),
                    rate=float(metadata["acquisition"]["sample_rate"]),
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
                rate=float(metadata["acquisition"]["sample_rate"]),
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
        
        
        