
# From package import all functions in functions.py
import sys
import os
import json

from tkinter import messagebox
from src.backend.thermalCamFunctions.ROI_Viz_Functions import overlay_rois_from_csv
# import src.backend.thermalCamFunctions.ThermalCamProcessing.Main_Functions as TCMainFunctions

from src.settings import absolute_path
sys.path.append(absolute_path)
# from ROI_Viz_Functions import overlay_rois_from_csv # TODO Works but the code needs changes
import Main_Functions as TCMainFunctions

def process_data(config_path: str):
    # Check that there is a config path 
    # if not os.path.isfile(config_path):
    #     return False

    # print(f'config path: {config_path}')
    # with open(config_path, 'r') as file:
    #     config = json.load(file)

    config = load_json(config_path)
    if config == False:
        return

    data_type = config['data']

    if data_type == 'image':
        # Initializing step
        Aux_Met_Data, FLUX_Met_Data, roi_masks, average_distances, Aux_Met_window, FLUX_Met_window, base_folder, output_csv_path, emissivity, elevation = TCMainFunctions.initialize_data_from_config_image(config_path)

        # Process images in the specified folder and obtain the resulting DataFrame
        TCMainFunctions.setup_gui_and_start(
            base_folder,               # Path to the folder containing thermal images
            roi_masks,                 # ROI masks for image processing
            average_distances,         # Average distances for ROIs
            Aux_Met_Data,              # Auxiliary meteorological data
            FLUX_Met_Data,             # FLUX meteorological data (optional)
            Aux_Met_window,            # Time window for auxiliary data matching
            FLUX_Met_window,           # Time window for FLUX data matching (optional)
            output_csv_path,           # Path to store final processed CSV file
            emissivity,                # Emissivity value
            elevation
        )
    elif data_type == 'point':
        # Initializing step
        Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window, point_dist, output_csv_path, emissivity, point_data_path, elevation = TCMainFunctions.initialize_data_from_config_point(config_path)
        
        # Process point data
        TCMainFunctions.setup_gui_and_start_point(
            point_data_path,            # Path to the CSV file containing point data
            point_dist,                 # Distance value for the point data
            Aux_Met_Data,               # Auxiliary meteorological data
            FLUX_Met_Data,              # FLUX meteorological data (optional)
            Aux_Met_window,             # Time window for auxiliary data matching
            FLUX_Met_window,            # Time window for FLUX data matching (optional)
            output_csv_path,            # Path to store final processed CSV file
            emissivity,                 # Emissivity value
            elevation
        )
    else:
        print(f"Unknown data type: {data_type}")

def open_overlay(config_path):
    if not os.path.isfile(config_path):
        messagebox.showerror("Error", "Config file not found.\n" \
        "Please load a config file in JSON format above.")
        return
    config = load_json(config_path)
    if config == False:
        return
    
    # def overlay_rois_from_csv(image_path, csv_path, output_image_path=None)
    overlay_rois_from_csv(config['first_image_path'], config['roi_path'])
    

def load_json(config_path):
    print(f'Config path: {config_path}')
    if not os.path.isfile(config_path):
        return False

    with open(config_path, 'r') as file:
        config = json.load(file)
    return config
