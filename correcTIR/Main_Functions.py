# =============================================
# LIBRARY IMPORTS
# =============================================
# Standard library imports
import os
import math
import json
from datetime import datetime
import csv
import time
import threading
import warnings

# Third-party package imports (ensure these are installed)
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
from PIL import Image  # Image handling
import cv2  # Computer vision processing
import tkinter as tk  # GUI components
from tkinter import messagebox
from tkinter import ttk  # Themed widgets for Tkinter
from collections import OrderedDict

# =============================================
# CONFIGURATION HANDLING
# =============================================
def load_json(config_path):
    """Load and return configuration settings from a JSON file."""
    print(f'Loading config from: {config_path}')
    if not os.path.isfile(config_path):
        print(f"Error: Config file not found at {config_path}")
        return None

    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# =============================================
# PROCESSING PIPELINE
# =============================================
def process_data(config_path):
    """Runs the processing pipeline based on data type (image or point)."""
    config = load_json(config_path)
    if config is None:
        return

    data_type = config.get('data', None)

    if data_type == 'image':
        print("Processing data...")
        Aux_Met_Data, aux_met_window, FLUX_Met_Data, flux_met_window, sky_percent, emissivity_vf2, emissivity_target, elevation, win_transmittance, roi_masks, average_distances, base_folder, output_csv_path = initialize_data_from_config_image(config_path)

        setup_gui_and_start(
            Aux_Met_Data, aux_met_window, FLUX_Met_Data, flux_met_window, sky_percent, emissivity_vf2, emissivity_target, elevation, win_transmittance, roi_masks, average_distances, base_folder, output_csv_path)

    elif data_type == 'point':
        print("Processing data...")
        Aux_Met_Data, aux_met_window, FLUX_Met_Data, flux_met_window, sky_percent, emissivity_vf2, emissivity_target, elevation, win_transmittance, point_data_path, point_dist, output_csv_path = initialize_data_from_config_point(config_path)

        setup_gui_and_start_point(
            Aux_Met_Data, aux_met_window, FLUX_Met_Data, flux_met_window, sky_percent, emissivity_vf2,
            emissivity_target, elevation, win_transmittance, point_data_path, point_dist, output_csv_path
        )

    else:
        print(f"Unknown data type (must be image or point): {data_type}")

def run_pipeline(config_path):
    """Runs the entire pipeline including data processing and optional ROI visualization."""
    process_data(config_path)

    print("\nProcessing complete.")

def start_processing_thread(*args):
    """
    Starts a separate thread for processing images.

    This function initializes a new thread targeting the 'process_images_in_folders'
    function and passes any arguments it receives to this target function. The thread
    is then started to run parallel to the main program, allowing for asynchronous processing.

    Parameters:
    *args: Variable length argument list to be passed to the 'process_images_in_folders' function.
    """
    processing_thread = threading.Thread(target=process_images_in_folders, args=args)
    processing_thread.start()

def setup_gui_and_start(Aux_Met_Data, aux_met_window, FLUX_Met_Data, flux_met_window, sky_percent, emissivity_vf2, emissivity_target, elevation, win_transmittance, roi_masks, average_distances, base_folder, output_csv_path):
    """
    Sets up the graphical user interface (GUI) and starts the image processing in a separate thread.

    This function initializes the GUI for the image processing application. It creates a
    main window with a progress bar and then starts the image processing in a separate thread
    using the provided parameters. The GUI provides visual feedback on the progress of the
    image processing.

    Parameters:
    Aux_Met_Data (pd.DataFrame): DataFrame containing preprocessed auxiliary meteorological data.
    aux_met_window (int): Time window in minutes for searching auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing preprocessed FLUX meteorological data.
    flux_met_window (int): Time window in minutes for searching FLUX meteorological data.
    sky_percent (int): Percent of target's view factor that is composed of sky.
    emissivity_vf2 (float): Emissivity of the dominant surrounding object other than sky.
    emissivity_target (float): Emissivity value from the configuration.
    elevation (float): Site elevation.
    win_transmittance (float): The transmittance value of the enclosure window.
    roi_masks (dict): A dictionary of Region of Interest (ROI) masks.
    average_distances (dict): A dictionary of average distances for each ROI.
    base_folder (str): The base folder path where the images are located.
    output_csv_path (str): Path to the output CSV file where the results will be saved.
    """
    root = tk.Tk()
    root.title("Processing Images")

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")

    # Function to close the window when processing is complete
    def close_window():
        root.destroy()
        messagebox.showinfo("Image", "Image processing complete!")

    # Check if the progress bar is complete and close the window
    def check_progress():
        if progress_bar['value'] >= progress_bar['maximum']:
            status_label.config(text="Processing complete.")
            root.after(1000, close_window)
        else:
            root.after(500, check_progress)

    root.after(500, check_progress)
    progress_bar.pack(pady=10)

    # Label for showing the processing status
    status_label = tk.Label(root, text="Starting...")
    status_label.pack(pady=10)

    # Start the processing in a separate thread
    start_args = (Aux_Met_Data, aux_met_window, FLUX_Met_Data, flux_met_window, sky_percent, emissivity_vf2, emissivity_target, elevation, win_transmittance, roi_masks, average_distances, base_folder, output_csv_path, progress_bar, status_label, root)
    start_processing_thread(*start_args)

    root.mainloop()

def setup_gui_and_start_point(Aux_Met_Data, aux_met_window, FLUX_Met_Data, flux_met_window, sky_percent, emissivity_vf2, emissivity_target, elevation, win_transmittance, point_data_path, point_dist, output_csv_path):
    """
    Sets up the graphical user interface (GUI) and starts the point data processing in a separate thread.

    This function initializes the GUI for the point data processing application. It creates a
    main window with a progress bar and then starts the point data processing in a separate thread
    using the provided parameters. The GUI provides visual feedback on the progress of the
    point data processing.

    Parameters:
    Aux_Met_Data (pd.DataFrame): DataFrame containing preprocessed auxiliary meteorological data.
    aux_met_window (int): Time window in minutes for searching auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing preprocessed FLUX meteorological data.
    flux_met_window (int): Time window in minutes for searching FLUX meteorological data.
    sky_percent (int): Percent of target's view factor that is composed of sky.
    emissivity_vf2 (float): Emissivity of the dominant surrounding object other than sky.
    emissivity_target (float): Emissivity value from the configuration.
    elevation (float): Site elevation.
    win_transmittance (float): The transmittance value of the enclosure window.
    point_data_path (str): Path to the CSV file containing point data.
    point_dist (float): Distance value for the point data.
    output_csv_path (str): Path to the output CSV file where the results will be saved.
    """
    root = tk.Tk()
    root.title("Processing Point Data")

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=10)

    # Label for showing the processing status
    status_label = tk.Label(root, text="Starting...")
    status_label.pack(pady=10)

    # Load point data to determine the number of rows
    point_data = pd.read_csv(point_data_path)
    total_rows = len(point_data)

    # Set progress bar maximum value
    progress_bar['maximum'] = total_rows

    # Function to update progress
    def update_progress(processed_df):
        result_df = processed_df
        if not result_df.empty:
            result_df.to_csv(output_csv_path, index=False)
        status_label.config(text="Processing complete.")
        root.quit()

    # Function to process point data and update the progress bar
    def process_and_update_progress():
        try:
            processed_df = process_and_export_corrected_point_data(Aux_Met_Data, aux_met_window, FLUX_Met_Data, flux_met_window, sky_percent, emissivity_vf2, emissivity_target, elevation, win_transmittance, point_data_path, point_dist, output_csv_path)
            update_progress(processed_df)

        except Exception as e:
            print(f"Error during processing: {e}")
            status_label.config(text="Processing failed.")
            root.quit()

    # Start the processing in a separate thread
    processing_thread = threading.Thread(target=process_and_update_progress)
    processing_thread.start()

    # Periodically update the progress bar
    def update_progress_bar():
        processed_rows = min(progress_bar['value'] + 1, total_rows)
        progress_bar['value'] = processed_rows
        status_label.config(text=f"Processing row {processed_rows} of {total_rows}")
        root.update_idletasks()
        if progress_bar['value'] < total_rows:
            root.after(100, update_progress_bar)

    update_progress_bar()

    # Function to close the window when processing is complete
    def close_window():
        root.destroy()
        messagebox.showinfo("Point Data", "Point data processing complete!")

    # Check if the progress bar is complete and close the window
    def check_progress():
        if progress_bar['value'] >= progress_bar['maximum']:
            status_label.config(text="Processing complete.")
            root.after(1000, close_window)
        else:
            root.after(500, check_progress)

    root.after(500, check_progress)

    root.mainloop()

# =============================================
# IMAGE PROCESSING FUNCTIONS (ROI FUNCTIONS)
# =============================================
def initialize_data_from_config_image(config_path):
    """
    Initialize data for the application from a configuration file.

    This function reads configuration settings from a JSON file and uses them to
    load and preprocess auxiliary and FLUX meteorological data, initialize ROI masks,
    and calculate average distances.

    Parameters:
    config_path (str): The file path to the configuration JSON file.

    Returns:
    tuple: A tuple containing:
        - Aux_Met_Data (DataFrame or None): Preprocessed auxiliary meteorological data.
        - aux_met_window (int): Time window for auxiliary meteorological data.
        - FLUX_Met_Data (DataFrame or None): Preprocessed FLUX meteorological data.
        - flux_met_window (int): Time window for FLUX meteorological data.
        - sky_percent (int): Percent of target's view factor that is composed of sky.
        - emissivity_vf2 (float): Emissivity of the dominant surrounding object other than sky.
        - emissivity_target (float): Emissivity value from the configuration.
        - elevation (float): Site elevation.
        - win_transmittance (float): The transmittance value of the enclosure window.
        - roi_masks (dict): ROI masks initialized from the configuration.
        - average_distances (dict): Average distances for each ROI.
        - base_folder (str): Base folder path from the configuration.
        - output_csv_path (str): Output CSV file path from the configuration.
    """
    print("Initializing image-based process...")

    # Load the configuration from the JSON file
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Extract and validate config values
    sky_percent = config.get('sky_percent')
    emissivity_vf2 = config.get('emissivity_vf2')
    emissivity_target = config.get('emissivity_target')
    win_transmittance = config.get('win_transmittance')

    # Validation
    if sky_percent is None or not (0 <= sky_percent <= 100):
        raise ValueError(f"Invalid 'sky_percent': {sky_percent}. It must be between 0 and 100.")

    if emissivity_vf2 is None or not (0 <= emissivity_vf2 <= 1):
        raise ValueError(f"Invalid 'emissivity_vf2': {emissivity_vf2}. It must be between 0 and 1.")

    if emissivity_target is None or not (0 <= emissivity_target <= 1):
        raise ValueError(f"Invalid 'emissivity_target': {emissivity_target}. It must be between 0 and 1.")

    if win_transmittance is None or not (0 <= win_transmittance <= 1):
        raise ValueError(f"Invalid 'win_transmittance': {win_transmittance}. It must be between 0 and 1.")

    # Define necessary columns for auxiliary meteorological data
    aux_columns = ["TIMESTAMP_END", "T_air", "RH", "LW_IN", "VF_2", "T_win"]

    # Conditionally exclude columns
    if sky_percent == 100 or emissivity_vf2 == 1:
        aux_columns = [col for col in aux_columns if col != "VF_2"]
    if win_transmittance == 1:
        aux_columns = [col for col in aux_columns if col != "T_win"]

    # FLUX dataset columns
    flux_columns = ["TIMESTAMP_END", "LW_IN"]

    # Load and preprocess auxiliary meteorological data
    Aux_Met_Data = None
    if config.get('aux_met_data_path'):
        Aux_Met_Data = pd.read_csv(config['aux_met_data_path'], usecols=lambda x: x in aux_columns)
        Aux_Met_Data = preprocess_dataframe(Aux_Met_Data)

    # Load FLUX_Met_Data **only if LW_IN is missing from Aux_Met_Data
    FLUX_Met_Data = None
    if config.get('flux_met_data_path') and (Aux_Met_Data is None or 'LW_IN' not in Aux_Met_Data.columns):
        FLUX_Met_Data = pd.read_csv(config['flux_met_data_path'], usecols=lambda x: x in flux_columns)
        FLUX_Met_Data = preprocess_dataframe(FLUX_Met_Data)
    else:
        print("Skipping FLUX_Met_Data as LW_IN is already present in Aux_Met_Data.")

    # Initialize ROI masks and distances
    roi_masks, average_distances = initialize_roi_masks_and_distances(
        config['roi_path'], config['roi_dist_path'], config['first_image_path'], config['img_dist_type'])

    return (
        Aux_Met_Data,
        config.get('aux_met_window'),
        FLUX_Met_Data,
        config.get('flux_met_window'),
        sky_percent,
        emissivity_vf2,
        config.get('emissivity_target'),
        config.get('elevation'),
        win_transmittance,
        roi_masks,
        average_distances,
        config.get('base_folder'),
        config.get('output_csv_path')
    )

def get_image_shape(image_path):
    """
    Get the dimensions of the image.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    tuple: A tuple representing the shape of the image in the form (height, width).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}. Ensure the file is valid and in a supported format.")
   
    return image.shape[:2]  # Return only height and width

def read_and_convert_image(image_path):
    """
    Read an image and convert it to 8-bit if it's 32-bit.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    np.ndarray: Image array suitable for processing with OpenCV.
    """
    with Image.open(image_path) as img:
        if img.mode == 'I':  # This mode corresponds to 32-bit unsigned integer pixels
            # Convert to 16-bit
            img = img.convert('I;16')
        
        # Convert Pillow image to OpenCV format
        return np.array(img)

def create_roi_masks(csv_path, image_shape, image_path):
    """
    Create a dictionary with masks for each ROI based on the CSV definitions.
    Converts the image to a compatible format if it's a 32-bit TIFF image.

    Parameters:
    csv_path (str): Path to the CSV file containing ROI definitions.
    image_shape (tuple): Shape of the image (height, width).
    image_path (str): Path to the image file.

    Returns:
    dict: A dictionary with ROI labels as keys and corresponding masks as values.
    """
    # Read and convert the image if necessary
    image = read_and_convert_image(image_path)

    # Ensure the image shape matches the expected shape
    if image.shape[:2] != image_shape:
        raise ValueError("The shape of the loaded image does not match the expected shape.")

    rois = {}
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        for row in reader:
            label = row[0]
            points_data = row[1:]
            
            # Convert string points to integers, filtering out any non-numeric values
            points = [(int(points_data[i]), int(points_data[i+1]))
                      for i in range(0, len(points_data), 2)
                      if points_data[i].isdigit() and points_data[i+1].isdigit()]

            if len(points) < 3:  # Check if there are enough points to form a polygon
                continue  # Skip this iteration if not enough points

            # Create a mask for this ROI
            mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(points, np.int32)], color=1)

            # Store the mask with the ROI label as the key
            rois[label] = mask.astype(bool)  # Convert to boolean mask
    return rois

def calculate_average_distances(distance_csv, roi_masks=None, data_type='pixeldistance'):
    """
    Calculate or read the average distance for each ROI.

    Parameters:
    distance_csv (str): Path to the CSV file containing distances.
    roi_masks (dict, optional): A dictionary containing masks for each ROI (used if data_type is 'pixeldistance').
    data_type (str): Type of data in the CSV file ('pixeldistance' or 'average').

    Returns:
    dict: A dictionary with ROI labels as keys and their average distances as values.
    """
    if data_type == 'pixeldistance':
        # Load the pixel distances from the CSV file
        pixel_distances = np.loadtxt(distance_csv, delimiter=',', encoding='utf-8-sig')
        
        # Calculate the average distance for each ROI
        average_distances = {}
        for label, mask in roi_masks.items():
            # Apply the mask to the distances array
            masked_distances = pixel_distances[mask]

            # Calculate the average distance within this ROI
            average_distance = np.mean(masked_distances)
            
            # Store the average distance with the ROI label as the key
            average_distances[label] = average_distance

    elif data_type == 'average':
        # Read the CSV file with label and average distance
        df = pd.read_csv(distance_csv)
        # Convert the DataFrame to a dictionary
        average_distances = df.set_index('label')['average_distance'].to_dict()

    else:
        raise ValueError("data_type must be either 'pixeldistance' or 'average'.")

    return average_distances

def initialize_roi_masks_and_distances(roi_csv_path, distance_csv_path, image_path, data_type='pixeldistance'):
    """
    Initialize ROI masks and calculate or read the average distances.

    Parameters:
    roi_csv_path (str): Path to the CSV file containing ROI information.
    distance_csv_path (str): Path to the CSV file containing distances.
    image_path (str): Path to the image file.
    data_type (str): Type of data in the distance CSV file ('pixeldistance' or 'average').

    Returns:
    tuple: A tuple containing ROI masks and average distances.
    """
    # Load the image shape
    image_shape = get_image_shape(image_path)
    
    # Create the ROI masks
    roi_masks = create_roi_masks(roi_csv_path, image_shape, image_path)
    
    # Calculate the average distances
    average_distances = calculate_average_distances(distance_csv_path, roi_masks, data_type)
    
    return roi_masks, average_distances

def calculate_roi_means_for_tiff(tiff_image_path, roi_masks):
    """
    Calculate the mean and the 1st, 5th, 10th, 25th, 50th, 75th, 90th, 95th, and 99th percentiles 
    for each ROI in a TIFF image.

    Parameters:
    tiff_image_path (str): Path to the .tiff image file.
    roi_masks (dict): A dictionary containing masks for each ROI.

    Returns:
    dict: A dictionary with ROI labels as keys and another dictionary with 
          percentiles and mean as values.
    """
    # Open the .tiff file
    with Image.open(tiff_image_path) as img:
        image_array = np.array(img)

        # Dictionary to store percentiles for each ROI
        roi_stats = {}

        for label, mask in roi_masks.items():
            # Extract only the values within the ROI (ignoring NaNs)
            selected_values = image_array[mask]

            # Remove NaN values (if any)
            selected_values = selected_values[~np.isnan(selected_values)]

            if selected_values.size > 0:
                # Compute percentiles
                percentiles = np.percentile(selected_values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
                mean_value = np.mean(selected_values)  # Compute mean

                # Store results in a dictionary
                roi_stats[label] = {
                    "mean": mean_value,
                    "p1": percentiles[0],
                    "p5": percentiles[1],
                    "p10": percentiles[2],
                    "p25": percentiles[3],
                    "p50": percentiles[4],
                    "p75": percentiles[5],
                    "p90": percentiles[6],
                    "p95": percentiles[7],
                    "p99": percentiles[8]
                }
            else:
                # If no valid pixels in ROI, store NaNs
                roi_stats[label] = {"mean": np.nan, **{f"p{p}": np.nan for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}}

    return roi_stats

def extract_timestamp(tiff_image_path):
    """
    Extract the timestamp from a TIFF image filename and convert it to a Pandas datetime object.

    Parameters:
    tiff_image_path (str): Path to the TIFF image file.

    Returns:
    pd.Timestamp: Timestamp extracted from the filename as a Pandas datetime object.
    """
    # Extract the filename from the full file path
    filename = os.path.basename(tiff_image_path)
    
    # Assuming the filename format is 'sitename_YYYYMMDD_HHmmss.tiff'
    filename_short = filename[:-5]  # Removing '.tiff'
    parts = filename_short.split('_')
    date_part, time_part = parts[-2], parts[-1]  # Get the date and time parts

    year = int(date_part[:4])
    month = int(date_part[4:6])
    day = int(date_part[6:8])
    hour = int(time_part[:2])
    minute = int(time_part[2:4])
    second = int(time_part[4:6])

    datetime_var = datetime(year, month, day, hour, minute, second)
    datetime_var = pd.to_datetime(datetime_var)
    
    return datetime_var

def find_closest_row(df, timestamp, time_window):
    """
    Find the closest row in a DataFrame based on a given timestamp and time window.

    Parameters:
    df (pd.DataFrame): The DataFrame to search.
    timestamp (pd.Timestamp): The target timestamp.
    time_window (pd.Timedelta): The time window within which to search for the closest row.

    Returns:
    pd.Series or None: The closest row in the DataFrame or None if no rows are found within the time window.
    """
    lower_bound = timestamp
    upper_bound = timestamp + time_window
    
    # Select rows within the future time window
    closest_rows = df[(df.index >= lower_bound) & (df.index <= upper_bound)].copy()
    
    if closest_rows.empty:
        return None

    # Calculate the difference from the current timestamp
    closest_rows['diff'] = closest_rows.index.to_series().subtract(timestamp).abs()
    
    # Find the row with the minimum difference (closest in time)
    closest_row = closest_rows.loc[closest_rows['diff'].idxmin()]

    # Drop the 'diff' column (if it exists) before returning

    return closest_row.drop('diff', errors='ignore')

def find_matching_logger_data(tiff_image_path, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window, elevation):
    """
    Find and extract matching logger data for a given TIFF image.

    Parameters:
    tiff_image_path (str): Path to the TIFF image.
    Aux_Met_Data (pd.DataFrame): DataFrame containing auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing FLUX meteorological data.
    Aux_Met_window (int): Time window in minutes for searching auxiliary meteorological data.
    FLUX_Met_window (int): Time window in minutes for searching FLUX meteorological data.

    Returns:
    dict: Dictionary containing extracted data.
    """
    # Extract the timestamp from the TIFF image filename
    timestamp = extract_timestamp(tiff_image_path)

    # Convert time windows to Timedelta objects
    Aux_Met_window_td = pd.to_timedelta(Aux_Met_window, unit='min')

    # Find the closest row in Aux_Met_Data based on the timestamp and time window
    closest_aux = find_closest_row(Aux_Met_Data, timestamp, Aux_Met_window_td)

    # Extract just the filename without the extension
    filename_short = os.path.basename(tiff_image_path).split('.')[0]

    # Initialize the file_data dictionary with common data
    file_data = {'filepath': tiff_image_path,
                 'filename_short': filename_short,
                 'Timestamp': timestamp}

    # Include 'T_air' and 'RH' from closest_aux
    if closest_aux is not None:
        file_data['T_air'] = closest_aux.get('T_air')
        file_data['RH'] = closest_aux.get('RH')

        # First check for 'sky_temp' in Aux_Met_Data
        file_data['sky_temp'] = closest_aux.get('sky_temp')
        file_data['LW_IN'] = closest_aux.get('LW_IN')

        # Optionally include VF_2 and T_win if they exist
        if 'VF_2' in closest_aux:
            file_data['VF_2'] = closest_aux['VF_2']
        if 'T_win' in closest_aux:
            file_data['T_win'] = closest_aux['T_win']

    # If 'sky_temp' is not in Aux_Met_Data, then check FLUX_Met_Data
    if (file_data.get('sky_temp') is None) and FLUX_Met_Data is not None and FLUX_Met_window is not None:
        FLUX_Met_window_td = pd.to_timedelta(FLUX_Met_window, unit='min')
        closest_flux = find_closest_row(FLUX_Met_Data, timestamp, FLUX_Met_window_td)
        if closest_flux is not None:
            file_data['sky_temp'] = closest_flux.get('sky_temp')
            file_data['LW_IN'] = closest_flux.get('LW_IN')

    T_air = file_data.get('T_air')
    RH = file_data.get('RH')

    if T_air is not None and RH is not None:
        if isinstance(T_air, pd.Series):
            T_air = T_air.iloc[0]  # Get the first value if it's a Series
        if isinstance(RH, pd.Series):
            RH = RH.iloc[0]  # Get the first value if it's a Series
        P = pressure_at_elevation(elevation)
        file_data['rho_v'] = vapor_density(T_air, RH, P)
    else:
        file_data['rho_v'] = None


    return file_data

def correct_integer_image(integerImg, file_data, emissivity_target, sky_percent, emissivity_vf2, win_transmittance):
    """
    Correct an integer image using provided meteorological data.

    Parameters:
    integerImg (numpy.ndarray): Integer image data to be corrected.
    file_data (dict): Dictionary containing meteorological data.
    emissivity_target (float): Emissivity value of the object.
    sky_percent (float): Percent view of sky (0–100).
    emissivity_vf2 (float): Emissivity of surrounding objects.
    win_transmittance (float): Window transmittance (0–1).

    Returns:
    tuple: Four corrected image datasets:
        - integerObj (standard correction)
        - integerObj_twin1 (correction with twin = 1)
        - integerObj_tau1 (correction with tau = 1)
        - integerObj_emiss1 (correction with emissivity = 1)
    """
    tau = file_data['tau']
    airT = file_data['T_air']

    integerImgSB = radiance_from_temperature(integerImg)
    integerAtmSB = radiance_from_temperature(airT)
    
    # Handle reflected radiation
    if sky_percent != 100:
        vf2 = file_data['VF_2']
        vf2_energy = radiance_from_temperature(vf2)
        vf1_energy = file_data['LW_IN']
        sky_frac = sky_percent / 100
        integerReflectSB = (
            sky_frac * vf1_energy +
            (1 - sky_frac) * (emissivity_vf2 * vf2_energy + vf1_energy * (1 - emissivity_vf2))
        )
    else:
        integerReflectSB = file_data['LW_IN']

    if win_transmittance != 1:
        twin = file_data['T_win']
        integerWindowSB = radiance_from_temperature(twin)

        integerObj = (
            integerImgSB / (tau * win_transmittance * emissivity_target)
            - (integerReflectSB * (1 - emissivity_target)) / emissivity_target
            - (integerAtmSB * (1 - tau)) / (tau * emissivity_target)
            - (integerWindowSB * (1 - win_transmittance)) / (tau * win_transmittance * emissivity_target)
        )

        integerObj_twin1 = (
            integerImgSB / (tau * emissivity_target)
            - (integerReflectSB * (1 - emissivity_target)) / emissivity_target
            - (integerAtmSB * (1 - tau)) / (tau * emissivity_target)
        )

        integerObj_tau1 = (
            integerImgSB / (win_transmittance * emissivity_target)
            - (integerReflectSB * (1 - emissivity_target)) / emissivity_target
            - (integerWindowSB * (1 - win_transmittance)) / (win_transmittance * emissivity_target)
        )

        integerObj_emiss1 = (
            integerImgSB / (tau * win_transmittance)
            - (integerAtmSB * (1 - tau)) / tau
            - (integerWindowSB * (1 - win_transmittance)) / (tau * win_transmittance)
        )

    else:
        integerObj = (
            integerImgSB / (tau * emissivity_target)
            - (integerReflectSB * (1 - emissivity_target)) / emissivity_target
            - (integerAtmSB * (1 - tau)) / (tau * emissivity_target)
        )

        integerObj_twin1 = integerObj  # No difference when win_transmittance == 1

        integerObj_tau1 = (
            integerImgSB / emissivity_target
            - (integerReflectSB * (1 - emissivity_target)) / emissivity_target
        )

        integerObj_emiss1 = (
            integerImgSB / tau
            - (integerAtmSB * (1 - tau)) / tau
        )

    # Convert all to temperature
    return (
        radiance_to_temp(integerObj),
        radiance_to_temp(integerObj_twin1),
        radiance_to_temp(integerObj_tau1),
        radiance_to_temp(integerObj_emiss1)
    )

def process_and_export_corrected_roi_means(image_path, roi_masks, average_distances, Aux_Met_Data, FLUX_Met_Data, aux_met_window, flux_met_window, emissivity_target, elevation, sky_percent, emissivity_vf2, win_transmittance):
    """
    Process and export corrected ROI percentiles and mean for a TIFF image while retaining uncorrected values.

    Parameters:
    image_path (str): Path to the TIFF image.
    roi_masks (dict): Dictionary of ROI masks.
    average_distances (dict): Dictionary of average distances for each ROI.
    Aux_Met_Data (pd.DataFrame): DataFrame containing auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing FLUX meteorological data.
    aux_met_window (int): Time window in minutes for searching auxiliary meteorological data.
    flux_met_window (int): Time window in minutes for searching FLUX meteorological data.
    emissivity_target (float): Emissivity value of target object.
    elevation (float): Site elevation value.
    sky_percent (int): Percent of target's view factor that is composed of sky.
    emissivity_vf2 (float): Emissivity of the dominant surrounding object other than sky.
    win_transmittance (float): The transmittance value of the enclosure window.

    Returns:
    OrderedDict: Dictionary containing processed data, structured correctly with all ROI 
                 values grouped together, including the mean and percentiles.
    """
    try:
        # 1. Extract necessary information from the filename and get file_data
        file_data = find_matching_logger_data(image_path, Aux_Met_Data, FLUX_Met_Data, aux_met_window, flux_met_window, elevation)
        
        #Validate required fields are present
        required_fields = ['T_air', 'RH', 'sky_temp', 'LW_IN','rho_v']

        if sky_percent != 100:
            required_fields.append('VF_2')
        if win_transmittance != 1:
            required_fields.append('T_win')

        missing_fields = [field for field in required_fields if file_data.get(field) is None]

        if missing_fields:
            print(f"Skipping {image_path} due to missing fields: {', '.join(missing_fields)}")
            return None

        # 2. Extract mean and percentiles for each ROI from the TIFF image
        roi_stats = calculate_roi_means_for_tiff(image_path, roi_masks)

        # 3. Define the order of stored values (mean + percentiles)
        percentiles_list = ["mean", 1, 5, 10, 25, 50, 75, 90, 95, 99]

        # 4. Initialize an ordered dictionary for structured output
        ordered_file_data = OrderedDict(file_data)  # Preserve initial metadata order

        # 5. Process and structure the ROI values
        for label in sorted(roi_stats.keys()):  # Ensure ROIs are processed in order
            dist = average_distances.get(label)

            rho_v = file_data.get('rho_v')
            if dist is None:
                raise ValueError(f"Missing distance for ROI label '{label}'")

            file_data['tau'] = atm_trans(dist, rho_v)
            ordered_file_data['tau'] = file_data['tau']

            for perc in percentiles_list:  # Include both mean and percentiles
                key = f"p{perc}" if perc != "mean" else "mean"
                value = roi_stats[label][key]  # Extract mean or percentile value

                # Store uncorrected first
                ordered_file_data[f"{label}_{key}_uncorrected"] = value

                # Get four corrected outputs from correct_integer_image
                corrected_value, corrected_value_twin1, corrected_value_tau1, corrected_value_emiss1 = correct_integer_image(
                    value, file_data, emissivity_target, sky_percent, emissivity_vf2, win_transmittance
                )

                # Store corrected values in the correct order
                ordered_file_data[f"{label}_{key}_fully_corrected"] = corrected_value
                ordered_file_data[f"{label}_{key}_tau1"] = corrected_value_tau1
                ordered_file_data[f"{label}_{key}_twin1"] = corrected_value_twin1
                ordered_file_data[f"{label}_{key}_emiss1"] = corrected_value_emiss1

        return ordered_file_data  # Returns an OrderedDict to maintain structured output

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_images_in_folders(Aux_Met_Data, aux_met_window, FLUX_Met_Data, flux_met_window, sky_percent, emissivity_vf2, emissivity_target, elevation, win_transmittance, roi_masks, average_distances, base_folder, output_csv_path, progress_bar, status_label, root):
    """
    Process images in folders and export the results to a CSV file with a progress bar and time tracking.

    Parameters:
    Aux_Met_Data (pd.DataFrame): DataFrame containing preprocessed auxiliary meteorological data.
    aux_met_window (int): Time window in minutes for searching auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing preprocessed FLUX meteorological data.
    flux_met_window (int): Time window in minutes for searching FLUX meteorological data.
    sky_percent (int): Percent of target's view factor that is composed of sky.
    emissivity_vf2 (float): Emissivity of the dominant surrounding object other than sky.
    emissivity_target (float): Emissivity value from the configuration.
    elevation (float): Site elevation.
    win_transmittance (float): The transmittance value of the enclosure window.
    roi_masks (dict): A dictionary of Region of Interest (ROI) masks.
    average_distances (dict): A dictionary of average distances for each ROI.
    base_folder (str): The base folder path where the images are located.
    output_csv_path (str): Path to the output CSV file where the results will be saved.
    progress_bar (tk.Progressbar): GUI progress bar.
    status_label (tk.Label): GUI status label.
    root (tk.Tk): Tkinter main window.

    Returns:
    pd.DataFrame: DataFrame containing processed data.
    Time it took for processing to occur.
    """
    # Record the start time
    start_time = time.time()

    file_data_list = []

    # List all year folders in the base folder
    year_folders = [year for year in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, year))]

    # Calculate total number of files for progress bar
    total_files = sum([len(os.listdir(os.path.join(base_folder, year, month))) for year in year_folders for month in os.listdir(os.path.join(base_folder, year)) if os.path.isdir(os.path.join(base_folder, year, month))])

    # Set progress bar maximum value
    progress_bar['maximum'] = total_files
    progress = 0

    for year_folder in year_folders:
        year_path = os.path.join(base_folder, year_folder)
        month_folders = [month for month in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, month))]

        for month_folder in month_folders:
            month_path = os.path.join(year_path, month_folder)

            for image_file in os.listdir(month_path):
                if image_file.lower().endswith('.tiff') and not image_file.startswith('._'):  # Skip hidden macOS metadata files
                    image_path = os.path.join(month_path, image_file)
                    file_data = process_and_export_corrected_roi_means(image_path, roi_masks, average_distances, Aux_Met_Data, FLUX_Met_Data, aux_met_window, flux_met_window, emissivity_target, elevation, sky_percent, emissivity_vf2, win_transmittance)
                    if file_data is None:
                        #print("Skipping this TIFF due to processing error.")
                        continue
                    file_data_list.append(file_data)

                # Update the progress bar
                progress += 1
                progress_bar['value'] = progress
                status_label.config(text=f"Processing file {progress} of {total_files}")
                root.update_idletasks()

    # Convert the list of file_data dictionaries to a DataFrame
    result_df = pd.DataFrame(file_data_list)

    # Sort the DataFrame by 'Timestamp' in ascending order
    if not result_df.empty and 'Timestamp' in result_df.columns:
        result_df = result_df.sort_values(by='Timestamp')
    else:
        print("Warning: result_df is empty or missing 'Timestamp' column. Skipping sort.")


    # Export the DataFrame to a CSV file
    result_df.to_csv(output_csv_path, index=False)

    # Record the end time
    end_time = time.time()

    # Calculate and print the total runtime
    total_time = end_time - start_time
    print(f"The function took {human_readable_time(total_time)} to run.")

    return result_df

# =============================================
# POINT DATA PROCESSING
# =============================================
def initialize_data_from_config_point(config_path):
    """
    Initialize data for the application from a configuration file.

    This function reads configuration settings from a JSON file and uses them to
    load and preprocess auxiliary and FLUX meteorological data. If LW_IN is present
    in Aux_Met_Data, FLUX_Met_Data is not loaded to avoid redundancy.

    Parameters:
    config_path (str): The file path to the configuration JSON file.

    Returns:
    tuple: A tuple containing:
        - Aux_Met_Data (DataFrame or None): Preprocessed auxiliary meteorological data.
        - aux_met_window (int): Time window for auxiliary meteorological data.
        - FLUX_Met_Data (DataFrame or None): Preprocessed FLUX meteorological data.
        - flux_met_window (int): Time window for FLUX meteorological data.
        - sky_percent (int): Percent of target's view factor that is composed of sky.
        - emissivity_vf2 (float): Emissivity of the dominant surrounding object other than sky.
        - emissivity_target (float): Emissivity value from the configuration.
        - elevation (float): Site elevation.
        - win_transmittance (float): The transmittance value of the enclosure window.
        - point_data_path (str): Path to the CSV file containing point data.
        - point_dist (float): Distance value for ROI.
        - output_csv_path (str): Output CSV file path from the configuration.
    """
    print("Initializing point-based process...")

    # Load the configuration from the JSON file
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Extract and validate key physical parameters
    emissivity_target = config.get('emissivity_target')
    elevation = config.get('elevation')
    sky_percent = config.get('sky_percent')
    emissivity_vf2 = config.get('emissivity_vf2')
    win_transmittance = config.get('win_transmittance')

    # Validation
    if sky_percent is None or not (0 <= sky_percent <= 100):
        raise ValueError(f"Invalid 'sky_percent': {sky_percent}. It must be between 0 and 100.")

    if emissivity_vf2 is None or not (0 <= emissivity_vf2 <= 1):
        raise ValueError(f"Invalid 'emissivity_vf2': {emissivity_vf2}. It must be between 0 and 1.")

    if emissivity_target is None or not (0 <= emissivity_target <= 1):
        raise ValueError(f"Invalid 'emissivity_target': {emissivity_target}. It must be between 0 and 1.")

    if win_transmittance is None or not (0 <= win_transmittance <= 1):
        raise ValueError(f"Invalid 'win_transmittance': {win_transmittance}. It must be between 0 and 1.")

    # Define necessary columns for auxiliary meteorological data
    aux_columns = ["TIMESTAMP_END", "T_air", "RH", "LW_IN", "VF_2", "T_win"]

    # Conditionally exclude columns
    if sky_percent == 100 or emissivity_vf2 == 1:
        aux_columns = [col for col in aux_columns if col != "VF_2"]
    if win_transmittance == 1:
        aux_columns = [col for col in aux_columns if col != "T_win"]

    # FLUX dataset columns
    flux_columns = ["TIMESTAMP_END", "LW_IN"]

    # Load and preprocess auxiliary meteorological data
    Aux_Met_Data = None
    if config.get('aux_met_data_path'):
        Aux_Met_Data = pd.read_csv(config['aux_met_data_path'], usecols=lambda x: x in aux_columns)
        Aux_Met_Data = preprocess_dataframe(Aux_Met_Data)

    # Load FLUX_Met_Data **only if LW_IN is missing from Aux_Met_Data
    FLUX_Met_Data = None
    if config.get('flux_met_data_path') and (Aux_Met_Data is None or 'LW_IN' not in Aux_Met_Data.columns):
        FLUX_Met_Data = pd.read_csv(config['flux_met_data_path'], usecols=lambda x: x in flux_columns)
        FLUX_Met_Data = preprocess_dataframe(FLUX_Met_Data)
    else:
        print("Skipping FLUX_Met_Data as LW_IN is already present in Aux_Met_Data.")

    return (
        Aux_Met_Data,
        config['aux_met_window'],
        FLUX_Met_Data,
        config['flux_met_window'],
        sky_percent,
        emissivity_vf2,
        emissivity_target,
        elevation,
        win_transmittance,
        config['point_data_path'],
        config['point_dist'],
        config['output_csv_path']
    )

def correct_point_data(point_value, file_data, emissivity_target, sky_percent, emissivity_vf2, win_transmittance):
    """
    Correct point data using provided meteorological data.

    Parameters:
    point_value (float): Point data value to be corrected.
    file_data (dict): Dictionary containing meteorological data.
    emissivity_target (float): Emissivity value of the object.
    sky_percent (float): Percent view of sky (0–100).
    emissivity_vf2 (float): Emissivity of surrounding objects.
    win_transmittance (float): Window transmittance (0–1).

    Returns:
    tuple: Four corrected point data values:
        - correctedPoint (standard correction)
        - correctedPoint_twin1 (correction with twin = 1)
        - correctedPoint_tau1 (correction with tau = 1)
        - correctedPoint_emiss1 (correction with emiss = 1)
    """
    # Extract values from file_data
    tau = file_data['tau']
    airT = file_data['T_air']

    # Convert point data to radiance
    pointRadiance = radiance_from_temperature(point_value)
    atmRadiance = radiance_from_temperature(airT)

    # Handle reflected radiation
    if sky_percent != 100:
        vf2 = file_data['VF_2']
        vf2_energy = radiance_from_temperature(vf2)
        vf1_energy = file_data['LW_IN']
        sky_frac = sky_percent / 100
        reflRadiance = (
            sky_frac * vf1_energy +
            (1 - sky_frac) * (emissivity_vf2 * vf2_energy + vf1_energy * (1 - emissivity_vf2))
        )
    else:
        reflRadiance = file_data['LW_IN']

    if win_transmittance != 1:
        twin = file_data['T_win']
        winRadiance = radiance_from_temperature(twin)

        correctedPoint = (
            pointRadiance / (tau * win_transmittance * emissivity_target)
            - (reflRadiance * (1 - emissivity_target)) / emissivity_target
            - (atmRadiance * (1 - tau)) / (tau * emissivity_target)
            - (winRadiance * (1 - win_transmittance)) / (tau * win_transmittance * emissivity_target)
        )

        correctedPoint_twin1 = (
            pointRadiance / (tau * emissivity_target)
            - (reflRadiance * (1 - emissivity_target)) / emissivity_target
            - (atmRadiance * (1 - tau)) / (tau * emissivity_target)
        )

        correctedPoint_tau1 = (
            pointRadiance / (win_transmittance * emissivity_target)
            - (reflRadiance * (1 - emissivity_target)) / emissivity_target
            - (winRadiance * (1 - win_transmittance)) / (win_transmittance * emissivity_target)
        )

        correctedPoint_emiss1 = (
            pointRadiance / (tau * win_transmittance)
            - (atmRadiance * (1 - tau)) / tau
            - (winRadiance * (1 - win_transmittance)) / (tau * win_transmittance)
        )

    else:
        correctedPoint = (
            pointRadiance / (tau * emissivity_target)
            - (reflRadiance * (1 - emissivity_target)) / emissivity_target
            - (atmRadiance * (1 - tau)) / (tau * emissivity_target)
        )

        correctedPoint_twin1 = correctedPoint  # No difference when win_transmittance == 1

        correctedPoint_tau1 = (
            pointRadiance / emissivity_target
            - (reflRadiance * (1 - emissivity_target)) / emissivity_target
        )

        correctedPoint_emiss1 = (
            pointRadiance / tau
            - (atmRadiance * (1 - tau)) / tau
        )

    # Convert all to temperature
    return (
        radiance_to_temp(correctedPoint),
        radiance_to_temp(correctedPoint_twin1),
        radiance_to_temp(correctedPoint_tau1),
        radiance_to_temp(correctedPoint_emiss1)
    )

def find_closest_row_point(df, timestamp, time_window):
    """
    Find the closest row in a DataFrame based on a given timestamp and time window.

    Parameters:
    df (pd.DataFrame): The DataFrame to search.
    timestamp (pd.Timestamp): The target timestamp.
    time_window (pd.Timedelta): The time window within which to search for the closest row.

    Returns:
    pd.Series or None: The closest row in the DataFrame or None if no rows are found within the time window.
    """
    lower_bound = timestamp
    upper_bound = timestamp + time_window
    
    # Select rows within the future time window
    closest_rows = df[(df.index >= lower_bound) & (df.index <= upper_bound)].copy()
    
    if closest_rows.empty:
        return None

    # Calculate the difference from the current timestamp
    closest_rows['diff'] = closest_rows.index.to_series().subtract(timestamp).abs()
    
    # Find the row with the minimum difference (closest in time)
    closest_row = closest_rows.loc[closest_rows['diff'].idxmin()]

    # Drop the 'diff' column (if it exists) before returning

    return closest_row.drop('diff', errors='ignore')

def find_matching_logger_data_point(timestamp, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window, elevation):
    """
    Find and extract matching logger data for a given timestamp.

    Parameters:
    timestamp (str): The timestamp to match.
    Aux_Met_Data (pd.DataFrame): DataFrame containing auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing FLUX meteorological data.
    Aux_Met_window (int): Time window in minutes for searching auxiliary meteorological data.
    FLUX_Met_window (int): Time window in minutes for searching FLUX meteorological data.

    Returns:
    dict: Dictionary containing extracted data.
    """

    timestamp_dt = pd.to_datetime(timestamp, format="%m/%d/%y %H:%M")
    time_window_td = pd.Timedelta(minutes=Aux_Met_window)
        
    # Find the closest row in Aux_Met_Data
    closest_aux = find_closest_row_point(Aux_Met_Data, timestamp_dt, time_window_td)

    # Initialize the file_data dictionary with common data
    file_data = {
                'Timestamp': timestamp}

    # Include 'T_air' and 'RH' from closest_aux
    if closest_aux is not None:
        file_data['T_air'] = closest_aux.get('T_air')
        file_data['RH'] = closest_aux.get('RH')

        # First check for 'sky_temp' in Aux_Met_Data
        file_data['sky_temp'] = closest_aux.get('sky_temp')
        file_data['LW_IN'] = closest_aux.get('LW_IN')

        # Optionally include VF_2 and T_win if they exist
        if 'VF_2' in closest_aux:
            file_data['VF_2'] = closest_aux['VF_2']
        if 'T_win' in closest_aux:
            file_data['T_win'] = closest_aux['T_win']

    # If 'sky_temp' is not in Aux_Met_Data, then check FLUX_Met_Data
    if (file_data.get('sky_temp') is None) and FLUX_Met_Data is not None and FLUX_Met_window is not None:
            closest_flux = find_closest_row_point(FLUX_Met_Data, timestamp_dt, pd.Timedelta(minutes=FLUX_Met_window))
            if closest_flux is not None:
                file_data['sky_temp'] = closest_flux.get('sky_temp')
                file_data['LW_IN'] = closest_flux.get('LW_IN')

    T_air = file_data.get('T_air')
    RH = file_data.get('RH')

    if T_air is not None and RH is not None:
        if isinstance(T_air, pd.Series):
            T_air = T_air.iloc[0]  # Get the first value if it's a Series
        if isinstance(RH, pd.Series):
            RH = RH.iloc[0]  # Get the first value if it's a Series
        P = pressure_at_elevation(elevation)
        file_data['rho_v'] = vapor_density(T_air, RH, P)
    else:
        file_data['rho_v'] = None


    return file_data

def process_and_export_corrected_point_data(Aux_Met_Data, aux_met_window, FLUX_Met_Data, flux_met_window, sky_percent, emissivity_vf2, emissivity_target, elevation, win_transmittance, point_data_path, point_dist, output_csv_path):
    """
    Process and export corrected point data, including meteorological values.

    Parameters:
    Aux_Met_Data (pd.DataFrame): DataFrame containing preprocessed auxiliary meteorological data.
    aux_met_window (int): Time window in minutes for searching auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing preprocessed FLUX meteorological data.
    flux_met_window (int): Time window in minutes for searching FLUX meteorological data.
    sky_percent (int): Percent of target's view factor that is composed of sky.
    emissivity_vf2 (float): Emissivity of the dominant surrounding object other than sky.
    emissivity_target (float): Emissivity value from the configuration.
    elevation (float): Site elevation.
    win_transmittance (float): The transmittance value of the enclosure window.
    point_data_path (str): Path to the CSV file containing point data.
    point_dist (float): Distance value for the point data.
    output_csv_path (str): Path to the output CSV file where the results will be saved.

    Returns:
    pd.DataFrame: DataFrame containing processed data, including meteorological data and corrected point values.
    """
    try:
        # Record the start time
        start_time = time.time()
        
        # Load the point data
        point_data = pd.read_csv(point_data_path)
        
        point_data['timestamp_dt'] = pd.to_datetime(point_data['timestamp'], format="%m/%d/%y %H:%M")

        # Initialize a list to hold the processed data
        processed_data = []

        for _, row in point_data.iterrows():
            timestamp = row['timestamp']
            point_value = row['temp_value']

            # Extract necessary information from the file_data
            file_data = find_matching_logger_data_point(timestamp, Aux_Met_Data, FLUX_Met_Data, aux_met_window, flux_met_window, elevation)
            
            #Validate required fields are present
            required_fields = ['T_air', 'RH', 'sky_temp', 'LW_IN','rho_v']

            if sky_percent != 100:
                required_fields.append('VF_2')
            if win_transmittance != 1:
                required_fields.append('T_win')

            missing_fields = [field for field in required_fields if file_data.get(field) is None]
            if missing_fields:
                print(f"Skipping {image_path} due to missing fields: {', '.join(missing_fields)}")
                return None

            # Calculate tau using the distance
            rho_v = file_data.get('rho_v')
            if point_dist is None:
                raise ValueError(f"Missing distance value 'point_dist' for processing.")

            file_data['tau'] = atm_trans(point_dist, rho_v)

            try:
                # Get four corrected outputs from correct_point_data
                corrected_value, corrected_value_twin1, corrected_value_tau1, corrected_value_emiss1 = correct_point_data(point_value, file_data, emissivity_target, sky_percent, emissivity_vf2, win_transmittance)

                # Initialize with timestamp first
                data_entry = OrderedDict()

                # Add all file_data entries
                for key, value in file_data.items():
                    data_entry[key] = value

                # Add corrected temperature values
                data_entry['temp_value_uncorrected'] = point_value
                data_entry['temp_value_fully_corrected'] = corrected_value
                data_entry['temp_value_tau1'] = corrected_value_tau1
                data_entry['temp_value_twin1'] = corrected_value_twin1
                data_entry['temp_value_emiss1'] = corrected_value_emiss1

                # Append to the list
                processed_data.append(data_entry)

            except Exception as err:
                print(f"There was an exception correcting point data: {err}")
                continue

        # Convert the list of dictionaries to a DataFrame
        result_df = pd.DataFrame(processed_data)

        # Save to CSV
        result_df.to_csv(output_csv_path, index=False)

        # Record the end time
        end_time = time.time()

        # Calculate and print the total runtime
        total_time = end_time - start_time

        print(f"The function took {human_readable_time(total_time)} to run.")
        return result_df
    
    except Exception as e:
        print(f"Error processing {point_data_path}: {e}")
        return pd.DataFrame()

# =============================================
# DATAFRAME PROCESSING
# =============================================
def preprocess_dataframe(df):
    """
    Preprocess a DataFrame by performing the following operations:
    1. Convert the 'TIMESTAMP_END' column to datetime and set it as the index.
    2. If 'LW_IN' column exists, calculate 'sky_temp'.

    Parameters:
    df (DataFrame): The input DataFrame to be preprocessed.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    # Step 1: Convert timestamp column to datetime and set as index
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        df['Timestamp'] = pd.to_datetime(df['TIMESTAMP_END'])
    df.set_index('Timestamp', inplace=True)

    # Step 2: Calculate 'sky_temp' if 'LW_IN' column exists
    if 'LW_IN' in df.columns:
        df['sky_temp'] = radiance_to_temp(df['LW_IN'])

    return df

# =============================================
# TIME PROCESSING
# =============================================
def human_readable_time(seconds):
    """
    Converts seconds to a more human-readable format of hours, minutes, and seconds.
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds"
    elif minutes > 0:
        return f"{int(minutes)} minutes, {seconds:.2f} seconds"
    else:
        return f"{seconds:.2f} seconds"

# =============================================
# ATMOSPHERIC FUNCTIONS
# =============================================
def pressure_at_elevation(h):
    """
    Calculate atmospheric pressure at a given elevation using the simplified barometric formula.
    
    Parameters:
    h (float): Elevation in meters.
    
    Returns:
    float: Atmospheric pressure in Pascals (Pa).
    """
    P0 = 101325  # Sea-level pressure in Pascals (Pa)
    T0 = 288.15  # Sea-level standard temperature in Kelvin (K)
    L = 0.0065   # Temperature lapse rate in K/m
    g = 9.80665  # Gravitational acceleration in m/s^2
    M = 0.0289644  # Molar mass of Earth's air in kg/mol
    R = 8.31447   # Universal gas constant in J/(mol·K)
    
    # Compute the exponent (g * M) / (R * L)
    exponent = (g * M) / (R * L)  # ~5.25588

    # Apply the barometric formula
    P = P0 * (1 - (L * h) / T0) ** exponent
    
    return P

def vapor_density(T_air, RH, P):
    """
    Calculate actual vapor density given air temperature, relative humidity, and pressure.

    Parameters:
    T_air (float): Air temperature in Celsius
    RH (float): Relative humidity in percentage (0 to 100)
    P (float): Atmospheric pressure in Pascals (Pa).

    Returns:
    float: Actual vapor density in g/m^3
    """
    # Constants for Tetens equation
    a = 610.94  # Pa
    b = 17.625
    c = 243.04  # °C
    
    # Step 1: Compute saturation vapor pressure at standard pressure (Pa)
    es_SL = a * math.exp(b * T_air / (c + T_air))
    
    # Step 2: Adjust for actual atmospheric pressure
    es = es_SL * (P / 101325)
    
    # Step 3: Compute actual vapor pressure (Pa)
    ea = es * (RH / 100)
    
    # Step 4: Compute vapor density using the Ideal Gas Law
    M_w = 18.015  # g/mol (Molar mass of water)
    R = 8.314472  # J/(mol*K) (Universal gas constant)
    T_K = T_air + 273.15  # Convert to Kelvin
    
    rho_v = (ea * M_w) / (R * T_K)  # g/m^3

    return rho_v

def atm_trans (dist, rho_v):
    """
    Calculate atmsopheric transmittance based on abbreviated LOWTRAN model.

    Parameters:
    dist (float): Distance to object of interest in UNIT??
    rho_v (float): Vapor density in g/m^3

    Returns:
    float: Atmsopheric transmittance as ratio between 0 and 1
    """
    # LOWTRAN constants
    alpha1 = 0.00656899996101856
    alpha2 = 0.0126200001686811
    beta1 = -0.00227600010111928
    beta2 = -0.00667000003159046
    X = 1.89999997

    term1 = X * math.exp(-math.sqrt(dist) * (alpha1 + beta1 * math.sqrt(rho_v)))
    term2 = (1 - X) * math.exp(-math.sqrt(dist) * (alpha2 + beta2 * math.sqrt(rho_v)))
    tau = term1 + term2
    
    return tau

# =============================================
# STEFAN-BOTLZMANN LAW FUNCTIONS
# =============================================
def radiance_from_temperature(T):
    """
    Calculate radiant flux from temperature based on Stefan-Boltzmann Law.

    Parameters:
    T (float): Temperature in Celsius

    Returns:
    float: Radiant flux in W/m^-2
    """
    # Stefan-Boltzmann constant in W/m^2⋅K^4
    sigma = 5.67e-8

    # Convert temperature to Kelvin
    T_kelvin = T + 273.15

    return sigma * T_kelvin**4

def radiance_to_temp(E):
    """
    Calculate temperature from radiant flux based on Stefan-Boltzmann Law, using the absolute value of radiant flux.

    Parameters:
    E (float): Radiant flux in W/m^-2

    Returns:
    float: Temperature in Celsius
    """
    # Stefan-Boltzmann constant (in W/m^2K^4)
    sigma = 5.67e-8  

    # Use the absolute value of E to ensure the calculation is always valid
    T = (abs(E) / sigma) ** 0.25
    
    # Convert temperature to Celsius
    T_celsius = T - 273.15
    
    return T_celsius
