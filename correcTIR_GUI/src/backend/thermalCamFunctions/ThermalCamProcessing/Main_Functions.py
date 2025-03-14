
# Standard library imports
import os
import math
import json
from datetime import datetime
import csv
import time
import threading
import warnings

# Third-party imports
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import tkinter as tk
from tkinter import ttk

##### Prep Functions
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
        - FLUX_Met_Data (DataFrame or None): Preprocessed FLUX meteorological data.
        - roi_masks (dict): ROI masks initialized from the configuration.
        - average_distances (dict): Average distances for each ROI.
        - Aux_Met_window (int): Time window for auxiliary meteorological data.
        - FLUX_Met_window (int): Time window for FLUX meteorological data.
        - base_folder (str): Base folder path from the configuration.
        - output_csv_path (str): Output CSV file path from the configuration.
        - emissivity (float): Emissivity value from the configuration.
    """
    print("Initializing process...")

    # Load the configuration from the JSON file
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Load and preprocess data based on the configuration
    Aux_Met_Data = pd.read_csv(config['aux_met_data_path']) if config['aux_met_data_path'] else None
    if Aux_Met_Data is not None:
        Aux_Met_Data = preprocess_dataframe(Aux_Met_Data)
    
    FLUX_Met_Data = pd.read_csv(config['flux_met_data_path']) if config['flux_met_data_path'] else None
    # FLUX_Met_Data = pd.read_csv(config['flux_met_data_path']) if config.get['flux_met_data_path'] else None
    if FLUX_Met_Data is not None:
        FLUX_Met_Data = preprocess_dataframe(FLUX_Met_Data)

    roi_masks, average_distances = initialize_roi_masks_and_distances(
        config['roi_path'], config['roi_dist_path'], config['first_image_path'], config['data_type']
    )

    return (Aux_Met_Data, FLUX_Met_Data, roi_masks, average_distances,
            config['Aux_Met_window'], config['FLUX_Met_window'],
            config['base_folder'], config['output_csv_path'],
            config['emissivity'])


def initialize_data_from_config_point(config_path):
    """
    Initialize data for the application from a configuration file.

    This function reads configuration settings from a JSON file and uses them to
    load and preprocess auxiliary and FLUX meteorological data.

    Parameters:
    config_path (str): The file path to the configuration JSON file.

    Returns:
    tuple: A tuple containing:
        - Aux_Met_Data (DataFrame or None): Preprocessed auxiliary meteorological data.
        - FLUX_Met_Data (DataFrame or None): Preprocessed FLUX meteorological data.
        - Aux_Met_window (int): Time window for auxiliary meteorological data.
        - FLUX_Met_window (int): Time window for FLUX meteorological data.
        - point_dist (float): Distance value for ROI
        - output_csv_path (str): Output CSV file path from the configuration.
        - emissivity (float): Emissivity value from the configuration.
        - point_data_path (str): Path to the CSV file containing point data.
    """
    print("Initializing process...")

    # Load the configuration from the JSON file
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Load and preprocess auxiliary meteorological data based on the configuration
    Aux_Met_Data = pd.read_csv(config['aux_met_data_path']) if config['aux_met_data_path'] else None
    if Aux_Met_Data is not None:
        Aux_Met_Data = preprocess_dataframe(Aux_Met_Data)
    
    # Load and preprocess FLUX meteorological data based on the configuration
    FLUX_Met_Data = pd.read_csv(config['flux_met_data_path']) if config['flux_met_data_path'] else None
    if FLUX_Met_Data is not None:
        FLUX_Met_Data = preprocess_dataframe(FLUX_Met_Data)

    return (Aux_Met_Data, FLUX_Met_Data,
            config['Aux_Met_window'], config['FLUX_Met_window'],
            config['point_dist'], config['output_csv_path'],
            config['emissivity'], config['point_data_path'])


##### Time Functions
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


##### Atmsopheric Functions
def vapor_density(T_air, RH):
    """
    Calculate actual vapor density given air temperature and relative humidity.

    Parameters:
    T_air (float): Air temperature in Celsius
    RH (float): Relative humidity in percentage (0 to 100)

    Returns:
    float: Actual vapor density in g/m^3
    """
    ## Step 1: Use Clausius-Clapeyron relation to determine expected vapor pressure
    # Constants
    a = 610.94
    b = 17.625
    c = 243.04
    # Saturation vapor pressure (in Pa)
    es = a * math.exp(b * T_air / (c + T_air))

    ## Step 2: Actual saturation vapor pressure
    ea = es * (RH/100)

    ## Step 3: Use ideal gas law to determine density
    # Molar mass of water (g/mol)
    M_w = 18.015
    # Universal Gas constant (J/mol*K)
    R = 8.314472
    # Absolute temperature (K)
    T_K = T_air + 273.15
    # Vapor density (g/m^3)
    rho_v = (ea * M_w)/(R * T_K)
    
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


##### Stefan-Boltzman Law Functions
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
    Calculate temperature from radiant flux based on Stefan-Boltzmann Law.

    Parameters:
    E (float): Radiant flux in W/m^-2

    Returns:
    float: Temperature in Celsius
    """
    # Stefan-Boltzmann constant (in W/m^2K^4)
    sigma = 5.67e-8  
    
    T = (E / sigma) ** 0.25
    
    #Convert temperature to Celsius
    T_celsius = T - 273.15
    
    return T_celsius


##### ROI and Image Functions
def get_image_shape(image_path):
    """
    Get the dimensions of the image.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    tuple: A tuple representing the shape of the image in the form (height, width).
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Read the image
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

def calculate_average_distances(distance_csv, roi_masks=None, data_type='pointcloud'):
    """
    Calculate or read the average distance for each ROI.

    Parameters:
    distance_csv (str): Path to the CSV file containing distances.
    roi_masks (dict, optional): A dictionary containing masks for each ROI (used if data_type is 'pointcloud').
    data_type (str): Type of data in the CSV file ('pointcloud' or 'average').

    Returns:
    dict: A dictionary with ROI labels as keys and their average distances as values.
    """
    if data_type == 'pointcloud':
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
        raise ValueError("data_type must be either 'pointcloud' or 'average'.")

    return average_distances

def initialize_roi_masks_and_distances(roi_csv_path, distance_csv_path, image_path, data_type='pointcloud'):
    """
    Initialize ROI masks and calculate or read the average distances.

    Parameters:
    roi_csv_path (str): Path to the CSV file containing ROI information.
    distance_csv_path (str): Path to the CSV file containing distances.
    image_path (str): Path to the image file.
    data_type (str): Type of data in the distance CSV file ('pointcloud' or 'average').

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
    Calculate the mean values for each ROI in a TIFF image.

    Parameters:
    tiff_image_path (str): Path to the .tiff image file.
    roi_masks (dict): A dictionary containing masks for each ROI.

    Returns:
    dict: A dictionary with ROI labels as keys and their mean values as values.
    """
    # Open the .tiff file
    with Image.open(tiff_image_path) as img:
        image_array = np.array(img)
        
        # Dictionary to hold the mean values of each ROI
        roi_means = {}
        
        for label, mask in roi_masks.items():
            # Create a mask for the current ROI
            selected_region = np.where(mask, image_array, np.nan)
            
            # Calculate the mean value, ignoring NaN values
            mean_value = np.nanmean(selected_region)
            
            # Store the mean value in the dictionary with the label
            roi_means[label] = mean_value
    
    return roi_means


##### Processing Functions
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

def find_matching_logger_data(tiff_image_path, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window):
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

        file_data['rho_v'] = vapor_density(T_air, RH)
    else:
        file_data['rho_v'] = None


    return file_data


##### Correcting Single Image Functions
def correct_integer_image(integerImg, file_data, emiss):
    """
    Correct an integer image using provided meteorological data.

    Parameters:
    integerImg (numpy.ndarray): Integer image data to be corrected.
    file_data (dict): Dictionary containing meteorological data including 'skyTemp', 'tau', and 'T_air'.

    Returns:
    numpy.ndarray: Corrected image data.
    """
    # Extract the values from file_data
    skyTemp = file_data['sky_temp']
    tau = file_data['tau']
    airT = file_data['T_air']

    # Convert integer image to radiance
    integerImgSB = radiance_from_temperature(integerImg)
    integerReflectSB = radiance_from_temperature(skyTemp)
    integerAtmSB = radiance_from_temperature(airT)

    # Constants
    rEmiss = 1.0  # Assuming sky has emissivity of 1

    # Perform the correction calculation
    integerObj = (1.0 / (emiss * tau)) * (integerImgSB - tau * (1 - emiss) * rEmiss * integerReflectSB - (1 - tau) * integerAtmSB)

    # Convert the corrected radiance back to temperature
    integerObj = radiance_to_temp(integerObj)

    return integerObj

def process_and_export_corrected_roi_means(tiff_image_path, roi_masks, average_distances, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window, emiss):
    """
    Process and export corrected ROI means for a TIFF image.

    Parameters:
    tiff_image_path (str): Path to the TIFF image.
    roi_masks (dict): Dictionary of ROI masks.
    average_distances (dict): Dictionary of average distances for each ROI.
    Aux_Met_Data (pd.DataFrame): DataFrame containing auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing FLUX meteorological data.
    Aux_Met_window (int): Time window in minutes for searching auxiliary meteorological data.
    FLUX_Met_window (int): Time window in minutes for searching FLUX meteorological data.

    Returns:
    dict: Dictionary containing processed data, including corrected ROI means.
    """
    try:
        # 1. Extract necessary information from the filename and get file_data
        file_data = find_matching_logger_data(tiff_image_path, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window)

        # 2. Extract mean values of ROIs from the TIFF
        roi_means = calculate_roi_means_for_tiff(tiff_image_path, roi_masks)

        # 3. Correct these mean values
        corrected_roi_means = {}
        for label, mean_value in roi_means.items():

            dist = average_distances.get(label)
            if file_data['rho_v'] is not None and dist is not None:
                file_data['tau'] = atm_trans(dist, file_data['rho_v']) 

            file_data[label + "_uncorrected"] = mean_value
            corrected_value = correct_integer_image(mean_value, file_data, emiss)
            corrected_roi_means[label] = corrected_value

        # 4. Add these corrected values to the file_data
        for label, corrected_value in corrected_roi_means.items():
            file_data[label + "_corrected"] = corrected_value

        return file_data
    
    except Exception as e:
        # Log the exception message
        print(f"Error processing {tiff_image_path}: {e}")

        # Conditional logging for debugging
        #print(f"Type of rho_v: {type(file_data.get('rho_v'))}")
        #for label, mean_value in roi_means.items():
            #print(f"Type of mean value for {label}: {type(mean_value)}")

        # You can add more debug info here as needed

        # Skip further processing and return None or an empty dict
        return None  # Or return {}


##### Correcting Image Path Functions
def process_images_in_folders(base_folder, roi_masks, average_distances, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window, output_csv_path, emissivity, progress_bar, status_label, root):
    """
    Process images in folders and export the results to a CSV file with a progress bar and time tracking.

    Parameters:
    base_folder (str): Base folder containing year and month subfolders.
    roi_masks (dict): Dictionary of ROI masks.
    average_distances (dict): Dictionary of average distances for each ROI.
    Aux_Met_Data (pd.DataFrame): DataFrame containing auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing FLUX meteorological data.
    Aux_Met_window (int): Time window in minutes for searching auxiliary meteorological data.
    FLUX_Met_window (int): Time window in minutes for searching FLUX meteorological data.
    output_csv_path (str): Path to the output CSV file.

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
                    file_data = process_and_export_corrected_roi_means(image_path, roi_masks, average_distances, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window, emissivity)
                    if file_data is None:
                        print("Skipping this TIFF due to processing error.")
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
    result_df = result_df.sort_values(by='Timestamp')

    # Export the DataFrame to a CSV file
    result_df.to_csv(output_csv_path, index=False)

    # Record the end time
    end_time = time.time()

    # Calculate and print the total runtime
    total_time = end_time - start_time
    print(f"The function took {human_readable_time(total_time)} to run.")

    return result_df

##### GUI Functions
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

def setup_gui_and_start(base_folder, roi_masks, average_distances, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window, output_csv_path, emissivity):
    """
    Sets up the graphical user interface (GUI) and starts the image processing in a separate thread.

    This function initializes the GUI for the image processing application. It creates a
    main window with a progress bar and then starts the image processing in a separate thread
    using the provided parameters. The GUI provides visual feedback on the progress of the
    image processing.

    Parameters:
    base_folder (str): The base folder path where the images are located.
    roi_masks (dict): A dictionary of Region of Interest (ROI) masks.
    average_distances (dict): A dictionary of average distances for each ROI.
    Aux_Met_Data (pd.DataFrame): DataFrame containing auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing FLUX meteorological data.
    Aux_Met_window (int): Time window in minutes for searching auxiliary meteorological data.
    FLUX_Met_window (int): Time window in minutes for searching FLUX meteorological data.
    output_csv_path (str): Path to the output CSV file where the results will be saved.
    """
    root = tk.Tk()
    root.title("Processing Images")

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress_bar.pack(pady=10)

    # Label for showing the processing status
    status_label = tk.Label(root, text="Starting...")
    status_label.pack(pady=10)

    # Start the processing in a separate thread
    start_args = (base_folder, roi_masks, average_distances, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window, output_csv_path, emissivity, progress_bar, status_label, root)
    start_processing_thread(*start_args)

    root.mainloop()

#### POINT DATA FUNCTIONS 
def correct_point_data(point_data, file_data, emiss):
    """
    Correct point data using provided meteorological data.

    Parameters:
    point_data (float): Point data value to be corrected.
    file_data (dict): Dictionary containing meteorological data including 'sky_temp', 'tau', and 'T_air'.
    emiss (float): Emissivity value.

    Returns:
    float: Corrected point data value.
    """
    # Extract the values from file_data
    skyTemp = file_data['sky_temp']
    tau = file_data['tau']
    airT = file_data['T_air']

    # Convert point data to radiance
    pointRadiance = radiance_from_temperature(point_data)
    skyRadiance = radiance_from_temperature(skyTemp)
    atmRadiance = radiance_from_temperature(airT)

    # Constants
    rEmiss = 1.0  # Assuming sky has emissivity of 1

    # Perform the correction calculation
    correctedPoint = (1.0 / (emiss * tau)) * (pointRadiance - tau * (1 - emiss) * rEmiss * skyRadiance - (1 - tau) * atmRadiance)

    # Convert the corrected radiance back to temperature
    correctedPoint = radiance_to_temp(correctedPoint)

    return correctedPoint

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


def find_matching_logger_data_point(timestamp, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window):
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

    #print(f"Finding matching data for timestamp: {timestamp}")
    timestamp_dt = pd.to_datetime(timestamp, format="%m/%d/%y %H:%M")
    time_window_td = pd.Timedelta(minutes=Aux_Met_window)
        
    # Find the closest row in Aux_Met_Data
    closest_aux = find_closest_row_point(Aux_Met_Data, timestamp_dt, time_window_td)
    #print(f"Found Aux Met row: {closest_aux}")

        # Initialize the file_data dictionary with common data
    file_data = {
                'Timestamp': timestamp}

        # Include 'T_air' and 'RH' from closest_aux
    if closest_aux is not None:
            file_data['T_air'] = closest_aux.get('T_air')
            file_data['RH'] = closest_aux.get('RH')

            # First check for 'sky_temp' in Aux_Met_Data
            file_data['sky_temp'] = closest_aux.get('sky_temp')

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

            file_data['rho_v'] = vapor_density(T_air, RH)
    else:
            file_data['rho_v'] = None


    return file_data



def process_and_export_corrected_point_data(output_csv_path, point_data_path, point_dist, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window, emissivity):
    """
    Process and export corrected point data.

    Parameters:
    point_data_path (str): Path to the CSV file containing point data.
    point_dist (float): Distance value for the point data.
    Aux_Met_Data (pd.DataFrame): DataFrame containing auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing FLUX meteorological data.
    Aux_Met_window (int): Time window in minutes for searching auxiliary meteorological data.
    FLUX_Met_window (int): Time window in minutes for searching FLUX data.
    emissivity (float): Emissivity value.

    Returns:
    pd.DataFrame: DataFrame containing processed data, including corrected and uncorrected point values.
    """
    try:
        # Load the point data
        point_data = pd.read_csv(point_data_path)
        # print("Point data head:")
        # print(point_data.head())

        # print('point data: ')
        # print(point_data)

        point_data['timestamp_dt'] = pd.to_datetime(point_data['timestamp'], format="%m/%d/%y %H:%M")
        print("Point data head with dimestamp_dt:")
        print(point_data.head())

        # Initialize a list to hold the processed data
        processed_data = []

        for _, row in point_data.iterrows():
            #print(f"Processing row: {row}")
            timestamp = row['timestamp']
            point_value = row['temp_value']
            #print(f"Timestamp: {timestamp}, Point Value: {point_value}")

            # Extract necessary information from the file_data
            file_data = find_matching_logger_data_point(timestamp, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window)
            if file_data is None:
                print(f"Skipping timestamp {timestamp} due to error in finding matching logger data.")
                continue

            print(f"File Data: {file_data}")

            # Calculate tau using the distance
            if file_data['rho_v'] is not None and point_dist is not None:
                file_data['tau'] = atm_trans(point_dist, file_data['rho_v'])

            print(f"File Data with Tau: {file_data}")

            try:
                # Correct the point value
                corrected_value = correct_point_data(point_value, file_data, emissivity)
                #print(f"Corrected Value: {corrected_value}")

                # Append corrected and uncorrected data to the processed data list
                processed_data.append({
                    'timestamp': timestamp,
                    'temp_value_uncorrected': point_value,
                    'temp_value_corrected': corrected_value
                })
            except Exception as err:
                print(f"There was an exception correcting point data: {err}")
                continue

        # Convert the list of dictionaries to a DataFrame
        result_df = pd.DataFrame(processed_data)
        # print("Processed DataFrame:")
        # print(result_df.head())
        result_df.to_csv(output_csv_path, index=False)
        
        return result_df
    
    except Exception as e:
        # Log the exception message
        print(f"Error processing {point_data_path}: {e}")
        return pd.DataFrame()



def setup_gui_and_start_point(point_data_path, point_dist, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window, output_csv_path, emissivity):
    """
    Sets up the graphical user interface (GUI) and starts the point data processing in a separate thread.

    This function initializes the GUI for the point data processing application. It creates a
    main window with a progress bar and then starts the point data processing in a separate thread
    using the provided parameters. The GUI provides visual feedback on the progress of the
    point data processing.

    Parameters:
    point_data_path (str): Path to the CSV file containing point data.
    point_dist (float): Distance value for the point data.
    Aux_Met_Data (pd.DataFrame): DataFrame containing auxiliary meteorological data.
    FLUX_Met_Data (pd.DataFrame): DataFrame containing FLUX meteorological data.
    Aux_Met_window (int): Time window in minutes for searching auxiliary meteorological data.
    FLUX_Met_window (int): Time window in minutes for searching FLUX meteorological data.
    output_csv_path (str): Path to the output CSV file where the results will be saved.
    emissivity (float): Emissivity value.
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
            processed_df = process_and_export_corrected_point_data(output_csv_path, point_data_path, point_dist, Aux_Met_Data, FLUX_Met_Data, Aux_Met_window, FLUX_Met_window, emissivity)
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

    root.mainloop()
