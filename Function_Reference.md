# Function Reference

* [Pre-Processing](#pre-processing)
* [Image Processing](#image-processing)
* [Point Processing](#point-processing)
* [Atmospheric Functions](#atmospheric-functions)
* [Stefan-Boltzmann Law Functions](#stefan-boltzmann-law-functions)
* [Region(s) of Interest](#regions-of-interest)
* [Image Display](#image-display)
* [Miscellaneous](#miscellaneous)

## Pre-Processing (located in Main_Functions.py)

| Function Name | Arguments | Description | Returns |
|--------------|-----------|-------------|---------|
| **run_pipeline** | `config_path` (str): Path to the JSON configuration file. | Runs the **entire processing pipeline**, calling `process_data()`, and prints confirmation upon completion. | `None` |
| **load_json** | `config_path` (str): Path to the JSON configuration file. | Reads the **configuration settings** from a JSON file and returns them as a dictionary. Prints an error if the file is missing. | `dict`: Parsed JSON configuration settings. Returns `None` if the file is not found. |
| **process_data** | `config_path` (str): Path to the JSON configuration file. | Determines whether **image-based** or **point-based** data is being processed and calls the appropriate functions to initialize data and start the GUI. | `None` |
| **preprocess_dataframe** | `df` (DataFrame): Input DataFrame containing meteorological data. | Preprocesses the DataFrame by: <br> 1. Converting `'TIMESTAMP_END'` to datetime and setting it as the index. <br> 2. If `'LW_IN'` exists, computing `'sky_temp'` using `radiance_to_temp()`. | `DataFrame`: The preprocessed DataFrame with the updated index and optional `'sky_temp'` column. |


## Image Processing (located in Main_Functions.py)

| Function Name | Arguments | Description | Returns |
|--------------|-----------|-------------|---------|
| **start_processing_thread** | `*args`: Variable-length arguments to be passed to `process_images_in_folders`. | Initializes and starts a new **thread** for image processing, allowing asynchronous execution. This prevents the main program from freezing during long processing tasks. | `None` |
| **setup_gui_and_start** | `base_folder` (str): Path to the image dataset. `roi_masks` (dict): ROI masks. `average_distances` (dict): Average distances for each ROI. `Aux_Met_Data` (pd.DataFrame): Auxiliary meteorological data. `FLUX_Met_Data` (pd.DataFrame): FLUX meteorological data. `Aux_Met_window` (int): Time window for auxiliary data matching. `FLUX_Met_window` (int): Time window for FLUX data matching. `output_csv_path` (str): Output CSV file path. `emissivity` (float): Emissivity value. | Initializes the GUI for image processing, providing progress tracking while running the processing in a separate thread. | `None` |
| **initialize_data_from_config_image** | `config_path` (str): Path to the configuration JSON file. | Reads configuration settings and loads meteorological data, initializes ROI masks, and calculates average distances. Ensures `FLUX_Met_Data` is only loaded if `LW_IN` is missing from `Aux_Met_Data`. | `tuple`: Contains `Aux_Met_Data`, `FLUX_Met_Data`, `roi_masks`, `average_distances`, `Aux_Met_window`, `FLUX_Met_window`, `base_folder`, `output_csv_path`, `emissivity`. |
| **get_image_shape** | `image_path` (str): Path to the image file. | Retrieves the dimensions (height, width) of an image. | `tuple`: Image dimensions (`height`, `width`). |
| **read_and_convert_image** | `image_path` (str): Path to the image file. | Reads an image and converts it to 16-bit format if it is 32-bit. | `np.ndarray`: Image array suitable for OpenCV. |
| **create_roi_masks** | `csv_path` (str): Path to the CSV file containing ROI definitions. `image_shape` (tuple): Shape of the image (`height`, `width`). `image_path` (str): Path to the image file. | Creates a dictionary containing masks for each ROI, ensuring image shape consistency. | `dict`: Dictionary of ROI labels and their corresponding masks. |
| **calculate_average_distances** | `distance_csv` (str): Path to the CSV file containing distances. `roi_masks` (dict, optional): ROI masks (used if `data_type='pointcloud'`). `data_type` (str): Data format ('pointcloud' or 'average'). | Reads or calculates average distances for each ROI based on the provided data type. | `dict`: Dictionary of ROI labels and their average distances. |
| **initialize_roi_masks_and_distances** | `roi_csv_path` (str): Path to the CSV file containing ROI definitions. `distance_csv_path` (str): Path to the CSV file containing distances. `image_path` (str): Path to the image file. `data_type` (str): Type of data in the distance CSV ('pointcloud' or 'average'). | Initializes ROI masks and calculates or reads average distances from a CSV file. | `tuple`: Contains `roi_masks` and `average_distances`. |
| **calculate_roi_means_for_tiff** | `tiff_image_path` (str): Path to the `.tiff` image file. `roi_masks` (dict): Dictionary of ROI masks. | Calculates the mean temperature values for each ROI in a `.tiff` image. | `dict`: Dictionary of ROI labels and their mean values. |
| **extract_timestamp** | `tiff_image_path` (str): Path to the `.tiff` image file. | Extracts and converts the timestamp from the TIFF filename. | `pd.Timestamp`: Extracted timestamp as a Pandas datetime object. |
| **find_closest_row** | `df` (pd.DataFrame): DataFrame to search. `timestamp` (pd.Timestamp): Target timestamp. `time_window` (pd.Timedelta): Time window for searching. | Finds the closest row in a DataFrame within a given timestamp range. | `pd.Series` or `None`: Closest matching row or `None` if no match is found. |
| **find_matching_logger_data** | `tiff_image_path` (str): Path to the TIFF image. `Aux_Met_Data` (pd.DataFrame): Auxiliary meteorological data. `FLUX_Met_Data` (pd.DataFrame): FLUX meteorological data. `Aux_Met_window` (int): Time window for auxiliary data. `FLUX_Met_window` (int): Time window for FLUX data. | Finds meteorological data that matches the TIFF timestamp. | `dict`: Extracted data containing `T_air`, `RH`, `sky_temp`, `LW_IN`, `rho_v`, and `tau`. |
| **correct_integer_image** | `integerImg` (np.ndarray): Integer image data. `file_data` (dict): Meteorological data dictionary. `emiss` (float): Emissivity value. | Applies temperature corrections to an integer image using multiple methods: standard, `tau = 1`, `emiss = 1`, and without reflection correction. | `tuple`: Contains four corrected image datasets (`integerObj`, `integerObj_tau1`, `integerObj_emiss1`, `integerObj_noReflect`). |
| **process_and_export_corrected_roi_means** | `tiff_image_path` (str): Path to the TIFF image. `roi_masks` (dict): ROI masks. `average_distances` (dict): Average distances. `Aux_Met_Data` (pd.DataFrame): Auxiliary meteorological data. `FLUX_Met_Data` (pd.DataFrame): FLUX meteorological data. `Aux_Met_window` (int): Time window for auxiliary data. `FLUX_Met_window` (int): Time window for FLUX data. `emiss` (float): Emissivity value. | Extracts meteorological data, corrects ROI mean values, and stores results for a TIFF image. | `dict`: Processed data, including corrected ROI means. |
| **process_images_in_folders** | `base_folder` (str): Folder containing images. `roi_masks` (dict): ROI masks. `average_distances` (dict): Average distances. `Aux_Met_Data` (pd.DataFrame): Auxiliary meteorological data. `FLUX_Met_Data` (pd.DataFrame): FLUX meteorological data. `Aux_Met_window` (int): Time window for auxiliary data. `FLUX_Met_window` (int): Time window for FLUX data. `output_csv_path` (str): Output CSV file path. `emissivity` (float): Emissivity value. `progress_bar` (tk.Progressbar): GUI progress bar. `status_label` (tk.Label): GUI status label. `root` (tk.Tk): Tkinter main window. | Iterates through images in the folder, processes each image, and tracks progress via a GUI. Saves processed data to CSV. | `pd.DataFrame`: Final processed data sorted by timestamp. |

## Point Processing (located in Main_Functions.py)

| Function Name | Arguments | Description | Returns |
|--------------|-----------|-------------|---------|
| **setup_gui_and_start_point** | `point_data_path` (str): Path to the CSV file containing point data. `point_dist` (float): Distance value for the point data. `Aux_Met_Data` (pd.DataFrame): Auxiliary meteorological data. `FLUX_Met_Data` (pd.DataFrame): FLUX meteorological data. `Aux_Met_window` (int): Time window for auxiliary data matching. `FLUX_Met_window` (int): Time window for FLUX data matching. `output_csv_path` (str): Path to store the output CSV. `emissivity` (float): Emissivity value. | Initializes a GUI for processing point data, providing visual progress tracking while running the processing in a separate thread. | `None` |
| **initialize_data_from_config_point** | `config_path` (str): Path to the configuration JSON file. | Reads configuration settings and loads necessary meteorological data. Ensures `FLUX_Met_Data` is only loaded if `LW_IN` is missing from `Aux_Met_Data`. | `tuple`: Contains `Aux_Met_Data`, `FLUX_Met_Data`, `Aux_Met_window`, `FLUX_Met_window`, `point_dist`, `output_csv_path`, `emissivity`, and `point_data_path`. |
| **correct_point_data** | `point_data` (float): Point data value. `file_data` (dict): Meteorological data dictionary. `emiss` (float): Emissivity value. | Applies temperature corrections to point data with multiple variations: standard, `tau = 1`, `emiss = 1`, and no reflectance. | `tuple`: Contains four corrected temperature values (`correctedPoint`, `correctedPoint_tau1`, `correctedPoint_emiss1`, `correctedPoint_noReflect`). |
| **find_closest_row_point** | `df` (pd.DataFrame): DataFrame to search. `timestamp` (pd.Timestamp): Target timestamp. `time_window` (pd.Timedelta): Time window for searching. | Finds the closest row in a DataFrame based on the timestamp and a given time window. | `pd.Series` or `None`: Returns the closest row or `None` if no match is found. |
| **find_matching_logger_data_point** | `timestamp` (str): Target timestamp. `Aux_Met_Data` (pd.DataFrame): Auxiliary meteorological data. `FLUX_Met_Data` (pd.DataFrame): FLUX meteorological data. `Aux_Met_window` (int): Time window for auxiliary data matching. `FLUX_Met_window` (int): Time window for FLUX data matching. | Extracts meteorological data matching the provided timestamp, prioritizing `Aux_Met_Data`. If `sky_temp` is missing, `FLUX_Met_Data` is checked. | `dict`: Contains extracted meteorological data (`T_air`, `RH`, `sky_temp`, `LW_IN`, `rho_v`, and `tau`). |
| **process_and_export_corrected_point_data** | `output_csv_path` (str): Path to store the processed CSV file. `point_data_path` (str): Path to the CSV file containing point data. `point_dist` (float): Distance value for the point data. `Aux_Met_Data` (pd.DataFrame): Auxiliary meteorological data. `FLUX_Met_Data` (pd.DataFrame): FLUX meteorological data. `Aux_Met_window` (int): Time window for auxiliary data matching. `FLUX_Met_window` (int): Time window for FLUX data matching. `emissivity` (float): Emissivity value. | Processes point data by applying temperature corrections and adding meteorological variables. Saves the processed data to a CSV file. | `pd.DataFrame`: Processed data containing meteorological values and corrected point values. |

## Atmospheric Functions (located in Main_Functions.py)

| Function Name           | Arguments                                                                 | Description                                                                                     | Returns                                         |
|------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------|
| **pressure_at_elevation** | `h` (float): Elevation in meters.                                      | Calculates atmospheric pressure at a given elevation using the simplified barometric formula.  | `float`: Atmospheric pressure in Pascals (Pa). |
| **vapor_density**      | `T_air` (float): Air temperature in Celsius. <br> `RH` (float): Relative humidity in percentage (0 to 100). <br> `P` (float): Atmospheric pressure in Pascals (Pa). | Computes the **actual vapor density** in the air using the **Clausius-Clapeyron relation** and the **ideal gas law**, adjusting for atmospheric pressure. | `float`: Actual vapor density in g/m³. |
| **atm_trans**         | `dist` (float): Distance to the object. <br> `rho_v` (float): Vapor density in g/m³. | Estimates **atmospheric transmittance** using an **abbreviated LOWTRAN model**, accounting for vapor density effects over distance. | `float`: Atmospheric transmittance (ratio between 0 and 1). |

## Stefan-Boltzmann Law Functions (located in Main_Functions.py)

| Function Name | Arguments | Description | Returns |
|--------------|-----------|-------------|---------|
| **radiance_from_temperature** | `T` (float): Temperature in Celsius. | Computes the **radiant flux** from temperature using the **Stefan-Boltzmann Law**. | `float`: Radiant flux in W/m². |
| **radiance_to_temp** | `E` (float): Radiant flux in W/m². | Computes the **temperature in Celsius** from radiant flux using the **Stefan-Boltzmann Law**. | `float`: Temperature in Celsius. |

## Region(s) of Interest (located in ROI_Viz_Functions.py)

| Function Name | Arguments | Description | Returns |
|--------------|-----------|-------------|---------|
| **draw_and_label_poly_rois** | `image_path` (str): Path to the thermal image file. | Allows users to **interactively draw and label polygonal ROIs** on a thermal image. Users can left-click to add points, right-click to complete an ROI, and press 'x' to exit. The labeled ROIs are saved to a CSV file. | `list`: A list of dictionaries containing ROI labels and their corresponding points. |
| **save_rois_to_csv** | `rois` (list): List of labeled ROIs. `filename` (str, optional): CSV filename (default: `"rois.csv"`). | Saves the labeled ROIs to a .csv file, where each row contains a label followed by a sequence of x, y coordinates. | `None` |
| **overlay_rois_from_csv** | `image_path` (str): Path to the thermal image. `csv_path` (str): Path to the .csv file with ROI data. `output_image_path` (str, optional): Path to save the overlay image. | Loads an image and **overlays labeled ROIs** from a .csv file. Optionally saves the modified image with the overlays. | `None` |

## Image Display (located in ROI_Viz_Functions.py)

|Function Name | Arguments | Description| Returns |
|------|------|-----|-----|
|**display_tiff_with_colormap**|`tiff_path` (str): Path to the TIFF image, `colormap` (str, optional): Colormap to use (default: `'inferno'`). | Converts a `.tiff` image to a format suitable for OpenCV, applying a colormap. | numpy.ndarray: The converted image.|
|**save_thermal_image** | `tiff_path` (str): Path to the input TIFF image, `save_path` (str): Path to save the output image, `colormap` (str, optional): Colormap to use (default: `'inferno'`).| Saves a `.tiff` image with a specified colormap, temperature color bar, and normalization. | None |

## Miscellaneous (located in Main_Functions.py)

| Function Name | Arguments | Description | Returns |
|--------------|-----------|-------------|---------|
| **human_readable_time** | `seconds` (float or int): Time duration in seconds. | Converts seconds into a **human-readable format**, displaying hours, minutes, and seconds when applicable. | `str`: Time formatted as `"X hours, Y minutes, Z.ZZ seconds"` or `"Y minutes, Z.ZZ seconds"`, or `"Z.ZZ seconds"` if less than a minute. |

