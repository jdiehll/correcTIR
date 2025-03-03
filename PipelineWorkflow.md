# TIRPost Detailed Pipeline Processing

## Summary
* [Entry Point](#entry-point): `run_pipeline()`
* [Determine Processing Type](#processing-type): `process_data()`
* [Process Images](#process-images): `setup_gui_and_start()` → `process_and_export_corrected_roi_means()` → `correct_integer_image()`
* [Process Points](#process-points): `setup_gui_and_start_point()` → `process_and_export_corrected_point_data()` → `correct_point_data()`

_**For more details on specific functions, refer to [Function Reference](Function_Reference.md).**_

## Entry Point

`run_pipeline()`

This function starts the processing pipeline and calls `process_data()`

## Processing Type

`process_data()`

This function reads the configuration .json and determines if **image** or **point** data is being processed.

Calls `load_json()`
* If data == ["image"](#process-images) → calls `initialize_data_from_config_image()` and `setup_gui_and_start()`
* If data == ["point"](#process-points) → calls `initialize_data_from_config_point()` and `setup_gui_and_start_point()`

## Process Images

### Load Data

`initialize_data_from_config_image(config_path)`

This function reads the configuration settings from a .json file and:
* Loads and preprocesses Auxiliary and FLUX meteorological data (when needed).
* Initializes ROI masks and calculates average distances.
  
These outputs are passed to `setup_gui_and_start()`, where processing begins.

Calls:
* `preprocess_dataframe()` → Cleans and prepares meteorological data.
* `initialize_roi_masks_and_distances()` → Loads ROI masks & distance values.

### Set-Up

`setup_gui_and_start(base_folder, roi_masks, ...)`

* Initializes the internal Python interface for processing .tiff images.
* Provides progress tracking for real-time feedback.
  
Calls:
* `start_processing_thread()` → Runs processing asynchronously.
---

`start_processing_thread()`

* Starts and creates a separate processing thread to prevent the main interface from freezing.
  
Calls:
* `process_images_in_folders()` → Iterates over images in the folder.
---

`process_images_in_folders()`

* Loops through all .tiff images in the specified folder.
* Applies temperature corrections to each ROI.
* Saves processed results to a .csv file.
* Tracks total processing time.

Calls:
* `process_and_export_corrected_roi_means()` → Runs for each image.
* `human_readable_time()` → Logs total processing time.

### Processing & Correction

`process_and_export_corrected_roi_means(tiff_image_path, ...)`

This function performs the core image correction, including:
* Finding meteorological data that matches the image timestamp.
* Extracting ROI mean values from the .tiff image.
* Applying temperature corrections using `correct_integer_image()`.

Calls:
* `find_matching_logger_data()` → Matches timestamps to meteorological data.
* `calculate_roi_means_for_tiff()` → Extracts ROI mean values.
* `atm_trans()` → Calculates atmospheric transmittance.
* `correct_integer_image()` → Applies multiple temperature corrections.

_**For more details on function behavior, see [Function Reference](Function_Reference.md): Image Processing.**_

## Process Points

### Load Data

`initialize_data_from_config_point(config_path)`

This function reads the configuration settings from a .json file and:
* Loads and preprocesses Auxiliary and FLUX meteorological data (when needed).
  
These outputs are passed to `setup_gui_and_start()`, where processing begins.

Calls:
* `preprocess_dataframe()` → Cleans and prepares meteorological data.

### Set-Up

`setup_gui_and_start_point(point_data_path, point_dist, ...)`

* Initializes the interal Python GUI interface for processing point data.
* Provides progress tracking for real-time feedback with internal functions.
  
Calls:
* `process_and_export_corrected_point_data()` → Runs processing asynchronously.
---

### Processing & Correction

`process_and_export_corrected_point_data(output_csv_path, ...)`

This function performs the core point correction, including:
* Finding meteorological data that matches the image timestamp.
* Applying temperature corrections using `correct_point_data()`.

Calls:
* `find_matching_logger_data_point()` → Matches timestamps to meteorological data.
* `atm_trans()` → Calculates atmospheric transmittance.
* `correct_point_data()` → Applies multiple temperature corrections.

_**For more details on function behavior, see [Function Reference](Function_Reference.md): Point Processing.**_
