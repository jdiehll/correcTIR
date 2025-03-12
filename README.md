<img src='./Figures/logo.png' align="right" height="175" />

<br><br>  <!-- Adds vertical space -->

# correcTIR

&nbsp;  <!-- Adds an empty space to create separation -->

## Python Package & GUI Documentation


### How to cite this package

>Placeholder


* [Overview](#overview)
* [Set-Up](#set-up)
* [Python Package](#python-package)
* [GUI](#GUI)
  * [Windows](#windows)
  * [MacOS](#macos)
  * [Linux](#linux) 
* [Requirements](#Requirements)
* [Output](#output-file)
* [Example Use](#Example-Use)
  * [Images](#images)
  * [Point](#point)
* [Additional Features](#Additional-Features)
* [Troubleshooting & FAQs](#Troubleshooting--FAQs)
* [Feedback](#feedback)

## Overview

correcTIR enhances the consistency and reliability of thermal measurements by standardizing the thermal data post-processing workflow. The GUI is designed to help new users swiftly process temperature data, while the Python package provides comprehensive customization options for more experienced users. correcTIR is capable of handling thermal data both as images (stored in .tiff format) and as point measurements (stored in .csv format with each entry as a row), with temperature values in degrees Celsius.

For detailed file formatting and setup instructions, see the [Set-Up](#set-up) section below.  
To learn more about the overall pipeline workflow, refer to [Pipeline Workflow](PipelineWorkflow.md).  
For specific function details, check the [Function Reference](Function_Reference.md).

The development of correcTIR is grounded in rigorous thermal theory, aiming to address the common challenges in thermal data accuracy and utility. For a comprehensive explanation of the theoretical foundations and expanded details on the functionalities of correcTIR, please consult:

>Placeholder for my paper

>Aubrecht, D. M., Helliker, B. R., Goulden, M. L., Roberts, D. A., Still, C. J., & Richardson, A. D. (2016). Continuous, long-term, high-frequency thermal imaging of vegetation: Uncertainties and recommended best practices. Agricultural and Forest Meteorology, 228–229, 315–326. https://doi.org/10.1016/j.agrformet.2016.07.017


## Set-Up

To ensure smooth use of correcTIR, your input data must follow specific structures. Below are the key requirements. Blank templates are available in the **"Template"** folder. For practical examples and additional guidance, see [Example Use](#example-use) section and **"Test Data"** folder.

For image-based processing, you will need:

✅ [Correct image folder structure and naming](#image-folder-structure--naming)

✅ [Auxiliary data](#auxiliary-data-files) (1 or 2 files, depending on measurement interval)

✅ [Distance data](#distance-data-file) (either a point cloud or an average distance file)

✅ [Region of Interest (ROI) file](#region-of-interest-roi-file)

📌 Total: 3–4 files

For point-based processing, you will need:

✅ [Point data](#point-data)

✅ [Auxiliary data](#auxiliary-data-files) (1 or 2 files, depending on measurement interval)

📌 Total: 2–3 files

⏳ Timezone: Your data must include a timestamp field in the required format to properly align images and point measurements. However, any timezone is acceptable as long as it is applied consistently across all input files. We recommend using local time.

🚀 Once your input data is in the correct format, you are ready to run the pipeline! You can now jump to the [Configuration File](#configuration-file) section to set up your processing for both Python package and GUI usage.

_Note: correcTIR is optimized for handling data from individual sites one at a time._

### Image Folder Structure & Naming

* Organize your thermal image measurements using the following folder structure for proper functioning:
  
Images/ Year/ Month/
  
**Example:** Images/ 2023/ 10/
  
* Each image should follow this naming convention:
  
sitename_YYYYMMDD_HHmmss.tiff

_Note: HHmmss can be in any timezone as long as it is applied consistently across all input files._

**Example:** NIWOT_20211002_141503.tiff
  
<figure>
    <img src='./Figures/FolderStructure.png' height="350" />
</figure>

### Point Data

* Organize your thermal point measurements using the following file structure for proper functioning:
  
  | timestamp      | temp_value      |
  |----------------|-----------------|
  | 1/16/17 14:30  | -5.1133         |
  | 1/16/17 14:45  | -5.1454         |
  | ...            | ...             |
  
**Column Descriptions:**

timestamp: Represents the date and time of the recorded measurement in MM/DD/YY HH:mm format. This timestamp uses the 24-hour time notation, where '14:30' indicates 2:30 PM. Any timezone is acceptable as long as it is applied consistently across all input files.
  
temp_value: Temperature measurements in degrees Celsius.

### Auxiliary Data File(s)

Processing thermal data requires additional environmental data, ideally collected at the same time interval as the thermal measurements. To align these datasets, you can specify a search window (in minutes), which defines how far forward in time the system will look to find the closest available auxiliary data point that matches each thermal data timestamp. If no auxiliary data point is found within the specified search window, the corresponding thermal measurement will not be processed. This ensures that only thermal data with valid environmental context are included in the analysis. See the [Example Use](#example-use) section for details.

| TIMESTAMP_END     | T_air    | RH      | LW_IN   |
|-------------------|----------|---------|---------|
| 8/30/17 14:35     | 14.33781 | 32.02635| 256.20  |
| 8/30/17 14:40     | 14.43806 | 31.93127| 255.05  |
| ...               | ...      | ...     | ...     |

**Column Descriptions:**

_Note: The order of the columns does not impact processing. Additional columns can be included in the dataset as long as the required columns are present and correctly labeled._

TIMESTAMP_END: Represents the end date and time of the recorded measurement in MM/DD/YY HH:mm format. This timestamp uses the 24-hour time notation, where '14:35' indicates 2:35 PM. Any timezone is acceptable as long as it is applied consistently across all input files.
  
T_air: Air temperature measurements in degrees Celsius.

RH: Relative humidity measurements in float percent (%).

LW_IN: Incoming longwave radiation measurements in W/m².

_Note: The software supports reading multiple auxiliary files when variables are recorded at different time intervals. For example:_

_* File 1 - Incoming longwave radiation averaged over 30-minute intervals from a flux tower._

_* File 2 - Air temperature and relative humidity recorded at 5-minute intervals._

_To accommodate such cases, you can provide the software with two separate auxiliary files. You can find blank templates in the Template folder._

### Distance Data File

_This file is **not** required if you're processing point-based measurements._

For image-based processing, the distance between the temperature measurement instrument and the region(s) of interest must be specified. You can provide this information using one of two methods:

* Point Cloud Projection: For more precise distance corrections, you can supply a point cloud (.csv) containing the distance **(in meters)** of each pixel from the camera within its field of view (FOV).

Example: If your thermal camera has a resolution of 480 × 640 pixels, your point cloud .csv should contain 480 rows and 640 columns, with each cell representing the distance (in meters) from the camera to that pixel. Examples files can be found in the "Test Data" folder. 

* Average Distance(s): Alternatively, you can manually enter the average distance(s) for each region of interest.
  
| label   | average_distance |
|---------|------------------|
| F1      | 28.56            |
| S1      | 18.31            |
| ...     | ...              |

**Column Descriptions:**

label: A unique identified for the region of interest. This label must be consistent across all processing steps and should match the label used for the Region of Interest (ROI) file in image processing.
  
average_distance: Average distance from the camera to specified region of interest in meters.
  
### Region of Interest (ROI) File

_This file is **not** required if you're processing point-based measurements._

The ROI file defines regions of interest (ROIs) within thermal image datasets. If the camera is shifted or moved, the ROIs must be updated to ensure accurate analysis. You can draw ROIs manually using the GUI (Draw ROI button in the Image Processing section) and in Python, use the function draw_and_label_poly_rois from the package. For additional ROI functions, such as saving an image with overlaid ROIs, see [Additional Features](#additional-features) section. 

| Label  | Point_1_x  | Point_1_y  | Point_2_x  | Point_2_y  | Point_3_x  | Point_3_y  | ...      | Point_X_x  | Point_X_y  |
|--------|------------|------------|------------|------------|------------|------------|----------|------------|------------|
| F1     | 298        | 231        | 294        | 247        | 299        | 279        |          |            |            |
| S1     | 287        | 244        | 283        | 284        | 289        | 312        |          |            |            |
| ...    | ...        | ...        | ...        | ...        | ...        | ...        | ...      | ...        | ...        |

**Column Descriptions:**

Label: A unique identified for the region of interest. This label must be consistent across all processing steps and should match the label used for the Average Distance file if using.
  
Point_1_x, Point_1_y: The x- and y-coordinates (in pixels) of the first point defining the ROI.

Additional Points: Each ROI consists of multiple points, with repeating Point_X_x, Point_X_y columns for however many points define the region.

The .csv file should start with a header row specifying the label and then pairs of x and y coordinates. Each subsequent row defines a labeled ROI with its set of coordinates.

### Configuration File

Now that your input files are in the correct format, you are ready to run the processing pipeline. The workflow for both the Python package and the GUI is illustrated below:
<figure>
    <img src="./Figures/TIRPostWorkflow.jpg" style="max-width: 80%; max-height: 600px; width: auto; height: auto; display: block; margin: auto;" />
</figure>

The GUI guides users through all steps and functions automatically, ensuring a seamless workflow.

For users running the Python package, some additional manual steps are required. Specifically, they must independently execute the overlay ROI and draw ROI functions before running the pipeline. These steps are detailed in the [Additional Features](#additional-features) section.

Whether you're using the Python package or the GUI to run the full pipeline and process your thermal data, you will need to create a .json configuration file.

This file serves two key purposes:

1. It is executed by the package to process your data.
2. It provides a record of the processing steps you've applied.

Below, we demonstrate what the "Set Inputs" stage of the workflow looks like across both platforms (Python package and GUI) and processing scenarios (images or points). Using the provided "Test Data", you should be able to replicate these steps, see the [Example Use](#example-use) section for details.

### Images

The following table lists and describes the required inputs for processing image data:
| Input                 | Description                                                                                  |
|-----------------------| ---------------------------------------------------------------------------------------------|
| `data`                  | image                                                                                        |
| `aux_met_data_path`     | File path to .csv file containing auxillary data                      |
| `aux_met_window`        | Search window for auxiliary data (integer, in minutes). This window determines how far **forward** in time to search for matching auxiliary data relative to the thermal data timestamp.                      |
| `flux_met_data_path`    | File path to .csv file containing flux data (**optional**, if at different interval)        |
| `flux_met_window`       | Search window for flux data (integer, in minutes, **optional**). This window determines how far **forward** in time to search for matching auxiliary data relative to the thermal data timestamp.      |
| `emissivity`            | Target emissivity value (float)           |
| `elevation`            | Site elevation for water density correction (float)           |
| `roi_path`              | File path to the .csv file containing ROI (Region of Interest) data                 |
| `roi_dist_path`         | File path to the .csv file containing ROI distance data             |
| `data_type`             | ROI distance data type (**point cloud** or **average**)           |
| `first_image_path`      | File path to a .tiff image used to set configuration **(doesn't have to be first image)**   |
| `base_folder`           | Path to base image folder |
| `output_csv_path`       | File path where the output .csv will be stored **(automatically created)**           |

_Note: The order of the inputs does not impact processing as long as the required input fields are present and correctly labeled. No additional parameters should be present._

**Python Package:** You must manually enter these inputs into a .json configuration file. The file must be named "config.json" for the package to work with the entire processing pipeline.

**GUI:** A built-in guide assists in creating the .json configuration file.

Once the .json file is created, you're ready to process your image data!

### Point

The following table lists and describes the required inputs for processing point data:
| Input                 | Description                                                            |
|-----------------------| -----------------------------------------------------------------------|
| `data`                  | point                                                                  |
| `aux_met_data_path`     | File path to .csv file containing auxillary data                              |
| `aux_met_window`        | Search window for auxiliary data (integer, in minutes). This window determines how far **forward** in time to search for matching auxiliary data relative to the thermal data timestamp.                    |
| `flux_met_data_path`    | File path to .csv file containing flux data (**optional**, if at different interval) |
| `flux_met_window`       | Search window for flux data (integer, in minutes, **optional**). This window determines how far **forward** in time to search for matching auxiliary data relative to the thermal data timestamp.              |
| `emissivity`            | Target emissivity value (float)                          |
| `elevation`            | Site elevation for water density correction (float)           |
| `point_data_path`       | File path to the .csv file containing point data                                      |
| `point_dist`           | Distance from the instrument to the target (float, in **meters**)                |
| `output_csv_path`       | File path where the output .csv will be stored **(automatically created)**                           |

_Note: The order of the inputs does not impact processing as long as the required input fields are present and correctly labeled. No additional parameters should be present._

**Python Package:** You must manually enter these inputs into a .json configuration file. The file must be named "config.json" for the package to work with the entire processing pipeline.

**GUI:** A built-in guide assists in creating the .json configuration file.

Once the .json file is created, you're ready to process your point data!

## Python Package

For users who want to run individual functions or the full pipeline in Python, the correcTIR package can be downloaded, installed, and executed as follows:

### Installation

1. Clone the Repository

To download the latest version of the repository, open a terminal or command prompt and run:

```
git clone [https://github.com/Thermal-Cam-Network/TIRPost.git](https://github.com/Thermal-Cam-Network/TIRPost.git)
```
If you don’t have Git installed, you can download the repository manually by clicking Code → Download ZIP on GitHub and extracting the contents.

2. Navigate into the Cloned Directory

Move into the project folder:

```
cd correcTIR
```

3. Create and Activate a Virtual Environment (Recommended)

To create a virtual environment:
```
python -m venv /path/to/new/virtual/environment
```
_Note it may be necessary to type python3 and not just python._

For instructions on creating a Conda environment, refer to the [conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 

Once the virtual environment is created, install all required correcTIR dependencies (see [Requirements](#requirements) section). 

4. Pull the Latest Updates (Optional but Recommended)
To update your local copy of the repository:
```
git pull origin main
```

### Running the Processing Pipeline

To successfully run the correcTIR package, follow these steps:

1. Prepare Your Files
* Ensure all required files are in the correct format, naming convention, and folder structure.
* Update and name your configuration file as "config.json" with your specific input details.
* Be sure all required correcTIR dependencies are installed (see [Requirements](#requirements) section).
_ Again we recommend working in a virtual environment._

2. Navigate to the Project Folder
Open a terminal and move into the project directory:
```
cd correcTIR
```

3. Run the Processing Pipeline
Execute the package:
```
python main.py
```
_Note: It may be necessary to type python3 and not just python._

This will begin processing your data using the inputs defined in "config.json".

4. Track Progress
A pop-up window will open, displaying real-time progress by tracking the number of processed images out of the total. This allows you to monitor the status of the processing pipeline.

5. Completion
Once processing is complete and the progress bar reaches 100%, you can close the pop-up window. Your processed data will now be available for review.

### Running Individual Functions

In addition to running the full correcTIR processing pipeline, you can also call individual functions from the Python package for custom workflows.

To do this, simply import and run the desired function in your Python script or interactive session.

Example: Converting Radiance to Temperature
```
from correcTIR.Main_Functions import radiance_to_temp

radiance_to_temp()
```

Example: Saving Thermal Image
```
from correcTIR.ROI_Viz_Functions import save_thermal_image

save_thermal_image()
```
For a full list of available functions and their required arguments, refer to the [Function Reference](Function_Reference.md).

## GUI
For users who prefer a graphical user interface (GUI), the correcTIR GUI is available as a pre-compiled application for the three most commonly used operating systems. The GUI is designed exclusively for running the entire processing pipeline—it does not support running individual functions separately. The GUI walks users through all necessary inputs, generates a config.json file, and then runs the pipeline. For more details, see [Example Use](#example-use) section.

_Note: correcTIR is not available on mobile devices (Android, iPhone)._

Regardless of the OS used either a Conda or Python virtual environment is recommended. Without a virtual environemt or Conda environment pyinstaller may not be available unless added to the path variable. 

Creating a virtual environment. Note it may be necessary to type python3 and not just python.
```
python -m venv /path/to/new/virtual/environment
```

To create a conda environment see the [conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 

Once a virtual environment is created install all of the TIR dependencies see [Requirements](#requirements) section. Run the following based on your OS in the terminal.

### Windows
```
python pyinstaller --windowed --onedir  ThermalCam.py
```

### MacOS
```
python pyinstaller -w --noupx --onedir ThermalCam.py
```

### Linux
```bash
python pyinstaller --windowed --onedir ThermalCam.py
```

## Requirements

The following requirements can be installed by running. 
_Note: Some packages may be downgraded on your system which is why having a virtual environemnt is key._

```
pip install -r requirements.txt
```

```
geopandas==1.0.1
matplotlib==3.9.1.post1
numpy==2.0.1
opencv-python==4.10.0.84
pandas==2.2.2
pillow==10.4.0
pyinstaller==6.10.0
ttkbootstrap==1.10.1
```

### Output File

You may be wondering what your output looks like. Your output file is a .csv with the follow variables:

### Images

| **Column Header**       | **Description**  |
|-------------------------|---------------------------------------------------------------------|
| `filepath`             | The complete file path where the original TIFF image is stored.   |
| `filename_short`       | A shortened or more convenient representation of the image file name. |
| `Timestamp`           | The date and time when the data were recorded, formatted as **MM/DD/YY HH:MM**. |
| `T_air`               | Air temperature at the time of image capture, measured in **°C**. |
| `RH`                 | Relative humidity at the time of data capture, expressed as a **percentage**. |
| `sky_temp`           | Temperature of the sky, inferred from the image, measured in **°C**. |
| `LW_IN`              | Longwave incoming radiation measured in **W/m²**. |
| `rho_v`              | Water vapor density in the air, measured in **g/m³**. |
| `tau`                | Atmospheric transmittance. |

For each Region of Interest (ROI_X), the following values are computed as additional columns in output .csv file:

| **Column Header**       | **Description**  |
|-------------------------|-----------------|
| `ROI_X_{stat}_uncorrected` |	Uncorrected thermal reading for {stat} (mean or percentile).   |
| `ROI_X_{stat}_fully_corrected` |	Fully corrected thermal reading for {stat}.|
| `ROI_X_{stat}_tau1` |	Thermal reading for {stat} with atmospheric corrections turned off. |
| `ROI_X_{stat}_objemiss1` |	Thermal reading for {stat} with reflected radiation corrections turned off.|

where {stat} represents one of the following statistical measures:

* mean: Mean value
* p1: 1st percentile
* p5: 5th percentile
* p10: 10th percentile
* p25: 25th percentile
* p50: 50th percentile (median)
* p75: 75th percentile
* p90: 90th percentile
* p95: 95th percentile
* p99: 99th percentile

These values are repeated for all available ROIs in the dataset.

### Points

| **Column Header**       | **Description**  |
|-------------------------|---------------------------------------------------------------------|
| `Timestamp`           | The date and time when the data were recorded, formatted as **MM/DD/YY HH:MM**. |
| `T_air`               | Air temperature at the time of image capture, measured in **°C**. |
| `RH`                 | Relative humidity at the time of data capture, expressed as a **percentage**. |
| `sky_temp`           | Temperature of the sky, inferred from the image, measured in **°C**. |
| `LW_IN`              | Longwave incoming radiation measured in **W/m²**. |
| `rho_v`              | Water vapor density in the air, measured in **g/m³**. |
| `tau`                | Atmospheric transmittance. |
| `temp_value__uncorrected`  | Uncorrected thermal reading. |
| `temp_value_corrected`    | Fully corrected thermal reading. |
| `temp_value_tau1`    | Corrected thermal reading if atmospheric corrections are turned off. |
| `temp_value_objemiss1`    | Corrected thermal reading if reflected radiation is turned off. |

## Example Use
I'm not exactly sure what this should look like. A video of me moving through the GUI?

## Additional Features

**Draw & Label ROI(s)**

Two functions allow users to interactively draw and label polygonal regions of interest (ROIs) on a thermal image and save them to a .csv file.

Input Parameters:
- image_path (str): Path to thermal image on which to draw the ROIs.
- rois (list): A list of dictionaries. Each dictionary contains the label and points of an ROI. Expected format for each dictionary: {'label': 'some_label', 'points': [(x1, y1), (x2, y2), ...]} **created by draw_and_label_poly_rois() function**
- filename (str, optional): Name of the .csv file to save the data to. Defaults to "rois.csv".

```
from correcTIR import draw_and_label_poly_rois, save_rois_to_csv

rois = draw_and_label_poly_rois('path/to/tiff/thermal/image/to/draw/ROIs.tiff')
save_rois_to_csv (rois, 'path/to/save/rois.csv')

```

**Checking ROI overlay is correct and option to save image with ROIs**

Display an image with ROIs and labels overlaid, as specified in a .csv file, and optionally save the overlay image to a specified path.

Input Parameters:
- image_path (str): Path to the image on which to overlay the ROIs.
- csv_path (str): Path to the CSV file containing ROI data. (Expected .csv format like ROI file expalined above)
- output_image_path (str, optional): Path to save the overlay image. If None, the image is not saved.

```
from correcTIR import overlay_rois_from_csv

overlay_rois_from_csv('path/to/tiff/thermal/image.tiff', 'path/to/csv/of/ROIs.csv','path/to/save/thermal/image.png')

```

**Saving thermal image without ROIs**

Save a .png image with a specified colormap, temperature color bar scale, and normalization.

Input Parameters:
- tiff_path (str): Path to the input .tiff image.
- save_path (str): Path to save the output image.
- colormap (str): The colormap to use. Default is 'inferno'.
```
from correcTIR import save_thermal_image

save_thermal_image('path/to/tiff/thermal/image.tiff','path/to/save/thermal/image.png')

```

**.seq files to .tiff files**

This process converts raw thermal images from .seq format into temperature-calibrated .tiff files for further analysis.
_Need to download exiftool and have it installed on computer @ https://exiftool.org/_

Assumes you have a folder for each month and running one year at a time.

1.  Update SEQ_Reformatting.ipynb with Your Paths
Open the script and update the following two directory paths:
```
base_dir = 'path/to/seq/image/folders'  # Path to raw .seq image folder  
output_base_dir = 'path/to/save/tiff/images'  # Path to save .tiff output files  
```
2. Run the Conversion Script
You can run SEQ_Reformatting.ipynb in any Python environment.
  
## Troubleshooting & FAQs

**I Don't Have All or Any of the Appropriate Auxiliary Data**

While it is best practice to collect auxiliary data near your temperature measurement instrument, we understand this is not always feasible.

In such cases, we recommend using ERA5-Land Hourly - ECMWF Climate Reanalysis reanalysis datasets for your site location and formatting them to match our required file structure. _Note: Make sure your search window is extended to capture this hourly dataset._
 
Platforms like Google Earth Engine can help retrieve relevant environmental data. An example of how to do this with Google can be found at this [link](https://code.earthengine.google.com/d5a4a985acef29b283d50e8dccaf0054).

> Muñoz Sabater, J., (2019): ERA5-Land monthly averaged data from 1981 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). (date of access), doi:10.24381/cds.68d2bb30

**My Output .csv Came Back Blank**

If your output file is empty, check the following troubleshooting steps:

  * Does your search window include the data?

Ensure that the time window you specified for matching thermal and auxiliary data actually contains valid data points.

  * Is your data in the correct timestamp format?

Confirm that timestamps match the expected format (e.g., YYYY-MM-DD HH:MM:SS).
Check for any timezone mismatches that might cause data misalignment.

  * Is your data in the correct number format?

Ensure numerical values are stored correctly (e.g., no missing values, text-formatted numbers, or decimal inconsistencies).
Check for unintended special characters (e.g., commas, spaces) that might interfere with parsing.

## Feedback
For inquiries, suggestions, or feedback, please contact the author at jdiehl@nau.edu.
