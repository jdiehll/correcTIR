
# Standard library imports
import csv

# Third-party imports
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cv2
import geopandas as gpd


##### Image Display Functions
def display_tiff_with_colormap(tiff_path, colormap='inferno'):
    """
    Convert a TIFF image with a specified colormap to a format suitable for OpenCV.

    Parameters:
    tiff_path (str): Path to the TIFF image.
    colormap (str): The colormap to use. Default is 'inferno'.

    Returns:
    numpy.ndarray: The converted image.
    """
    # Load the TIFF image
    with Image.open(tiff_path) as img:
        image_data = np.array(img)

    # Calculate mean and standard deviation
    mean_val = np.mean(image_data)
    std_val = np.std(image_data)

    # Normalize data within 2 std deviation
    normalized_data = (image_data - (mean_val-2*std_val)) / (4*std_val)
    normalized_data = np.clip(normalized_data, 0, 1)

    # Assuming 'colormap' is a variable holding the name of the colormap
    colored_image = (matplotlib.colormaps.get_cmap(colormap)(normalized_data)[:, :, :3] * 255).astype(np.uint8)


    return cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR)

def save_thermal_image(tiff_path, save_path, colormap='inferno'):
    """
    Save a TIFF image with a specified colormap, temperature color bar scale, and normalization.

    Parameters:
    tiff_path (str): Path to the input TIFF image.
    save_path (str): Path to save the output image.
    colormap (str): The colormap to use. Default is 'inferno'.

    Returns:
    None
    """
    # Load the TIFF image
    with Image.open(tiff_path) as img:
        image_data = np.array(img)

    # Define the range of values for the colormap based on the original data
    vmin = np.min(image_data)
    vmax = np.max(image_data)

    # Calculate mean and standard deviation for normalization
    mean_val = np.mean(image_data)
    std_val = np.std(image_data)

    # Normalize data within 2 std deviations
    normalized_data = (image_data - (mean_val - 2 * std_val)) / (4 * std_val)
    normalized_data = np.clip(normalized_data, 0, 1)

    # Create a figure for the color bar
    colorbar_fig, colorbar_ax = plt.subplots(figsize=(0.1, 6))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cbar = cm.ScalarMappable(norm=norm, cmap=colormap)
    cbar.set_array([])
    colorbar = colorbar_fig.colorbar(cbar, cax=colorbar_ax)
    colorbar.set_label('Temperature (°C)', rotation=90)

    # Create a figure for the normalized image
    image_fig, image_ax = plt.subplots(figsize=(8, 6))
    image_ax.imshow(normalized_data, cmap=colormap, vmin=0, vmax=1)
    image_ax.axis('off')

    # Save the colorbar and image separately
    colorbar_fig.savefig('colorbar_temp.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    image_fig.savefig('normalized_image_with_colorbar.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

    # Close the figures
    plt.close(colorbar_fig)
    plt.close(image_fig)

    # Load the saved colorbar and normalized image
    with Image.open('colorbar_temp.png') as colorbar_img:
        colorbar_data = np.array(colorbar_img)

    with Image.open('normalized_image_with_colorbar.png') as normalized_image_with_colorbar_img:
        normalized_image_with_colorbar_data = np.array(normalized_image_with_colorbar_img)

    # Combine the normalized image and colorbar
    combined_image = np.hstack((normalized_image_with_colorbar_data, colorbar_data))

    # Save the combined image
    plt.imsave(save_path, combined_image)


##### ROI Selection Functions (also saves drawn ROIs to csv for future use)
class DrawAndLabelPolyROIS:

    def __init__(self, image_path, roi_filepath='rois.csv'):
        self.image_path = image_path
        self.roi_filepath = roi_filepath
        self._rois = []
        self._roi_points = []
        self._image_data = None
        self._roi_counter = 1
    
    def draw_and_label_poly_rois(self):
        """
        Interactively draw and label polygonal regions of interest (ROIs) on a thermal image.

        Parameters:
        image_path (str): Path to the thermal image file.

        Returns:
        list: A list of dictionaries. Each dictionary contains the label and points of an ROI.
        
        Usage:
        - Left-click to add points to the polygon.
        - Right-click to complete the polygon and label it.
        - Press 'x' to exit and save the labeled polygons to a CSV file.

        Notes:
        The function will also display basic instructions on the image window.
        The drawn ROIs are saved to a CSV file using the save_rois_to_csv function.
        """
        image_data = display_tiff_with_colormap(self.image_path)
        self._image_data = image_data
        cv2.namedWindow('Thermal Image with ROIs', cv2.WINDOW_NORMAL)

        # rois = []
        # roi_points = []

        cv2.setMouseCallback('Thermal Image with ROIs', self.draw_roi)

        cv2.imshow('Thermal Image with ROIs', self.overlay_instructions(self._image_data))

        print("Draw ROIs on the image. Right-click to complete an ROI. Press 'x' to exit.")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                # cv2.destroyAllWindows() TODO remove this is not needed one statement works
                cv2.waitKey(1)
                break
            if cv2.getWindowProperty('Thermal Image with ROIs', cv2.WND_PROP_VISIBLE) < 1:
                cv2.waitKey(1)
                break
        cv2.destroyAllWindows()
    
        self.save_rois_to_csv()
        return self._rois

    def overlay_instructions(self, img):
        # Copy the original image so we don't modify it directly
        overlay_img = img.copy()
        instructions = [
            ("Left click: Add point", (10, 30)),
            ("Right click: Complete ROI", (10, 60)),
            ("Press 'x': Exit", (10, 90))
        ]

        for text, position in instructions:
            cv2.putText(overlay_img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return overlay_img

    def draw_roi(self, event, x, y, flags, param):
        # nonlocal roi_points
        
        # Get the dimensions of the displayed window
        window_width = cv2.getWindowImageRect('Thermal Image with ROIs')[2]
        window_height = cv2.getWindowImageRect('Thermal Image with ROIs')[3]
        
        # Normalize x and y coordinates to match image's actual size
        # x = int(x * (image_data.shape[1] / window_width))
        # y = int(y * (image_data.shape[0] / window_height))

        if event == cv2.EVENT_LBUTTONDOWN:
            self._roi_points.append((x, y))
            cv2.circle(self._image_data, (x, y), 5, (0, 0, 255), -1)
            if len(self._roi_points) > 1:
                cv2.line(self._image_data, self._roi_points[-2], self._roi_points[-1], (0, 0, 255), 2)
            # print(f"x: {x} y: {y}")
            cv2.imshow('Thermal Image with ROIs', self.overlay_instructions(self._image_data))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self._roi_points) > 2:
                cv2.line(self._image_data, self._roi_points[-1], self._roi_points[0], (0, 0, 255), 2)
                cv2.imshow('Thermal Image with ROIs', self.overlay_instructions(self._image_data))

                
                # label = input("Enter label for this ROI: ")

                self._rois.append({'label': f'roi_{self._roi_counter}', 'points': self._roi_points.copy()})
                self._roi_points = []
                self._roi_counter += 1



    def save_rois_to_csv(self):
        """
        Save labeled regions of interest (ROIs) to a CSV file.

        Parameters:
        rois (list): A list of dictionaries. Each dictionary contains the label and points of an ROI.
                    Expected format for each dictionary: {'label': 'some_label', 'points': [(x1, y1), (x2, y2), ...]}
        filename (str, optional): Name of the CSV file to save the data to. Defaults to "rois.csv".

        Returns:
        None

        Notes:
        The CSV file will have the following format:
        Label, Point_1_x, Point_1_y, Point_2_x, Point_2_y, ...
        label1, x1, y1, x2, y2, ...
        label2, x1, y1, x2, y2, ...
        ...
        
        The number of Point_x_x and Point_x_y columns will vary based on the maximum number of points in the provided ROIs.
        """
        with open(self.roi_filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            try:
                # Write header
                headers = ["Label"]
                max_points = max(len(roi['points']) for roi in self._rois)
                for i in range(max_points):
                    headers.extend([f"Point_{i+1}_x", f"Point_{i+1}_y"])
                csvwriter.writerow(headers)
                
                # Write ROI data
                for roi in self._rois:
                    row_data = [roi['label']]
                    for point in roi['points']:
                        row_data.extend([point[0], point[1]])
                    csvwriter.writerow(row_data)
            except Exception as err:
                print('DrawROI error or no roi produced.')


##### Load ROI from Shapefile Function
def convert_shapefile_to_csv(shapefile_path, csv_path):
    """
    Convert a shapefile to a CSV file in the specified format with vertically flipped coordinates.

    Parameters:
    shapefile_path (str): Path to the shapefile.
    csv_path (str): Path where the CSV file will be saved.

    The CSV file format will be:
    Label, Point_1_x, Point_1_y, Point_2_x, Point_2_y, ...
    """
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Determine the maximum height (for flipping the y-coordinate)
    max_height = max([row['geometry'].bounds[3] for _, row in gdf.iterrows()])

    # Open a CSV file to write the data
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write the header
        headers = ["Label"]
        max_points = max(len(row['geometry'].exterior.coords) for _, row in gdf.iterrows())
        for i in range(max_points):
            headers.extend([f"Point_{i+1}_x", f"Point_{i+1}_y"])
        csvwriter.writerow(headers)

        # Write the data
        for _, row in gdf.iterrows():
            label = row['crown']
            points = list(row['geometry'].exterior.coords)

            # Flip the y-coordinates and flatten the list of points to write to CSV
            row_data = [label]
            for point in points:
                flipped_y = max_height - point[1]  # Flip the y-coordinate
                row_data.extend([point[0], flipped_y])
            csvwriter.writerow(row_data)


##### ROI Review Functions from .csv

def overlay_rois_from_csv(image_path, csv_path, output_image_path=None):
    """
    Display an image with regions of interest (ROIs) and labels overlaid, as specified in a CSV file,
    and optionally save the overlay image to a specified path.

    Parameters:
    image_path (str): Path to the image on which to overlay the ROIs.
    csv_path (str): Path to the CSV file containing ROI data.
    output_image_path (str, optional): Path to save the overlay image. If None, the image is not saved.

    Expected CSV Format:
    Label, Point_1_x, Point_1_y, Point_2_x, Point_2_y, ...
    label1, x1, y1, x2, y2, ...
    label2, x1, y1, x2, y2, ...
    ...

    The CSV file should start with a header row specifying the label and then pairs of x and y coordinates.
    Each subsequent row defines a labeled ROI with its set of coordinates.

    Returns:
    None
    """
    def overlay_instructions(img):
        # Copy the original image so we don't modify it directly
        overlay_img = img.copy()
        instructions = [
            ("Press 'x': Exit", (10, 30))
        ]

        for text, position in instructions:
            cv2.putText(overlay_img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        return overlay_img

    image = display_tiff_with_colormap(image_path)
    
    rois = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip the header
        
        for row in reader:
            label = row[0]
            points_data = row[1:]

            # Filter out empty strings and convert points to integers
            points = [(int(float(points_data[i])), int(float(points_data[i+1])))
                      for i in range(0, len(points_data) - 1, 2)
                      if points_data[i] and points_data[i+1]]

            rois.append({'label': label, 'points': points})

    # Process and overlay each ROI
    for roi in rois:
        pts = np.array(roi['points'], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)  # Draw the ROI in green for visibility
        # Optional: Add text label
        if roi['points']:
            cv2.putText(image, roi['label'], roi['points'][0], cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)

    # Save or display the image
    if output_image_path:
        cv2.imwrite(output_image_path, image)
    else:
        # Display the image
        cv2.imshow('Thermal Image with ROIs', overlay_instructions(image))
    
        # Close window when 'x' is pressed
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                # cv2.destroyAllWindows()
                cv2.waitKey(1)
                break
            if cv2.getWindowProperty('Thermal Image with ROIs', cv2.WND_PROP_VISIBLE) < 1:
                cv2.waitKey(1)
                break
        cv2.destroyAllWindows()




