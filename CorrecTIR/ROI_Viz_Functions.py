# =============================================
# LIBRARY IMPORTSS
# =============================================
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

# =============================================
# IMAGE DISPLAY & SAVING FUNCTIONS
# =============================================
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

# =============================================
# ROI FUNCTIONS
# =============================================
def draw_and_label_poly_rois(image_path):
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
    image_data = display_tiff_with_colormap(image_path)
    cv2.namedWindow('Thermal Image with ROIs', cv2.WINDOW_NORMAL)

    rois = []
    roi_points = []

    def overlay_instructions(img):
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

    def draw_roi(event, x, y, flags, param):
        nonlocal roi_points
        
        # Get the dimensions of the displayed window
        window_width = cv2.getWindowImageRect('Thermal Image with ROIs')[2]
        window_height = cv2.getWindowImageRect('Thermal Image with ROIs')[3]
        
        # Normalize x and y coordinates to match image's actual size
        x = int(x * (image_data.shape[1] / window_width))
        y = int(y * (image_data.shape[0] / window_height))

        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
            cv2.circle(image_data, (x, y), 5, (0, 0, 255), -1)
            if len(roi_points) > 1:
                cv2.line(image_data, roi_points[-2], roi_points[-1], (0, 0, 255), 2)
            cv2.imshow('Thermal Image with ROIs', overlay_instructions(image_data))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(roi_points) > 2:
                cv2.line(image_data, roi_points[-1], roi_points[0], (0, 0, 255), 2)
                cv2.imshow('Thermal Image with ROIs', overlay_instructions(image_data))
                label = input("Enter label for this ROI: ")
                rois.append({'label': label, 'points': roi_points.copy()})
                roi_points = []

    cv2.setMouseCallback('Thermal Image with ROIs', draw_roi)

    cv2.imshow('Thermal Image with ROIs', overlay_instructions(image_data))

    print("Draw ROIs on the image. Right-click to complete an ROI. Press 'x' to exit.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break

    save_rois_to_csv(rois)
    return rois

def save_rois_to_csv(rois, filename="rois.csv"):
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
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write header
        headers = ["Label"]
        max_points = max(len(roi['points']) for roi in rois)
        for i in range(max_points):
            headers.extend([f"Point_{i+1}_x", f"Point_{i+1}_y"])
        csvwriter.writerow(headers)
        
        # Write ROI data
        for roi in rois:
            row_data = [roi['label']]
            for point in roi['points']:
                row_data.extend([point[0], point[1]])
            csvwriter.writerow(row_data)

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
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                break