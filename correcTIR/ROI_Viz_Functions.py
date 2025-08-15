
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
    """
    ROI drawer with:
      • Precise zoom/pan, cursor-anchored zoom
      • Inferno colormap + percentile stretch + optional CLAHE
      • Window overlay/status bar help (never overlaps image)
      • Persistent saved ROIs
    Controls:
      Left (click)=add point | Left-drag=pan | = in | - out
      Right=close ROI | f=fit | u=undo | c=clear | x/ESC=save & exit | h=toggle help
    """

    def __init__(self, image_path, roi_filepath='rois.csv'):
        self.image_path = image_path
        self.roi_filepath = roi_filepath
        self.win = (
            "INSTRUCTIONS | Left: add | Left-drag: pan | =: zoom in | -: zoom out | Right: close ROI | f: fit | u: undo | c: clear | x/ESC: save & exit"
        )

        # image/display
        self._raw_gray = None          # float32 grayscale source
        self._img = None               # BGR uint8 display image (contrast + inferno)
        self.scale = 1.0               # zoom scale
        self.offset = np.array([0.0, 0.0], dtype=np.float32)  # image top-left in window coords

        # contrast settings
        self.clahe_on = True
        self.clahe_clip = 4.0
        self.percentile_pairs = [(2.0, 98.0), (1.0, 99.0), (0.5, 99.5)]
        self.percentile_pair_idx = 1   # start at 1–99%

        # roi state
        self._rois = []                # [{'label': 'roi_1', 'points': [(x,y), ...]}, ...]
        self._roi_points = []          # current polygon (image coords)
        self._roi_counter = 1

        # mouse/gesture
        self.lbtn_down = False
        self.pan_active = False
        self.pan_last_win = None
        self.click_start_win = None
        self.click_start_time = 0.0
        self.pan_time_threshold = 0.15   # seconds to treat as long press
        self.pan_move_threshold = 4.0    # px movement to start pan

        # help overlay
        self._overlay_on = True

    # ---------- image loading & display build ----------
    def _load_image(self):
        with Image.open(self.image_path) as im:
            arr = np.array(im)

        # collapse to grayscale float32 for consistent contrast ops
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        arr = arr.astype(np.float32)

        self._raw_gray = arr
        self._img = self._make_display_image(self._raw_gray)

    def _make_display_image(self, gray_f32):
        # 1) percentile stretch
        lo_p, hi_p = self.percentile_pairs[self.percentile_pair_idx]
        lo = float(np.nanpercentile(gray_f32, lo_p))
        hi = float(np.nanpercentile(gray_f32, hi_p))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(gray_f32)), float(np.nanmax(gray_f32))
        x = np.clip((gray_f32 - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

        # 2) 8-bit
        x8 = (x * 255.0).astype(np.uint8)

        # 3) CLAHE (optional)
        if self.clahe_on:
            clahe = cv2.createCLAHE(clipLimit=float(self.clahe_clip), tileGridSize=(8, 8))
            x8 = clahe.apply(x8)

        # 4) inferno colormap (BGR)
        disp = cv2.applyColorMap(x8, cv2.COLORMAP_INFERNO)
        return disp

    def _rebuild_display(self):
        if self._raw_gray is None:
            return
        self._img = self._make_display_image(self._raw_gray)
        self._render()

    # ---------- coordinate transforms ----------
    def _img_to_win(self, pt_img):
        return (pt_img * self.scale + self.offset).astype(int)

    def _win_to_img(self, pt_win):
        p = (np.array(pt_win, dtype=np.float32) - self.offset) / max(self.scale, 1e-6)
        h, w = self._img.shape[:2]
        p[0] = np.clip(p[0], 0, w - 1)
        p[1] = np.clip(p[1], 0, h - 1)
        return p

    # ---------- drawing ----------
    def _marker_radius(self):
        return max(1, int(3 * (self.scale ** 0.5)))

    def _draw_saved_rois(self, canvas):
        for roi in self._rois:
            pts = np.array(roi['points'], dtype=np.float32)
            if pts.size == 0: 
                continue
            ptsw = np.array([self._img_to_win(p) for p in pts], dtype=int)
            for i in range(len(ptsw)):
                cv2.line(canvas, tuple(ptsw[i]), tuple(ptsw[(i+1) % len(ptsw)]), (0, 255, 0), 1, cv2.LINE_AA)
            lx, ly = ptsw[0]
            cv2.putText(canvas, roi['label'], (lx+5, ly-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    def _draw_current_roi(self, canvas):
        if not self._roi_points:
            return
        ptsw = np.array([self._img_to_win(np.array(p)) for p in self._roi_points], dtype=int)
        for i in range(1, len(ptsw)):
            cv2.line(canvas, tuple(ptsw[i-1]), tuple(ptsw[i]), (0, 0, 255), 1, cv2.LINE_AA)
        for p in ptsw:
            cv2.circle(canvas, tuple(p), self._marker_radius(), (0,0,255), -1, cv2.LINE_AA)

    # ---------- render ----------
    def _render(self):
        if self._img is None:
            return

        ih, iw = self._img.shape[:2]

        # Query current window size; if not ready, approximate from scale
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(self.win)
            if win_w <= 0 or win_h <= 0:
                win_w, win_h = int(iw * self.scale), int(ih * self.scale)
        except Exception:
            win_w, win_h = int(iw * self.scale), int(ih * self.scale)

        # Build scaled image
        disp_w, disp_h = max(1, int(iw * self.scale)), max(1, int(ih * self.scale))
        disp = cv2.resize(self._img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        # Canvas for viewing area
        canvas = np.zeros((max(1, win_h), max(1, win_w), 3), dtype=np.uint8)

        # Clamp panning so image remains reachable
        min_x = min(0, win_w - disp_w)
        min_y = min(0, win_h - disp_h)
        self.offset[0] = float(np.clip(self.offset[0], min_x, 0))
        self.offset[1] = float(np.clip(self.offset[1], min_y, 0))

        # Paste disp with offset
        x0 = int(round(self.offset[0])); y0 = int(round(self.offset[1]))
        sx0 = max(0, -x0); sy0 = max(0, -y0)
        sx1 = min(disp_w, win_w - x0); sy1 = min(disp_h, win_h - y0)
        dx0 = max(0, x0);  dy0 = max(0, y0)
        if sx1 > sx0 and sy1 > sy0:
            canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = disp[sy0:sy1, sx0:sx1]

        # Draw ROIs on top
        self._draw_saved_rois(canvas)
        self._draw_current_roi(canvas)

        cv2.imshow(self.win, canvas)
        self._view = canvas

        # Refresh overlay if enabled (keeps it visible on some backends)
        if self._overlay_on:
            self._show_controls_overlay()

    # ---------- overlay/status bar help ----------
    def _show_controls_overlay(self):
        help_text = (
            "Left: add | Left-drag: pan | Wheel/+/=: zoom in | -: zoom out\n"
            "Right: close ROI | f: fit | 1: 1:1 | u: undo | c: clear | x/ESC: save & exit"
        )
        try:
            cv2.displayOverlay(self.win, help_text, 0)  # persist
            cv2.displayStatusBar(self.win, "Press 'h' to hide/show controls", 0)
            # Optional: window title hint
            cv2.setWindowTitle(self.win, "Thermal Image with ROIs — press 'h' for controls")
        except Exception:
            # If your OpenCV build doesn't support overlay/status bar, just ignore.
            pass

    # ---------- zoom helpers ----------
    def _zoom_at_window_point(self, wx, wy, factor):
        prev = self.scale
        new  = float(np.clip(prev * factor, 0.02, 50.0))
        if new == prev:
            return
        cursor_img = self._win_to_img((wx, wy))
        self.scale = new
        cursor_win_new = self._img_to_win(cursor_img)
        self.offset += (np.array([wx, wy], dtype=np.float32) - cursor_win_new.astype(np.float32))
        self._render()

    def _window_center(self):
        try:
            _, _, ww, wh = cv2.getWindowImageRect(self.win)
        except Exception:
            wh, ww = self._img.shape[:2]
        return ww // 2, wh // 2

    def _fit_to_window(self):
        ih, iw = self._img.shape[:2]
        # Choose a reasonable base window (doesn't auto-resize later)
        base_w, base_h = 1100, 800
        try:
            cv2.resizeWindow(self.win, base_w, base_h)
        except Exception:
            pass
        fit_scale = min(base_w / max(1, iw), base_h / max(1, ih))
        self.scale = float(np.clip(fit_scale, 0.02, 50.0))
        disp_w, disp_h = int(iw * self.scale), int(ih * self.scale)
        self.offset = np.array([(base_w - disp_w) / 2.0, (base_h - disp_h) / 2.0], dtype=np.float32)
        self._render()

    def _one_to_one(self):
        ih, iw = self._img.shape[:2]
        try:
            _, _, ww, wh = cv2.getWindowImageRect(self.win)
        except Exception:
            ww, wh = iw, ih
        self.scale = 1.0
        self.offset = np.array([(ww - iw) / 2.0, (wh - ih) / 2.0], dtype=np.float32)
        self._render()

    # ---------- mouse handler ----------
    def _on_mouse(self, event, x, y, flags, param):
        # Robust wheel handling (varies by platform)
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = flags >> 16
            if delta == 0:
                delta = flags
            zoom_in = delta > 0
            self._zoom_at_window_point(x, y, 1.15 if zoom_in else 1/1.15)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.lbtn_down = True
            self.pan_active = False
            self.click_start_win = np.array([x, y], dtype=np.float32)
            self.pan_last_win = self.click_start_win.copy()
            self.click_start_time = time.time()
            return

        if event == cv2.EVENT_MOUSEMOVE and self.lbtn_down:
            cur = np.array([x, y], dtype=np.float32)
            dt = time.time() - self.click_start_time
            moved = np.linalg.norm(cur - self.click_start_win)
            if (dt >= self.pan_time_threshold) or (moved >= self.pan_move_threshold):
                self.pan_active = True
                delta = cur - self.pan_last_win
                self.pan_last_win = cur
                self.offset += delta
                self._render()
            return

        if event == cv2.EVENT_LBUTTONUP:
            was_pan = self.pan_active
            self.lbtn_down = False
            self.pan_active = False
            if not was_pan:
                p_img = self._win_to_img((x, y))
                self._roi_points.append((float(p_img[0]), float(p_img[1])))
                self._render()
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self._roi_points) >= 3:
                self._rois.append({
                    'label': f'roi_{self._roi_counter}',
                    'points': [(int(round(px)), int(round(py))) for (px, py) in self._roi_points]
                })
                self._roi_points = []
                self._roi_counter += 1
                self._render()
            return

    # ---------- main ----------
    def draw_and_label_poly_rois(self):
        self._load_image()
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.win, self._on_mouse)

        # initial fit + overlay
        self._fit_to_window()
        if self._overlay_on:
            self._show_controls_overlay()

        print("Controls: Left click: add, Left-drag: pan, = zoom in, -: zoom out, Right click:close, "
              "f:fit, , u:undo, c:clear, x/ESC:save & exit")

        while True:
            key = cv2.waitKey(10) & 0xFF
            if key in (ord('x'), 27):
                break
            elif key == ord('u'):
                if self._roi_points:
                    self._roi_points.pop()
                    self._render()
            elif key == ord('c'):
                self._roi_points = []
                self._render()
            elif key in (ord('+'), ord('=')):
                wx, wy = self._window_center()
                self._zoom_at_window_point(wx, wy, 1.15)
            elif key == ord('-'):
                wx, wy = self._window_center()
                self._zoom_at_window_point(wx, wy, 1/1.15)
            elif key == ord('f'):
                self._fit_to_window()
            elif key == ord('1'):
                self._one_to_one()
            elif key == ord('h'):
                self._overlay_on = not self._overlay_on
                try:
                    if self._overlay_on:
                        self._show_controls_overlay()
                    else:
                        cv2.displayOverlay(self.win, "", 1)
                        cv2.displayStatusBar(self.win, "", 1)
                except Exception:
                    pass
            # Optional contrast hotkeys if you want live control:
            elif key == ord('a'):  # toggle CLAHE
                self.clahe_on = not self.clahe_on
                self._rebuild_display()
            elif key == ord('p'):  # cycle percentile pairs
                self.percentile_pair_idx = (self.percentile_pair_idx + 1) % len(self.percentile_pairs)
                self._rebuild_display()
            elif key == ord('['):
                self.clahe_clip = max(1.0, self.clahe_clip - 0.5)
                if self.clahe_on: self._rebuild_display()
            elif key == ord(']'):
                self.clahe_clip = min(12.0, self.clahe_clip + 0.5)
                if self.clahe_on: self._rebuild_display()

            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        self.save_rois_to_csv()
        return self._rois

    # ---------- CSV ----------
    def save_rois_to_csv(self):
        with open(self.roi_filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            headers = ["Label"]
            max_pts = max((len(r['points']) for r in self._rois), default=0)
            for i in range(max_pts):
                headers += [f"Point_{i+1}_x", f"Point_{i+1}_y"]
            csvwriter.writerow(headers)
            for roi in self._rois:
                row = [roi['label']]
                for x, y in roi['points']:
                    row += [int(round(x)), int(round(y))]
                csvwriter.writerow(row)


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




