# Load Dependencies
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont 

from tkinter import messagebox

import src.backend.thermalCamFunctions.ROI_Viz_Functions as roiviz
from src.file_functions import get_path, get_dir
from src.input_checks import check_string, check_value

# from src.settings import absolute_path
# sys.path.append(absolute_path)
# import ROI_Viz_Functions as roiviz

class DrawROI(ttk.Frame):
    def __init__(self, parent, window):
        """Page containing the form for the creation of ROI(s) and the Draw ROI Function."""
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Draw ROI")
        label.grid(row=0, column=0, padx=10, pady=10)

        roi_frame = ttk.LabelFrame(self, text="")
        roi_frame.grid(row=1, column=0, padx=10, pady=10)

        tiff_path = ttk.Entry(roi_frame)
        tiff_path.grid(row=0, column=1, padx=10, pady=10)
        tiff_path.bind('<KeyRelease>', lambda e: check_value(self, e.widget, button_drawROI))

        button_get_ROI_Path = ttk.Button(
            roi_frame,
            text='Select Image Path To Draw ROI On',
            command=lambda: get_path(self, tiff_path, file_type="*.tiff")
        )
        button_get_ROI_Path.grid(row=0, column=0, padx=10, pady=10)

        output_roi_dir = ttk.Entry(roi_frame)
        output_roi_dir.grid(row=2, column=1, padx=10, pady=10)
        
        button_file = ttk.Button(
            roi_frame,
            text="Output Dir for ROI(s) Created",
            command=lambda: get_dir(self, output_roi_dir)
        )
        button_file.grid(row=2, column=0, padx=10, pady=10)

        output_roi_file_name_label = ttk.Label(roi_frame, text="Output CSV Name")
        output_roi_file_name = ttk.Entry(roi_frame)
        output_roi_file_name_label.grid(row=2, column=2, padx=10, pady=10)
        output_roi_file_name.grid(row=2, column=3, padx=10, pady=10)
        output_roi_file_name.bind('<KeyRelease>', lambda e: check_string(self, e.widget, ".csv"))

        button_drawROI = ttk.Button(
            roi_frame,
            text='Run Draw ROI Function',
            command=lambda: self.draw_roi_submit(tiff_path, output_roi_dir, output_roi_file_name),
            state='normal'
        )
        button_drawROI.grid(row=3, column=0, padx=10, pady=10)

        # Buttons for returning to the image or main pages.
        button_frame = ttk.LabelFrame(self, text="")
        button_frame.grid(row=2, column=0)

        button_image_page = ttk.Button(
            button_frame, 
            text="Back to Image Data Page",
            command=lambda: window.show_frame("ImageData"),
        )
        button_image_page.grid(row=0, column=0, padx=10, pady=10)

        button_main_page = ttk.Button(
            button_frame, 
            text="Main Page",
            command=lambda: window.show_frame("MainPage"),
        )
        button_main_page.grid(row=0, column=1, padx=10, pady=1)

    def draw_roi_submit(self, tiff_path, roi_dir, roi_filename):
        try:
            print('submit pressed')
            roi_output_filepath = roi_dir.get() + '/' + roi_filename.get()
            run_draw_roi = roiviz.DrawAndLabelPolyROIS(tiff_path.get(), roi_output_filepath)
            run_draw_roi.draw_and_label_poly_rois()
            return
        except:
            if tiff_path.get() == None or tiff_path.get() == '':
                messagebox.showerror(title="Error in draw ROI!", message=f'Please select a tiff image.')
            elif roi_dir.get() == None or roi_dir.get() == '' or roi_filename.get() == None or roi_filename.get() == '':
                messagebox.showerror(title="Error in draw ROI!", message=f'Please select an roi directory and filename with the csv extension.')
            else:
                messagebox.showerror(title="Error in draw ROI!", message=f'There was an error submitting the tiff to draw ROI: {tiff_path.get()}.')
            return
