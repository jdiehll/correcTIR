# Load Dependencies
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont 
from tkinter import filedialog


from src.backend.process_data import process_data, open_overlay
from src.file_functions import get_path

class ProcessImageData(ttk.Frame):
    def __init__(self, parent, window):
        """The main page for image data allowing creation and selection of configuration files, checking ROI(s) and running the processing script."""
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="correcTIR Image Data Processing")
        label.grid(row=0, column=0, padx=10, pady=10)

        image_data_frame = ttk.LabelFrame(self, text="")
        image_data_frame.grid(row=1, column=0, padx=10, pady=10)

        button_point_data_inputs_page = ttk.Button(
            image_data_frame, 
            text="Create Config File To Process Data",
            command=lambda: window.show_frame("ImageDataInputs"),
        )
        button_point_data_inputs_page.grid(row=0, column=0, padx=10, pady=10)

        config_path = ttk.Entry(image_data_frame)
        config_path.grid(row=2, column=0, padx=10, pady=1)

        button_config_path = ttk.Button(
            image_data_frame,
            text="Select Already Created Config File",
            command=lambda: get_path(self, config_path, file_type="*.json")
        )
        button_config_path.grid(row=1, column=0, padx=10, pady=10)

        open_overlay
        button_open_overlay = ttk.Button(
            image_data_frame, 
            text="Check ROI(s)",
            command=lambda: open_overlay(config_path.get())
        )
        button_open_overlay.grid(row=3, column=0, padx=10, pady=10)

        button_processing_page = ttk.Button(
            image_data_frame, 
            text="Process Image Data",
            command=lambda: process_data(config_path.get())
        )
        button_processing_page.grid(row=4, column=0, padx=10, pady=10)

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
