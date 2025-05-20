import sys
import os

# Load Dependencies
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont 
from tkinter import filedialog


from src.backend.process_data import open_overlay
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../correcTIR/")))
from src.file_functions import get_path
from Main_Functions import process_data

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
        button_point_data_inputs_page.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        config_path = ttk.Entry(image_data_frame)
        config_path.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        button_config_path = ttk.Button(
            image_data_frame,
            text="Select Already Created Config File",
            command=lambda: get_path(self, config_path, file_type="*.json")
        )
        button_config_path.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        help_icon = ttk.Button(
            image_data_frame,
            text="?",
            width=2,
            command=lambda: self.helpButton(
                title='Config File Selection Help',
                message="Select the config file you want to use for processing or create a new config file above. \
                    \n\nNote: The config file must be in JSON format (.json)."
            )
        )
        help_icon.grid(row=1, column=1, sticky="w", padx=(0, 2), pady=10)

        button_open_overlay = ttk.Button(
            image_data_frame, 
            text="Check ROI(s)",
            command=lambda: open_overlay(config_path.get())
        )
        button_open_overlay.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        help_icon = ttk.Button(
            image_data_frame,
            text="?",
            width=2,
            command=lambda: self.helpButton(
            title='Check ROI Help',
            message="Check the ROI specified in teh config file. \
                \n\nNote: The config file must be in JSON format."
            )
        )
        help_icon.grid(row=3, column=1, sticky="w", padx=(0, 2), pady=10)

        button_processing_page = ttk.Button(
            image_data_frame, 
            text="Process Image Data",
            command=lambda: self.process_data_with_check(config_path.get())
        )
        button_processing_page.grid(row=4, column=0, padx=10, pady=10, sticky="w")

        help_icon = ttk.Button(
            image_data_frame,
            text="?",
            width=2,
            command=lambda: self.helpButton(
            title='Process Image Help',
            message='Process the images specified in the config file. \
                \n\nNote: The config file must be specified above using "Select Already Created Config File," and the file should be in JSON format (.json).'
            )
        )
        help_icon.grid(row=4, column=1, sticky="w", padx=(0, 2), pady=10)

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
        button_main_page.grid(row=0, column=1, padx=10, pady=10)

    # Help button"
    def helpButton(self, message: str, title: str = "Help"):
        """
        Display a help message box with the given message and title."""
        tk.messagebox.showinfo(title=title, message=message)

    def process_data_with_check(self, config_path: str):
        """
        Process the data with a check for the config path.
        """
        if not os.path.isfile(config_path):
            tk.messagebox.showerror("Error", "Config file not found.\n" \
            "Please load a config file in JSON format above.")
            return
        if not config_path.endswith('.json'):
            tk.messagebox.showerror("Error", "Config file must be in JSON format.\n" \
            "Please load a config file in JSON format above.")
            return
        process_data(config_path)
