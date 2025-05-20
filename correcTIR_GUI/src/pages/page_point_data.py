# Load Dependencies
import sys
import os

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont 

from tkinter import messagebox

# from src.backend.process_data import process_data
from src.file_functions import get_path

# add the path for Main_Functions
from src.backend.process_data import open_overlay
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../correcTIR/")))
from Main_Functions import process_data

class PointData(tk.Frame):
    def __init__(self, parent, window):
        """
        The main page for point data allowing creation and selection of configuration files as well as running the processing script.
        """
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="correcTIR Point Data")
        label.grid(row=0, column=0, padx=10, pady=10)

        point_data_frame = ttk.LabelFrame(self, text="")
        point_data_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        button_point_data_inputs_page = ttk.Button(
            point_data_frame, 
            text="Create Config File To Process Data",
            command=lambda: window.show_frame("PointDataInputs"),
        )
        button_point_data_inputs_page.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        config_path = ttk.Entry(point_data_frame)
        config_path.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        button_config_path = ttk.Button(
            point_data_frame,
            text="Select Already Created Config File",
            command=lambda: get_path(self, config_path, file_type="*.json")
        )
        button_config_path.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        help_icon = ttk.Button(
            point_data_frame,
            text="?",
            width=2,
            command=lambda: self.helpButton(
            title='Config File Selection Help',
            message='Select the config file you want to use for processing or create a new config file above.\
                \n\nNote: The config file must be in JSON format (.json).'
            )
        )
        help_icon.grid(row=1, column=1, sticky="w", padx=(0, 2), pady=10)

        button_process_point_data = ttk.Button(
            point_data_frame, 
            text="Process Point Data",
            command=lambda: self.process_data_with_check(config_path.get())
        )
        button_process_point_data.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        button_frame = ttk.LabelFrame(self, text="")
        button_frame.grid(row=2, column=0)

        help_icon = ttk.Button(
            point_data_frame,
            text="?",
            width=2,
            command=lambda: self.helpButton(
            title='Process Point Data Help',
            message='Process the point data specified in the config file. \
                \n\nNote: The config file must be specified above using "Select Already Created Config File," and the file should be in JSON format (.json).'
            )
        )
        help_icon.grid(row=3, column=1, sticky="w", padx=(0, 2), pady=10)

        button_main_page = ttk.Button(
            button_frame, 
            text="Main Page",
            command=lambda: window.show_frame("MainPage"),
        )
        button_main_page.grid(row=0, column=0, padx=10, pady=10)

    # Help button"
    def helpButton(self, message: str, title: str = "Help"):
        """
        Display a help message box with the given message and title.
        """
        tk.messagebox.showinfo(title=title, message=message)

    def process_data_with_check(self, config_path: str):
        """
        Process the data with a check for the config path.
        """
        if not os.path.isfile(config_path):
            messagebox.showerror("Error", "Config file not found.\n" \
            "Please load a config file in JSON format above.")
            return
        if not config_path.endswith('.json'):
            messagebox.showerror("Error", "Config file must be JSON format.\n" \
            "Please load a config file in JSON format above.")
            return
        process_data(config_path)