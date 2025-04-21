# Load Dependencies
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont 

from tkinter import messagebox

from src.backend.process_data import process_data
from src.file_functions import get_path

class PointData(tk.Frame):
    def __init__(self, parent, window):
        """
        The main page for point data allowing creation and selection of configuration files as well as running the processing script.
        """
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="correcTIR Point Data")
        label.grid(row=0, column=0, padx=10, pady=10)

        point_data_frame = ttk.LabelFrame(self, text="")
        point_data_frame.grid(row=1, column=0, padx=10, pady=10)

        button_point_data_inputs_page = ttk.Button(
            point_data_frame, 
            text="Create Config File To Process Data",
            command=lambda: window.show_frame("PointDataInputs"),
        )
        button_point_data_inputs_page.grid(row=0, column=0, padx=10, pady=10)

        config_path = ttk.Entry(point_data_frame)
        config_path.grid(row=2, column=0, padx=10, pady=10)

        button_config_path = ttk.Button(
            point_data_frame,
            text="Select Already Created Config File",
            command=lambda: get_path(self, config_path, file_type="*.json")
        )
        button_config_path.grid(row=1, column=0, padx=10, pady=10)

        button_process_point_data = ttk.Button(
            point_data_frame, 
            text="Process Point Data",
            command=lambda: process_data(config_path.get())
        )
        button_process_point_data.grid(row=3, column=0, padx=10, pady=10)

        button_frame = ttk.LabelFrame(self, text="")
        button_frame.grid(row=2, column=0)

        button_main_page = ttk.Button(
            button_frame, 
            text="Main Page",
            command=lambda: window.show_frame("MainPage"),
        )
        button_main_page.grid(row=0, column=0, padx=10, pady=1)