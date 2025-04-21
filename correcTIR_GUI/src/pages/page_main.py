import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont 
from tkinter import filedialog

class MainPage(ttk.Frame):
    def __init__(self, parent, window):
        """
        The startup page for the correcTIR app.
        """
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="correcTIR Post Processing")
        label.grid(row=0, column=0, padx=10, pady=10)

        data_type_frame = ttk.LabelFrame(self, text="Select Data Type")
        data_type_frame.grid(row=1, column=0, padx=10, pady=10)

        image_data = ttk.Button(
            data_type_frame, 
            text="Image",
            command=lambda: window.show_frame("ImageData"),
        )
        image_data.grid(row=0, column=0, padx=10, pady=10)
        
        point_data = ttk.Button(
            data_type_frame, 
            text="Point",
            command=lambda: window.show_frame("PointData"),
        )
        point_data.grid(row=0, column=1, padx=10, pady=10)
