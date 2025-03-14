# Load Dependencies
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont 

class ImageData(ttk.Frame):
    def __init__(self, parent, window):
        """Main page for image data allowing navigation to the Draw ROI or Processing page."""
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="correcTIR Image Data")
        label.grid(row=0, column=0, padx=10, pady=10)

        data_type_frame = ttk.LabelFrame(self, text="Draw or Begin Processing Region(s) of Interest (ROI)")
        data_type_frame.grid(row=1, column=0, padx=10, pady=10)

        button_draw_roi = ttk.Button(
            data_type_frame, 
            text="Draw ROI",
            command=lambda: window.show_frame("DrawROI"),
        )
        button_draw_roi.grid(row=0, column=0, padx=10, pady=10)
        
        button_roi_overlay = ttk.Button(
            data_type_frame, 
            text="Process Images",
            command=lambda: window.show_frame("ProcessImageData"),
        )
        button_roi_overlay.grid(row=0, column=1, padx=10, pady=10)

        button_frame = ttk.LabelFrame(self, text="")
        button_frame.grid(row=2, column=0)

        button_main_page = ttk.Button(
            button_frame, 
            text="Main Page",
            command=lambda: window.show_frame("MainPage"),
        )
        button_main_page.grid(row=0, column=0, padx=10, pady=1)