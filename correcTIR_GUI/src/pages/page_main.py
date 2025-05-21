import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import webbrowser


class MainPage(ttk.Frame):
    def __init__(self, parent, window):
        """
        The startup page for the correcTIR app.
        """
        super().__init__(parent)

        # === HEADER with Title Only ===
        header_frame = ttk.Frame(self)
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        title_label = ttk.Label(
            header_frame,
            text="correcTIR Post Processing",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, sticky="w")

        # === DATA TYPE SELECTION ===
        data_type_frame = ttk.LabelFrame(self, text="Select Data Type")
        data_type_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")

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

        # === CITATION SECTION (TEXT ONLY) ===
        cite_frame = ttk.Frame(self)
        cite_frame.grid(row=2, column=0, padx=10, pady=(30, 10), sticky="w")

        cite_label = ttk.Label(
            cite_frame,
            text="How to cite this package:",
            font=tkfont.Font(size=14, weight = 'bold')
        )
        cite_label.grid(row=0, column=0, sticky="w")

        citation_link = ttk.Label(
            cite_frame,
            text="DOI: 10.5281/zenodo.15446495",
            foreground="white",
            cursor="hand2"
        )
        citation_link.grid(row=1, column=0, pady=5, sticky="w")
        citation_link.bind(
            "<Button-1>",
            lambda e: webbrowser.open("https://doi.org/10.5281/zenodo.15446495")
        )
