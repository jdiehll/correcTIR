# Load Dependencies
import tkinter as tk
import platform

from tkinter import ttk
from tkinter import font as tkfont 
from ttkbootstrap import Style

# Page imports
from src.pages.page_main import MainPage
from src.pages.page_image_data import ImageData
from src.pages.page_point_data import PointData
from src.pages.page_point_data_inputs import PointDataInputs
from src.pages.page_image_data_process import ProcessImageData
from src.pages.page_image_data_draw_roi import DrawROI
from src.pages.page_image_data_inputs import ImageDataInputs

# Create a class allowing instantiation of the app as a whole. All other
# Classes will be nested within this class.

class ThermoCam(tk.Tk):
    """
    Setup the root for all other pages in the Thermal Cam app. The primary window is created here and styles are defined.
    """

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("800x600")  # width x height in pixels

        self.title_font = tkfont.Font(family='Helvetica', size=14, weight="bold", slant="italic")

        self.platform = platform.system()

        # Check if the computer is a Mac and set a theme
        if self.platform == "Darwin":
            # self.style = Style('journal')
            self.style = Style('darkly')
            self.style = ttk.Style(self)
            # self.style.theme_use('aqua')
        else: 
            # self.style = Style('darkly')
            self.style = Style('darkly')
            self.style = ttk.Style(self)
            self.style.theme_use('clam')

        # Configure element styles
        if self.platform == "Darwin":
            pass
        if self.platform == "Windows":
            pass
        if self.platform == "Linux":
            pass
        
        # create a container that all pages are displayed in.
        self.title('correcTIR')
        container = tk.Frame(self)
        container.pack()

        self.frames = {}
        for F in (MainPage, 
                  ImageData, 
                  PointData, 
                  ProcessImageData, 
                  DrawROI, 
                  PointDataInputs,
                  ImageDataInputs,
                  ):
            page_name = F.__name__
            frame = F(parent=container, window=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame('MainPage')

    def show_frame(self, page_name):
        """
        Show a frame for the given page name.

        Parameters:
        page_name (str): The name of the class representing a page in the app. 
        """
        frame = self.frames[page_name]
        frame.tkraise()
    
if __name__ == "__main__":
    """ Start the Thermal Cam app. """
    app = ThermoCam()
    app.mainloop()