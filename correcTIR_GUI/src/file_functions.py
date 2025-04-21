
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory    

def get_path(self, filename, initial_dir = "~/", file_type:str = "*.txt*"):
    """
    Get the absolute path of a file.

    Parameters:
    self: The self value from the class calling get_path.
    filename: The entry widget that should recieve the filename.
    initial_dir: The initial directory to show the user.
    file_type: The file type to show the user.

    Returns:
    The absolute path to the file selected.
    """

    file_explorer = tk.Label(self, text="Explore files",
        font=("Verdana", 14, "bold"),
        width=100,
        height=4, fg="white", bg="gray")
            
    f_path = askopenfilename(initialdir=initial_dir,
        title="Select File", filetypes=(("File type", file_type),("All Files","*.*")))
    file_explorer.configure(text="File Opened: "+ f_path)

    filename.delete(0, "end")
    filename.insert(0, f_path)
    return f_path
    
def get_dir(self, directory, initial_dir = "~/"):
    """
    Get the absolute path of a directory.

    Parameters:
    self: The self value from the class calling get_path.
    directory: The entry widget that should recieve the filename.
    initial_dir: The initial directory to show the user.

    Returns:
    The absolute path to the directory selected.
    """
        
    explorer = tk.Label(self, text="Get Directory",
        font=("Verdana", 14, "bold"),
        width=100,
        height=4, fg="white", bg="gray")
    f_path = askdirectory(initialdir=initial_dir)

    directory.delete(0, "end")
    directory.insert(0, f_path)
    return f_path