import tkinter as tk
from tkinter import ttk

def check_value(root, entry, min_value, max_value):
    """
    Check an tk or ttk entry to see if it is a float between a minimum and maximum value.

    Parameters:
    root: The self value for the page class
    entry: The tk or tkk entry widget value.
    min_value: The minimum acceptable value.
    max_value: The maximum acceptable value.

    Returns:
    True or False and changes the text color to red if invalid.
    """
    style = ttk.Style()
    try:
        value = float(entry.get().strip())
        valid = min_value <= value <= max_value
    except ValueError:
        valid = False

    # Configure style for valid and invalid entries
    if valid:
        style.configure('Valid.TEntry', foreground='white')
        entry.config(style='Valid.TEntry')
    else:
        style.configure('Invalid.TEntry', foreground='red')
        entry.config(style='Invalid.TEntry')
    return valid

def check_value_int(root, entry, min_value, max_value):
    """
    Check an tk or ttk entry to see if it is an integer between a minimum and maximum value.

    Parameters:
    root: The self value for the page class
    entry: The tk or tkk entry widget value.
    min_value: The minimum acceptable value.
    max_value: The maximum acceptable value.

    Returns:
    True or False and changes the text color to red if invalid.
    """

    style = ttk.Style()
    try:
        is_int = isinstance(entry, int)
        value = int(entry.get().strip())
        valid = min_value <= value <= max_value and is_int
    except ValueError:
        valid = False

    # Configure style for valid and invalid entries
    if valid:
        style.configure('Valid.TEntry', foreground='white')
        entry.config(style='Valid.TEntry')
    else:
        style.configure('Invalid.TEntry', foreground='red')
        entry.config(style='Invalid.TEntry')
    return valid
    
def check_string(root, entry, check_text):
    """
    Check an tk or ttk entry to see if it is a string containing a given substring.

    Parameters:
    root: The self value for the page class
    entry: The tk or tkk entry widget value.
    check_text: A string that should be a substring of the tk or ttk entry value.

    Returns:
    True or False and changes the text color to red if invalid.
    """

    style = ttk.Style()
    try:
        value = str(entry.get().strip())
        valid = value.count(check_text) >= 1
    except ValueError:
        valid = False

    # Configure style for valid and invalid entries
    if valid:
        style.configure('Valid.TEntry', foreground='white')
        entry.config(style='Valid.TEntry')
    else:
        style.configure('Invalid.TEntry', foreground='red')
        entry.config(style='Invalid.TEntry')
    return valid