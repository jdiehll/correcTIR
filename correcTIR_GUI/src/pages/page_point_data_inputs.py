# Load Dependencies
import os
import json
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont 
from tkinter import messagebox

from tkinter.filedialog import askopenfilename, askdirectory
from src.file_functions import get_dir, get_path
from src.input_checks import check_string, check_value

class PointDataInputs(tk.Frame):
    def __init__(self, parent, window):
        """Page containing the form for the creation of point data configuration files."""
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="correcTIR Point Data Inputs")
        label.grid(row=0, column=0, padx=10, pady=10)

        path_variables_frame = ttk.LabelFrame(self, text="Set Path Variables")
        path_variables_frame.grid(row=1, column=0, padx=10, pady=10)

        ##### Start of path variable form #####
        # row 0
        aux_met_data_path = ttk.Entry(path_variables_frame)
        aux_met_data_path.grid(row=0, column=1, padx=10, pady=10)

        button_file = ttk.Button(
            path_variables_frame,
            text="File Path For Auxillary Data",
            command=lambda: get_path(self, aux_met_data_path, file_type="*.csv")
            # command=filename.create_browser
        )
        button_file.grid(row=0, column=0, padx=10, pady=10)


        flux_met_data_path = ttk.Entry(path_variables_frame)
        flux_met_data_path.grid(row=0, column=3, padx=10, pady=10)

        button_file = ttk.Button(
            path_variables_frame,
            text="File Path For Flux Data (if different interval)",
            command=lambda: get_path(self, flux_met_data_path, file_type="*.csv")
            # command=filename.create_browser
        )
        button_file.grid(row=0, column=2, padx=10, pady=10)

        # row 1
        output_csv_dir = ttk.Entry(path_variables_frame)
        output_csv_dir.grid(row=1, column=1, padx=10, pady=10)
        
        button_file = ttk.Button(
            path_variables_frame,
            text="Output Dir for Processed Data",
            command=lambda: get_dir(self, output_csv_dir)
            # command=filename.create_browser
        )
        button_file.grid(row=1, column=0, padx=10, pady=10)

        output_file_name_label = ttk.Label(path_variables_frame, text="Output CSV Name")
        output_file_name = ttk.Entry(path_variables_frame)
        output_file_name_label.grid(row=1, column=2, padx=10, pady=10)
        output_file_name.grid(row=1, column=3, padx=10, pady=10)
        output_file_name.bind('<KeyRelease>', lambda e: check_string(self, e.widget, ".csv"))
        
        #row 2
        point_data_path = ttk.Entry(path_variables_frame)
        point_data_path.grid(row=2, column=1, padx=10, pady=10)

        button_file = ttk.Button(
            path_variables_frame,
            text="Unprocessed/Raw Point Data CSV",
            command=lambda: get_path(self, point_data_path, file_type="*.csv")
            # command=filename.create_browser
        )
        button_file.grid(row=2, column=0, padx=10, pady=10)


        ##### Start of Simulation Variable Form #####
        simulation_variable_frame = ttk.LabelFrame(self, text="Processing Variables")
        simulation_variable_frame.grid(row=2, column=0, padx=10, pady=10)
        
        aux_met_window_label = ttk.Label(simulation_variable_frame, text="Auxillary Data Search Window (Minutes)")
        aux_met_window = ttk.Entry(simulation_variable_frame)
        aux_met_window_label.grid(row=0, column=0, padx=10, pady=10)
        aux_met_window.grid(row=0, column=1, padx=10, pady=10)
        aux_met_window.bind('<KeyRelease>', lambda e: check_value(self, e.widget, 1, float('inf')))

        flux_met_window_label = ttk.Label(simulation_variable_frame, text="Flux Data Search Window (Minutes)")
        flux_met_window = ttk.Entry(simulation_variable_frame)
        flux_met_window_label.grid(row=0, column=2, padx=10, pady=10)
        flux_met_window.grid(row=0, column=3)
        flux_met_window.bind('<KeyRelease>', lambda e: check_value(self, e.widget, 1, float('inf')))

        emissivity_label = ttk.Label(simulation_variable_frame, text="Object Emissivity Value")
        emissivity = ttk.Entry(simulation_variable_frame)
        emissivity_label.grid(row=1, column=0, padx=10, pady=10)
        emissivity.grid(row=1, column=1, padx=10, pady=10)
        emissivity.bind('<KeyRelease>', lambda e: check_value(self, e.widget, 0, 1))

        point_dist_label = ttk.Label(simulation_variable_frame, text="ROI Distance (meters)")
        point_dist = ttk.Entry(simulation_variable_frame)
        point_dist_label.grid(row=1, column=2, padx=10, pady=10)
        point_dist.grid(row=1, column=3, padx=10, pady=10)
        point_dist.bind('<KeyRelease>', lambda e: check_value(self, e.widget, 0, float('inf')))

        elevation_label = ttk.Label(simulation_variable_frame, text="Elevation (m)")
        elevation = ttk.Entry(simulation_variable_frame)
        elevation_label.grid(row=2, column=2, padx=10, pady=10)
        elevation.grid(row=2, column=3, padx=10, pady=10)
        elevation.bind('<KeyRelease>', lambda e: check_value(self, e.widget, 1, float('inf')))

        # Config File Details
        configuration_frame = ttk.LabelFrame(self, text="Configuration File")
        configuration_frame.grid(row=3, column=0, padx=10, pady=10)

        config_json_dir = ttk.Entry(configuration_frame)
        config_json_dir.grid(row=0, column=1, padx=10, pady=10)
        
        button_file = ttk.Button(
            configuration_frame,
            text="Output Dir for Config File",
            command=lambda: get_dir(self, config_json_dir)
            # command=filename.create_browser
        )
        button_file.grid(row=0, column=0, padx=10, pady=10)

        config_file_name_label = ttk.Label(configuration_frame, text="Output JSON Name")
        config_file_name = ttk.Entry(configuration_frame)
        config_file_name_label.grid(row=0, column=2, padx=10, pady=10)
        config_file_name.grid(row=0, column=3, padx=10, pady=10)
        config_file_name.bind('<KeyRelease>', lambda e: check_string(self, e.widget, ".json"))

        ##### Data Prep Section #####
        form_data = {
            "aux_met_data_path": aux_met_data_path,
            "flux_met_data_path": flux_met_data_path,

            "emissivity": emissivity,
            "Aux_Met_window": aux_met_window,
            "FLUX_Met_window":flux_met_window,

            "output_csv_dir": output_csv_dir,
            "output_csv_file_name": output_file_name,
            "point_data_path": point_data_path,
            "point_dist": point_dist,
            "config_json_dir": config_json_dir,
            "config_json_file_name": config_file_name,

            "elevation": elevation,
        }

        # Add form here? 
        button_frame = ttk.LabelFrame(self, text="")
        button_frame.grid(row=4, column=0)

        submit_form = ttk.Button(
            button_frame,
            text="Compile Config File",
            command=lambda: self.submit_form(form_data)
        )
        submit_form.grid(row=0, column=0, padx=10, pady=10)

        button_page = ttk.Button(
            button_frame, 
            text="Back to Point Data Page",
            command=lambda: window.show_frame("PointData"),
        )
        button_page.grid(row=2, column=0, padx=10, pady=10)


    def submit_form(self, form_data):
        try:
            output_csv_path = form_data['output_csv_dir'].get() + '/' + form_data['output_csv_file_name'].get()
            output_config_json_path = form_data['config_json_dir'].get() + '/' + form_data['config_json_file_name'].get()
            try:
                with open(output_csv_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows([])
            except:
                messagebox.showerror(title="Error in Submission!", message=f'There was an error writing the output csv file: {output_csv_path}.')
                return

            config_data = {
                "data": "point",
                "Aux_Met_window": float(form_data['Aux_Met_window'].get()),
                "FLUX_Met_window": float(form_data['FLUX_Met_window'].get()),
                "output_csv_path": output_csv_path,
                "emissivity": float(form_data['emissivity'].get()),
                "aux_met_data_path": form_data['aux_met_data_path'].get(),
                "flux_met_data_path": form_data['flux_met_data_path'].get(),
                "point_data_path": form_data["point_data_path"].get(),
                "point_dist": float(form_data["point_dist"].get()),
                "elevation": form_data['elevation'].get(), 
            }

            os.makedirs(form_data['output_csv_dir'].get(), exist_ok=True)

            print(f'config_data: {config_data}')

            # configuration_json = form_data['config_json_dir'].get() + os.path.sep + 'point_config.json'
            configuration_json = output_config_json_path
            with open(configuration_json, 'w') as json_file:
                json.dump(config_data, json_file, indent=4)
            messagebox.showinfo(title="Submission", message="Configuration file created successfully.")

        except Exception as exp:
            messagebox.showerror(title="Error in Submission!", message='Please make sure all fields are filled out with the correct data.')
            print(f'error message: {exp}')
