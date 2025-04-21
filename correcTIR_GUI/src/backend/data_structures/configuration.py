import os
import json

class ConfigurationFile:

    def __init__(self):
        self._filename = ''
        self._json_data = {}

    @property
    def filename(self):
        return self._filename

    @property
    def filename(self, name):
        self._filename = name

    @property
    def json_data(self):
        return self._json_data

    def read_json_file(self):
        with open(self._filename, 'r') as f:
            data = json.load(f)
        return data

    def _load_json(self):
        if self._filename != '':
            try:
                self._json_data = self.read_json_file()
            except Exception as err:
                print('There was an exception loading the json data: ', err)
        else:
            print('No filename has been set.')

class JsonData:
    def __init__(self):
        self._config = {
            "data": None, 
            "Aux_Met_window": None,
            "FLUX_Met_window": None,
            "output_csv_path": None,
            "emissivity": None,
            "aux_met_data_path": None,
            "flux_met_data_path": None,
            "first_image_path": None,
            "roi_path": None,
            "roi_dist_path": None,
            "data_type": None,
            "base_folder": None,
            "point_data_path": None,
            "point_dist": None,
        }

    @property
    def config(self):
        return self._config
    
    def add_config_dict(self, data):
        """Set configuration values for values with keys matching expected configuration keys."""
        if type(data) is dict:
            for key in data.keys():
                self.add_key_value(key, data[key])

    def add_key_value(self, key, data):
        if key in self._config:
            self._config[key] = data
        else:
            print(f'Key did not match {key}.')

    def find_set_keys(self):
        """Return a list of keys that have been set."""
        set_keys = []
        for key in self._config.keys():
            if self._config[key] != None and self._config[key] != '':
                set_keys.append(key)
        return set_keys

    def find_unset_keys(self):
        """Return a list of keys that have not been set."""
        unset_keys = []
        for key in self._config.keys():
            if self._config[key] == None or self._config[key] == '':
                unset_keys.append(key)
        return unset_keys
    
    def __str__(self):
        """Print a summary of the config data."""
        return f"Set Keys: {self.find_set_keys()} \n" \
               f"Unset Keys: {self.find_unset_keys()} \n" \
               f"\n" \
               f"Json Data: \n" \
               f"{self._config}"


                    



    
