 # Import the main pipeline function from correcTIR package
from correcTIR.Main_Functions import run_pipeline

# Define the config path
config_path = './config.json'

# Run the full pipeline
run_pipeline(config_path)
