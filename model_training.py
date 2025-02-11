import pickle
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import torch
from neuralhydrology.nh_run import start_run

def main():
    # Define the command-line arguments
    parser = argparse.ArgumentParser(description='Run the model training with a specified config file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')

    # Parse the arguments
    args = parser.parse_args()

    # Use the provided config file path
    config_file_path = Path(args.config)

    # Start the run with the specified config file
    start_run(config_file=config_file_path)

if __name__ == "__main__":
    main()