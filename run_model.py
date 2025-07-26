"""
run_model.py

Main entry point for the PM10 forecasting system.
This script orchestrates the modular components and handles command-line interface.

Usage:
    python run_model.py --data-file data.json [--landuse-pbf landuse.pbf] --output-file output.json
"""

import argparse
from src.io_utils import load_json_data, save_json_output
from src.landuse import load_landuse_data
from src.predictor import generate_output


def main():
    parser = argparse.ArgumentParser(description="Generate PM10 forecasts using trained models.")
    parser.add_argument("--data-file", required=True, help="Path to input data.json")
    parser.add_argument("--landuse-pbf", required=False, help="Path to landuse.pbf")
    parser.add_argument("--output-file", required=True, help="Path to write output.json")
    args = parser.parse_args()

    # Load input data using modular utility
    print(f"Reading input data from: {args.data_file}")
    data = load_json_data(args.data_file)

    # Load landuse data if provided
    landuse_data = None
    if args.landuse_pbf:
        print(f"Reading landuse data from: {args.landuse_pbf}")
        landuse_data = load_landuse_data(args.landuse_pbf)

    # Generate forecasts using the prediction module
    output = generate_output(data, landuse_data=landuse_data)

    # Save output using modular utility
    save_json_output(output, args.output_file)

    print(f"Wrote forecasts to: {args.output_file}")
    if args.landuse_pbf:
        print(f"Land use PBF processed from: {args.landuse_pbf}")


if __name__ == "__main__":
    main()