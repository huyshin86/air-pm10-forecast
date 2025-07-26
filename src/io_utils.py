"""
io_utils.py

Handles JSON input/output operations and data validation.
"""
import json

def load_json_data(file_path):
    """
    Load and validate JSON input data containing cases, stations, and targets.

    Args:
        file_path (str): Path to the input JSON file

    Returns:
        dict: Parsed JSON data

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValueError: If required fields are missing
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")
    
    # Validate required fields

    return data

def save_json_output(output_data, file_path):
    """
    Save prediction results to a JSON file.

    Args:
        output_data (dict): Dictionary containing predictions
        file_path (str): Path where to save the output JSON
    """
    try:
        with open(file_path, "w") as f:
            json.dump(output_data, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to write output file {file_path}: {e}")