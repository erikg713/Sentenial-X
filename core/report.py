import os
import json
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)

def save_report(data, report_type="recon", directory="reports"):
    """
    Save a report as a JSON file.

    Args:
        data (dict or list): The data to save in the report.
        report_type (str): The type of report (default is "recon").
        directory (str): Directory where the report will be saved (default is "reports").

    Returns:
        str: The filename of the saved report, or None if an error occurred.
    """
    try:
        # Validate data
        if not isinstance(data, (dict, list, str, int, float, bool, type(None))):
            raise ValueError("Data must be JSON serializable.")
        
        # Create reports directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate unique filename
        filename = f"{directory}/{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.json"
        
        # Save JSON data to file
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        
        logging.info(f"Report successfully saved to {filename}")
        return filename
    
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logging.error(f"Error saving report: {e}")
        return None
