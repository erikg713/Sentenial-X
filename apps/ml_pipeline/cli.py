# apps/ml_pipeline/argparser.py
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a machine learning model using feedback data for Sentenial-X."
    )
    
    parser.add_argument(
        "feedback_file",
        type=str,
        help="Path to the JSON file containing feedback data."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="secure_db/model.pkl",
        help="Path where the trained model will be saved."
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline without actually training or saving the model."
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for detailed output."
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it already exists."
    )
    
    args = parser.parse_args()
    
    # Validate feedback file exists
    if not os.path.isfile(args.feedback_file):
        parser.error(f"Feedback file '{args.feedback_file}' does not exist.")
    
    # Validate output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    return args