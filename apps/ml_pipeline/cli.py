import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train model with feedback data.")
    parser.add_argument("feedback_file", type=str, help="Path to the feedback JSON file.")
    parser.add_argument("--output", type=str, default="secure_db/model.pkl", help="Path to save the trained model.")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline without training or saving.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()
