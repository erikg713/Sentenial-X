def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("feedback_file", type=str)
    parser.add_argument("--output", type=str, default="secure_db/model.pkl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

