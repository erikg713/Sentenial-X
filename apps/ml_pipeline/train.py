import logging
from datetime import datetime
from pathlib import Path

from ml_pipeline.config import DEFAULT_RANDOM_STATE
from ml_pipeline.cli import parse_args
from ml_pipeline.data_loader import load_feedback
from ml_pipeline.model_trainer import train_model
from ml_pipeline.save_utils import save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    args = parse_args()
    feedback_path = Path(args.feedback_file)
    output_path = Path(args.output)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    texts, labels = load_feedback(feedback_path)
    logger.info("Loaded %d feedback samples", len(texts))

    if args.dry_run:
        logger.info("Dry run: feedback loaded, skipping training and saving.")
        return

    model, vectorizer = train_model(texts, labels, random_state=DEFAULT_RANDOM_STATE)
    metadata = {
        "trained_on": datetime.now().isoformat(),
        "num_samples": len(texts),
        "features": len(vectorizer.get_feature_names_out())
    }
    save_model(model, vectorizer, output_path, metadata)

if __name__ == "__main__":
    main()
