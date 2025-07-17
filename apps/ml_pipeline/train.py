def main():
    args = parse_args()
    feedback_path = Path(args.feedback_file)
    output_path = Path(args.output)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    texts, labels = load_feedback(feedback_path)
    logger.info("Loaded %d samples", len(texts))

    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return

    model, vectorizer = train_model(texts, labels)
    metadata = {
        "trained_on": datetime.now().isoformat(),
        "num_samples": len(texts),
        "features": len(vectorizer.get_feature_names_out())
    }

    save_model(model, vectorizer, output_path, metadata)
    logger.info("Model saved to %s", output_path)

