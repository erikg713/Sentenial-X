# sentenial_x/core/cortex/cli.py

import argparse
from .stream_handler import StreamHandler
from .model_trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="🧠 Sentenial-X Cortex CLI - NLP Stream Processor")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train
    train_parser = subparsers.add_parser("train", help="Train the NLP intent classifier")
    train_parser.add_argument("--data", type=str, required=True, help="Path to training CSV")

    # Run Stream
    run_parser = subparsers.add_parser("run", help="Run real-time NLP processor")
    run_parser.add_argument("--mode", type=str, choices=["kafka", "websocket"], default="kafka", help="Stream mode")
    run_parser.add_argument("--topic", type=str, help="Kafka topic name")
    run_parser.add_argument("--kafka", type=str, help="Kafka bootstrap server")
    run_parser.add_argument("--ws", type=str, help="WebSocket server URL")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.data)
    elif args.command == "run":
        stream = StreamHandler(
            mode=args.mode,
            kafka_topic=args.topic,
            kafka_bootstrap=args.kafka,
            ws_url=args.ws
        )
        stream.start()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

