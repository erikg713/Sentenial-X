# sentenial_core/sentenial_main.py

import argparse
import asyncio
import logging
import sys
import threading

from sentenial_x.core.cortex.cli import main as cortex_cli
from sentenial_x.core.cortex.daemon import CortexDaemon
from sentenial_x.core.cortex.server import app as fastapi_app
from sentenial_x.core.cortex.gui import run_gui

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger("sentenial_main")

def run_api(host="0.0.0.0", port=8080):
    logger.info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port)

def run_daemon(mode, kafka_topic, kafka_bootstrap, ws_url):
    logger.info(f"Starting CortexDaemon with mode={mode}")
    daemon = CortexDaemon(
        mode=mode,
        kafka_topic=kafka_topic,
        kafka_bootstrap=kafka_bootstrap,
        ws_url=ws_url
    )
    daemon.run()

def run_gui_async():
    logger.info("Starting GUI")
    run_gui()

def main():
    parser = argparse.ArgumentParser(description="Sentenial-X Main Entrypoint")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # CLI passthrough
    cli_parser = subparsers.add_parser("cli", help="Run Cortex CLI")
    cli_parser.add_argument('args', nargs=argparse.REMAINDER)

    # API server
    api_parser = subparsers.add_parser("api", help="Run FastAPI server")
    api_parser.add_argument("--host", default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=8080)

    # Daemon runner
    daemon_parser = subparsers.add_parser("daemon", help="Run background daemon")
    daemon_parser.add_argument("--mode", choices=["kafka", "websocket"], default="kafka")
    daemon_parser.add_argument("--topic", default="pinet_logs")
    daemon_parser.add_argument("--kafka", default="localhost:9092")
    daemon_parser.add_argument("--ws", default=None)

    # GUI launcher
    gui_parser = subparsers.add_parser("gui", help="Run PyQt GUI")

    args = parser.parse_args()

    if args.command == "cli":
        # Pass through arguments to cortex.cli.main
        sys.argv = ["cli"] + args.args
        cortex_cli()
    elif args.command == "api":
        run_api(host=args.host, port=args.port)
    elif args.command == "daemon":
        run_daemon(mode=args.mode, kafka_topic=args.topic, kafka_bootstrap=args.kafka, ws_url=args.ws)
    elif args.command == "gui":
        run_gui_async()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

