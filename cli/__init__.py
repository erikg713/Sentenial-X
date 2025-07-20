import argparse import logging import sys from sentenial_core.simulator import wormgpt_clone, blind_spot_tracker

logging.basicConfig( level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', handlers=[ logging.StreamHandler(sys.stdout) ] )

def run_wormgpt_clone(args): prompt = args.prompt temperature = args.temperature logging.info("Running WormGPT Clone...") output = wormgpt_clone.generate_malicious_text(prompt=prompt, temperature=temperature) print("\n[ WormGPT Clone Output ]\n", output)

def run_blind_spot_tracker(args): logging.info("Running Blind Spot Tracker...") findings = blind_spot_tracker.scan_blind_spots() print("\n[ Blind Spot Tracker Results ]") for issue in findings: print(f"- {issue}")

def main(): parser = argparse.ArgumentParser(description="Sentenial-X CLI Control Panel") subparsers = parser.add_subparsers(dest="command", required=True)

# WormGPT Clone Command
worm_parser = subparsers.add_parser("wormgpt", help="Run WormGPT Clone module")
worm_parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt to feed into WormGPT clone")
worm_parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
worm_parser.set_defaults(func=run_wormgpt_clone)

# Blind Spot Tracker Command
blind_parser = subparsers.add_parser("blindspots", help="Run Blind Spot Tracker module")
blind_parser.set_defaults(func=run_blind_spot_tracker)

# Future commands for modules like cortex, analyzer, engine, etc. can go here

args = parser.parse_args()
args.func(args)

if name == "main": main()

