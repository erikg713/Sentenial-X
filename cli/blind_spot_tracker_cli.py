import argparse
from sentenial_core import cli
import simulator
import Blind_Spot_Tracker

def run_cli():
    tracker = BlindSpotTracker()
    tracker.start()
    try:
        input("🔍 Press Enter to stop the Blind Spot Tracker...\n")
    finally:
        tracker.stop()
        print("🧠 Anomalies Detected:")
        for a in tracker.get_anomalies():
            print(a)

if __name__ == "__main__":
    run_cli()
