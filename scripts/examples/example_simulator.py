# -*- coding: utf-8 -*-
"""
Example: Running a Simulator
----------------------------

Demonstrates how to initialize and run a simulator engine.
"""

from core.simulator import EmulationManager
from core.simulator.wormgpt_clone import WormGPTDetector
from core.simulator.blind_spot_tracker import BlindSpotTracker

def main():
    manager = EmulationManager()
    
    # Register simulators
    worm_sim = WormGPTDetector()
    blind_sim = BlindSpotTracker()
    manager.register(worm_sim)
    manager.register(blind_sim)

    print("Simulators registered:", [sim.name for sim in manager.simulators])

    # Run all simulators
    print("Running all simulators sequentially...")
    results = manager.run_all(sequential=True)
    for res in results:
        print(res)

if __name__ == "__main__":
    main()
