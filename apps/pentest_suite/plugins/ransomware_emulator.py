# plugins/ransomware_emulator.py

from plugins.plugin_base import Plugin
from ransomware_emulator.emulator import RansomwareEmulator
from typing import Dict, Any, List

class RansomwareEmulatorPlugin(Plugin):
    name = "ransomware_emulator"
    description = "Run a synthetic ransomware campaign against your files"

    # describe the parameters for the GUI to render
    parameters: List[Dict[str, Any]] = [
        {
            "name": "payload_name",
            "type": "select",
            "label": "Select Payload",
            "choices": RansomwareEmulator().list_payloads().keys()
        },
        {
            "name": "file_count",
            "type": "int",
            "label": "Number of Test Files",
            "default": 10,
            "min": 1,
            "max": 100
        },
        {
            "name": "monitor",
            "type": "bool",
            "label": "Enable Monitoring",
            "default": True
        }
    ]

    def __init__(self):
        self.emulator = RansomwareEmulator()

    def run(self, payload_name: str, file_count: int = 10, monitor: bool = True) -> Dict:
        """
        Invokes the emulator and returns its result dict.
        """
        return self.emulator.run_campaign(
            payload_name=payload_name,
            file_count=file_count,
            monitor=monitor
        )

def register():
    return RansomwareEmulatorPlugin()
