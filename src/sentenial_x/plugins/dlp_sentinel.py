from sentenial_x.plugins import PluginBase
from hashlib import sha256

class DLPSentinelPlugin(PluginBase):
    """
    Fingerprint sensitive assets, monitor flows, enforce quarantine/redaction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fingerprints = {}  # file_hash -> metadata

    def on_data_discovered(self, event):
        content = event.payload["content"]
        h = sha256(content).hexdigest()
        sens_type = event.payload.get("type")
        self.fingerprints[h] = sens_type

    def on_file_event(self, event):
        content = event.payload["data"]
        h = sha256(content).hexdigest()
        if h in self.fingerprints:
            self.emit("on_dlp_violation", {
                "file_hash": h,
                "type": self.fingerprints[h],
                "action": "quarantine"
            })

    def on_scheduled_task(self, event):
        self.emit("on_dlp_report", {"count": len(self.fingerprints)})
