from sentenial_x.plugins import PluginBase
import statistics

class InsiderWatchPlugin(PluginBase):
    """
    Build user baselines & flag deviations, correlate with HR events.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baselines = {}  # user_id -> stats

    def on_user_activity(self, event):
        user, action = event.payload["user_id"], event.payload["bytes_transferred"]
        hist = self.baselines.setdefault(user, [])
        hist.append(action)
        if len(hist) >= 10:
            mean = statistics.mean(hist)
            stdev = statistics.pstdev(hist)
            if action > mean + 3 * stdev:
                self.emit("on_insider_anomaly", {"user": user, "value": action})

    def on_hr_event(self, event):
        # capture hires/transfers
        self.emit("on_hr_context", {"event": event.payload})

    def on_scheduled_task(self, event):
        # daily summary
        self.emit("on_insider_summary", {"baseline_stats": self.baselines})
