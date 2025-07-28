from sentenial_x.plugins import PluginBase

class PhishTrainerPlugin(PluginBase):
    """
    Launch simulated phishing campaigns, track user clicks, deliver adaptive coaching.
    """

    def on_scheduled_campaign(self, event):
        template = event.payload["template"]
        targets = event.payload["users"]
        self.emit("on_phish_sent", {"count": len(targets)})

    def on_user_response(self, event):
        user = event.payload["user_id"]
        action = event.payload["action"]  # clicked, reported, ignored
        self.emit("on_phish_feedback", {"user": user, "action": action})
