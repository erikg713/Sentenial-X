from sentenial_x.plugins import PluginBase

class SecChatbotPlugin(PluginBase):
    """
    Expose LLM-driven commands in Slack/Teams for scans, honeypots, playbooks.
    """

    def on_chat_command(self, event):
        command = event.payload["text"]
        # stub: simple dispatcher
        if command.startswith("/sentenial status"):
            resource = command.split()[-1]
            # request status from core
            self.emit("on_status_requested", {"resource": resource})
        else:
            self.emit("on_chat_response", {"text": "Unknown command"})
