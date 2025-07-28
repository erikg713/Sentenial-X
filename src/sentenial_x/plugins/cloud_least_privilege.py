from sentenial_x.plugins import PluginBase

class CloudLeastPrivilegePlugin(PluginBase):
    """
    Scan IAM configs, detect over-privilege, suggest least-priv diffs.
    """

    def on_iam_config_change(self, event):
        policies = event.payload["policies"]  # dict of role->policy
        overpriv = []
        for role, policy in policies.items():
            # stub: flag any wildcard in actions
            if any(stmt.get("Action") == "*" for stmt in policy.get("Statement", [])):
                overpriv.append(role)
        self.emit("on_overpriv_detected", {"roles": overpriv})

    def on_scheduled_task(self, event):
        self.emit("on_iam_recommendations_ready", {"timestamp": event.timestamp})
