from sentenial_x.plugins import PluginBase

class TwinLabPlugin(PluginBase):
    """
    Mirror network segments, run exploit chains off-line, produce risk narratives.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.twins = {}            # twin_id -> blueprint/state
        self.snapshots = {}        # twin_id -> list of snapshots
        self.scenarios = {}        # scenario_id -> details

    def on_deploy(self, event):
        """
        Deploy a new digital twin from a blueprint.
        payload: { twin_id, blueprint }
        """
        twin_id = event.payload["twin_id"]
        blueprint = event.payload["blueprint"]
        # stub: provision sandbox resources
        self.twins[twin_id] = {"blueprint": blueprint, "state": "running"}
        self.emit("on_twin_deployed", {"twin_id": twin_id})

    def on_snapshot(self, event):
        """
        Take a snapshot of the twin's current state.
        payload: { twin_id }
        """
        twin_id = event.payload["twin_id"]
        state = self.twins.get(twin_id, {}).get("state")
        snap = {"state": state, "timestamp": event.timestamp}
        self.snapshots.setdefault(twin_id, []).append(snap)
        self.emit("on_twin_snapshot", {"twin_id": twin_id, "snapshot": snap})

    def on_experiment(self, event):
        """
        Execute a custom experiment script in the sandbox.
        payload: { twin_id, steps: [...] }
        """
        twin_id = event.payload["twin_id"]
        steps = event.payload["steps"]
        # stub: run steps
        result = {"twin_id": twin_id, "outcome": "success", "details": steps}
        self.emit("on_experiment_result", result)

    def on_simulate_patch_rollout(self, event):
        """
        Simulate rolling out a patch across the twin.
        payload: { twin_id, patch_id, hosts: [...] }
        """
        twin_id = event.payload["twin_id"]
        patch_id = event.payload["patch_id"]
        hosts = event.payload["hosts"]
        # stub: apply patch in sandbox
        applied = len(hosts)
        narrative = f"Applied patch {patch_id} to {applied} hosts."
        self.emit("on_patch_simulation_complete", {
            "twin_id": twin_id,
            "patch_id": patch_id,
            "success_count": applied,
        })
        # auto-generate narrative
        self.emit("on_risk_narrative", {
            "twin_id": twin_id,
            "text": narrative
        })

    def on_simulate_attack_chain(self, event):
        """
        Run a predefined attack chain scenario.
        payload: { twin_id, scenario_id }
        """
        twin_id = event.payload["twin_id"]
        scenario_id = event.payload["scenario_id"]
        # stub: replay scenario
        outcome = "breach detected"  # or "contained"
        self.emit("on_attack_chain_complete", {
            "twin_id": twin_id,
            "scenario_id": scenario_id,
            "outcome": outcome
        })

    def on_generate_narrative(self, event):
        """
        Ask the LLM to produce a risk narrative for the twin.
        payload: { twin_id, context: {...} }
        """
        twin_id = event.payload["twin_id"]
        context = event.payload.get("context", {})
        # stub: call LLM
        text = f"Risk narrative for twin {twin_id}: system is resilient."
        self.emit("on_risk_narrative", {
            "twin_id": twin_id,
            "text": text
        })

    def on_list_scenarios(self, event):
        """
        Return all available attack or test scenarios.
        payload: {}
        """
        # stub: list hardcoded scenarios
        self.scenarios = {
            "phishing_lateral": "Phishing -> creds -> lateral movement",
            "ransomware": "Deploy ransomware, simulate backups"
        }
        self.emit("on_scenario_list", {"scenarios": self.scenarios})

    def on_teardown(self, event):
        """
        Tear down a twin and its resources.
        payload: { twin_id }
        """
        twin_id = event.payload["twin_id"]
        self.twins.pop(twin_id, None)
        self.snapshots.pop(twin_id, None)
        self.emit("on_twin_teardown", {"twin_id": twin_id})

    def on_scheduled_task(self, event):
        """
        Daily cleanup: snapshot all twins and remove >7-day-old snapshots.
        """
        for twin_id in list(self.twins.keys()):
            # take fresh snapshot
            self.on_snapshot(event)
            # prune old
            snaps = self.snapshots.get(twin_id, [])
            keep = [s for s in snaps if (event.timestamp - s["timestamp"]).days <= 7]
            self.snapshots[twin_id] = keep
        self.emit("on_twinlab_daily_maintenance", {
            "active_twins": list(self.twins.keys())
        })
