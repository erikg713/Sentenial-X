import requests
from sentenial_x.plugins import PluginBase

class ApiGatekeeperPlugin(PluginBase):
    """
    Crawl microservice specs, fuzz common endpoints, emit schema + playbook.
    """

    def on_service_discovered(self, event):
        swagger_url = event.payload.get("swagger_url")
        spec = requests.get(swagger_url).json()
        self.emit("on_api_spec_loaded", {"spec": spec})

    def on_api_spec_loaded(self, event):
        spec = event.payload["spec"]
        # TODO: iterate spec, run auth/injection tests
        findings = []
        for path, methods in spec.get("paths", {}).items():
            for method, details in methods.items():
                # stub: check if security defined
                if not details.get("security"):
                    findings.append({"path": path, "method": method, "issue": "no_auth"})
        self.emit("on_api_vulns_found", {"findings": findings})

    def on_scheduled_task(self, event):
        # regenerate hardened schemas / playbook each run
        self.emit("on_api_playbook_ready", {"timestamp": event.timestamp})
