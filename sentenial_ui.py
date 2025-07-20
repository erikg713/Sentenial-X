# sentenial_ui.py

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.reactive import reactive
from textual.containers import Container
from rich.panel import Panel
from time import sleep
from sentenialx.ai_core.datastore import get_recent_threats

class ThreatFeed(Static):
    threats = reactive([])

    def on_mount(self):
        self.set_interval(2, self.refresh_feed)

    def refresh_feed(self):
        new_threats = get_recent_threats(limit=5)
        if new_threats != self.threats:
            self.threats = new_threats
            self.update_feed()

    def update_feed(self):
        lines = []
        for row in self.threats:
            ts, ttype, src, payload, conf = row[1:6]
            lines.append(f"[cyan]{ts}[/cyan] [bold blue]{ttype}[/] | {src} | 🔥 {conf:.2f}\n[yellow]{payload}[/]")
        self.update(Panel("\n".join(lines), title="🛰️ LIVE THREAT FEED", border_style="cyan"))


class SentenialXDashboard(App):
    CSS_PATH = None
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(ThreatFeed(), id="main")
        yield Footer()

if __name__ == "__main__":
    SentenialXDashboard().run()