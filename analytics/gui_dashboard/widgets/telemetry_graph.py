# analytics/gui_dashboard/widgets/telemetry_graph.py

import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Dict, List
from core.cortex.orchestrator import AgentOrchestrator  # Your orchestrator module


class TelemetryGraphWidget(QWidget):
    """
    GUI widget to visualize real-time telemetry from agents.
    """

    def __init__(self, orchestrator: AgentOrchestrator, update_interval_ms: int = 1000, parent=None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.update_interval_ms = update_interval_ms

        # Layout
        self.layout = QVBoxLayout()
        self.title_label = QLabel("Telemetry Metrics")
        self.layout.addWidget(self.title_label)

        # Matplotlib figure
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Telemetry over time")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)

        # Data buffer
        self.data_buffer: Dict[str, List[float]] = {}

        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_graph)
        self.timer.start(self.update_interval_ms)

    def fetch_telemetry(self) -> Dict[str, float]:
        """
        Retrieve latest telemetry from all agents.
        Returns a dictionary of metric_name: value
        """
        telemetry_data = {}
        for agent_id, agent_obj in self.orchestrator.list_agents().items():
            metrics = agent_obj.get_telemetry()
            for key, value in metrics.items():
                telemetry_data[f"{agent_id}-{key}"] = value
        return telemetry_data

    def update_graph(self):
        """
        Pull telemetry and update the matplotlib graph.
        """
        telemetry = self.fetch_telemetry()
        for key, value in telemetry.items():
            if key not in self.data_buffer:
                self.data_buffer[key] = []
            self.data_buffer[key].append(value)
            # Keep only last 100 data points
            if len(self.data_buffer[key]) > 100:
                self.data_buffer[key] = self.data_buffer[key][-100:]

        self.ax.clear()
        for key, values in self.data_buffer.items():
            self.ax.plot(values, label=key)

        self.ax.set_title("Telemetry over time")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.legend(loc="upper left", fontsize=8)
        self.canvas.draw()


# Example usage
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    from core.cortex.orchestrator import AgentOrchestrator

    app = QApplication(sys.argv)
    orchestrator = AgentOrchestrator()
    widget = TelemetryGraphWidget(orchestrator)
    widget.show()
    sys.exit(app.exec_())
