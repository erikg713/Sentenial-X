# apps/dashboard/widgets/telemetry_chart.py
class TelemetryChartWidget:
    def __init__(self):
        self.data = []

    def add_point(self, telemetry):
        self.data.append(telemetry)

    def render(self):
        return self.data[-20:]
