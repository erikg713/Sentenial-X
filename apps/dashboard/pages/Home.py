# apps/dashboard/pages/Home.py
import asyncio
from apps.dashboard.pages.dashboard_page import DashboardPage

class HomePage:
    """
    Sentenial-X Dashboard Home Page
    Combines all widgets and provides live data rendering.
    """
    def __init__(self):
        self.dashboard = DashboardPage()

    def render(self):
        """
        Render the full dashboard with current widget states
        """
        rendered_data = self.dashboard.render()
        self.display(rendered_data)
        return rendered_data

    def display(self, data):
        """
        Pretty-print dashboard sections to console for CLI view
        """
        print("=== Sentenial-X Dashboard HOME ===")
        for section, content in data.items():
            print(f"\n[{section.upper()}]")
            if isinstance(content, dict):
                for key, value in content.items():
                    print(f"  {key}: {value}")
            elif isinstance(content, list):
                for item in content:
                    print(f"  - {item}")
            else:
                print(f"  {content}")

    async def live_update(self, interval=5):
        """
        Continuously updates the dashboard every `interval` seconds
        """
        while True:
            self.render()
            await asyncio.sleep(interval)

# Example run as CLI
if __name__ == "__main__":
    home = HomePage()
    try:
        asyncio.run(home.live_update(interval=5))
    except KeyboardInterrupt:
        print("\n[Dashboard terminated by user]")
