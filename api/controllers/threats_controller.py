from api.utils.logger import log_info

class ThreatsController:
    async def list_threats(self):
        log_info("Listing available threats")
        return {"threats": ["APT-Simulation", "EternalBlue", "StrutsRCE"]}

    async def simulate_threat(self, payload: dict):
        log_info(f"Simulating threat with payload: {payload}")
        # TODO: integrate with simulator engine
        return {"status": "success", "details": payload} 