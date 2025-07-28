# Simulated PiNet signal listener (replace with real Pi SDK once available)
import asyncio
import random

class PiNetSignalConsumer:
    def __init__(self):
        pass

    async def consume(self, callback):
        while True:
            fake_signal = {
                "id": f"pi-sig-{random.randint(100, 999)}",
                "threat_level": random.randint(1, 10),
                "description": random.choice([
                    "User gained root access",
                    "Base64 encoded command seen in logs",
                    "Normal user login",
                    "Privilege escalation attempt",
                ])
            }
            await callback(fake_signal)
            await asyncio.sleep(3)
