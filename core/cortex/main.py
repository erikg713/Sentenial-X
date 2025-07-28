import logging
import os
import asyncio
from sentenial_x.core.cortex import Brainstem, SemanticAnalyzer, DecisionEngine, SignalRouter
from sentenial_x.core.cortex.stream_processor import StreamProcessor

async def main():
    brainstem = Brainstem()
    analyzer = SemanticAnalyzer()
    engine = DecisionEngine()
    router = SignalRouter(brainstem, analyzer, engine)

    stream = StreamProcessor(router)
    asyncio.create_task(stream.start_stream())

    # Simulate incoming real-time signals
    test_signals = [
        {"id": "s1", "threat_level": 9, "description": "Detected RCE payload targeting Apache"},
        {"id": "s2", "threat_level": 4, "description": "Unusual process tree with encoded powershell"},
        {"id": "s3", "threat_level": 2, "description": "User logged in from new device"},
    ]

    for sig in test_signals:
        await stream.add_signal(sig)

    await asyncio.sleep(5)
    stream.stop_stream()

if __name__ == "__main__":
    asyncio.run(main())
