async def bind_stream(router, mode="kafka"):
    from sentenial_x.core.cortex.stream_processor import StreamProcessor
    stream = StreamProcessor(router)
    asyncio.create_task(stream.start_stream())

    async def handler(signal):
        await stream.add_signal(signal)

    if mode == "kafka":
        from stream.kafka_signal_consumer import KafkaSignalConsumer
        kafka = KafkaSignalConsumer()
        for signal in kafka.consume():
            await handler(signal)

    elif mode == "websocket":
        from stream.websocket_signal_consumer import WebSocketSignalConsumer
        ws = WebSocketSignalConsumer()
        await ws.consume(handler)

    elif mode == "pinet":
        from stream.pinet_signal_consumer import PiNetSignalConsumer
        pi = PiNetSignalConsumer()
        await pi.consume(handler)
