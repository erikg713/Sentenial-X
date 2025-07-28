from kafka import KafkaConsumer
import json

class KafkaSignalConsumer:
    def __init__(self, topic="sentenial-signals", bootstrap_servers="localhost:9092"):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

    def consume(self):
        for message in self.consumer:
            yield message.value
