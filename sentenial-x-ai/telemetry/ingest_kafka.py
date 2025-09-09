# telemetry/ingest_kafka.py
from confluent_kafka import Consumer
import json
from telemetry.feature_store import upsert_features

c = Consumer({
    'bootstrap.servers': "kafka:9092",
    'group.id': "sentenialx-features",
    'auto.offset.reset': 'earliest'
})
c.subscribe(["telemetry.events"])

try:
    while True:
        msg = c.poll(timeout=1.0)
        if msg is None: continue
        if msg.error():
            continue
        event = json.loads(msg.value())
        features = extract_features(event)   # implement mapping
        upsert_features(features)            # save to db/feature store
finally:
    c.close()
