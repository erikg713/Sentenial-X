# sentenial_x/core/cortex/config.py

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "model": {
        "name": "bert-base-uncased",
        "custom_model_path": os.path.join(BASE_DIR, "models", "cyber_intent_bert"),
        "max_seq_length": 128,
        "device": "cuda" if os.environ.get("USE_GPU", "1") == "1" else "cpu"
    },
    "training": {
        "epochs": 4,
        "batch_size": 16,
        "lr": 2e-5,
        "train_data": os.path.join(BASE_DIR, "data", "train.csv"),
        "val_data": os.path.join(BASE_DIR, "data", "val.csv")
    },
    "streaming": {
        "kafka": {
            "enabled": True,
            "topic": "pinet-threat-logs",
            "bootstrap_servers": "localhost:9092"
        },
        "websocket": {
            "enabled": True,
            "url": "ws://localhost:8765/logs"
        },
        "pinet": {
            "enabled": True,
            "log_path": "/var/log/pinet/realtime.log"
        }
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8080
    }
}
