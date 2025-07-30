### cortex/README FILE

# Train the model
python cortex/cli.py train --data datasets/threat_intents.csv

# Run real-time Kafka stream NLP
python cortex/cli.py run --mode kafka --topic pinet_logs --kafka localhost:9092

# Run real-time WebSocket NLP
python cortex/cli.py run --mode websocket --ws ws://localhost:8080/logs
