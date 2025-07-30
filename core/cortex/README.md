### cortex/README FILE

# Train the model
python cortex/cli.py train --data datasets/threat_intents.csv

# Run real-time Kafka stream NLP
python cortex/cli.py run --mode kafka --topic pinet_logs --kafka localhost:9092

# Run real-time WebSocket NLP
python cortex/cli.py run --mode websocket --ws ws://localhost:8080/logs

# Run as a Background Servicde #
python -m sentenial_x.core.cortex.daemon --mode kafka --topic pinet_logs --kafka localhost:9092

# RUN THE SERVER #
uvicorn sentenial_x.core.cortex.server:app --host 0.0.0.0 --port 8080
# test with
curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d '{"text":"Suspicious login attempt detected"}'

# RUN GUI #
python -m sentenial_x.core.cortex.gui

# BUILD DOCKER #
🖥️ Run steps
Build the container:

bash
Copy
Edit
docker build -t sentenialx-gui .
Run the container:

bash
Copy
Edit
docker run --gpus all --rm -it sentenialx-gui
Or with docker-compose:

bash
Copy
Edit
docker-compose up --build
