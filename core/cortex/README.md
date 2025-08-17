Cortex – Real-Time Threat Intelligence NLP

Cortex is a high-performance Natural Language Processing (NLP) engine for detecting and classifying threat intents from system logs.
It supports real-time streaming via Kafka and WebSocket, offers a REST API, provides a graphical user interface, and is fully containerized for production deployments.


---

📖 Table of Contents

Features

Installation

Project Structure

Configuration

Training

Real-Time Streaming

Background Service

API Server

GUI

Docker Deployment

Example Request



---

✨ Features

🔎 Threat Intent Classification from log data.

⚡ Real-time Streaming support via Kafka and WebSocket.

🌐 REST API for easy integration with external systems.

🖥️ GUI for visualization and interaction.

🐳 Dockerized Deployment for scalable production environments.



---

📦 Installation

Clone the repository and install dependencies:

git clone https://github.com/your-org/cortex.git
cd cortex
pip install -r requirements.txt


---

📂 Project Structure

cortex/
├── cli.py                  # Command line interface (train/run)
├── __init__.py
datasets/
├── threat_intents.csv       # Example dataset
sentenial_x/
└── core/
    └── cortex/
        ├── daemon.py       # Background service runner
        ├── server.py       # REST API server (FastAPI + Uvicorn)
        ├── gui.py          # GUI launcher
        ├── models/         # Trained model artifacts
        ├── utils/          # Helper utilities
        └── __init__.py
docker-compose.yml
Dockerfile
README.md
requirements.txt


---

⚙️ Configuration

Cortex can be configured via environment variables or command-line arguments.

Parameter	Description	Default

--mode	Run mode: kafka | websocket	kafka
--topic	Kafka topic to consume	pinet_logs
--kafka	Kafka broker address	localhost:9092
--ws	WebSocket endpoint	ws://localhost:8080/logs
--host	API server host	0.0.0.0
--port	API server port	8080
DATA_PATH	Path to training dataset	datasets/threat_intents.csv
MODEL_PATH	Path to save/load trained models	sentenial_x/core/cortex/models/


💡 You can define these in a .env file for Docker and Docker Compose deployments.


---

🧠 Training

python cortex/cli.py train --data datasets/threat_intents.csv


---

⚡ Real-Time Streaming

Kafka Stream

python cortex/cli.py run --mode kafka --topic pinet_logs --kafka localhost:9092

WebSocket Stream

python cortex/cli.py run --mode websocket --ws ws://localhost:8080/logs


---

🔧 Background Service

python -m sentenial_x.core.cortex.daemon --mode kafka --topic pinet_logs --kafka localhost:9092


---

🌐 API Server

uvicorn sentenial_x.core.cortex.server:app --host 0.0.0.0 --port 8080


---

📡 Example Request

curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Suspicious login attempt detected"}'


---

🖥️ GUI

python -m sentenial_x.core.cortex.gui


---

🐳 Docker Deployment

Build the Container

docker build -t sentenialx-gui .

Run the Container (with GPU support)

docker run --gpus all --rm -it sentenialx-gui

Or with Docker Compose

docker-compose up --build

---
