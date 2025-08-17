-------------
Cortex – Real-Time Threat Intelligence NLP
-------------
Cortex is a high-performance Natural Language Processing (NLP) engine for detecting and classifying threat intents from system logs.
It supports real-time streaming via Kafka and WebSocket, offers a REST API, provides a graphical user interface, and is fully containerized for production deployments.


---

📖 Table of Contents

Features

Installation

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

🧠 Training

Train the model on your dataset:

python cortex/cli.py train --data datasets/threat_intents.csv


---

⚡ Real-Time Streaming

Kafka Stream

python cortex/cli.py run --mode kafka --topic pinet_logs --kafka localhost:9092

WebSocket Stream

python cortex/cli.py run --mode websocket --ws ws://localhost:8080/logs


---

🔧 Background Service

Run Cortex as a daemonized background process:

python -m sentenial_x.core.cortex.daemon --mode kafka --topic pinet_logs --kafka localhost:9092


---

🌐 API Server

Start the REST API with Uvicorn:

uvicorn sentenial_x.core.cortex.server:app --host 0.0.0.0 --port 8080


---

📡 Example Request

Test prediction with curl:

curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Suspicious login attempt detected"}'


---

🖥️ GUI

Launch the graphical interface:

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

💡 Tip: For production setups, it’s recommended to use Docker Compose with environment variables for Kafka and API configuration.


---

Would you like me to also add a 📂 Project Structure diagram + a ⚙️ Configuration section (for setting Kafka brokers, WebSocket URLs, model paths, etc.) so new developers can jump in with zero friction?

