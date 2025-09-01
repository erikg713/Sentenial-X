### Sentenial-X API ###
---------------------------
Sentenial-X API is the production-ready REST API for the Sentenial-X Cybersecurity Platform.
It provides secure, modular, and high-performance endpoints for orchestrator, telemetry, cortex, WormGPT, and exploits.


---

🚀 Features

Built with FastAPI for high-performance asynchronous REST endpoints

Modular route system for easy extension

Threat analysis, orchestration, telemetry, WormGPT emulation, and exploit listing

Secure API key authentication

Fully CORS-enabled for frontend or GUI integration

Configurable via .env and api/config.py

Production-ready Uvicorn + Gunicorn deployment



---

📂 Project Structure

SentenialX/
 ├── api/
 │   ├── server.py          # Main FastAPI entrypoint
 │   ├── routes/            # Route handlers
 │   │   ├── telemetry.py
 │   │   ├── orchestrator.py
 │   │   ├── cortex.py
 │   │   ├── wormgpt.py
 │   │   └── exploits.py
 │   └── utils/             # Utilities: auth, logger, db, response
 │       ├── auth.py
 │       ├── logger.py
 │       ├── db.py
 │       └── response.py
 ├── cli/
 ├── chain-executor/
 ├── core/
 ├── gui/
 ├── requirements.txt
 └── README.md


---

⚡ Installation

git clone https://github.com/your-org/sentenialx.git
cd sentenialx
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt


---

🔧 Configuration

All configurable settings are stored in api/config.py or .env file.

Example .env:

API_KEY=super-secret-key
HOST=0.0.0.0
PORT=8000
WORKERS=4
LOG_LEVEL=INFO
ALLOWED_ORIGINS=*


---

🔑 Authentication

All endpoints require an API Key header:

x-api-key: super-secret-key

Example curl request:

curl -H "x-api-key: super-secret-key" http://localhost:8000/api/telemetry/


---

📡 API Endpoints

Method	Endpoint	Description

GET	/api/telemetry/	Check telemetry status
POST	/api/orchestrator/start	Start the orchestrator
POST	/api/orchestrator/stop	Stop the orchestrator
POST	/api/cortex/analyze	Analyze a threat
POST	/api/wormgpt/emulate	Execute WormGPT emulation
GET	/api/exploits/list	List all available exploits



---

🛡 Production Deployment

We recommend running in Docker + Uvicorn/Gunicorn for stability:

Dockerfile Example:

FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

Run container:

docker build -t sentenialx-api .
docker run -p 8000:8000 sentenialx-api


---

🛠 Development

Run in development mode with hot reload:

uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

Make sure PYTHONPATH includes your CLI modules:

export PYTHONPATH=$PYTHONPATH:$(pwd)


---

📈 Logging

All server logs are standardized with timestamps and log levels:

[2025-09-01 12:00:00] [INFO] - Sentenial-X API starting...


---

📜 License

Proprietary – Sentenial-X Core System


---
