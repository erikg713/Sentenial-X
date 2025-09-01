## 🛡 Sentenial-X API ##

The Sentenial-X API is the production-ready REST interface for the Sentenial-X Cybersecurity Platform.
It provides endpoints for threat analysis, orchestration, telemetry monitoring, exploit emulation, and WormGPT simulations.


---

🚀 Features

FastAPI-powered RESTful API (blazing fast, async-first).

Modular routing system for maintainability.

Secure API key authentication.

CORS-enabled for integration with GUI/Frontend dashboards.

Ready for Dockerized deployments with Gunicorn + Uvicorn workers.



---

📂 Project Structure

SentenialX/
 ├── api/
 │   ├── server.py           # FastAPI entrypoint
 │   ├── routes/             # Route handlers
 │   │   ├── telemetry.py    # System metrics
 │   │   ├── orchestrator.py # Orchestration control
 │   │   ├── cortex.py       # Threat intelligence analysis
 │   │   ├── wormgpt.py      # WormGPT emulation
 │   │   ├── exploits.py     # Exploit utilities
 │   └── utils/              # Core utilities
 │       ├── auth.py         # API key auth
 │       ├── logger.py       # Centralized logging
 │       ├── db.py           # Database integration placeholder
 │       ├── response.py     # Standardized responses
 ├── cli/                    # CLI tools
 ├── chain-executor/         # Execution engine
 ├── core/                   # Core detection logic
 ├── gui/                    # Dashboard frontend
 ├── requirements.txt        # Python dependencies
 └── README.md               # Documentation


---

⚡ Installation

Clone the repository and install dependencies:

git clone https://github.com/your-org/sentenialx.git
cd sentenialx
pip install -r requirements.txt


---

▶️ Running the API

Development (hot reload)

uvicorn api.server:app --reload

Production (multi-worker)

uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4

Or via Docker:

docker build -t sentenialx-api .
docker run -p 8000:8000 sentenialx-api


---

🔑 Authentication

All endpoints require an API Key via the x-api-key header.

Example:

curl -H "x-api-key: super-secret-key" http://localhost:8000/api/telemetry/


---

📡 Available Endpoints

Endpoint	Method	Description

/api/telemetry/	GET	Get system telemetry
/api/orchestrator/start	POST	Start orchestrator
/api/orchestrator/stop	POST	Stop orchestrator
/api/cortex/analyze	POST	Run threat analysis
/api/wormgpt/emulate	POST	Execute WormGPT emulation
/api/exploits/list	GET	List available exploits



---

🛡 Deployment Recommendations

For production we suggest:

Run behind NGINX or a reverse proxy.

Use Gunicorn + Uvicorn workers for scaling.

Secure API key management via environment variables.

Deploy via Docker/Kubernetes for consistency.



---

⚙️ Environment Setup

# Install dependencies
pip install fastapi uvicorn pydantic python-dotenv

# Add CLI modules to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set API key (replace in production)
export API_KEY="super-secret-key"

# Run the API
uvicorn api.server:app --host 0.0.0.0 --port 8000


---

📜 License

Proprietary – Part of the Sentenial-X Core System.
Unauthorized redistribution is prohibited.


---
