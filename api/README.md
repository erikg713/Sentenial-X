# Sentenial-X API

Sentenial-X API is the production-grade REST API for the Sentenial-X Cybersecurity Platform. It's built for high-performance, maintainability, and secure integrations with frontend, orchestration services, telemetry, threat analysis (cortex), LLM emulation (WormGPT), and exploit inventory tooling.

This README was updated to be concise, professional, and actionable: it includes clear installation and deployment steps, security recommendations, and a short reference for the available endpoints and environment variables.

---

## Key features

- Fast, asynchronous HTTP API using FastAPI and Uvicorn
- Modular route layout for easy extension and testing
- Secure API key authentication and request-level logging
- CORS-enabled for GUI or frontend integrations (configurable)
- Configurable via `.env` + `api/config.py`
- Production-ready deployment patterns (Docker, Gunicorn+Uvicorn)
- OpenAPI / Swagger UI automatically available via FastAPI

---

## Repository layout (top-level)
```
SentenialX/
 ├── api/
 │   ├── server.py          # Main FastAPI entrypoint
 │   ├── config.py          # Configuration and env parsing
 │   ├── routes/            # Route handlers (telemetry, orchestrator, etc.)
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
```
---

## Quick setup (local development)

1. Clone the repository:
   git clone https://github.com/erikg713/Sentenial-X.git
   cd Sentenial-X

2. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate

3. Install Python dependencies:
   pip install --upgrade pip
   pip install -r requirements.txt

4. Create a `.env` file (see configuration below), then run in dev mode:
   uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

FastAPI will expose interactive docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Configuration

Configuration is read from `api/config.py` and environment variables. Use a `.env` file for development; in production, prefer a secrets manager.

Example `.env`:

API_KEY=super-secret-key
HOST=0.0.0.0
PORT=8000
WORKERS=4
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000  # comma-separated origins
SENTRY_DSN=                           # optional: for error monitoring

Security notes:
- Never commit `.env` to source control.
- Use a secret manager (Vault, AWS Secrets Manager, GitHub Secrets) for production API keys.
- Serve the API behind TLS (e.g., Nginx or a load balancer) in production.

---

## Authentication

Every endpoint requires the `x-api-key` header or an equivalent token mechanism. This is enforced by `api/utils/auth.py`.

Example curl:

curl -H "x-api-key: super-secret-key" \
     -H "Content-Type: application/json" \
     http://localhost:8000/api/telemetry/

A typical JSON response:

{
  "status": "ok",
  "uptime_seconds": 1234,
  "services": {
    "cortex": "connected",
    "telemetry": "ok"
  }
}

Consider switching to bearer tokens or OAuth2 for multi-user or long-term deployments.

---

## API Reference (summary)

Method  Endpoint                         Description
GET     /api/telemetry/                   Check telemetry / health
POST    /api/orchestrator/start           Start the orchestrator
POST    /api/orchestrator/stop            Stop the orchestrator
POST    /api/cortex/analyze               Analyze a threat (JSON input)
POST    /api/wormgpt/emulate              Emulate LLM behavior / run a chain
GET     /api/exploits/list                List available exploits

Detailed request and response schema are available from the OpenAPI docs at /openapi.json and via the interactive UI at /docs.

---

## Supported LLM models (used by WormGPT emulation)

The platform can be configured to use or emulate the following LLMs (examples):

- meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
- meta-llama/Llama-4-Scout-17B-16E-Instruct
- meta-llama/Llama-3.3-70B-Instruct-Turbo
- meta-llama/Llama-3.3-70B-Instruct
- meta-llama/Meta-Llama-3.1-405B-Instruct
- meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
- meta-llama/Meta-Llama-3.1-70B-Instruct

Note: actual support depends on your infra and licensing — make sure you comply with the model vendor license before deploying.

---

## Production deployment

Recommended: Docker + process manager. Example Dockerfile:

FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.server:app", \
     "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]

Best practice:
- Use an upstream reverse proxy for TLS termination (Nginx, Traefik).
- Run multiple replicas behind a load balancer.
- Configure readiness and liveness probes.
- Mount secrets via environment variables or secret stores, not via files in the image.
- Limit container capabilities; run as non-root user.

Example docker-compose snippet (production-like):

version: "3.8"
services:
  api:
    build: .
    environment:
      - API_KEY=${API_KEY}
      - LOG_LEVEL=INFO
      - ALLOWED_ORIGINS=${ALLOWED_ORIGINS}
    ports:
      - "8000:8000"
    restart: unless-stopped

---

## Observability & monitoring

- Structured logs (JSON or timestamped text) are produced by `api/utils/logger.py`.
- Enable Sentry (SENTRY_DSN) or a similar APM for error tracking and traces.
- Add metrics (Prometheus) for request latency, error rates, and uptime.

---

## Development tips

- Keep PYTHONPATH include to project root if importing local modules:
  export PYTHONPATH="$PYTHONPATH:$(pwd)"
- Write integration tests against a test instance using TestClient from FastAPI.
- Use black/isort/ruff for consistent formatting and linting.
- Add pre-commit hooks to prevent accidental commits of secrets.

---

## Examples & quick tests

Health check:

curl -H "x-api-key: ${API_KEY}" http://localhost:8000/api/telemetry/

Start orchestrator (POST):
```
curl -X POST -H "x-api-key: ${API_KEY}" \
     -H "Content-Type: application/json" \
     -d '{"mode":"standard"}' \
```     http://localhost:8000/api/orchestrator/start

Analyze with cortex:

curl -X POST -H "x-api-key: ${API_KEY}" \
     -H "Content-Type: application/json" \
     -d '{"indicator":"bad.domain", "type":"domain"}' \
     http://localhost:8000/api/cortex/analyze

---

## Recommended next steps (engineering)

- Add contract tests for each route (input validation + error cases)
- Add automated CI that runs lint, tests, and builds a Docker image
- Harden auth (rotate API keys, implement rate-limiting per key)
- Integrate OpenAPI schema validation in CI and ensure docs are kept up to date
- Add an examples folder with sample curl/HTTPie requests and Postman collection

---

## Logging example

Standardized log format:

[2025-09-01 12:00:00] [INFO] Sentenial-X API starting...
[2025-09-01 12:00:01] [WARN] Invalid API key from 10.0.0.1
[2025-09-01 12:00:02] [ERROR] Cortex analyze error: <trace>

---

## License

-----------------
