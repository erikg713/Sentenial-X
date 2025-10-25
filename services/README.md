# Sentenial-X

A comprehensive cybersecurity platform for threat detection, simulation, and response in a SOC environment.

## Overview
- **apps/**: Core applications like API gateway and dashboard.
- **services/**: Microservices for auth, threat analysis, etc.
- **core/**: ML engine and models.
- **libs/**: Shared libraries.
- **infra/**: Deployment tools.
- **tests/**: Unit and integration tests.
- **scripts/**: Utility scripts.

## Setup
1. Copy `.env.example` to `.env` and fill in values.
2. Install dependencies: `pip install -r requirements.txt` and `npm install`.
3. Run services: Use `docker-compose up` from `infra/docker/`.
4. Access dashboard at http://localhost:3000.

## Development
- Backend: Python 3.10+, FastAPI, PyTorch.
- Frontend: Next.js 14+.
- ML: BERT-based models for threat intent.
- Deployment: Docker, Kubernetes, Terraform (AWS EKS).

## License
Proprietary - For educational/simulation purposes only.
