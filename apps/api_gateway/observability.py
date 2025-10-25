from fastapi import FastAPI
import logging

def init_observability(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    # Mock observability setup (e.g., Prometheus integration)
    logging.info("Observability initialized")
