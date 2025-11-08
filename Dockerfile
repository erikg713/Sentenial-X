# syntax=docker/dockerfile:1.6
FROM python:3.12-slim-bookworm AS base
WORKDIR /app
RUN pip install --no-cache-dir "optimum[onnxruntime]" numba scikit-learn numpy torch fastapi uvicorn[standard] pydantic aiohttp requests python-json-config opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

FROM base AS prod
COPY sentenialx.py .
COPY models ./models
COPY config ./config
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s CMD ["curl", "-f", "http://localhost:8000/readyz"]
CMD ["uvicorn", "sentenialx:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--limit-concurrency", "1000"]
