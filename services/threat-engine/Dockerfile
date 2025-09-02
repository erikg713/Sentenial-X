FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn joblib pyyaml python-dotenv

EXPOSE 8081

CMD ["uvicorn", "services.threat-engine.api:app", "--host", "0.0.0.0", "--port", "8081"]
