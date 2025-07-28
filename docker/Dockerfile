ENTRYPOINT ["sh", "-c", "echo 'Sentenial-X: Crafted for resilience, Engineered for vengeance.' && uvicorn api_server:app --host 0.0.0.0 --port 8000"]
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ api_server.py ./
COPY models/optimized ./models/optimized
EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

