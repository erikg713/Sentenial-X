ENTRYPOINT ["sh", "-c", "echo 'Sentenial-X: Crafted for resilience, Engineered for vengeance.' && uvicorn api_server:app --host 0.0.0.0 --port 8000"]

FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
