FROM python:3.10-slim
WORKDIR /app
COPY apps/ransomware-sim/requirements.txt .
RUN pip install -r requirements.txt
COPY apps/ransomware-sim/ .
EXPOSE 8011
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
