FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt config.yml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src

# Expose inference port
EXPOSE 8000

CMD ["python", "-u", "src/api_server.py"]

