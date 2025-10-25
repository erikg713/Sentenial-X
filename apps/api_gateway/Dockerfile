FROM python:3.10-slim
WORKDIR /app
COPY apps/api-gateway/requirements.txt .
RUN pip install -r requirements.txt
COPY apps/api-gateway/ .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
