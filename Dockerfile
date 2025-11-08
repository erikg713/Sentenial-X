FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/readyz || exit 1
CMD ["python", "-m", "sentenialx"]
