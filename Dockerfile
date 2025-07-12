# Build the image
docker build -t sentenialx .

# Run in passive monitoring mode
docker run --rm -p 8000:8000 --env-file .env sentenialxFROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "sentinel_main.py", "--mode=passive"]
# Build the image
docker build -t sentenialx .

# Run in passive monitoring mode
docker run --rm -p 8000:8000 --env-file .env sentenialx