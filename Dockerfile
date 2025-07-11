FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["python", "sentinel_main.py", "--mode=passive"]
