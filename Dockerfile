FROM python:3.11-slim

WORKDIR /app

COPY . /app
COPY .env /app/.env

RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=sentinel_core.py
ENV FLASK_ENV=development
ENV FLASK_DEBUG=1

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
