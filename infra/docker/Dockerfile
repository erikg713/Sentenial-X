# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (Python package manager)
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set work directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY chain-executor ./chain-executor
COPY sentenial_core ./sentenial_core
COPY main.py .  # Or your actual entrypoint

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main

# Optional: copy entrypoint or configs
# COPY infra/docker/entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh

# Default command
CMD ["python", "main.py"]

