# Use official CUDA image with Ubuntu 22.04 and PyTorch support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-venv \
    libgl1-mesa-glx libxrender1 libxext6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 \
    xvfb wget curl git && \
    rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

RUN python3.11 -m pip install --upgrade pip setuptools wheel
RUN python3.11 -m pip install -r requirements.txt

# Copy app source
COPY . .

# Environment variables for PyQt display and GPU
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99

# Start Xvfb (virtual framebuffer) and run the GUI app
CMD Xvfb :99 -screen 0 1024x768x24 & python3 -m sentenial_x.core.cortex.gui
