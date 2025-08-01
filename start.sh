#!/bin/bash

# ------------------------------------------
# 🚀 Crypto Trading Bot Startup Script (Enhanced)
# ------------------------------------------

set -e  # Exit on error

echo "=========================================="
echo "🚀 Launching Crypto Trading Bot with Python"
echo "=========================================="

# Load environment variables
if [ -f .env ]; then
  echo "🔧 Loading environment variables from .env..."
  export $(grep -v '^#' .env | xargs)
else
  echo "⚠️  .env file not found. Continuing without it."
fi

# Setup Python virtual environment
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
fi

echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
if [ -f "requirements.txt" ]; then
  echo "📚 Installing Python dependencies..."
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "❌ requirements.txt missing!"
  exit 1
fi

# Prepare logs
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/bot_$(date +%F_%H-%M-%S).log"

# Graceful shutdown trap
trap "echo '🛑 Shutdown requested. Killing bot...'; pkill -f main.py; exit 0" SIGINT SIGTERM

# Run bot with logging and backgrounding
if [ -f "main.py" ]; then
  echo "🤖 Starting bot in background. Logging to $LOG_FILE"
  nohup python3 main.py >> "$LOG_FILE" 2>&1 &
  echo "✅ Bot is running with PID $!"
else
  echo "❌ main.py not found!"
  exit 1
fi