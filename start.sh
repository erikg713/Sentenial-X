#!/bin/bash

# Start the full dev environment
echo "🚀 Building and launching Sentenial-X stack..."
docker-compose up --build -d

echo "✅ Sentenial-X is now running!"
echo " - Flask API:      http://localhost:5000"
echo " - PostgreSQL:     localhost:5432 (user:password)"
echo " - Redis:          localhost:6379"
echo " - pgAdmin:        http://localhost:5050 (admin@sentenialx.local / securepassword)"