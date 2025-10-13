#!/bin/bash
# Quick start script for TTS service with Docker

set -e

echo "🚀 Starting LLM Test-Time Scaling Service"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker and try again"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Creating .env from .env.example..."

    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file"
        echo ""
        echo "📝 Please edit .env and add your OPENROUTER_API_KEY"
        echo "   Then run this script again"
        exit 0
    else
        echo "❌ Error: .env.example not found"
        exit 1
    fi
fi

# Check if OPENROUTER_API_KEY is set
source .env
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ Error: OPENROUTER_API_KEY not set in .env"
    echo "Please add your API key to .env file"
    exit 1
fi

echo "✅ Environment configured"
echo ""

# Build and start
echo "🔨 Building Docker image..."
docker-compose build --no-cache

echo ""
echo "🎯 Starting service..."
docker-compose up -d

echo ""
echo "⏳ Waiting for service to be ready..."
sleep 5

# Wait for health check
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "✅ Service is healthy!"
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "   Attempt $RETRY_COUNT/$MAX_RETRIES..."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Service failed to start"
    echo "Check logs with: docker-compose logs tts-service"
    exit 1
fi

echo ""
echo "=========================================="
echo "✨ TTS Service is running!"
echo "=========================================="
echo ""
echo "📍 Service URL: http://localhost:8001"
echo "📚 API Docs:    http://localhost:8001/docs"
echo "💚 Health:      http://localhost:8001/health"
echo ""
echo "Test with:"
echo "  curl http://localhost:8001/v1/models"
echo ""
echo "View logs:"
echo "  docker-compose logs -f tts-service"
echo ""
echo "Stop service:"
echo "  docker-compose down"
echo ""
