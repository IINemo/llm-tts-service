# Multi-stage build for smaller image size
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY service/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy lm-polygraph and install it
COPY lm-polygraph/ ./lm-polygraph/
RUN pip install --no-cache-dir --user ./lm-polygraph/

# Final stage
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    PYTHONPATH=/app

# Create non-root user for security
RUN useradd -m -u 1000 tts && \
    mkdir -p /app && \
    chown -R tts:tts /app

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/tts/.local

# Copy application code
COPY --chown=tts:tts llm_tts/ ./llm_tts/
COPY --chown=tts:tts service/ ./service/

# Switch to non-root user
USER tts
ENV PATH=/home/tts/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the service
CMD ["python", "service/main.py"]
