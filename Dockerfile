# Production Dockerfile for BorderGuard AI
# Compatible with Render, Railway, Hugging Face Spaces, and Fly.io
FROM python:3.11-slim

# Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Install system dependencies: Tesseract OCR, OpenCV graphics libs, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (required for Hugging Face Spaces & security best practices)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /home/appuser/.insightface && \
    chown -R appuser:appuser /app /home/appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER appuser

# Pre-download InsightFace buffalo_l models during build so the live app starts instantly
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']).prepare(ctx_id=0, det_size=(640, 640))" || true

# Copy project source code
COPY --chown=appuser:appuser . .

# Expose the port (dynamically overridden by cloud provider via $PORT)
EXPOSE 8000

# Start Uvicorn bound to 0.0.0.0 and dynamically using $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
