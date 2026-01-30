# Backend Dockerfile
# Used for both FastAPI server and Celery worker
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy engine package (now part of backend)
COPY engine/ ./engine/

# Expose port for FastAPI
EXPOSE 8000

# Default command - FastAPI server with gunicorn for production
# Override in docker-compose for Celery worker
# Timeout set to 600s for long-running chart-data requests (30m timeframe)
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "600", "--graceful-timeout", "300"]
