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

# Copy engine folder (mounted via docker-compose, but include for standalone builds)
# Note: In docker-compose, engine is mounted from parent directory

# Expose port for FastAPI
EXPOSE 8000

# Default command - FastAPI server
# Override in docker-compose for Celery worker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
