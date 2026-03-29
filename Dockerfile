FROM python:3.11-slim

LABEL maintainer="ml-engineering"
LABEL description="Production ML Training Pipeline"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories
RUN mkdir -p checkpoints data/versioned evaluation_results mlruns

# Default: run full pipeline
CMD ["python", "scripts/run_pipeline.py"]
