Production-Grade ML Training Pipeline

A modular, distributed, and production-ready ML training pipeline with experiment tracking, automated orchestration, and real-time monitoring.

Features
End-to-end ML pipeline: data loading → preprocessing → training → evaluation → deployment
Distributed training with PyTorch DDP (multi-process/GPU simulation)
Automated experiment tracking with MLflow
Real-time monitoring with Prometheus
REST API serving via FastAPI
Configurable hyperparameter tuning
Tech Stack
Backend: Python, PyTorch, FastAPI
Monitoring: MLflow, Prometheus
Deployment: Docker, docker-compose
Data: CSV, JSON, HuggingFace datasets
Quick Start

Using Docker (Recommended)

git clone <repo-url>
cd ml_pipeline
docker-compose up --build
docker-compose exec trainer python scripts/run_pipeline.py

Local Setup (Python 3.10+)

pip install -r requirements.txt
python scripts/run_pipeline.py           # Run full pipeline
python scripts/tune.py                   # Hyperparameter tuning
uvicorn api.serve:app --host 0.0.0.0 --port 8080  # Serve model
API Endpoints
POST /predict – Single prediction
POST /predict/batch – Batch predictions
GET /health – Health check
GET /model/info – Model info
GET /metrics – Prometheus metrics
Author

Vandana K H
