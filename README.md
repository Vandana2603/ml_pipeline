# 🚀 Production-Grade ML Training Pipeline

A fully modular, distributed, and production-ready ML training pipeline with experiment tracking, automated orchestration, and real-time monitoring.

---

## 🏗️ System Architecture

```
ml_pipeline/
├── data_pipeline/          # Data loading, preprocessing, versioning
│   ├── loader.py           # Dataset loading (CSV/JSON/HuggingFace)
│   ├── preprocessor.py     # Cleaning, tokenization, feature engineering
│   └── versioning.py       # Dataset versioning with hashing
├── training/               # Distributed training engine
│   ├── trainer.py          # DDP trainer with checkpointing
│   ├── model.py            # Model definitions (MLP, LSTM, Transformer)
│   └── distributed.py      # DDP setup and utilities
├── evaluation/             # Automated evaluation pipeline
│   └── evaluator.py        # Metrics: accuracy, F1, precision, recall
├── orchestration/          # Pipeline orchestration
│   └── pipeline.py         # End-to-end workflow controller
├── monitoring/             # Real-time tracking & logging
│   └── tracker.py          # MLflow integration + metrics tracking
├── api/                    # FastAPI model serving
│   └── serve.py            # REST API for model inference
├── configs/                # All configuration files
│   └── config.yaml         # Central config
├── scripts/                # Entry-point scripts
│   ├── run_pipeline.py     # Run full pipeline
│   └── tune.py             # Hyperparameter tuning
├── docker-compose.yml      # Full stack: training + MLflow + monitoring
├── Dockerfile              # Reproducible training container
└── requirements.txt        # All dependencies
```

---

## ⚡ Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and enter the project
cd ml_pipeline

# Build and start all services
docker-compose up --build

# In another terminal, run the full pipeline
docker-compose exec trainer python scripts/run_pipeline.py

# View MLflow UI
open http://localhost:5000

# View Prometheus metrics
open http://localhost:8001/metrics

# View FastAPI model server
open http://localhost:8080/docs
```

### Option 2: Local (Python 3.10+)

```bash
# Install dependencies
pip install -r requirements.txt

# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000 &

# Run the full pipeline
python scripts/run_pipeline.py

# OR run individual stages
python scripts/run_pipeline.py --stage data        # Data only
python scripts/run_pipeline.py --stage train       # Training only
python scripts/run_pipeline.py --stage evaluate    # Evaluation only

# Hyperparameter tuning
python scripts/tune.py

# Serve trained model
uvicorn api.serve:app --host 0.0.0.0 --port 8080
```

---

## 🔧 Configuration

Edit `configs/config.yaml` to control every aspect of the pipeline:

```yaml
data:
  source: "sklearn"          # sklearn | csv | json | huggingface
  dataset_name: "breast_cancer"
  test_size: 0.2

model:
  type: "mlp"               # mlp | lstm | transformer
  hidden_dims: [256, 128, 64]

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  distributed: true         # Enable DDP
  num_processes: 2          # Simulated GPUs/processes

tracking:
  mlflow_uri: "http://localhost:5000"
  experiment_name: "production-ml-pipeline"
```

---

## 🌐 Distributed Training (DDP)

The pipeline uses **PyTorch Distributed Data Parallel**:

- **Multi-process simulation**: Runs N processes on CPU to simulate N GPUs
- **DistributedSampler**: Partitions data across workers — no overlap
- **Gradient sync**: Automatic all-reduce across workers via `torch.distributed`
- **Checkpointing**: Saves every N epochs; resumes seamlessly

```bash
# Explicit multi-process launch (2 workers)
python -m torch.distributed.launch --nproc_per_node=2 scripts/run_pipeline.py --stage train
```

---

## 📊 Experiment Tracking (MLflow)

Every run logs:
- **Hyperparameters**: LR, batch size, architecture, epochs
- **Metrics per epoch**: train loss, val loss, val accuracy
- **Artifacts**: model checkpoint, preprocessor, evaluation report
- **Tags**: git hash, Python version, dataset version

Compare experiments at `http://localhost:5000`.

---

## 📈 Scalability Considerations

| Concern | Solution |
|---|---|
| Data at scale | Chunked loading, versioned datasets |
| Multi-GPU training | PyTorch DDP with NCCL backend |
| Experiment explosion | MLflow with tagging + search |
| Pipeline failures | Retry logic + stage-level checkpointing |
| Model serving | FastAPI with async inference |
| Monitoring | Prometheus metrics endpoint |

---

## 🔌 API Endpoints

Once the model server is running:

```
POST /predict          - Single prediction
POST /predict/batch    - Batch predictions
GET  /health           - Health check
GET  /model/info       - Model metadata
GET  /metrics          - Prometheus metrics
```

---

## 🧪 Hyperparameter Tuning

```bash
python scripts/tune.py
```

Runs a configurable grid/random search over:
- Learning rates: [0.1, 0.01, 0.001]
- Hidden dimensions: [[256,128], [512,256,128]]
- Batch sizes: [32, 64, 128]

All trials logged to MLflow for comparison.
