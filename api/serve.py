"""
api/serve.py
FastAPI model server with single + batch prediction, health check,
Prometheus metrics, and model metadata endpoints.
"""

import time
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Prometheus metrics
# ------------------------------------------------------------------ #
REQUEST_COUNT  = Counter("ml_api_requests_total", "Total API requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("ml_api_latency_seconds", "Request latency", ["endpoint"])

# ------------------------------------------------------------------ #
# App
# ------------------------------------------------------------------ #
app = FastAPI(
    title="ML Pipeline Model Server",
    description="Production model serving API",
    version="1.0.0",
)

# Global state
_model = None
_config = None
_preprocessor = None
_model_info: Dict[str, Any] = {}


# ------------------------------------------------------------------ #
# Startup
# ------------------------------------------------------------------ #

@app.on_event("startup")
async def startup():
    global _model, _config, _preprocessor, _model_info
    import yaml
    import joblib

    with open("configs/config.yaml") as f:
        _config = yaml.safe_load(f)

    # Load preprocessor
    prep_path = "checkpoints/preprocessor.joblib"
    if Path(prep_path).exists():
        saved = joblib.load(prep_path)
        _preprocessor = saved
        logger.info("Preprocessor loaded")

    # Load model
    ckpt_dir = Path(_config["training"]["checkpoint_dir"])
    ckpt_path = None
    for candidate in ["checkpoint_best.pt", "checkpoint_final.pt"]:
        p = ckpt_dir / candidate
        if p.exists():
            ckpt_path = p
            break
    if not ckpt_path:
        candidates = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
        if candidates:
            ckpt_path = candidates[-1]

    if ckpt_path:
        from training.model import build_model
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("config", _config)
        history = ckpt.get("history", {})
        # Infer dims from weights
        state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
        first_weight = next(iter(state.values()))
        # Get input_dim from first linear weight
        input_key = [k for k in state if "weight" in k][0]
        input_dim = state[input_key].shape[1]
        last_key = [k for k in state if "weight" in k][-1]
        n_classes = state[last_key].shape[0]

        _model = build_model(cfg, input_dim, n_classes)
        _model.load_state_dict(state)
        _model.eval()

        _model_info = {
            "checkpoint": str(ckpt_path),
            "epoch": ckpt.get("epoch", "unknown"),
            "input_dim": input_dim,
            "n_classes": n_classes,
            "best_val_acc": max(history.get("val_acc", [0])),
        }
        logger.info(f"Model loaded: {ckpt_path}")
    else:
        logger.warning("No checkpoint found — prediction endpoints unavailable")


# ------------------------------------------------------------------ #
# Schemas
# ------------------------------------------------------------------ #

class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Feature vector")

class BatchPredictRequest(BaseModel):
    samples: List[List[float]] = Field(..., description="List of feature vectors")

class PredictResponse(BaseModel):
    prediction: int
    probabilities: List[float]
    latency_ms: float

class BatchPredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]
    latency_ms: float


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "no_model",
        "model_loaded": _model is not None,
    }


@app.get("/model/info")
def model_info():
    if not _model_info:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model_info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    try:
        x = torch.FloatTensor(req.features).unsqueeze(0)
        with torch.no_grad():
            logits = _model(x)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
            pred = int(torch.argmax(logits).item())

        latency = (time.time() - t0) * 1000
        REQUEST_COUNT.labels(endpoint="predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="predict").observe(latency / 1000)

        return PredictResponse(prediction=pred, probabilities=probs, latency_ms=round(latency, 2))
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    try:
        x = torch.FloatTensor(req.samples)
        with torch.no_grad():
            logits = _model(x)
            probs = torch.softmax(logits, dim=1).tolist()
            preds = torch.argmax(logits, dim=1).tolist()

        latency = (time.time() - t0) * 1000
        REQUEST_COUNT.labels(endpoint="predict_batch", status="success").inc()

        return BatchPredictResponse(predictions=preds, probabilities=probs, latency_ms=round(latency, 2))
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="predict_batch", status="error").inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
