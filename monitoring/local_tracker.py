"""
monitoring/local_tracker.py
File-based experiment tracker — used when MLflow is not reachable.
Writes JSON experiment logs to ./experiments/ for offline comparison.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class LocalTracker:
    """
    Stores experiments as JSON files in ./experiments/<run_id>/.
    Drop-in replacement for ExperimentTracker when MLflow is offline.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exp_dir = Path("experiments")
        self.exp_dir.mkdir(exist_ok=True)
        self._run_id: Optional[str] = None
        self._run_dir: Optional[Path] = None
        self._params: Dict = {}
        self._metrics: Dict[str, list] = {}
        self._tags: Dict = {}
        self._t0: float = 0.0

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None) -> str:
        self._run_id = str(uuid.uuid4())[:8]
        self._run_dir = self.exp_dir / self._run_id
        self._run_dir.mkdir(parents=True)
        self._tags = tags or {}
        self._t0 = time.time()
        logger.info(f"[LocalTracker] Run started: {self._run_id} ({run_name or ''})")
        return self._run_id

    def log_params(self, params: Dict[str, Any]):
        self._params.update(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append({"step": step, "value": value})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            if v is not None:
                self.log_metric(k, float(v), step=step)

    def log_artifact(self, local_path: str):
        pass  # no-op for local tracker

    def log_model(self, model_path: str, artifact_name: str = "model"):
        pass

    def set_tag(self, key: str, value: str):
        self._tags[key] = value

    def end_run(self, status: str = "FINISHED"):
        if not self._run_dir:
            return
        summary = {
            "run_id": self._run_id,
            "status": status,
            "duration_s": round(time.time() - self._t0, 2),
            "timestamp": datetime.now().isoformat(),
            "params": self._params,
            "tags": self._tags,
            "metrics_final": {
                k: v[-1]["value"] for k, v in self._metrics.items() if v
            },
        }
        with open(self._run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(self._run_dir / "metrics_history.json", "w") as f:
            json.dump(self._metrics, f, indent=2)
        logger.info(f"[LocalTracker] Run {self._run_id} saved → {self._run_dir}/summary.json")

    def epoch_callback(self) -> Callable:
        def _cb(epoch, train_loss, val_loss, val_acc, lr):
            step = epoch + 1
            self.log_metrics({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "train/lr": lr,
            }, step=step)
        return _cb

    def log_config_as_params(self):
        self.log_params({
            "model.type": self.config["model"]["type"],
            "model.hidden_dims": str(self.config["model"]["hidden_dims"]),
            "training.epochs": self.config["training"]["epochs"],
            "training.batch_size": self.config["training"]["batch_size"],
            "training.learning_rate": self.config["training"]["learning_rate"],
            "data.source": self.config["data"]["source"],
            "data.dataset_name": self.config["data"].get("dataset_name", ""),
        })
