"""
monitoring/tracker.py
MLflow-based experiment tracking.
Logs: hyperparameters, per-epoch metrics, artifacts, model versions.
Falls back gracefully when MLflow server is unavailable.
"""

import logging
import platform
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Wraps MLflow with a clean interface.
    All public methods are safe to call even if MLflow is unreachable.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        track_cfg = config.get("tracking", {})
        self.mlflow_uri = track_cfg.get("mlflow_uri", "http://localhost:5000")
        self.experiment_name = track_cfg.get("experiment_name", "ml-pipeline")
        self.tags = track_cfg.get("tags", {})
        self._run = None
        self._mlflow = None
        self._active = False

        self._try_init_mlflow()

    def _try_init_mlflow(self):
        try:
            import mlflow
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            self._active = True
            logger.info(f"MLflow tracking: {self.mlflow_uri} | experiment: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow unavailable — tracking disabled. ({e})")
            self._active = False

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None) -> str:
        if not self._active:
            return "no-tracking"

        try:
            all_tags = {
                "python_version": sys.version.split()[0],
                "platform": platform.system(),
                **self.tags,
                **(tags or {}),
            }
            self._run = self._mlflow.start_run(run_name=run_name, tags=all_tags)
            run_id = self._run.info.run_id
            logger.info(f"MLflow run started: {run_id}")
            return run_id
        except Exception as e:
            logger.warning(f"MLflow start_run failed: {e}")
            return "no-tracking"

    def log_params(self, params: Dict[str, Any]):
        if not self._active or not self._run:
            return
        try:
            flat = self._flatten(params)
            self._mlflow.log_params(flat)
        except Exception as e:
            logger.warning(f"MLflow log_params failed: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if not self._active or not self._run:
            return
        try:
            self._mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.warning(f"MLflow log_metric failed: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            if v is not None:
                self.log_metric(k, float(v), step=step)

    def log_artifact(self, local_path: str):
        if not self._active or not self._run:
            return
        try:
            self._mlflow.log_artifact(local_path)
        except Exception as e:
            logger.warning(f"MLflow log_artifact failed: {e}")

    def log_model(self, model_path: str, artifact_name: str = "model"):
        """Log a PyTorch model checkpoint as an artifact."""
        self.log_artifact(model_path)

    def set_tag(self, key: str, value: str):
        if not self._active or not self._run:
            return
        try:
            self._mlflow.set_tag(key, value)
        except Exception as e:
            logger.warning(f"MLflow set_tag failed: {e}")

    def end_run(self, status: str = "FINISHED"):
        if not self._active or not self._run:
            return
        try:
            self._mlflow.end_run(status=status)
            logger.info(f"MLflow run ended: {status}")
        except Exception as e:
            logger.warning(f"MLflow end_run failed: {e}")
        finally:
            self._run = None

    def epoch_callback(self) -> Callable:
        """
        Returns a callable (epoch, train_loss, val_loss, val_acc, lr) → None
        for use as an on_epoch_end hook during training.
        """
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
        """Flatten and log the entire config as MLflow parameters."""
        params = {
            "model.type": self.config["model"]["type"],
            "model.hidden_dims": str(self.config["model"]["hidden_dims"]),
            "model.dropout": self.config["model"]["dropout"],
            "training.epochs": self.config["training"]["epochs"],
            "training.batch_size": self.config["training"]["batch_size"],
            "training.learning_rate": self.config["training"]["learning_rate"],
            "training.distributed": self.config["training"]["distributed"],
            "training.num_processes": self.config["training"].get("num_processes", 1),
            "data.source": self.config["data"]["source"],
            "data.dataset_name": self.config["data"].get("dataset_name", ""),
        }
        self.log_params(params)

    @staticmethod
    def _flatten(d: Dict, prefix: str = "") -> Dict[str, str]:
        result = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(ExperimentTracker._flatten(v, key))
            else:
                result[key] = str(v)
        return result
