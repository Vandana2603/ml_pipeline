"""
orchestration/pipeline.py
End-to-end pipeline controller with retry logic and stage isolation.
Auto-detects PyTorch availability; falls back to sklearn gracefully.
"""

import time
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _retry(fn, name: str, retries: int = 3, delay: float = 2.0):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            logger.error(f"[{name}] Attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(delay)


class Pipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self._setup_logging()
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.n_features = self.n_classes = None
        self.dataset_version = None
        self.run_id = None
        self.history = None
        self._tracker = None
        logger.info(f"Backend: {'PyTorch DDP' if TORCH_AVAILABLE else 'scikit-learn'}")
        logger.info(f"Tracking: {'MLflow' if MLFLOW_AVAILABLE else 'LocalTracker'}")

    def run_data_stage(self) -> str:
        logger.info("=" * 60)
        logger.info("STAGE 1: DATA PIPELINE")
        logger.info("=" * 60)

        from data_pipeline.loader import DataLoader
        from data_pipeline.preprocessor import Preprocessor
        from data_pipeline.versioning import DatasetVersioner

        def _run():
            loader = DataLoader(self.config)
            X_raw, y_raw, metadata = loader.load()
            preprocessor = Preprocessor(self.config)
            (self.X_train, self.X_val, self.X_test,
             self.y_train, self.y_val, self.y_test,
             split_info) = preprocessor.fit_transform(X_raw, y_raw)
            self.n_features = split_info["n_features"]
            self.n_classes  = split_info["n_classes"]
            Path("checkpoints").mkdir(exist_ok=True)
            preprocessor.save("checkpoints/preprocessor.joblib")
            versioner = DatasetVersioner(self.config["data"]["version_dir"])
            version_id = versioner.save(
                self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test,
                {**metadata, **split_info}, self.config,
            )
            return version_id

        self.dataset_version = _retry(_run, "DATA")
        logger.info(f"Data stage complete. Version: {self.dataset_version}")
        return self.dataset_version

    def run_train_stage(self) -> str:
        logger.info("=" * 60)
        logger.info("STAGE 2: TRAINING PIPELINE")
        logger.info("=" * 60)
        if self.X_train is None:
            raise RuntimeError("Run data stage first.")

        if MLFLOW_AVAILABLE:
            from monitoring.tracker import ExperimentTracker
            tracker = ExperimentTracker(self.config)
        else:
            from monitoring.local_tracker import LocalTracker
            tracker = LocalTracker(self.config)

        self.run_id = tracker.start_run(
            run_name=f"run-{self.dataset_version or 'local'}",
            tags={"dataset_version": self.dataset_version or "unknown",
                  "backend": "pytorch" if TORCH_AVAILABLE else "sklearn"},
        )
        tracker.log_config_as_params()
        self._tracker = tracker

        def _run():
            if TORCH_AVAILABLE:
                from training.trainer import Trainer
                trainer = Trainer(self.config)
            else:
                from training.sklearn_trainer import SklearnTrainer
                trainer = SklearnTrainer(self.config)
            return trainer.train(
                self.X_train, self.y_train,
                self.X_val,   self.y_val,
                n_features=self.n_features,
                n_classes=self.n_classes,
                on_epoch_end=tracker.epoch_callback(),
            )

        self.history = _retry(_run, "TRAIN")

        if self.history:
            tracker.log_metrics({
                "final/train_loss":        self.history["train_loss"][-1],
                "final/val_loss":          self.history["val_loss"][-1],
                "final/val_accuracy":      self.history["val_acc"][-1],
                "final/best_val_accuracy": max(self.history["val_acc"]),
            })
        tracker.set_tag("status", "trained")
        logger.info(f"Training complete. Run ID: {self.run_id}")
        return self.run_id

    def run_evaluate_stage(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("STAGE 3: EVALUATION PIPELINE")
        logger.info("=" * 60)
        if self.X_test is None:
            raise RuntimeError("Run data stage first.")

        def _run():
            if TORCH_AVAILABLE:
                from evaluation.evaluator import Evaluator
                evaluator = Evaluator(self.config)
            else:
                from evaluation.sklearn_evaluator import SklearnEvaluator
                evaluator = SklearnEvaluator(self.config)
            return evaluator.evaluate(
                self.X_test, self.y_test,
                n_features=self.n_features,
                n_classes=self.n_classes,
                run_id=self.run_id,
            )

        result = _retry(_run, "EVALUATE")
        metrics = result["metrics"]

        if self._tracker:
            self._tracker.log_metrics({f"test/{k}": v for k, v in metrics.items() if v is not None})
            self._tracker.set_tag("status", "evaluated")
            self._tracker.end_run()

        logger.info(f"Evaluation | accuracy={metrics['accuracy']:.4f} | f1={metrics['f1_weighted']:.4f}")
        return result

    def run_all(self) -> Dict[str, Any]:
        t0 = time.time()
        logger.info("Starting full ML pipeline")
        self.run_data_stage()
        self.run_train_stage()
        result = self.run_evaluate_stage()
        logger.info(f"Full pipeline complete in {time.time()-t0:.1f}s")
        return result

    def _setup_logging(self):
        level = self.config.get("monitoring", {}).get("log_level", "INFO")
        logging.basicConfig(
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            level=getattr(logging, level, logging.INFO),
            force=True,
        )
