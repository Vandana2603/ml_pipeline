"""
evaluation/sklearn_evaluator.py
Evaluation pipeline for sklearn-trained models (joblib checkpoints).
Uses the optimal threshold saved during training.
"""

import json
import logging
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
)

logger = logging.getLogger(__name__)


class SklearnEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config["evaluation"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, X_test, y_test, n_features, n_classes,
                 checkpoint_path=None, run_id=None) -> Dict[str, Any]:
        if not checkpoint_path:
            checkpoint_path = self._find_best_checkpoint()

        logger.info(f"Evaluating sklearn model: {checkpoint_path}")
        data = joblib.load(checkpoint_path)
        model = data["model"]
        threshold = data.get("threshold", 0.5)

        probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Use saved threshold for binary classification
        if probs is not None and probs.shape[1] == 2:
            y_pred = (probs[:, 1] >= threshold).astype(int)
            logger.info(f"Using threshold={threshold:.2f}")
        elif probs is not None:
            y_pred = np.argmax(probs, axis=1)
        else:
            y_pred = model.predict(X_test)

        metrics = self._compute_metrics(y_test, y_pred, probs, n_classes)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        result = {
            "run_id": run_id,
            "checkpoint": str(checkpoint_path),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "classification_report": report,
            "confusion_matrix": cm,
            "n_test_samples": len(y_test),
        }

        result_path = self.results_dir / f"eval_{run_id or 'latest'}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(
            f"Evaluation | accuracy={metrics['accuracy']:.4f} | "
            f"f1={metrics['f1_weighted']:.4f} | "
            f"roc_auc={metrics.get('roc_auc', 'N/A')}"
        )
        return result

    def _compute_metrics(self, y_true, y_pred, probs, n_classes) -> Dict[str, Any]:
        metrics = {
            "accuracy":           float(accuracy_score(y_true, y_pred)),
            "f1_weighted":        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro":           float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted":    float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        if probs is not None:
            try:
                if n_classes == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, probs, multi_class="ovr", average="weighted"))
            except Exception:
                metrics["roc_auc"] = None
        return metrics

    def _find_best_checkpoint(self) -> str:
        ckpt_dir = Path(self.config["training"]["checkpoint_dir"])
        for name in ["checkpoint_best.pt", "checkpoint_final.pt"]:
            p = ckpt_dir / name
            if p.exists():
                return str(p)
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
