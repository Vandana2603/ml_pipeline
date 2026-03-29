"""
evaluation/evaluator.py
Automated post-training evaluation with full metric suite.
Stores results as JSON linked to the experiment run.
"""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Loads the best checkpoint and evaluates on the held-out test set.
    Computes: accuracy, F1, precision, recall, ROC-AUC.
    Writes results to disk and returns dict for MLflow logging.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config["evaluation"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_features: int,
        n_classes: int,
        checkpoint_path: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation. Returns metrics dict.
        """
        from training.model import build_model

        # Load model
        if not checkpoint_path:
            checkpoint_path = self._find_best_checkpoint()

        logger.info(f"Evaluating from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        model = build_model(self.config, n_features, n_classes)
        state = ckpt["model_state_dict"]
        # Strip DDP 'module.' prefix if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()

        # Inference
        X_tensor = torch.FloatTensor(X_test)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
            y_pred = logits.argmax(dim=1).numpy()

        # Metrics
        metrics = self._compute_metrics(y_test, y_pred, probs, n_classes)

        # Full classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        result = {
            "run_id": run_id,
            "checkpoint": checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "classification_report": report,
            "confusion_matrix": cm,
            "n_test_samples": len(y_test),
        }

        # Save to disk
        result_path = self.results_dir / f"eval_{run_id or 'latest'}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(
            f"Evaluation complete | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"f1={metrics['f1_weighted']:.4f} | "
            f"roc_auc={metrics.get('roc_auc', 'N/A')}"
        )

        return result

    def _compute_metrics(
        self, y_true, y_pred, probs, n_classes
    ) -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        # ROC-AUC (binary or multiclass OvR)
        try:
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")
                )
        except Exception:
            metrics["roc_auc"] = None

        return metrics

    def _find_best_checkpoint(self) -> str:
        ckpt_dir = Path(self.config["training"]["checkpoint_dir"])
        best = ckpt_dir / "checkpoint_best.pt"
        if best.exists():
            return str(best)

        final = ckpt_dir / "checkpoint_final.pt"
        if final.exists():
            return str(final)

        candidates = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
        if candidates:
            return str(candidates[-1])

        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
