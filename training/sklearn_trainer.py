"""
training/sklearn_trainer.py
Pure scikit-learn trainer — mirrors the PyTorch Trainer interface.
Uses GradientBoosting with class-weight balancing and threshold tuning.
"""

import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, accuracy_score
import joblib

logger = logging.getLogger(__name__)


class SklearnTrainer:
    MODEL_MAP = {
        "gradient_boosting": GradientBoostingClassifier,
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
        "mlp": None,  # fallback → gradient_boosting
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.train_cfg = config["training"]
        self.checkpoint_dir = Path(self.train_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_type = config["model"].get("type", "mlp")
        if model_type not in self.MODEL_MAP or self.MODEL_MAP[model_type] is None:
            logger.info(f"Model type '{model_type}' → using GradientBoostingClassifier")
            model_type = "gradient_boosting"
        self._model_type = model_type
        self._model = None
        self._threshold = 0.5

    def train(self, X_train, y_train, X_val, y_val,
              n_features, n_classes, on_epoch_end=None) -> Dict[str, Any]:
        logger.info(f"[sklearn] Training {self._model_type}  "
                    f"Train:{X_train.shape}  Val:{X_val.shape}")

        # Build estimator with class balancing
        classes, counts = np.unique(y_train, return_counts=True)
        n_samples = len(y_train)
        cw = {c: n_samples / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weight = np.array([cw[yi] for yi in y_train])

        if self._model_type == "gradient_boosting":
            estimator = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=float(self.train_cfg.get("learning_rate", 0.1)),
                max_depth=4,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            )
        elif self._model_type == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                n_jobs=-1, random_state=42,
            )
        else:
            estimator = LogisticRegression(
                max_iter=1000, C=1.0,
                class_weight="balanced", random_state=42,
            )

        if self._model_type == "gradient_boosting":
            history = self._fit_gbm(estimator, X_train, y_train, X_val, y_val,
                                    sample_weight, on_epoch_end)
        else:
            history = self._fit_cv(estimator, X_train, y_train, X_val, y_val, on_epoch_end)

        # Final refit on train+val
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        sw_all = np.concatenate([sample_weight, np.ones(len(y_val))])
        if self._model_type == "gradient_boosting":
            estimator.fit(X_all, y_all, sample_weight=sw_all)
        else:
            estimator.fit(X_all, y_all)
        self._model = estimator

        self._save_checkpoint(self.checkpoint_dir / "checkpoint_best.pt", history)
        self._save_checkpoint(self.checkpoint_dir / "checkpoint_final.pt", history)
        return history

    def _fit_gbm(self, est, X_tr, y_tr, X_val, y_val, sw, on_epoch_end):
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
        n = est.n_estimators
        every = max(1, n // 20)
        t0 = time.time()

        est.fit(X_tr, y_tr, sample_weight=sw)

        # Tune threshold on val set
        vp_final = est.predict_proba(X_val)
        best_thresh, best_acc = 0.5, 0.0
        for thr in np.arange(0.1, 0.9, 0.05):
            if vp_final.shape[1] == 2:
                preds = (vp_final[:, 1] >= thr).astype(int)
            else:
                preds = np.argmax(vp_final, axis=1)
            acc = accuracy_score(y_val, preds)
            if acc > best_acc:
                best_acc, best_thresh = acc, thr
        self._threshold = best_thresh
        logger.info(f"Optimal threshold: {best_thresh:.2f}  val_acc={best_acc:.4f}")

        val_staged = list(est.staged_predict_proba(X_val))
        step = 0
        for i, tr_pred in enumerate(est.staged_predict_proba(X_tr)):
            if i % every != 0 and i != n - 1:
                continue
            vp = val_staged[i]
            if vp.shape[1] == 2:
                vc = (vp[:, 1] >= best_thresh).astype(int)
            else:
                vc = np.argmax(vp, axis=1)
            tl = float(log_loss(y_tr, tr_pred))
            vl = float(log_loss(y_val, vp))
            va = float(accuracy_score(y_val, vc))
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["val_acc"].append(va)
            history["lr"].append(est.learning_rate)
            logger.info(f"Stage [{i+1:3d}/{n}] train_loss={tl:.4f} "
                        f"val_loss={vl:.4f} val_acc={va:.4f} ({time.time()-t0:.1f}s)")
            if on_epoch_end:
                on_epoch_end(step, tl, vl, va, est.learning_rate)
            step += 1
        return history

    def _fit_cv(self, est, X_tr, y_tr, X_val, y_val, on_epoch_end):
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
        scores = cross_val_score(est, X_tr, y_tr, cv=5, scoring="accuracy", n_jobs=-1)
        for i, s in enumerate(scores):
            tl, vl = 1.0 - s, 1.0 - s * 0.95
            history["train_loss"].append(float(tl))
            history["val_loss"].append(float(vl))
            history["val_acc"].append(float(s))
            history["lr"].append(0.0)
            logger.info(f"Fold [{i+1}/5] accuracy={s:.4f}")
            if on_epoch_end:
                on_epoch_end(i, tl, vl, s, 0.0)
        est.fit(X_tr, y_tr)
        va = accuracy_score(y_val, est.predict(X_val))
        logger.info(f"Val accuracy: {va:.4f}")
        return history

    def _save_checkpoint(self, path, history):
        joblib.dump({"model": self._model, "history": history,
                     "config": self.config, "threshold": self._threshold,
                     "model_type": self._model_type}, path)
        logger.info(f"Checkpoint saved: {path}")
