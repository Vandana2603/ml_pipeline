"""
data_pipeline/loader.py
Loads datasets from multiple sources: sklearn, CSV, JSON.
Returns raw DataFrames with a unified interface.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class DataLoader:
    """Unified data loader supporting sklearn, CSV, and JSON sources."""

    SKLEARN_DATASETS = {
        "breast_cancer": "load_breast_cancer",
        "iris": "load_iris",
        "wine": "load_wine",
        "diabetes": "load_diabetes",
        "digits": "load_digits",
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.source = config["data"]["source"]
        self.dataset_name = config["data"].get("dataset_name", "breast_cancer")
        self.csv_path = config["data"].get("csv_path")
        self.json_path = config["data"].get("json_path")

    def load(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Load dataset from configured source.
        Returns: (features_df, labels_series, metadata_dict)
        """
        logger.info(f"Loading dataset from source='{self.source}'")

        if self.source == "sklearn":
            return self._load_sklearn()
        elif self.source == "csv":
            return self._load_csv()
        elif self.source == "json":
            return self._load_json()
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

    def _load_sklearn(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        from sklearn import datasets as sk_datasets

        loader_name = self.SKLEARN_DATASETS.get(self.dataset_name)
        if not loader_name:
            raise ValueError(
                f"Unknown sklearn dataset '{self.dataset_name}'. "
                f"Available: {list(self.SKLEARN_DATASETS.keys())}"
            )

        loader_fn = getattr(sk_datasets, loader_name)
        raw = loader_fn()

        X = pd.DataFrame(raw.data, columns=raw.feature_names if hasattr(raw, "feature_names") else None)
        y = pd.Series(raw.target, name="target")

        metadata = {
            "source": "sklearn",
            "dataset_name": self.dataset_name,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "class_names": raw.target_names.tolist() if hasattr(raw, "target_names") else None,
            "feature_names": X.columns.tolist(),
        }

        logger.info(
            f"Loaded sklearn/{self.dataset_name}: "
            f"{metadata['n_samples']} samples, "
            f"{metadata['n_features']} features, "
            f"{metadata['n_classes']} classes"
        )
        return X, y, metadata

    def _load_csv(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        if not self.csv_path or not Path(self.csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        target_col = df.columns[-1]  # assume last column is target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        metadata = {
            "source": "csv",
            "path": self.csv_path,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": y.nunique(),
            "feature_names": X.columns.tolist(),
        }
        logger.info(f"Loaded CSV: {metadata['n_samples']} samples, {metadata['n_features']} features")
        return X, y, metadata

    def _load_json(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        if not self.json_path or not Path(self.json_path).exists():
            raise FileNotFoundError(f"JSON not found: {self.json_path}")

        with open(self.json_path) as f:
            records = json.load(f)

        df = pd.DataFrame(records)
        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        metadata = {
            "source": "json",
            "path": self.json_path,
            "n_samples": len(X),
            "n_features": X.shape[1],
        }
        logger.info(f"Loaded JSON: {metadata['n_samples']} samples")
        return X, y, metadata
