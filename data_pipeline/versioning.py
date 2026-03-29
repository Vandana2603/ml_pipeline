"""
data_pipeline/versioning.py
Dataset versioning: store processed datasets with hash-based IDs.
Allows reproducibility and dataset comparison across experiments.
"""

import hashlib
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DatasetVersioner:
    """
    Stores and retrieves versioned datasets.
    Version ID = MD5 hash of (config + data shape + first-row checksum).
    """

    def __init__(self, version_dir: str = "data/versioned"):
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.version_dir / "index.json"

    def save(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        metadata: Dict[str, Any],
        config: Dict[str, Any],
    ) -> str:
        """
        Save processed dataset splits and return a version ID.
        """
        version_id = self._compute_version_id(X_train, metadata, config)
        version_path = self.version_dir / version_id

        if version_path.exists():
            logger.info(f"Dataset version {version_id} already exists — skipping save")
            return version_id

        version_path.mkdir(parents=True)

        # Save arrays
        np.save(version_path / "X_train.npy", X_train)
        np.save(version_path / "X_val.npy", X_val)
        np.save(version_path / "X_test.npy", X_test)
        np.save(version_path / "y_train.npy", y_train)
        np.save(version_path / "y_val.npy", y_val)
        np.save(version_path / "y_test.npy", y_test)

        # Save manifest
        manifest = {
            "version_id": version_id,
            "created_at": datetime.now().isoformat(),
            "shapes": {
                "X_train": list(X_train.shape),
                "X_val": list(X_val.shape),
                "X_test": list(X_test.shape),
            },
            "metadata": metadata,
            "config_snapshot": {
                "source": config["data"]["source"],
                "dataset_name": config["data"].get("dataset_name"),
                "test_size": config["data"]["test_size"],
                "val_size": config["data"]["val_size"],
                "random_seed": config["data"]["random_seed"],
            },
        }

        with open(version_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        self._update_index(version_id, manifest)
        logger.info(f"Dataset version saved: {version_id}")
        return version_id

    def load(self, version_id: str) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        Dict
    ]:
        """Load a specific dataset version."""
        version_path = self.version_dir / version_id
        if not version_path.exists():
            raise FileNotFoundError(f"Dataset version not found: {version_id}")

        X_train = np.load(version_path / "X_train.npy")
        X_val   = np.load(version_path / "X_val.npy")
        X_test  = np.load(version_path / "X_test.npy")
        y_train = np.load(version_path / "y_train.npy")
        y_val   = np.load(version_path / "y_val.npy")
        y_test  = np.load(version_path / "y_test.npy")

        with open(version_path / "manifest.json") as f:
            manifest = json.load(f)

        logger.info(f"Loaded dataset version: {version_id}")
        return X_train, X_val, X_test, y_train, y_val, y_test, manifest

    def load_latest(self) -> Optional[Tuple]:
        """Load the most recently saved dataset version."""
        index = self._load_index()
        if not index:
            return None
        latest = sorted(index.values(), key=lambda x: x["created_at"])[-1]
        return self.load(latest["version_id"])

    def list_versions(self) -> list:
        index = self._load_index()
        return list(index.values())

    def _compute_version_id(self, X_train: np.ndarray, metadata: Dict, config: Dict) -> str:
        """Compute a stable hash-based version ID."""
        h = hashlib.md5()
        h.update(str(X_train.shape).encode())
        h.update(str(X_train[0] if len(X_train) > 0 else "").encode())
        h.update(json.dumps({
            "source": config["data"]["source"],
            "dataset_name": config["data"].get("dataset_name"),
            "random_seed": config["data"]["random_seed"],
            "n_samples": metadata.get("n_samples"),
        }, sort_keys=True).encode())
        return h.hexdigest()[:12]

    def _load_index(self) -> Dict:
        if not self.index_path.exists():
            return {}
        with open(self.index_path) as f:
            return json.load(f)

    def _update_index(self, version_id: str, manifest: Dict):
        index = self._load_index()
        index[version_id] = {
            "version_id": version_id,
            "created_at": manifest["created_at"],
            "dataset": manifest["config_snapshot"].get("dataset_name", "unknown"),
            "n_train": manifest["shapes"]["X_train"][0],
        }
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)
