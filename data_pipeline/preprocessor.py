"""
data_pipeline/preprocessor.py
Preprocessing: cleaning, feature engineering, normalization, train/val/test splitting.
Returns ready-to-use tensors and fitted preprocessor for inference.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Modular preprocessing pipeline:
    1. Drop duplicates & handle missing values
    2. Feature engineering (polynomial, interaction terms)
    3. Normalize features
    4. Encode labels
    5. Split into train / val / test
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_size = config["data"]["test_size"]
        self.val_size = config["data"]["val_size"]
        self.random_seed = config["data"]["random_seed"]
        self.fitted = False

        # Build sklearn pipeline
        self._sklearn_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        self._label_encoder = LabelEncoder()

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Full preprocessing with fitting.
        Returns: X_train, X_val, X_test, y_train, y_val, y_test, split_info
        """
        logger.info("Starting preprocessing pipeline")

        # Step 1: Clean
        X, y = self._clean(X, y)

        # Step 2: Feature engineering
        X = self._feature_engineering(X)

        # Step 3: Split first (to avoid data leakage)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X.values, y.values,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=self._safe_stratify(y.values),
        )

        val_frac = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_frac,
            random_state=self.random_seed,
            stratify=self._safe_stratify(y_trainval),
        )

        # Step 4: Fit transform features
        X_train = self._sklearn_pipe.fit_transform(X_train)
        X_val = self._sklearn_pipe.transform(X_val)
        X_test = self._sklearn_pipe.transform(X_test)

        # Step 5: Encode labels
        y_train = self._label_encoder.fit_transform(y_train)
        y_val = self._label_encoder.transform(y_val)
        y_test = self._label_encoder.transform(y_test)

        self.n_features = X_train.shape[1]
        self.n_classes = len(self._label_encoder.classes_)
        self.fitted = True

        split_info = {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "classes": self._label_encoder.classes_.tolist(),
        }

        logger.info(
            f"Preprocessing complete | "
            f"train={split_info['n_train']}, "
            f"val={split_info['n_val']}, "
            f"test={split_info['n_test']} | "
            f"features={self.n_features}, classes={self.n_classes}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test, split_info

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted pipeline (for inference)."""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted yet. Call fit_transform first.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._sklearn_pipe.transform(X)

    def _clean(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove duplicates, handle obvious outliers."""
        original_len = len(X)
        combined = pd.concat([X, y.rename("__target__")], axis=1).drop_duplicates()
        X = combined.drop(columns=["__target__"])
        y = combined["__target__"]

        removed = original_len - len(X)
        if removed:
            logger.info(f"Removed {removed} duplicate rows")

        # Drop columns with >50% missing
        missing_ratio = X.isnull().mean()
        drop_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
        if drop_cols:
            logger.info(f"Dropping high-missing columns: {drop_cols}")
            X = X.drop(columns=drop_cols)

        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])

        return X.reset_index(drop=True), y.reset_index(drop=True)

    def _feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features:
        - Squared terms for top-variance features
        - Log transform for skewed features
        """
        engineered = X.copy()

        # Log1p for positive skewed columns
        skewed = X.apply(lambda col: col.skew()).abs()
        skewed_cols = skewed[skewed > 2].index.tolist()
        for col in skewed_cols[:5]:  # limit to top 5
            if (X[col] >= 0).all():
                engineered[f"{col}_log1p"] = np.log1p(X[col])

        # Squared terms for top-variance columns
        variance = X.var().sort_values(ascending=False)
        top_cols = variance.head(3).index.tolist()
        for col in top_cols:
            engineered[f"{col}_sq"] = X[col] ** 2

        added = engineered.shape[1] - X.shape[1]
        if added:
            logger.info(f"Feature engineering added {added} new features ({engineered.shape[1]} total)")

        return engineered

    def _safe_stratify(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Use stratify only if all classes have >=2 samples."""
        unique, counts = np.unique(y, return_counts=True)
        if np.all(counts >= 2):
            return y
        return None

    def save(self, path: str):
        """Persist fitted preprocessor to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self._sklearn_pipe, "label_encoder": self._label_encoder},
            path
        )
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str, config: Dict) -> "Preprocessor":
        """Load a previously fitted preprocessor."""
        obj = cls(config)
        saved = joblib.load(path)
        obj._sklearn_pipe = saved["pipeline"]
        obj._label_encoder = saved["label_encoder"]
        obj.n_features = obj._sklearn_pipe.named_steps["scaler"].n_features_in_
        obj.n_classes = len(obj._label_encoder.classes_)
        obj.fitted = True
        logger.info(f"Preprocessor loaded from {path}")
        return obj
