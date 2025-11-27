"""
Ensemble classifier for discrepancy detection.

This module implements XGBoost + LightGBM ensemble classification
with support for class imbalance handling.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.model_selection import cross_val_score

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("XGBoost/LightGBM not available")


class DiscrepancyType(str, Enum):
    """Types of discrepancies."""

    ALIGNED = "aligned"
    GEOMETRIC_MISALIGNMENT = "geometric_misalignment"
    SEMANTIC_CHANGE = "semantic_change"
    BOUNDARY_CHANGE = "boundary_change"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class PredictionResult:
    """Result of discrepancy prediction."""

    label: DiscrepancyType
    confidence: float
    probabilities: dict[str, float]


class DiscrepancyClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier combining XGBoost and LightGBM.

    Attributes
    ----------
    n_estimators : int
        Number of boosting rounds per model
    learning_rate : float
        Learning rate for boosting
    max_depth : int
        Maximum tree depth
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        class_weight: dict | None = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize ensemble classifier.

        Parameters
        ----------
        n_estimators : int, optional
            Number of estimators, defaults to 100
        learning_rate : float, optional
            Learning rate, defaults to 0.1
        max_depth : int, optional
            Max tree depth, defaults to 6
        class_weight : dict, optional
            Class weights for imbalance handling
        random_state : int, optional
            Random seed, defaults to 42
        """
        if not ML_AVAILABLE:
            raise RuntimeError("XGBoost and LightGBM are required")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state

        self._xgb_model: xgb.XGBClassifier | None = None
        self._lgb_model: lgb.LGBMClassifier | None = None
        self._classes: np.ndarray | None = None

        logger.info(f"DiscrepancyClassifier initialized with n_estimators={n_estimators}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "DiscrepancyClassifier":
        """
        Fit the ensemble classifier.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target labels

        Returns
        -------
        DiscrepancyClassifier
            Fitted classifier
        """
        self._classes = np.unique(y)

        self._xgb_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        self._xgb_model.fit(X, y)

        self._lgb_model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbose=-1,
        )
        self._lgb_model.fit(X, y)

        logger.info(f"Fitted ensemble on {len(y)} samples with {len(self._classes)} classes")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Predicted labels
        """
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_classes)
        """
        if self._xgb_model is None or self._lgb_model is None:
            raise RuntimeError("Classifier must be fitted before prediction")

        xgb_proba = self._xgb_model.predict_proba(X)
        lgb_proba = self._lgb_model.predict_proba(X)

        return (xgb_proba + lgb_proba) / 2

    def predict_single(
        self,
        features: np.ndarray,
    ) -> PredictionResult:
        """
        Predict for a single sample with detailed result.

        Parameters
        ----------
        features : np.ndarray
            Feature vector of shape (n_features,)

        Returns
        -------
        PredictionResult
            Detailed prediction result
        """
        X = features.reshape(1, -1)
        proba = self.predict_proba(X)[0]
        label_idx = np.argmax(proba)

        return PredictionResult(
            label=DiscrepancyType(self._classes[label_idx]),
            confidence=float(proba[label_idx]),
            probabilities={str(c): float(p) for c, p in zip(self._classes, proba, strict=False)},
        )

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> dict[str, float]:
        """
        Perform cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        cv : int, optional
            Number of folds, defaults to 5

        Returns
        -------
        dict[str, float]
            Cross-validation scores
        """
        scores = cross_val_score(self, X, y, cv=cv, scoring="f1_weighted")
        return {
            "mean_f1": float(np.mean(scores)),
            "std_f1": float(np.std(scores)),
            "scores": scores.tolist(),
        }

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "xgb": self._xgb_model,
                "lgb": self._lgb_model,
                "classes": self._classes,
                "params": {
                    "n_estimators": self.n_estimators,
                    "learning_rate": self.learning_rate,
                    "max_depth": self.max_depth,
                },
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> "DiscrepancyClassifier":
        """Load model from disk."""
        import joblib

        data = joblib.load(path)
        self._xgb_model = data["xgb"]
        self._lgb_model = data["lgb"]
        self._classes = data["classes"]
        logger.info(f"Model loaded from {path}")
        return self
