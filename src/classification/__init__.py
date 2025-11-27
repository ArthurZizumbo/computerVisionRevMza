"""
Classification module for ML-based discrepancy detection.

This module provides:
- Geometric metrics calculation
- DINOv2 embedding extraction
- Feature concatenation
- XGBoost/LightGBM ensemble classification
"""

from src.classification.dino_extractor import DINOv2Extractor
from src.classification.ensemble import DiscrepancyClassifier
from src.classification.feature_builder import FeatureBuilder
from src.classification.metrics import GeometricMetrics, compute_metrics

__all__ = [
    "GeometricMetrics",
    "compute_metrics",
    "DINOv2Extractor",
    "FeatureBuilder",
    "DiscrepancyClassifier",
]
