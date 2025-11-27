"""
Feature builder for combining geometric metrics and embeddings.

This module concatenates the 7 geometric metrics with DINOv2
embeddings to create the final feature vector for classification.
"""

import numpy as np
from loguru import logger

from src.classification.dino_extractor import DINOv2Extractor
from src.classification.metrics import compute_metrics


class FeatureBuilder:
    """
    Builds combined feature vectors from images and metrics.

    The final feature vector consists of:
    - 7 geometric metrics
    - 384 DINOv2 embeddings (satellite)
    - 384 DINOv2 embeddings (vector) [optional]
    Total: 391 or 775 dimensions

    Attributes
    ----------
    include_vector_embedding : bool
        Whether to include vector image embedding
    dino_extractor : DINOv2Extractor
        Embedding extractor instance
    """

    def __init__(
        self,
        include_vector_embedding: bool = False,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """
        Initialize feature builder.

        Parameters
        ----------
        include_vector_embedding : bool, optional
            Include vector embedding, defaults to False
        device : str, optional
            Device for DINOv2, defaults to auto-detect
        cache_dir : str, optional
            Cache directory for model weights
        """
        self.include_vector_embedding = include_vector_embedding
        self._dino_extractor: DINOv2Extractor | None = None
        self._device = device
        self._cache_dir = cache_dir

        n_features = 7 + 384
        if include_vector_embedding:
            n_features += 384
        self.n_features = n_features

        logger.info(f"FeatureBuilder initialized with {n_features} features")

    @property
    def dino_extractor(self) -> DINOv2Extractor:
        """Lazy-load DINOv2 extractor."""
        if self._dino_extractor is None:
            self._dino_extractor = DINOv2Extractor(device=self._device, cache_dir=self._cache_dir)
        return self._dino_extractor

    def build(
        self,
        satellite: np.ndarray,
        aligned_vector: np.ndarray,
        satellite_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build feature vector for a single sample.

        Parameters
        ----------
        satellite : np.ndarray
            Satellite image (RGB)
        aligned_vector : np.ndarray
            Aligned vector image
        satellite_mask : np.ndarray, optional
            Satellite-derived mask for metrics

        Returns
        -------
        np.ndarray
            Feature vector of shape (n_features,)
        """
        if satellite_mask is None:
            satellite_mask = aligned_vector

        metrics = compute_metrics(aligned_vector, satellite_mask)
        metrics_array = metrics.to_array()

        satellite_embedding = self.dino_extractor.extract(satellite)

        features = [metrics_array, satellite_embedding]

        if self.include_vector_embedding:
            vector_embedding = self.dino_extractor.extract(aligned_vector)
            features.append(vector_embedding)

        return np.concatenate(features)

    def build_batch(
        self,
        satellites: list[np.ndarray],
        aligned_vectors: list[np.ndarray],
        satellite_masks: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Build feature vectors for a batch of samples.

        Parameters
        ----------
        satellites : list[np.ndarray]
            List of satellite images
        aligned_vectors : list[np.ndarray]
            List of aligned vector images
        satellite_masks : list[np.ndarray], optional
            List of satellite-derived masks

        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_samples, n_features)
        """
        if satellite_masks is None:
            satellite_masks = aligned_vectors

        n_samples = len(satellites)
        features = np.zeros((n_samples, self.n_features))

        for i in range(n_samples):
            features[i] = self.build(satellites[i], aligned_vectors[i], satellite_masks[i])

        logger.debug(f"Built features for {n_samples} samples")
        return features

    def get_feature_names(self) -> list[str]:
        """Get names of all features."""
        names = [
            "iou",
            "hausdorff_distance",
            "dice_coefficient",
            "area_ratio_error",
            "centroid_distance",
            "angular_difference",
            "contour_correlation",
        ]

        names.extend([f"sat_emb_{i}" for i in range(384)])

        if self.include_vector_embedding:
            names.extend([f"vec_emb_{i}" for i in range(384)])

        return names
