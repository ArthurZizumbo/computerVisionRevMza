"""
SAM (Segment Anything Model) validator for semantic segmentation.

This module uses SAM for validating alignment results and
detecting semantic changes in satellite imagery.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

try:
    from ultralytics import SAM

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning("Ultralytics SAM not available, SAMValidator will be disabled")


@dataclass
class SAMResult:
    """Result of SAM segmentation."""

    masks: list[np.ndarray]
    scores: list[float]
    num_segments: int
    coverage_ratio: float


class SAMValidator:
    """
    Validates alignment using SAM semantic segmentation.

    Attributes
    ----------
    model_name : str
        SAM model variant to use
    device : str
        Device for inference
    """

    def __init__(
        self,
        model_name: str = "sam_b.pt",
        device: str | None = None,
    ) -> None:
        """
        Initialize SAM validator.

        Parameters
        ----------
        model_name : str, optional
            SAM model name, defaults to "sam_b.pt"
        device : str, optional
            Device for inference, defaults to auto-detect
        """
        if not SAM_AVAILABLE:
            raise RuntimeError("Ultralytics SAM is required for SAMValidator")

        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = SAM(model_name)
        logger.info(f"SAMValidator initialized with {model_name} on {device}")

    def segment(
        self,
        image: np.ndarray,
        points: list[tuple[int, int]] | None = None,
        boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> SAMResult:
        """
        Segment image using SAM.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR format)
        points : list[tuple[int, int]], optional
            Point prompts for segmentation
        boxes : list[tuple[int, int, int, int]], optional
            Box prompts for segmentation

        Returns
        -------
        SAMResult
            Segmentation result with masks and scores
        """
        results = self.model(image, points=points, boxes=boxes, device=self.device)

        masks = []
        scores = []

        for result in results:
            if result.masks is not None:
                for mask, score in zip(result.masks.data, result.boxes.conf, strict=False):
                    masks.append(mask.cpu().numpy())
                    scores.append(float(score))

        total_coverage = sum(m.sum() for m in masks) / (image.shape[0] * image.shape[1])

        logger.debug(f"SAM produced {len(masks)} segments with {total_coverage:.2%} coverage")
        return SAMResult(
            masks=masks,
            scores=scores,
            num_segments=len(masks),
            coverage_ratio=total_coverage,
        )

    def validate_alignment(
        self,
        satellite: np.ndarray,
        vector_mask: np.ndarray,
        iou_threshold: float = 0.5,
    ) -> tuple[bool, float]:
        """
        Validate alignment by comparing SAM segments with vector mask.

        Parameters
        ----------
        satellite : np.ndarray
            Satellite image
        vector_mask : np.ndarray
            Rasterized vector mask
        iou_threshold : float, optional
            IoU threshold for validation, defaults to 0.5

        Returns
        -------
        tuple[bool, float]
            Validation result (passed, IoU score)
        """
        sam_result = self.segment(satellite)

        if not sam_result.masks:
            return False, 0.0

        combined_mask = np.zeros(satellite.shape[:2], dtype=np.uint8)
        for mask in sam_result.masks:
            resized_mask = cv2.resize(
                mask.astype(np.uint8), (satellite.shape[1], satellite.shape[0])
            )
            combined_mask = np.maximum(combined_mask, resized_mask)

        vector_binary = (vector_mask > 127).astype(np.uint8)
        if len(vector_binary.shape) == 3:
            vector_binary = vector_binary[:, :, 0]

        intersection = np.logical_and(combined_mask, vector_binary).sum()
        union = np.logical_or(combined_mask, vector_binary).sum()
        iou = intersection / union if union > 0 else 0.0

        passed = iou >= iou_threshold
        logger.debug(f"SAM validation: IoU={iou:.4f}, passed={passed}")

        return passed, iou
