"""
Geometric metrics for discrepancy classification.

This module implements the 7 geometric metrics used for
comparing aligned vectors with satellite imagery.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from scipy.spatial.distance import directed_hausdorff


@dataclass
class GeometricMetrics:
    """
    Seven geometric metrics for classification.

    Attributes
    ----------
    iou : float
        Intersection over Union [0, 1]
    hausdorff_distance : float
        Normalized Hausdorff distance [0, inf)
    dice_coefficient : float
        Dice/F1 coefficient [0, 1]
    area_ratio_error : float
        Relative area difference [0, inf)
    centroid_distance : float
        Normalized centroid distance [0, inf)
    angular_difference : float
        Rotation angle difference [0, 180] degrees
    contour_correlation : float
        Contour shape matching score [0, inf)
    """

    iou: float
    hausdorff_distance: float
    dice_coefficient: float
    area_ratio_error: float
    centroid_distance: float
    angular_difference: float
    contour_correlation: float

    def to_array(self) -> np.ndarray:
        """Convert metrics to numpy array for ML input."""
        return np.array(
            [
                self.iou,
                self.hausdorff_distance,
                self.dice_coefficient,
                self.area_ratio_error,
                self.centroid_distance,
                self.angular_difference,
                self.contour_correlation,
            ]
        )

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "iou": self.iou,
            "hausdorff_distance": self.hausdorff_distance,
            "dice_coefficient": self.dice_coefficient,
            "area_ratio_error": self.area_ratio_error,
            "centroid_distance": self.centroid_distance,
            "angular_difference": self.angular_difference,
            "contour_correlation": self.contour_correlation,
        }


def compute_metrics(
    aligned_vector: np.ndarray,
    satellite_mask: np.ndarray,
) -> GeometricMetrics:
    """
    Compute the 7 geometric metrics between aligned vector and satellite mask.

    Parameters
    ----------
    aligned_vector : np.ndarray
        Aligned vector image (grayscale or binary)
    satellite_mask : np.ndarray
        Satellite-derived mask (grayscale or binary)

    Returns
    -------
    GeometricMetrics
        Computed metrics dataclass
    """
    vec_binary = _binarize(aligned_vector)
    sat_binary = _binarize(satellite_mask)

    iou = _compute_iou(vec_binary, sat_binary)
    hausdorff = _compute_hausdorff(vec_binary, sat_binary)
    dice = _compute_dice(vec_binary, sat_binary)
    area_error = _compute_area_ratio_error(vec_binary, sat_binary)
    centroid_dist = _compute_centroid_distance(vec_binary, sat_binary)
    angular_diff = _compute_angular_difference(vec_binary, sat_binary)
    contour_corr = _compute_contour_correlation(vec_binary, sat_binary)

    logger.debug(f"Computed metrics: IoU={iou:.4f}, Dice={dice:.4f}")

    return GeometricMetrics(
        iou=iou,
        hausdorff_distance=hausdorff,
        dice_coefficient=dice,
        area_ratio_error=area_error,
        centroid_distance=centroid_dist,
        angular_difference=angular_diff,
        contour_correlation=contour_corr,
    )


def _binarize(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Convert image to binary mask."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (image > threshold).astype(np.uint8)


def _compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def _compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice coefficient."""
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    return float(2 * intersection / total) if total > 0 else 0.0


def _compute_hausdorff(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute normalized Hausdorff distance."""
    coords1 = np.argwhere(mask1)
    coords2 = np.argwhere(mask2)

    if len(coords1) == 0 or len(coords2) == 0:
        return 1.0

    d1 = directed_hausdorff(coords1, coords2)[0]
    d2 = directed_hausdorff(coords2, coords1)[0]
    max_dist = max(mask1.shape)

    return float(max(d1, d2) / max_dist)


def _compute_area_ratio_error(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute relative area difference."""
    area1 = mask1.sum()
    area2 = mask2.sum()
    max_area = max(area1, area2, 1)
    return float(abs(area1 - area2) / max_area)


def _compute_centroid_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute normalized centroid distance."""
    coords1 = np.argwhere(mask1)
    coords2 = np.argwhere(mask2)

    if len(coords1) == 0 or len(coords2) == 0:
        return 1.0

    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    distance = np.linalg.norm(centroid1 - centroid2)
    max_dist = max(mask1.shape)

    return float(distance / max_dist)


def _compute_angular_difference(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute orientation angle difference using image moments."""
    moments1 = cv2.moments(mask1)
    moments2 = cv2.moments(mask2)

    if moments1["mu20"] == moments1["mu02"] or moments2["mu20"] == moments2["mu02"]:
        return 0.0

    angle1 = 0.5 * np.arctan2(2 * moments1["mu11"], moments1["mu20"] - moments1["mu02"])
    angle2 = 0.5 * np.arctan2(2 * moments2["mu11"], moments2["mu20"] - moments2["mu02"])

    diff = abs(np.degrees(angle1 - angle2))
    return float(min(diff, 180 - diff))


def _compute_contour_correlation(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute contour shape matching score."""
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours1 or not contours2:
        return 1.0

    largest1 = max(contours1, key=cv2.contourArea)
    largest2 = max(contours2, key=cv2.contourArea)

    return float(cv2.matchShapes(largest1, largest2, cv2.CONTOURS_MATCH_I2, 0))
