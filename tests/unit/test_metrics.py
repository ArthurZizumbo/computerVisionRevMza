"""
Unit tests for geometric metrics module.
"""

import numpy as np
import pytest

from src.classification.metrics import (
    GeometricMetrics,
    _binarize,
    _compute_dice,
    _compute_iou,
    compute_metrics,
)


class TestGeometricMetrics:
    """Tests for GeometricMetrics dataclass."""

    def test_to_array(self):
        """Test conversion to numpy array."""
        metrics = GeometricMetrics(
            iou=0.8,
            hausdorff_distance=0.1,
            dice_coefficient=0.85,
            area_ratio_error=0.05,
            centroid_distance=0.02,
            angular_difference=5.0,
            contour_correlation=0.01,
        )
        arr = metrics.to_array()
        assert arr.shape == (7,)
        assert arr[0] == 0.8
        assert arr[1] == 0.1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = GeometricMetrics(
            iou=0.8,
            hausdorff_distance=0.1,
            dice_coefficient=0.85,
            area_ratio_error=0.05,
            centroid_distance=0.02,
            angular_difference=5.0,
            contour_correlation=0.01,
        )
        d = metrics.to_dict()
        assert d["iou"] == 0.8
        assert "hausdorff_distance" in d


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_identical_masks(self):
        """Test metrics for identical masks."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255

        metrics = compute_metrics(mask, mask)

        assert metrics.iou == pytest.approx(1.0, abs=0.01)
        assert metrics.dice_coefficient == pytest.approx(1.0, abs=0.01)
        assert metrics.hausdorff_distance == pytest.approx(0.0, abs=0.01)

    def test_non_overlapping_masks(self):
        """Test metrics for non-overlapping masks."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[10:30, 10:30] = 255

        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[70:90, 70:90] = 255

        metrics = compute_metrics(mask1, mask2)

        assert metrics.iou == 0.0
        assert metrics.dice_coefficient == 0.0

    def test_partial_overlap(self):
        """Test metrics for partially overlapping masks."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[20:60, 20:60] = 255

        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[40:80, 40:80] = 255

        metrics = compute_metrics(mask1, mask2)

        assert 0 < metrics.iou < 1
        assert 0 < metrics.dice_coefficient < 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_binarize_grayscale(self):
        """Test binarization of grayscale image."""
        img = np.array([[100, 150], [200, 50]], dtype=np.uint8)
        binary = _binarize(img, threshold=127)

        assert binary[0, 0] == 0
        assert binary[0, 1] == 1
        assert binary[1, 0] == 1
        assert binary[1, 1] == 0

    def test_compute_iou_perfect(self):
        """Test IoU for identical masks."""
        mask = np.ones((10, 10), dtype=np.uint8)
        iou = _compute_iou(mask, mask)
        assert iou == 1.0

    def test_compute_iou_empty(self):
        """Test IoU for empty masks."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        iou = _compute_iou(mask, mask)
        assert iou == 0.0

    def test_compute_dice_perfect(self):
        """Test Dice for identical masks."""
        mask = np.ones((10, 10), dtype=np.uint8)
        dice = _compute_dice(mask, mask)
        assert dice == 1.0
