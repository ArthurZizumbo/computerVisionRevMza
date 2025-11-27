"""
Unit tests for ECC aligner module.
"""

import numpy as np
import pytest

from src.alignment.ecc_aligner import ECCAligner, ECCResult


class TestECCAligner:
    """Tests for ECCAligner class."""

    def test_initialization_default(self):
        """Test default initialization."""
        aligner = ECCAligner()
        assert aligner.num_iterations == 5000
        assert aligner.termination_eps == 1e-10

    def test_initialization_custom(self):
        """Test custom initialization."""
        aligner = ECCAligner(
            motion_type="affine",
            num_iterations=1000,
            termination_eps=1e-5,
        )
        assert aligner.num_iterations == 1000
        assert aligner.termination_eps == 1e-5

    def test_invalid_motion_type(self):
        """Test invalid motion type raises error."""
        with pytest.raises(ValueError):
            ECCAligner(motion_type="invalid")

    def test_align_identical_images(self):
        """Test alignment of identical images."""
        aligner = ECCAligner()

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = aligner.align(img, img)

        assert isinstance(result, ECCResult)
        assert result.converged or result.correlation_coefficient > 0

    def test_to_grayscale_bgr(self):
        """Test grayscale conversion for BGR image."""
        aligner = ECCAligner()
        bgr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        gray = aligner._to_grayscale(bgr)
        assert gray.shape == (50, 50)

    def test_to_grayscale_already_gray(self):
        """Test grayscale conversion for already gray image."""
        aligner = ECCAligner()
        gray_in = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        gray_out = aligner._to_grayscale(gray_in)
        assert np.array_equal(gray_in, gray_out)

    def test_apply_warp(self):
        """Test warp application."""
        aligner = ECCAligner()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        warped = aligner.apply_warp(img, warp_matrix)

        assert warped.shape == img.shape

    def test_get_confidence_threshold(self):
        """Test confidence threshold getter."""
        aligner = ECCAligner()
        threshold = aligner.get_confidence_threshold()
        assert threshold == 0.85


class TestECCResult:
    """Tests for ECCResult dataclass."""

    def test_result_creation(self):
        """Test ECCResult creation."""
        warp = np.eye(2, 3, dtype=np.float32)
        result = ECCResult(
            warp_matrix=warp,
            correlation_coefficient=0.95,
            converged=True,
            iterations=100,
        )

        assert result.converged is True
        assert result.correlation_coefficient == 0.95
        assert result.iterations == 100
