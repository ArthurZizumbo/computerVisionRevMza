"""
ECC (Enhanced Correlation Coefficient) aligner for rigid image registration.

This module implements ECC-based alignment using OpenCV for
efficient rigid transformation estimation.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger


@dataclass
class ECCResult:
    """Result of ECC alignment."""

    warp_matrix: np.ndarray
    correlation_coefficient: float
    converged: bool
    iterations: int


class ECCAligner:
    """
    Aligns images using Enhanced Correlation Coefficient maximization.

    Attributes
    ----------
    motion_type : int
        Type of motion model (MOTION_EUCLIDEAN, MOTION_AFFINE, etc.)
    num_iterations : int
        Maximum number of iterations
    termination_eps : float
        Termination epsilon for convergence
    """

    MOTION_TYPES = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
        "homography": cv2.MOTION_HOMOGRAPHY,
    }

    def __init__(
        self,
        motion_type: str = "euclidean",
        num_iterations: int = 5000,
        termination_eps: float = 1e-10,
    ) -> None:
        """
        Initialize ECC aligner.

        Parameters
        ----------
        motion_type : str, optional
            Motion model type, defaults to "euclidean"
        num_iterations : int, optional
            Maximum iterations, defaults to 5000
        termination_eps : float, optional
            Convergence threshold, defaults to 1e-10
        """
        if motion_type not in self.MOTION_TYPES:
            raise ValueError(f"motion_type must be one of {list(self.MOTION_TYPES.keys())}")

        self.motion_type = self.MOTION_TYPES[motion_type]
        self.num_iterations = num_iterations
        self.termination_eps = termination_eps
        logger.info(f"ECCAligner initialized with motion_type={motion_type}")

    def align(
        self,
        template: np.ndarray,
        source: np.ndarray,
        initial_warp: np.ndarray | None = None,
    ) -> ECCResult:
        """
        Align source image to template using ECC.

        Parameters
        ----------
        template : np.ndarray
            Template image (reference)
        source : np.ndarray
            Source image to be aligned
        initial_warp : np.ndarray, optional
            Initial warp matrix, defaults to identity

        Returns
        -------
        ECCResult
            Alignment result with warp matrix and confidence
        """
        template_gray = self._to_grayscale(template)
        source_gray = self._to_grayscale(source)

        if self.motion_type == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        if initial_warp is not None:
            warp_matrix = initial_warp.astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.num_iterations,
            self.termination_eps,
        )

        try:
            cc, warp_matrix_result = cv2.findTransformECC(
                templateImage=template_gray,
                inputImage=source_gray,
                warpMatrix=warp_matrix,
                motionType=self.motion_type,
                criteria=criteria,
            )
            logger.debug(f"ECC converged with cc={cc:.4f}")
            return ECCResult(
                warp_matrix=np.asarray(warp_matrix_result, dtype=np.float32),
                correlation_coefficient=cc,
                converged=True,
                iterations=self.num_iterations,
            )
        except cv2.error as e:
            logger.warning(f"ECC failed to converge: {e}")
            return ECCResult(
                warp_matrix=warp_matrix,
                correlation_coefficient=0.0,
                converged=False,
                iterations=0,
            )

    def apply_warp(
        self,
        image: np.ndarray,
        warp_matrix: np.ndarray,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Apply warp transformation to an image.

        Parameters
        ----------
        image : np.ndarray
            Image to transform
        warp_matrix : np.ndarray
            Warp matrix (2x3 or 3x3)
        output_size : tuple[int, int], optional
            Output size, defaults to input size

        Returns
        -------
        np.ndarray
            Warped image
        """
        h, w = image.shape[:2]
        if output_size is None:
            output_size = (w, h)

        if warp_matrix.shape[0] == 3:
            return cv2.warpPerspective(image, warp_matrix, output_size)
        else:
            return cv2.warpAffine(image, warp_matrix, output_size)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def get_confidence_threshold(self) -> float:
        """Get recommended confidence threshold for this aligner."""
        return 0.85
