"""
LoFTR (Local Feature Transformer) aligner for robust feature matching.

This module implements LoFTR-based alignment using Kornia for
deformable matching in challenging conditions.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

try:
    import kornia.feature as KF
    import torch

    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    logger.warning("Kornia not available, LoFTRAligner will be disabled")


@dataclass
class LoFTRResult:
    """Result of LoFTR matching."""

    keypoints_source: np.ndarray
    keypoints_target: np.ndarray
    confidence_scores: np.ndarray
    homography: np.ndarray | None
    num_matches: int
    mean_confidence: float


class LoFTRAligner:
    """
    Aligns images using LoFTR feature matching.

    Attributes
    ----------
    pretrained : str
        Pretrained model name ("outdoor" or "indoor")
    device : str
        Device to run inference on
    confidence_threshold : float
        Minimum confidence for matches
    """

    def __init__(
        self,
        pretrained: str = "outdoor",
        device: str | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        """
        Initialize LoFTR aligner.

        Parameters
        ----------
        pretrained : str, optional
            Pretrained model, defaults to "outdoor"
        device : str, optional
            Device for inference, defaults to auto-detect
        confidence_threshold : float, optional
            Confidence threshold, defaults to 0.5
        """
        if not KORNIA_AVAILABLE:
            raise RuntimeError("Kornia is required for LoFTRAligner")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.matcher = KF.LoFTR(pretrained=pretrained).to(self.device)
        self.matcher.eval()
        logger.info(f"LoFTRAligner initialized on {device} with pretrained={pretrained}")

    @torch.inference_mode()
    def find_matches(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> LoFTRResult:
        """
        Find correspondences between two images.

        Parameters
        ----------
        source : np.ndarray
            Source image (BGR format)
        target : np.ndarray
            Target image (BGR format)

        Returns
        -------
        LoFTRResult
            Matching result with keypoints and homography
        """
        tensor_source = self._preprocess(source)
        tensor_target = self._preprocess(target)

        input_dict = {"image0": tensor_source, "image1": tensor_target}
        correspondences = self.matcher(input_dict)

        kpts_source = correspondences["keypoints0"].cpu().numpy()
        kpts_target = correspondences["keypoints1"].cpu().numpy()
        confidence = correspondences["confidence"].cpu().numpy()

        mask = confidence >= self.confidence_threshold
        kpts_source = kpts_source[mask]
        kpts_target = kpts_target[mask]
        confidence = confidence[mask]

        homography = None
        if len(kpts_source) >= 4:
            homography, _ = cv2.findHomography(
                kpts_source,
                kpts_target,
                cv2.RANSAC,
                ransacReprojThreshold=5.0,
            )

        logger.debug(f"LoFTR found {len(kpts_source)} matches")
        return LoFTRResult(
            keypoints_source=kpts_source,
            keypoints_target=kpts_target,
            confidence_scores=confidence,
            homography=homography,
            num_matches=len(kpts_source),
            mean_confidence=float(np.mean(confidence)) if len(confidence) > 0 else 0.0,
        )

    def apply_homography(
        self,
        image: np.ndarray,
        homography: np.ndarray,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Apply homography transformation to an image.

        Parameters
        ----------
        image : np.ndarray
            Image to transform
        homography : np.ndarray
            3x3 homography matrix
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

        return cv2.warpPerspective(image, homography, output_size)

    def _preprocess(self, img: np.ndarray) -> "torch.Tensor":
        """Convert image to normalized tensor."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(gray).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def get_min_matches_threshold(self) -> int:
        """Get recommended minimum matches threshold."""
        return 50
