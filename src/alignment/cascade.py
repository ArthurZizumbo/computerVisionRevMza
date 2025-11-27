"""
Alignment cascade orchestrator for fallback logic.

This module implements the cascading alignment strategy:
ECC -> LoFTR -> SAM with configurable thresholds.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from loguru import logger

from src.alignment.ecc_aligner import ECCAligner
from src.alignment.loftr_aligner import LoFTRAligner


class AlignmentMethod(str, Enum):
    """Alignment method used in cascade."""

    ECC = "ecc"
    LOFTR = "loftr"
    SAM = "sam"
    FAILED = "failed"


@dataclass
class CascadeResult:
    """Result of cascade alignment."""

    method_used: AlignmentMethod
    warp_matrix: np.ndarray | None
    confidence: float
    aligned_image: np.ndarray | None
    details: dict


class AlignmentCascade:
    """
    Orchestrates cascading alignment with fallback logic.

    The cascade attempts alignment in order:
    1. ECC (fast, rigid transformation)
    2. LoFTR (robust, handles more deformation)
    3. SAM validation (semantic fallback)

    Attributes
    ----------
    ecc_threshold : float
        Minimum ECC correlation for success
    loftr_min_matches : int
        Minimum LoFTR matches for success
    """

    def __init__(
        self,
        ecc_threshold: float = 0.85,
        loftr_min_matches: int = 50,
        device: str | None = None,
    ) -> None:
        """
        Initialize alignment cascade.

        Parameters
        ----------
        ecc_threshold : float, optional
            ECC confidence threshold, defaults to 0.85
        loftr_min_matches : int, optional
            Minimum LoFTR matches, defaults to 50
        device : str, optional
            Device for deep learning models
        """
        self.ecc_threshold = ecc_threshold
        self.loftr_min_matches = loftr_min_matches
        self.device = device

        self._ecc_aligner: ECCAligner | None = None
        self._loftr_aligner: LoFTRAligner | None = None

        logger.info(
            f"AlignmentCascade initialized with ecc_threshold={ecc_threshold}, "
            f"loftr_min_matches={loftr_min_matches}"
        )

    @property
    def ecc_aligner(self) -> ECCAligner:
        """Lazy-load ECC aligner."""
        if self._ecc_aligner is None:
            self._ecc_aligner = ECCAligner()
        return self._ecc_aligner

    @property
    def loftr_aligner(self) -> LoFTRAligner:
        """Lazy-load LoFTR aligner."""
        if self._loftr_aligner is None:
            self._loftr_aligner = LoFTRAligner(device=self.device)
        return self._loftr_aligner

    def align(
        self,
        satellite: np.ndarray,
        vector_image: np.ndarray,
        skip_ecc: bool = False,
        skip_loftr: bool = False,
    ) -> CascadeResult:
        """
        Perform cascading alignment.

        Parameters
        ----------
        satellite : np.ndarray
            Satellite image (reference)
        vector_image : np.ndarray
            Vector rasterized image (to be aligned)
        skip_ecc : bool, optional
            Skip ECC step, defaults to False
        skip_loftr : bool, optional
            Skip LoFTR step, defaults to False

        Returns
        -------
        CascadeResult
            Alignment result with method used and transformation
        """
        if not skip_ecc:
            ecc_result = self._try_ecc(satellite, vector_image)
            if ecc_result is not None:
                return ecc_result

        if not skip_loftr:
            loftr_result = self._try_loftr(satellite, vector_image)
            if loftr_result is not None:
                return loftr_result

        logger.warning("All alignment methods failed")
        return CascadeResult(
            method_used=AlignmentMethod.FAILED,
            warp_matrix=None,
            confidence=0.0,
            aligned_image=None,
            details={"reason": "All methods exhausted"},
        )

    def _try_ecc(
        self,
        satellite: np.ndarray,
        vector_image: np.ndarray,
    ) -> CascadeResult | None:
        """Try ECC alignment."""
        logger.debug("Attempting ECC alignment")
        result = self.ecc_aligner.align(satellite, vector_image)

        if result.converged and result.correlation_coefficient >= self.ecc_threshold:
            aligned = self.ecc_aligner.apply_warp(vector_image, result.warp_matrix)
            logger.info(f"ECC succeeded with cc={result.correlation_coefficient:.4f}")
            return CascadeResult(
                method_used=AlignmentMethod.ECC,
                warp_matrix=result.warp_matrix,
                confidence=result.correlation_coefficient,
                aligned_image=aligned,
                details={
                    "iterations": result.iterations,
                    "converged": result.converged,
                },
            )

        logger.debug(
            f"ECC failed: converged={result.converged}, " f"cc={result.correlation_coefficient:.4f}"
        )
        return None

    def _try_loftr(
        self,
        satellite: np.ndarray,
        vector_image: np.ndarray,
    ) -> CascadeResult | None:
        """Try LoFTR alignment."""
        logger.debug("Attempting LoFTR alignment")

        try:
            result = self.loftr_aligner.find_matches(satellite, vector_image)
        except Exception as e:
            logger.error(f"LoFTR failed with error: {e}")
            return None

        if result.num_matches >= self.loftr_min_matches and result.homography is not None:
            aligned = self.loftr_aligner.apply_homography(vector_image, result.homography)
            logger.info(f"LoFTR succeeded with {result.num_matches} matches")
            return CascadeResult(
                method_used=AlignmentMethod.LOFTR,
                warp_matrix=result.homography,
                confidence=result.mean_confidence,
                aligned_image=aligned,
                details={
                    "num_matches": result.num_matches,
                    "mean_confidence": result.mean_confidence,
                },
            )

        logger.debug(f"LoFTR failed: matches={result.num_matches}")
        return None
