"""
Alignment module for computer vision-based image registration.

This module provides:
- ECC (Enhanced Correlation Coefficient) alignment
- LoFTR (Local Feature Transformer) alignment
- SAM (Segment Anything Model) validation
- Cascade orchestrator for fallback logic
"""

from src.alignment.cascade import AlignmentCascade
from src.alignment.ecc_aligner import ECCAligner
from src.alignment.loftr_aligner import LoFTRAligner
from src.alignment.sam_validator import SAMValidator

__all__ = ["ECCAligner", "LoFTRAligner", "SAMValidator", "AlignmentCascade"]
