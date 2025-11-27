"""
DINOv2 embedding extractor for transfer learning.

This module extracts 384-dimensional embeddings from images
using the pretrained DINOv2 ViT-S/14 model.
"""

import cv2
import numpy as np
from loguru import logger

try:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, DINOv2Extractor will be disabled")


class DINOv2Extractor:
    """
    Extracts embeddings using DINOv2 ViT-S/14.

    Attributes
    ----------
    model_name : str
        HuggingFace model name
    device : str
        Device for inference
    embedding_dim : int
        Dimension of output embeddings (384 for ViT-S/14)
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """
        Initialize DINOv2 extractor.

        Parameters
        ----------
        model_name : str, optional
            HuggingFace model name, defaults to "facebook/dinov2-small"
        device : str, optional
            Device for inference, defaults to auto-detect
        cache_dir : str, optional
            Directory to cache model weights
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library is required for DINOv2Extractor")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model_name = model_name
        self.embedding_dim = 384

        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.model.eval()

        logger.info(f"DINOv2Extractor initialized on {device}")

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single image.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR format from OpenCV)

        Returns
        -------
        np.ndarray
            Embedding vector of shape (384,)
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding.squeeze()

    @torch.inference_mode()
    def extract_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from a batch of images.

        Parameters
        ----------
        images : list[np.ndarray]
            List of input images (BGR format)

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (n_images, 384)
        """
        rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        logger.debug(f"Extracted embeddings for {len(images)} images")
        return embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
