"""
Tile stitcher for assembling multiple satellite tiles into a single image.

This module handles the assembly of multiple Google Maps tiles into
a seamless mosaic for larger geographic areas.
"""

import numpy as np
from loguru import logger
from PIL import Image


class TileStitcher:
    """
    Assembles multiple satellite tiles into a single mosaic image.

    Attributes
    ----------
    tile_size : tuple[int, int]
        Size of individual tiles in pixels
    overlap : int
        Overlap between adjacent tiles in pixels
    """

    def __init__(
        self,
        tile_size: tuple[int, int] = (640, 640),
        overlap: int = 0,
    ) -> None:
        """
        Initialize tile stitcher.

        Parameters
        ----------
        tile_size : tuple[int, int], optional
            Size of each tile, defaults to (640, 640)
        overlap : int, optional
            Overlap between tiles, defaults to 0
        """
        self.tile_size = tile_size
        self.overlap = overlap
        logger.info(f"TileStitcher initialized with tile_size={tile_size}, overlap={overlap}")

    def stitch(
        self,
        tiles: list[list[np.ndarray]],
        normalize_brightness: bool = True,
    ) -> np.ndarray:
        """
        Stitch a grid of tiles into a single image.

        Parameters
        ----------
        tiles : list[list[np.ndarray]]
            2D grid of tile images (rows x cols)
        normalize_brightness : bool, optional
            Whether to normalize brightness across tiles, defaults to True

        Returns
        -------
        np.ndarray
            Stitched mosaic image (BGR format)

        Raises
        ------
        ValueError
            If tiles grid is empty or tiles have inconsistent sizes
        """
        if not tiles or not tiles[0]:
            raise ValueError("Tiles grid cannot be empty")

        n_rows = len(tiles)
        n_cols = len(tiles[0])

        tile_h, tile_w = tiles[0][0].shape[:2]
        effective_h = tile_h - self.overlap
        effective_w = tile_w - self.overlap

        mosaic_h = n_rows * effective_h + self.overlap
        mosaic_w = n_cols * effective_w + self.overlap

        mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

        for row_idx, row in enumerate(tiles):
            for col_idx, tile in enumerate(row):
                if normalize_brightness:
                    tile = self._normalize_tile(tile)

                y_start = row_idx * effective_h
                x_start = col_idx * effective_w

                mosaic[y_start : y_start + tile_h, x_start : x_start + tile_w] = tile

        logger.info(f"Stitched {n_rows}x{n_cols} tiles into {mosaic_w}x{mosaic_h} image")
        return mosaic

    def _normalize_tile(self, tile: np.ndarray) -> np.ndarray:
        """
        Normalize tile brightness to reduce seams.

        Parameters
        ----------
        tile : np.ndarray
            Input tile image

        Returns
        -------
        np.ndarray
            Brightness-normalized tile
        """
        lab = Image.fromarray(tile).convert("LAB")
        l_channel = np.array(lab)[:, :, 0]
        mean_l = np.mean(l_channel)
        target_mean = 128

        if mean_l > 0:
            scale = target_mean / mean_l
            l_channel = np.clip(l_channel * scale, 0, 255).astype(np.uint8)
            lab_array = np.array(lab)
            lab_array[:, :, 0] = l_channel
            return np.array(Image.fromarray(lab_array, mode="LAB").convert("RGB"))

        return tile

    def calculate_grid_size(
        self,
        bbox: tuple[float, float, float, float],
        zoom: int,
    ) -> tuple[int, int]:
        """
        Calculate number of tiles needed to cover a bounding box.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Bounding box (min_lon, min_lat, max_lon, max_lat)
        zoom : int
            Zoom level for tile calculation

        Returns
        -------
        tuple[int, int]
            Number of tiles (rows, cols) needed
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        meters_per_pixel = 156543.03392 * np.cos(np.radians((min_lat + max_lat) / 2)) / (2**zoom)

        width_meters = self._haversine_distance(min_lat, min_lon, min_lat, max_lon)
        height_meters = self._haversine_distance(min_lat, min_lon, max_lat, min_lon)

        effective_tile_w = (self.tile_size[0] - self.overlap) * meters_per_pixel
        effective_tile_h = (self.tile_size[1] - self.overlap) * meters_per_pixel

        n_cols = int(np.ceil(width_meters / effective_tile_w))
        n_rows = int(np.ceil(height_meters / effective_tile_h))

        return max(1, n_rows), max(1, n_cols)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in meters between two coordinates."""
        earth_radius = 6371000  # Earth radius in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return earth_radius * c
