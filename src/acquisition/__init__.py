"""
Acquisition module for downloading and processing geospatial data.

This module handles:
- Google Maps API tile downloads
- Tile stitching for large areas
- Vector rasterization from GeoJSON
"""

from src.acquisition.google_maps_client import GoogleMapsClient
from src.acquisition.tile_stitcher import TileStitcher
from src.acquisition.vector_rasterizer import VectorRasterizer

__all__ = ["GoogleMapsClient", "TileStitcher", "VectorRasterizer"]
