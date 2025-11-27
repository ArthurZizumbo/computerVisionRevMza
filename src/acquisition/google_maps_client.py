"""
Google Maps Static API client for satellite tile downloads.

This module provides async HTTP client with retry logic and disk caching
for downloading satellite imagery tiles from Google Maps API.
"""

from pathlib import Path

import httpx
from diskcache import Cache
from loguru import logger


class GoogleMapsClient:
    """
    Async client for Google Maps Static API with caching and retry.

    Attributes
    ----------
    api_key : str
        Google Maps API key
    cache_dir : Path
        Directory for disk cache
    max_retries : int
        Maximum number of retry attempts
    """

    BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"

    def __init__(
        self,
        api_key: str,
        cache_dir: Path | None = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize Google Maps client.

        Parameters
        ----------
        api_key : str
            Google Maps Static API key
        cache_dir : Path, optional
            Cache directory path, defaults to data/cache/tiles
        max_retries : int, optional
            Maximum retry attempts, defaults to 3
        """
        self.api_key = api_key
        self.cache_dir = cache_dir or Path("data/cache/tiles")
        self.max_retries = max_retries
        self._cache = Cache(str(self.cache_dir))
        logger.info(f"GoogleMapsClient initialized with cache at {self.cache_dir}")

    async def fetch_tile(
        self,
        center_lat: float,
        center_lon: float,
        zoom: int = 18,
        size: tuple[int, int] = (640, 640),
        maptype: str = "satellite",
    ) -> bytes:
        """
        Fetch a satellite tile from Google Maps API.

        Parameters
        ----------
        center_lat : float
            Center latitude coordinate
        center_lon : float
            Center longitude coordinate
        zoom : int, optional
            Zoom level (1-21), defaults to 18
        size : tuple[int, int], optional
            Image size in pixels, defaults to (640, 640)
        maptype : str, optional
            Map type, defaults to "satellite"

        Returns
        -------
        bytes
            Raw image bytes (PNG format)

        Raises
        ------
        httpx.HTTPError
            If all retry attempts fail
        """
        cache_key = f"{center_lat:.6f}_{center_lon:.6f}_{zoom}_{size[0]}x{size[1]}_{maptype}"

        if cache_key in self._cache:
            logger.debug(f"Cache hit for tile at ({center_lat}, {center_lon})")
            return self._cache[cache_key]

        params = {
            "center": f"{center_lat},{center_lon}",
            "zoom": zoom,
            "size": f"{size[0]}x{size[1]}",
            "maptype": maptype,
            "key": self.api_key,
        }

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(self.BASE_URL, params=params)
                    response.raise_for_status()
                    image_data = response.content
                    self._cache[cache_key] = image_data
                    logger.debug(f"Downloaded tile at ({center_lat}, {center_lon})")
                    return image_data
            except httpx.HTTPError as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise

        raise RuntimeError("Unexpected state: should have raised or returned")

    def clear_cache(self) -> None:
        """Clear all cached tiles."""
        self._cache.clear()
        logger.info("Tile cache cleared")

    def close(self) -> None:
        """Close cache connection."""
        self._cache.close()
