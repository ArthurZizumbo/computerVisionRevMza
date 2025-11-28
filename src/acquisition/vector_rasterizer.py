"""
Vector rasterizer for converting GeoJSON polygons to images.

This module handles the conversion of vector geometries (GeoJSON)
to raster images while preserving georeferencing information.
"""

import numpy as np
import pyproj
from loguru import logger
from PIL import Image, ImageDraw
from shapely.geometry import mapping, shape
from shapely.ops import transform


class VectorRasterizer:
    """
    Rasterizes GeoJSON geometries to images.

    Attributes
    ----------
    target_crs : str
        Target coordinate reference system
    default_color : tuple[int, int, int]
        Default fill color for polygons
    """

    def __init__(
        self,
        target_crs: str = "EPSG:4326",
        default_color: tuple[int, int, int] = (255, 0, 0),
    ) -> None:
        """
        Initialize vector rasterizer.

        Parameters
        ----------
        target_crs : str, optional
            Target CRS, defaults to "EPSG:4326"
        default_color : tuple[int, int, int], optional
            Default polygon fill color (RGB), defaults to red
        """
        self.target_crs = target_crs
        self.default_color = default_color
        logger.info(f"VectorRasterizer initialized with CRS={target_crs}")

    def rasterize(
        self,
        geojson: dict,
        target_size: tuple[int, int],
        bbox: tuple[float, float, float, float] | None = None,
        color: tuple[int, int, int] | None = None,
        fill_alpha: int = 255,
        outline_width: int = 2,
    ) -> np.ndarray:
        """
        Rasterize a GeoJSON geometry to an image.

        Parameters
        ----------
        geojson : dict
            GeoJSON geometry or feature
        target_size : tuple[int, int]
            Target image size (width, height)
        bbox : tuple[float, float, float, float], optional
            Bounding box (min_lon, min_lat, max_lon, max_lat)
        color : tuple[int, int, int], optional
            Fill color, defaults to default_color
        fill_alpha : int, optional
            Fill transparency (0-255), defaults to 255
        outline_width : int, optional
            Outline width in pixels, defaults to 2

        Returns
        -------
        np.ndarray
            Rasterized image (RGBA format)
        """
        geometry = shape(geojson["geometry"]) if "geometry" in geojson else shape(geojson)

        if bbox is None:
            bounds = geometry.bounds
            bbox = (bounds[0], bounds[1], bounds[2], bounds[3])

        width, height = target_size
        min_lon, min_lat, max_lon, max_lat = bbox

        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        fill_color = color or self.default_color
        fill_with_alpha = (*fill_color, fill_alpha)

        pixel_coords = self._geo_to_pixel(geometry, bbox, target_size)

        if geometry.geom_type == "Polygon" and isinstance(pixel_coords, dict):
            self._draw_polygon(draw, pixel_coords, fill_with_alpha, outline_width)
        elif geometry.geom_type == "MultiPolygon" and isinstance(pixel_coords, list):
            for poly_coords in pixel_coords:
                self._draw_polygon(draw, poly_coords, fill_with_alpha, outline_width)

        logger.debug(f"Rasterized {geometry.geom_type} to {width}x{height} image")
        return np.array(img)

    def _geo_to_pixel(
        self,
        geometry,
        bbox: tuple[float, float, float, float],
        size: tuple[int, int],
    ) -> dict[str, list] | list[dict[str, list]]:
        """Convert geometry coordinates to pixel coordinates."""
        min_lon, min_lat, max_lon, max_lat = bbox
        width, height = size

        def transform_coords(lon: float, lat: float) -> tuple[float, float]:
            x = (lon - min_lon) / (max_lon - min_lon) * width
            y = (max_lat - lat) / (max_lat - min_lat) * height
            return (x, y)

        if geometry.geom_type == "Polygon":
            exterior = [transform_coords(lon, lat) for lon, lat in geometry.exterior.coords]
            interiors = [
                [transform_coords(lon, lat) for lon, lat in interior.coords]
                for interior in geometry.interiors
            ]
            return {"exterior": exterior, "interiors": interiors}
        elif geometry.geom_type == "MultiPolygon":
            result: list[dict[str, list]] = []
            for poly in geometry.geoms:
                exterior = [transform_coords(lon, lat) for lon, lat in poly.exterior.coords]
                interiors = [
                    [transform_coords(lon, lat) for lon, lat in interior.coords]
                    for interior in poly.interiors
                ]
                result.append({"exterior": exterior, "interiors": interiors})
            return result

        return {"exterior": [], "interiors": []}

    def _draw_polygon(
        self,
        draw: ImageDraw.ImageDraw,
        coords: dict[str, list],
        fill_color: tuple[int, int, int, int],
        outline_width: int,
    ) -> None:
        """Draw a polygon on the image."""
        exterior = coords["exterior"]
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=fill_color, outline=fill_color[:3])
            if outline_width > 1:
                draw.line(exterior + [exterior[0]], fill=fill_color[:3], width=outline_width)

    def reproject_geometry(
        self,
        geojson: dict,
        source_crs: str,
        target_crs: str,
    ) -> dict[str, object]:
        """
        Reproject a GeoJSON geometry to a different CRS.

        Parameters
        ----------
        geojson : dict
            GeoJSON geometry or feature
        source_crs : str
            Source CRS (e.g., "EPSG:4326")
        target_crs : str
            Target CRS (e.g., "EPSG:32614")

        Returns
        -------
        dict[str, object]
            Reprojected GeoJSON
        """
        geometry = shape(geojson["geometry"]) if "geometry" in geojson else shape(geojson)

        transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

        reprojected = transform(transformer.transform, geometry)

        result: dict[str, object] = mapping(reprojected)
        return result
