"""
Geospatial helper functions.

This module provides utility functions for common geospatial operations
including bounding box calculations and coordinate transformations.
"""

import numpy as np
from shapely.geometry import shape


def calculate_bbox(
    geojson: dict,
    padding: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Calculate bounding box from GeoJSON with optional padding.

    Parameters
    ----------
    geojson : dict
        GeoJSON geometry or feature
    padding : float, optional
        Padding ratio (0.1 = 10% padding), defaults to 0.0

    Returns
    -------
    tuple[float, float, float, float]
        Bounding box (min_lon, min_lat, max_lon, max_lat)
    """
    geometry = shape(geojson["geometry"]) if "geometry" in geojson else shape(geojson)

    min_lon, min_lat, max_lon, max_lat = geometry.bounds

    if padding > 0:
        width = max_lon - min_lon
        height = max_lat - min_lat
        min_lon -= width * padding
        max_lon += width * padding
        min_lat -= height * padding
        max_lat += height * padding

    return (min_lon, min_lat, max_lon, max_lat)


def calculate_union_bbox(
    geojson1: dict,
    geojson2: dict,
    padding: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Calculate union bounding box from two GeoJSON geometries.

    Parameters
    ----------
    geojson1 : dict
        First GeoJSON geometry
    geojson2 : dict
        Second GeoJSON geometry
    padding : float, optional
        Padding ratio, defaults to 0.0

    Returns
    -------
    tuple[float, float, float, float]
        Union bounding box
    """
    bbox1 = calculate_bbox(geojson1, padding=0)
    bbox2 = calculate_bbox(geojson2, padding=0)

    min_lon = min(bbox1[0], bbox2[0])
    min_lat = min(bbox1[1], bbox2[1])
    max_lon = max(bbox1[2], bbox2[2])
    max_lat = max(bbox1[3], bbox2[3])

    if padding > 0:
        width = max_lon - min_lon
        height = max_lat - min_lat
        min_lon -= width * padding
        max_lon += width * padding
        min_lat -= height * padding
        max_lat += height * padding

    return (min_lon, min_lat, max_lon, max_lat)


def meters_per_pixel(latitude: float, zoom: int) -> float:
    """
    Calculate meters per pixel at given latitude and zoom level.

    Parameters
    ----------
    latitude : float
        Latitude in degrees
    zoom : int
        Map zoom level (1-21)

    Returns
    -------
    float
        Meters per pixel
    """
    return 156543.03392 * np.cos(np.radians(latitude)) / (2**zoom)


def calculate_optimal_zoom(
    bbox: tuple[float, float, float, float],
    target_size: tuple[int, int] = (640, 640),
    min_zoom: int = 15,
    max_zoom: int = 20,
) -> int:
    """
    Calculate optimal zoom level to fit bounding box in target image size.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box (min_lon, min_lat, max_lon, max_lat)
    target_size : tuple[int, int], optional
        Target image size, defaults to (640, 640)
    min_zoom : int, optional
        Minimum zoom level, defaults to 15
    max_zoom : int, optional
        Maximum zoom level, defaults to 20

    Returns
    -------
    int
        Optimal zoom level
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2

    width_meters = haversine_distance(center_lat, min_lon, center_lat, max_lon)
    height_meters = haversine_distance(min_lat, min_lon, max_lat, min_lon)

    for zoom in range(max_zoom, min_zoom - 1, -1):
        mpp = meters_per_pixel(center_lat, zoom)
        image_width_meters = target_size[0] * mpp
        image_height_meters = target_size[1] * mpp

        if width_meters <= image_width_meters and height_meters <= image_height_meters:
            return zoom

    return min_zoom


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate distance in meters between two coordinates.

    Parameters
    ----------
    lat1 : float
        First latitude
    lon1 : float
        First longitude
    lat2 : float
        Second latitude
    lon2 : float
        Second longitude

    Returns
    -------
    float
        Distance in meters
    """
    earth_radius = 6371000  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return earth_radius * c
