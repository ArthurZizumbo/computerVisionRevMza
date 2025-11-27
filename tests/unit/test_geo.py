"""
Unit tests for geo utilities module.
"""

import pytest

from src.utils.geo import (
    calculate_bbox,
    calculate_optimal_zoom,
    calculate_union_bbox,
    haversine_distance,
    meters_per_pixel,
)


class TestCalculateBbox:
    """Tests for calculate_bbox function."""

    def test_simple_polygon(self):
        """Test bbox calculation for simple polygon."""
        geojson = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }

        bbox = calculate_bbox(geojson)

        assert bbox == (0, 0, 1, 1)

    def test_with_padding(self):
        """Test bbox with padding."""
        geojson = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
        }

        bbox = calculate_bbox(geojson, padding=0.1)

        assert bbox[0] < 0
        assert bbox[1] < 0
        assert bbox[2] > 10
        assert bbox[3] > 10

    def test_feature_input(self):
        """Test bbox from GeoJSON feature."""
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        }

        bbox = calculate_bbox(feature)

        assert bbox == (0, 0, 1, 1)


class TestCalculateUnionBbox:
    """Tests for calculate_union_bbox function."""

    def test_union_of_two_polygons(self):
        """Test union bbox of two polygons."""
        geojson1 = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }
        geojson2 = {
            "type": "Polygon",
            "coordinates": [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]],
        }

        bbox = calculate_union_bbox(geojson1, geojson2)

        assert bbox == (0, 0, 3, 3)


class TestMetersPerPixel:
    """Tests for meters_per_pixel function."""

    def test_equator_zoom_0(self):
        """Test meters per pixel at equator, zoom 0."""
        mpp = meters_per_pixel(0, 0)
        assert mpp == pytest.approx(156543.03392, rel=0.01)

    def test_higher_zoom(self):
        """Test that higher zoom has fewer meters per pixel."""
        mpp_low = meters_per_pixel(0, 10)
        mpp_high = meters_per_pixel(0, 15)
        assert mpp_high < mpp_low


class TestHaversineDistance:
    """Tests for haversine_distance function."""

    def test_same_point(self):
        """Test distance between same point is zero."""
        dist = haversine_distance(0, 0, 0, 0)
        assert dist == 0

    def test_known_distance(self):
        """Test known distance between two cities."""
        # Approximate distance NYC to LA
        dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        # Should be roughly 3935 km
        assert 3900000 < dist < 4000000


class TestCalculateOptimalZoom:
    """Tests for calculate_optimal_zoom function."""

    def test_small_bbox(self):
        """Test zoom for small bounding box."""
        bbox = (0, 0, 0.001, 0.001)
        zoom = calculate_optimal_zoom(bbox)
        assert 15 <= zoom <= 20

    def test_returns_integer(self):
        """Test that zoom is an integer."""
        bbox = (0, 0, 0.01, 0.01)
        zoom = calculate_optimal_zoom(bbox)
        assert isinstance(zoom, int)
