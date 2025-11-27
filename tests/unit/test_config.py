"""
Unit tests for configuration module.
"""

from src.utils.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.gcp_project_id == "geo-rect-prod"
        assert settings.gcp_region == "us-central1"
        assert settings.log_level == "INFO"
        assert settings.debug is False
        assert settings.ecc_threshold == 0.85
        assert settings.loftr_min_matches == 50

    def test_custom_values(self):
        """Test settings with custom values."""
        settings = Settings(
            log_level="DEBUG",
            debug=True,
            ecc_threshold=0.90,
        )

        assert settings.log_level == "DEBUG"
        assert settings.debug is True
        assert settings.ecc_threshold == 0.90


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings(self):
        """Test that get_settings returns Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_cached(self):
        """Test that settings are cached."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
