"""Integration tests for DVC and GCS configuration."""

import subprocess
from pathlib import Path


class TestDVCIntegration:
    """Integration tests for DVC and GCS."""

    def test_dvc_is_initialized(self):
        """Verify DVC is initialized in the repository."""
        assert Path(".dvc").exists()
        assert Path(".dvc/config").exists()

    def test_dvc_remote_configured(self):
        """Verify DVC remote is configured."""
        result = subprocess.run(
            ["dvc", "remote", "list"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "storage" in result.stdout
        assert "gs://geo-rect-artifacts" in result.stdout

    def test_dvc_version_has_gs_support(self):
        """Verify DVC has GCS support installed."""
        result = subprocess.run(
            ["dvc", "version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "gs" in result.stdout

    def test_dvc_status_runs(self):
        """Verify DVC status command runs without errors."""
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_credentials_file_exists(self):
        """Verify credentials file exists."""
        cred_path = Path("credentials/gcp-dvc-key.json")
        assert cred_path.exists(), "Credentials file not found"

    def test_dvcignore_exists(self):
        """Verify .dvcignore file exists."""
        assert Path(".dvcignore").exists()
