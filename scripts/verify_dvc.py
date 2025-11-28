"""
Script to verify DVC and GCS configuration.

This script checks DVC initialization, remote configuration, and GCS connectivity.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"


def check_dvc_init() -> bool:
    """Check if DVC is initialized."""
    dvc_dir = Path(".dvc")
    if dvc_dir.exists() and dvc_dir.is_dir():
        print("DVC initialized: YES")
        return True
    print("DVC initialized: NO")
    return False


def check_dvc_remote() -> bool:
    """Check DVC remote configuration."""
    success, output = run_command(["dvc", "remote", "list"])
    if success and "storage" in output:
        print("DVC remote configured: YES")
        print(f"  Remote: {output}")
        return True
    print("DVC remote configured: NO")
    return False


def check_gcs_connectivity() -> bool:
    """Check GCS bucket connectivity via DVC version with gs support."""
    success, output = run_command(["dvc", "version"])
    if success and "gs" in output:
        print("GCS connectivity: OK (dvc-gs installed)")
        return True
    print("GCS connectivity: FAILED - dvc-gs not installed")
    return False


def check_dvc_status() -> bool:
    """Check DVC status."""
    success, output = run_command(["dvc", "status"])
    if success:
        print("DVC status: OK")
        if output:
            print(f"  {output}")
        return True
    print(f"DVC status: FAILED - {output}")
    return False


def check_credentials() -> bool:
    """Check if credentials file exists."""
    cred_path = Path("credentials/gcp-dvc-key.json")
    if cred_path.exists():
        print("Credentials file: EXISTS")
        return True
    print("Credentials file: MISSING")
    return False


def main() -> int:
    """Run all DVC verification checks."""
    print("=" * 50)
    print("Geo-Rect DVC Verification Script")
    print("=" * 50)

    checks = [
        ("DVC Initialization", check_dvc_init),
        ("DVC Remote", check_dvc_remote),
        ("Credentials File", check_credentials),
        ("GCS Connectivity", check_gcs_connectivity),
        ("DVC Status", check_dvc_status),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        results.append((name, check_func()))

    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
