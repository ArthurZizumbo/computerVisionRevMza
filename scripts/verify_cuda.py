"""
Script to verify CUDA installation and GPU availability.

This script checks the NVIDIA GPU, CUDA toolkit, and PyTorch CUDA support.
"""

import subprocess
import sys


def check_nvidia_smi() -> bool:
    """Check NVIDIA driver installation via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("NVIDIA Driver Information:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"nvidia-smi failed: {e}")
        return False


def check_pytorch_cuda() -> bool:
    """Check PyTorch CUDA support."""
    try:
        import torch

        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Device Count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nDevice {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")

            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x)
            print(f"\nGPU Tensor Test: PASSED (result shape: {y.shape})")
            return True
        else:
            print("CUDA is not available")
            return False
    except ImportError:
        print("PyTorch is not installed")
        return False


def check_kornia() -> bool:
    """Check Kornia installation for LoFTR."""
    try:
        import kornia

        print(f"\nKornia Version: {kornia.__version__}")
        return True
    except ImportError:
        print("Kornia is not installed")
        return False


def check_opencv() -> bool:
    """Check OpenCV installation."""
    try:
        import cv2

        print(f"\nOpenCV Version: {cv2.__version__}")
        return True
    except ImportError:
        print("OpenCV is not installed")
        return False


def check_geospatial() -> bool:
    """Check geospatial libraries."""
    try:
        import geopandas
        import rasterio
        import shapely

        print(f"\nGeoPandas Version: {geopandas.__version__}")
        print(f"Shapely Version: {shapely.__version__}")
        print(f"Rasterio Version: {rasterio.__version__}")
        return True
    except ImportError as e:
        print(f"Geospatial library missing: {e}")
        return False


def check_ml_libraries() -> bool:
    """Check ML libraries."""
    try:
        import lightgbm
        import sklearn
        import xgboost

        print(f"\nXGBoost Version: {xgboost.__version__}")
        print(f"LightGBM Version: {lightgbm.__version__}")
        print(f"Scikit-learn Version: {sklearn.__version__}")
        return True
    except ImportError as e:
        print(f"ML library missing: {e}")
        return False


def main() -> int:
    """Run all verification checks."""
    print("=" * 60)
    print("Geo-Rect Environment Verification Script")
    print("=" * 60)

    checks = [
        ("NVIDIA Driver", check_nvidia_smi),
        ("PyTorch CUDA", check_pytorch_cuda),
        ("Kornia", check_kornia),
        ("OpenCV", check_opencv),
        ("Geospatial", check_geospatial),
        ("ML Libraries", check_ml_libraries),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n--- {name} Check ---")
        results.append((name, check_func()))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
