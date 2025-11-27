"""
Test core imports to verify installation.

Run with: poetry run python scripts/test_imports.py
"""

import sys


def test_imports() -> bool:
    """Test all core imports."""
    print("Testing core imports...")
    errors = []

    try:
        import torch

        print(f"  torch: {torch.__version__}")
        assert torch.cuda.is_available(), "CUDA not available"
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        errors.append(f"torch: {e}")

    try:
        import cv2

        print(f"  opencv: {cv2.__version__}")
    except Exception as e:
        errors.append(f"cv2: {e}")

    try:
        import kornia

        print(f"  kornia: {kornia.__version__}")
    except Exception as e:
        errors.append(f"kornia: {e}")

    try:
        import geopandas

        print(f"  geopandas: {geopandas.__version__}")
    except Exception as e:
        errors.append(f"geopandas: {e}")

    try:
        import shapely

        print(f"  shapely: {shapely.__version__}")
    except Exception as e:
        errors.append(f"shapely: {e}")

    try:
        import rasterio

        print(f"  rasterio: {rasterio.__version__}")
    except Exception as e:
        errors.append(f"rasterio: {e}")

    try:
        import xgboost

        print(f"  xgboost: {xgboost.__version__}")
    except Exception as e:
        errors.append(f"xgboost: {e}")

    try:
        import lightgbm

        print(f"  lightgbm: {lightgbm.__version__}")
    except Exception as e:
        errors.append(f"lightgbm: {e}")

    try:
        import sklearn

        print(f"  sklearn: {sklearn.__version__}")
    except Exception as e:
        errors.append(f"sklearn: {e}")

    try:
        import mlflow

        print(f"  mlflow: {mlflow.__version__}")
    except Exception as e:
        errors.append(f"mlflow: {e}")

    try:
        import fastapi

        print(f"  fastapi: {fastapi.__version__}")
    except Exception as e:
        errors.append(f"fastapi: {e}")

    try:
        import pydantic

        print(f"  pydantic: {pydantic.__version__}")
    except Exception as e:
        errors.append(f"pydantic: {e}")

    try:
        import transformers

        print(f"  transformers: {transformers.__version__}")
    except Exception as e:
        errors.append(f"transformers: {e}")

    if errors:
        print("\nImport errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("\nAll imports successful!")
    return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
