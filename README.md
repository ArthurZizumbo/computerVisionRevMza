# Geo-Rect: Sistema Híbrido de Validación Geoespacial

> Sistema de detección de discrepancias geométricas y semánticas entre vectores catastrales e imágenes satelitales usando Computer Vision + Machine Learning.

## 🚀 Quick Start

### Prerrequisitos

- Windows 11
- Python 3.11+
- Poetry >= 1.7.0
- NVIDIA GPU con CUDA 12.4
- Docker Desktop (opcional, para MLflow)

### Instalación

```powershell
# Navegar al directorio del proyecto
cd c:\Users\arthu\Proyectos\INE\manzanasDispares

# Instalar dependencias con Poetry
poetry install --with dev

# Verificar CUDA e instalación
poetry run python scripts/verify_cuda.py

# Configurar pre-commit hooks
poetry run pre-commit install

# Copiar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

### Verificar Instalación

```powershell
# Verificar PyTorch con CUDA
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Ejecutar tests
poetry run pytest tests/ -v

# Verificar todas las importaciones
poetry run python scripts/test_imports.py
```

### Levantar Servicios (Opcional)

```powershell
# Iniciar MLflow con Docker
docker-compose -f docker/docker-compose.yml up -d

# MLflow UI: http://localhost:5000
```

## 📁 Estructura del Proyecto

```
geo-rect/
├── src/                    # Código fuente
│   ├── acquisition/        # Fase 1: Adquisición de datos
│   │   ├── google_maps_client.py
│   │   ├── tile_stitcher.py
│   │   └── vector_rasterizer.py
│   ├── alignment/          # Fase 2: Alineación CV
│   │   ├── ecc_aligner.py
│   │   ├── loftr_aligner.py
│   │   ├── sam_validator.py
│   │   └── cascade.py
│   ├── classification/     # Fase 3: Clasificación ML
│   │   ├── metrics.py
│   │   ├── dino_extractor.py
│   │   ├── feature_builder.py
│   │   └── ensemble.py
│   ├── api/                # FastAPI backend
│   └── utils/              # Utilidades compartidas
├── data/                   # Datos (DVC tracked)
├── models/                 # Modelos entrenados (DVC tracked)
├── notebooks/              # Jupyter notebooks
├── tests/                  # Tests unitarios e integración
├── scripts/                # Scripts de utilidad
└── docs/                   # Documentación
```

## 🛠️ Stack Tecnológico

| Categoría | Tecnología |
|-----------|------------|
| **Deep Learning** | PyTorch 2.5.1 + CUDA 12.4 |
| **Computer Vision** | OpenCV, Kornia (LoFTR) |
| **Machine Learning** | XGBoost, LightGBM, DINOv2 |
| **Geospatial** | GeoPandas, Shapely, Rasterio |
| **MLOps** | DVC, MLflow |
| **API** | FastAPI, Pydantic |

## 🔧 Comandos Útiles

```powershell
# Tests con coverage
poetry run pytest tests/ -v --cov=src

# Linting y formateo
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/

# Type checking
poetry run mypy src/

# Pre-commit en todos los archivos
poetry run pre-commit run --all-files
```

## 📊 Pipeline de 3 Fases

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   FASE 1    │    │      FASE 2      │    │       FASE 3        │
│ Adquisición │───▶│  Alineación CV   │───▶│  Clasificación ML   │
└─────────────┘    └──────────────────┘    └─────────────────────┘
      │                    │                         │
      ▼                    ▼                         ▼
• Google Maps API    • ECC (rígido)           • DINOv2 embeddings
• Vector→Raster      • LoFTR (deformable)     • 7 métricas geom.
• Tile stitching     • SAM validation         • XGBoost ensemble
```

## 📝 Licencia

MIT License

---
**Autor:** Arthur Zizumbo
**Proyecto:** INE - Validación Cartográfica
