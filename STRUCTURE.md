# Estructura del Proyecto Geo-Rect

Este documento describe la estructura completa del proyecto y el propósito de cada directorio.

## Directorios Principales

### `/src`
Código fuente principal del proyecto, organizado en módulos:

- **`acquisition/`** - Fase 1: Adquisición de datos
  - `google_maps_client.py`: Cliente async para Google Maps Static API
  - `tile_stitcher.py`: Ensamblador de tiles satelitales
  - `vector_rasterizer.py`: Conversión de GeoJSON a imágenes

- **`alignment/`** - Fase 2: Alineación por Computer Vision
  - `ecc_aligner.py`: Alineación ECC (Enhanced Correlation Coefficient)
  - `loftr_aligner.py`: Matching con LoFTR (Local Feature Transformer)
  - `sam_validator.py`: Validación con SAM (Segment Anything Model)
  - `cascade.py`: Orquestador de cascada ECC → LoFTR → SAM

- **`classification/`** - Fase 3: Clasificación ML
  - `metrics.py`: 7 métricas geométricas (IoU, Hausdorff, Dice, etc.)
  - `dino_extractor.py`: Extractor de embeddings DINOv2
  - `feature_builder.py`: Constructor de features combinados
  - `ensemble.py`: Clasificador XGBoost + LightGBM

- **`api/`** - FastAPI endpoints (por implementar)

- **`pipelines/`** - Dagster pipelines (por implementar)

- **`utils/`** - Utilidades compartidas
  - `config.py`: Configuración con Pydantic Settings
  - `logging.py`: Logging estructurado con Loguru
  - `geo.py`: Helpers geoespaciales

### `/data`
Datos del proyecto (versionados con DVC):
- `raw/`: Datos originales
- `processed/`: Datos preprocesados
- `labeled/`: Datos etiquetados para entrenamiento
- `cache/`: Cache de tiles descargados

### `/models`
Modelos entrenados (versionados con DVC)

### `/tests`
Tests del proyecto:
- `unit/`: Tests unitarios
- `integration/`: Tests de integración
- `fixtures/`: Datos de prueba

### `/notebooks`
Jupyter notebooks para exploración y experimentación

### `/scripts`
Scripts de utilidad:
- `verify_cuda.py`: Verificación de GPU y CUDA
- `test_imports.py`: Test de importaciones

### `/docker`
Archivos de Docker:
- `Dockerfile`: Imagen de producción
- `docker-compose.yml`: Servicios auxiliares (MLflow)

### `/docs`
Documentación del proyecto:
- `referencia/`: Documentos de referencia
- `us-planning/`: Planificación de User Stories
- `us-resolved/`: User Stories implementadas

## Archivos de Configuración

| Archivo | Propósito |
|---------|-----------|
| `pyproject.toml` | Dependencias y configuración de Poetry |
| `.pre-commit-config.yaml` | Hooks de pre-commit |
| `.gitignore` | Archivos ignorados por Git |
| `.env.example` | Plantilla de variables de entorno |
| `AGENTS.md` | Guía para agentes de código IA |
