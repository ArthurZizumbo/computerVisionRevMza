# Contexto del Proyecto: Geo-Rect - Sistema Híbrido de Validación Geoespacial

> **Propósito de este documento:** Servir como contexto completo para agentes de código IA en múltiples iteraciones. Contiene toda la información necesaria para entender el proyecto sin necesidad de consultar otros archivos.

---

## 1. Resumen Ejecutivo

**Nombre del Proyecto:** Geo-Rect
**Autor:** Arthur Zizumbo
**Duración:** 6 semanas (3 sprints × 2 semanas)
**Presupuesto Cloud:** $300 USD GCP (estimado real: ~$26.44)
**Paradigma:** Local-First, Cloud-Deploy

### Problema Central
Detectar discrepancias entre vectores catastrales (polígonos de manzanas) e imágenes satelitales reales en **5,000 manzanas urbanas** utilizando Computer Vision + Machine Learning.

### Tipos de Discrepancias a Detectar
| Categoría | Descripción | Ejemplos |
|-----------|-------------|----------|
| **Geométrica** | Desalineación vector-ráster | Vectores desfasados, rotados o escalados incorrectamente |
| **Semántica** | Cambios en el contenido | Construcciones nuevas, demoliciones, expansiones urbanas |
| **Mixta** | Combinación de ambas | Cambio de uso de suelo + desplazamiento de linderos |

### Desafío Crítico: Few-Shot Learning
- **Solo 300 manzanas etiquetadas** de 5,000 totales (6%)
- Solución: Transfer Learning con DINOv2 (embeddings pre-entrenados)
- Data Augmentation geométrica para aumentar variabilidad

---

## 2. Restricciones Fundamentales

### 2.1 Hardware Local (Desarrollo y Entrenamiento)
| Componente | Especificación | Restricción |
|------------|----------------|-------------|
| GPU | NVIDIA RTX 4070 Laptop | **8 GB VRAM** - Modelos deben caber en memoria |
| CPU | 22 cores | Paralelización disponible para preprocessing |
| RAM | No especificada | Asumir ~64GB disponible |
| Almacenamiento | SSD local | ~50GB para datos + modelos |

### 2.2 Presupuesto Cloud ($300 USD GCP)
```
Distribución Estimada:
├── Cloud Run API (FastAPI)      : $12.00/mes × 1.5 meses = $18.00
├── Cloud Storage (datos)        : $0.022/GB × 20GB       = $0.44
├── Artifact Registry (Docker)   : $0.10/GB × 5GB         = $0.50
├── Eventarc + Pub/Sub          : $1.00
├── Contingencia (20%)          : $6.50
└── TOTAL ESTIMADO              : $26.44 (91% bajo presupuesto)
```

### 2.3 Paradigma Local-First
```
┌─────────────────────────────────────────────────────────────┐
│                    PRINCIPIO LOCAL-FIRST                     │
├─────────────────────────────────────────────────────────────┤
│  ✅ LOCAL (RTX 4070):          │  ☁️ CLOUD (GCP):            │
│  • Entrenamiento completo      │  • Solo inferencia          │
│  • Experimentación             │  • API REST (FastAPI)       │
│  • Feature extraction          │  • Serving modelos ligeros  │
│  • Validación de datos         │  • Almacenamiento DVC       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Arquitectura del Pipeline (3 Fases)

### Vista General
```
┌──────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE GEO-RECT                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐     │
│  │   FASE 1    │    │      FASE 2      │    │       FASE 3        │     │
│  │ Adquisición │───▶│  Alineación CV   │───▶│  Clasificación ML   │     │
│  └─────────────┘    └──────────────────┘    └─────────────────────┘     │
│        │                    │                         │                  │
│        ▼                    ▼                         ▼                  │
│  • Google Maps API    • ECC (rígido)           • DINOv2 embeddings      │
│  • Vector→Raster      • LoFTR (deformable)     • 7 métricas geom.       │
│  • Tile stitching     • SAM validation         • XGBoost ensemble       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Fase 1: Adquisición de Datos

**Objetivo:** Obtener tripletas (imagen_satelital, vector_anterior, vector_nuevo) para cada manzana.

```python
# Pseudocódigo del proceso de adquisición
def acquire_block_data(block_geojson_old: dict, block_geojson_new: dict) -> BlockData:
    # 1. Calcular bounding box unificado
    bbox = calculate_union_bbox(block_geojson_old, block_geojson_new, padding=0.1)

    # 2. Descargar imagen satelital (Google Maps Static API)
    satellite_img = fetch_google_maps_tile(
        center=bbox.center,
        zoom=calculate_optimal_zoom(bbox),
        size=(640, 640),
        maptype="satellite"
    )

    # 3. Rasterizar vectores a imágenes separadas
    vector_old_img = rasterize_geojson(
        geojson=block_geojson_old,
        target_size=satellite_img.shape[:2],
        color=(0, 0, 255)  # Rojo para anterior
    )

    vector_new_img = rasterize_geojson(
        geojson=block_geojson_new,
        target_size=satellite_img.shape[:2],
        color=(0, 255, 0)  # Verde para nuevo
    )

    return BlockData(
        block_id=block_geojson_new["properties"]["id"],
        satellite=satellite_img,
        vector_old=vector_old_img,
        vector_new=vector_new_img
    )
```

**Componentes Clave:**
| Componente | Tecnología | Propósito |
|------------|-----------|-----------|
| API Cliente | `httpx` async | Descarga paralela de tiles |
| Tile Stitching | `rasterio`, `PIL` | Ensamblar múltiples tiles |
| Rasterización | `rasterio`, `shapely` | Vector GeoJSON → imagen PNG |
| Cache | `diskcache` | Evitar re-descargas |

### 3.2 Fase 2: Alineación por Computer Vision

**Objetivo:** Corregir desalineación de la **Capa Nueva** usando el satélite y la **Capa Anterior** como contexto.

**Estrategia Cascada (Fallback Progresivo):**
```
┌─────────────────────────────────────────────────────────────┐
│                  CASCADA DE ALINEACIÓN                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Intento 1: ECC (Enhanced Correlation Coefficient)          │
│  ├─ Alinea Capa Nueva vs Satélite                          │
│  ├─ Si confianza ≥ 0.85 → ✅ Usar resultado                │
│  └─ Si confianza < 0.85 → Fallback a LoFTR                 │
│                                                              │
│  Intento 2: LoFTR (Local Feature TRansformer)               │
│  ├─ Matching robusto (Nueva vs Satélite)                   │
│  ├─ Si matches ≥ 50 → ✅ Usar resultado                    │
│  └─ Si matches < 50 → Fallback a SAM                       │
│                                                              │
│  Intento 3: SAM (Segment Anything Model)                    │
│  ├─ Segmentación semántica para validar topología          │
│  └─ Usa Capa Anterior para detectar cambios drásticos      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Código de Referencia - ECC:**
```python
import cv2
import numpy as np

def align_with_ecc(
    satellite: np.ndarray,
    vector_new: np.ndarray,
    num_iterations: int = 5000,
    termination_eps: float = 1e-10
) -> tuple[np.ndarray, float]:
    """
    Alinea imagen satelital con vector nuevo usando ECC.

    Returns:
        tuple: (warp_matrix, correlation_coefficient)
    """
    # Convertir a escala de grises
    sat_gray = cv2.cvtColor(satellite, cv2.COLOR_BGR2GRAY)
    vec_gray = cv2.cvtColor(vector_new, cv2.COLOR_BGR2GRAY)

    # Definir matriz de transformación inicial (identidad)
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Criterio de terminación
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        num_iterations,
        termination_eps
    )

    # Ejecutar ECC
    try:
        cc, warp_matrix = cv2.findTransformECC(
            templateImage=vec_gray,
            inputImage=sat_gray,
            warpMatrix=warp_matrix,
            motionType=cv2.MOTION_EUCLIDEAN,
            criteria=criteria
        )
        return warp_matrix, cc
    except cv2.error:
        return warp_matrix, 0.0  # Fallo de convergencia
```

**Código de Referencia - LoFTR:**
```python
import torch
import kornia.feature as KF

class LoFTRAligner:
    def __init__(self, pretrained: str = "outdoor", device: str = "cuda"):
        self.device = torch.device(device)
        self.matcher = KF.LoFTR(pretrained=pretrained).to(self.device)
        self.matcher.eval()

    @torch.inference_mode()
    def find_matches(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encuentra correspondencias entre dos imágenes.

        Returns:
            tuple: (keypoints1, keypoints2, confidence_scores)
        """
        # Preprocesar imágenes
        tensor1 = self._preprocess(img1)
        tensor2 = self._preprocess(img2)

        # Ejecutar matching
        input_dict = {"image0": tensor1, "image1": tensor2}
        correspondences = self.matcher(input_dict)

        return (
            correspondences["keypoints0"].cpu().numpy(),
            correspondences["keypoints1"].cpu().numpy(),
            correspondences["confidence"].cpu().numpy()
        )

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Convierte imagen a tensor normalizado."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(gray).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)
```

### 3.3 Fase 3: Clasificación ML

**Objetivo:** Clasificar cada manzana como: `aligned`, `misaligned`, `semantic_change`, `requires_review`.

**Arquitectura del Clasificador:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE CLASIFICACIÓN                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────┐    ┌────────────────┐    ┌──────────────────┐   │
│  │   DINOv2       │    │  7 Métricas    │    │    Ensemble      │   │
│  │  Embeddings    │───▶│  Geométricas   │───▶│   XGBoost +      │   │
│  │  (384-dim)     │    │  (7-dim)       │    │   LightGBM       │   │
│  └────────────────┘    └────────────────┘    └──────────────────┘   │
│         │                     │                      │              │
│         └──────────┬──────────┘                      │              │
│                    ▼                                 ▼              │
│           Feature Vector                      Predicción +         │
│           (391 dimensiones)                   Confianza            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Las 7 Métricas Geométricas:**
```python
from dataclasses import dataclass
import numpy as np

@dataclass
class GeometricMetrics:
    """7 métricas geométricas para clasificación."""

    # 1. IoU (Intersection over Union) - Jaccard Index
    iou: float  # [0, 1] - 1.0 = superposición perfecta

    # 2. Distancia de Hausdorff normalizada
    hausdorff_distance: float  # [0, ∞) - 0 = idénticos

    # 3. Dice Coefficient (F1 de áreas)
    dice_coefficient: float  # [0, 1] - 1.0 = idénticos

    # 4. Error de área relativo
    area_ratio_error: float  # [0, ∞) - 0 = áreas iguales

    # 5. Diferencia de centroide normalizada
    centroid_distance: float  # [0, ∞) - 0 = centros alineados

    # 6. Diferencia angular (rotación)
    angular_difference: float  # [0, 180] grados

    # 7. Coeficiente de correlación de contornos
    contour_correlation: float  # [-1, 1] - 1.0 = formas idénticas

def compute_metrics(
    aligned_vector: np.ndarray,
    satellite_mask: np.ndarray
) -> GeometricMetrics:
    """
    Calcula las 7 métricas geométricas entre vector alineado y máscara satelital.
    """
    # Binarizar imágenes
    vec_binary = (aligned_vector > 127).astype(np.uint8)
    sat_binary = (satellite_mask > 127).astype(np.uint8)

    # 1. IoU
    intersection = np.logical_and(vec_binary, sat_binary).sum()
    union = np.logical_or(vec_binary, sat_binary).sum()
    iou = intersection / union if union > 0 else 0.0

    # 2. Hausdorff Distance (usando scipy o cv2)
    from scipy.spatial.distance import directed_hausdorff
    vec_coords = np.argwhere(vec_binary)
    sat_coords = np.argwhere(sat_binary)
    hausdorff = max(
        directed_hausdorff(vec_coords, sat_coords)[0],
        directed_hausdorff(sat_coords, vec_coords)[0]
    ) / max(vec_binary.shape)  # Normalizar

    # 3. Dice Coefficient
    dice = 2 * intersection / (vec_binary.sum() + sat_binary.sum()) if (vec_binary.sum() + sat_binary.sum()) > 0 else 0.0

    # 4. Area Ratio Error
    vec_area = vec_binary.sum()
    sat_area = sat_binary.sum()
    area_ratio_error = abs(vec_area - sat_area) / max(vec_area, sat_area, 1)

    # 5. Centroid Distance
    vec_centroid = np.mean(np.argwhere(vec_binary), axis=0) if vec_area > 0 else np.array([0, 0])
    sat_centroid = np.mean(np.argwhere(sat_binary), axis=0) if sat_area > 0 else np.array([0, 0])
    centroid_distance = np.linalg.norm(vec_centroid - sat_centroid) / max(vec_binary.shape)

    # 6. Angular Difference (usando momentos)
    vec_moments = cv2.moments(vec_binary)
    sat_moments = cv2.moments(sat_binary)
    vec_angle = 0.5 * np.arctan2(2 * vec_moments['mu11'], vec_moments['mu20'] - vec_moments['mu02'])
    sat_angle = 0.5 * np.arctan2(2 * sat_moments['mu11'], sat_moments['mu20'] - sat_moments['mu02'])
    angular_diff = abs(np.degrees(vec_angle - sat_angle))

    # 7. Contour Correlation
    vec_contours, _ = cv2.findContours(vec_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sat_contours, _ = cv2.findContours(sat_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if vec_contours and sat_contours:
        contour_corr = cv2.matchShapes(vec_contours[0], sat_contours[0], cv2.CONTOURS_MATCH_I2, 0)
    else:
        contour_corr = 1.0  # Máxima diferencia

    return GeometricMetrics(
        iou=iou,
        hausdorff_distance=hausdorff,
        dice_coefficient=dice,
        area_ratio_error=area_ratio_error,
        centroid_distance=centroid_distance,
        angular_difference=angular_diff,
        contour_correlation=contour_corr
    )
```

**Extractor de Embeddings DINOv2:**
```python
import torch
from transformers import AutoModel, AutoImageProcessor

class DINOv2Extractor:
    """Extractor de embeddings usando DINOv2 ViT-S/14."""

    def __init__(self, model_name: str = "facebook/dinov2-small", device: str = "cuda"):
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = 384  # ViT-S/14

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae embedding de 384 dimensiones de una imagen.

        Args:
            image: Imagen BGR (numpy array)

        Returns:
            np.ndarray: Vector de embeddings (384,)
        """
        # Preprocesar
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)

        # Usar CLS token como embedding global
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings.squeeze()
```

---

## 4. Stack Tecnológico Completo

### 4.1 Núcleo del Sistema
| Categoría | Tecnología | Versión | Propósito |
|-----------|-----------|---------|-----------|
| **Runtime** | Python | 3.11+ | Compatibilidad PyTorch, typing moderno |
| **Gestión deps** | Poetry | 1.7+ | Lockfile reproducible, grupos dev/prod |
| **Deep Learning** | PyTorch | 2.2+ | Backend GPU, CUDA 12.x |
| **CV Clásico** | OpenCV | 4.9+ | ECC, morfología, contornos |
| **CV Diferenciable** | Kornia | 0.7+ | LoFTR, augmentations GPU |
| **Segmentación** | SAM (Ultralytics) | latest | Segment Anything Model |

### 4.2 Machine Learning
| Categoría | Tecnología | Versión | Propósito |
|-----------|-----------|---------|-----------|
| **Embeddings** | DINOv2 | ViT-S/14 | Transfer learning, 384-dim features |
| **Clasificador 1** | XGBoost | 2.0+ | Gradient boosting, manejo de imbalance |
| **Clasificador 2** | LightGBM | 4.0+ | Alternativa rápida, leaf-wise |
| **HPO** | Optuna | 3.5+ | Optimización bayesiana hiperparámetros |
| **Explicabilidad** | SHAP | 0.44+ | Interpretación de predicciones |

### 4.3 Datos Geoespaciales
| Categoría | Tecnología | Versión | Propósito |
|-----------|-----------|---------|-----------|
| **Vectores** | GeoPandas | 0.14+ | Manipulación GeoDataFrames |
| **Geometría** | Shapely | 2.0+ | Operaciones geométricas |
| **Rasterización** | Rasterio | 1.3+ | Vector → Raster, CRS handling |
| **Proyecciones** | PyProj | 3.6+ | Transformaciones CRS |

### 4.4 MLOps y Orquestación
| Categoría | Tecnología | Versión | Propósito |
|-----------|-----------|---------|-----------|
| **Versionado datos** | DVC | 3.30+ | Data/model versioning con GCS remote |
| **Tracking experimentos** | MLflow | 2.10+ | Métricas, parámetros, artefactos |
| **Orquestación** | Dagster | 1.5+ | DAGs de pipelines, scheduling |
| **Validación datos** | Pydantic | 2.5+ | Schemas, validación en runtime |

### 4.5 API y Despliegue
| Categoría | Tecnología | Versión | Propósito |
|-----------|-----------|---------|-----------|
| **Framework API** | FastAPI | 0.109+ | REST async, OpenAPI docs |
| **Server ASGI** | Uvicorn | 0.27+ | Server producción |
| **Contenedores** | Docker | 24+ | Empaquetado reproducible |
| **Cloud Run** | GCP | - | Serverless containers |
| **Storage** | GCS | - | Almacenamiento de datos DVC |

### 4.6 Frontend (Visualización)
| Categoría | Tecnología | Versión | Propósito |
|-----------|-----------|---------|-----------|
| **Framework** | Nuxt | 4.x | Vue 3, SSR, file-based routing |
| **Mapas** | MapLibre GL JS | 4.x | Renderizado vectorial WebGL |
| **UI Components** | Nuxt UI | 2.x | Componentes Tailwind |
| **HTTP Client** | ofetch | - | Cliente HTTP isomórfico |

### 4.7 Calidad de Código
| Categoría | Tecnología | Versión | Propósito |
|-----------|-----------|---------|-----------|
| **Linting** | Ruff | 0.1+ | Linter + formatter ultrarrápido |
| **Type Checking** | Mypy | 1.8+ | Verificación estática de tipos |
| **Testing** | Pytest | 8.0+ | Framework de tests |
| **Coverage** | pytest-cov | 4.1+ | Cobertura de código |
| **Pre-commit** | pre-commit | 3.6+ | Hooks de calidad |

---

## 5. Estructura del Proyecto

```
geo-rect/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Tests + linting en PRs
│       └── deploy.yml             # Deploy a Cloud Run
├── data/
│   ├── raw/                       # Datos originales (DVC tracked)
│   │   ├── manzanas.geojson       # 5,000 polígonos de manzanas
│   │   └── satellite_tiles/       # Imágenes descargadas
│   ├── processed/                 # Datos preprocesados
│   │   ├── pairs/                 # Pares (satellite, vector) alineados
│   │   └── features/              # Features extraídos
│   └── labeled/                   # 300 manzanas etiquetadas
│       ├── train/                 # 240 samples (80%)
│       └── val/                   # 60 samples (20%)
├── src/
│   ├── __init__.py
│   ├── acquisition/               # Fase 1: Descarga de datos
│   │   ├── __init__.py
│   │   ├── google_maps_client.py  # Cliente API Google Maps
│   │   ├── tile_stitcher.py       # Ensamblador de tiles
│   │   └── vector_rasterizer.py   # GeoJSON → imagen
│   ├── alignment/                 # Fase 2: Alineación CV
│   │   ├── __init__.py
│   │   ├── ecc_aligner.py         # Alineador ECC (OpenCV)
│   │   ├── loftr_aligner.py       # Alineador LoFTR (Kornia)
│   │   ├── sam_validator.py       # Validación con SAM
│   │   └── cascade.py             # Orquestador de cascada
│   ├── classification/            # Fase 3: Clasificación ML
│   │   ├── __init__.py
│   │   ├── metrics.py             # 7 métricas geométricas
│   │   ├── dino_extractor.py      # Extractor DINOv2
│   │   ├── feature_builder.py     # Concatenador de features
│   │   └── ensemble.py            # XGBoost + LightGBM
│   ├── api/                       # FastAPI endpoints
│   │   ├── __init__.py
│   │   ├── main.py                # App FastAPI
│   │   ├── routes/
│   │   │   ├── health.py          # /health, /ready
│   │   │   ├── predict.py         # /predict, /batch
│   │   │   └── explain.py         # /explain (SHAP)
│   │   └── schemas/               # Pydantic models
│   │       ├── request.py
│   │       └── response.py
│   ├── pipelines/                 # Dagster pipelines
│   │   ├── __init__.py
│   │   ├── acquisition_pipeline.py
│   │   ├── training_pipeline.py
│   │   └── inference_pipeline.py
│   └── utils/                     # Utilidades compartidas
│       ├── __init__.py
│       ├── config.py              # Configuración con Pydantic
│       ├── logging.py             # Logging estructurado
│       └── geo.py                 # Helpers geoespaciales
├── models/                        # Modelos entrenados
│   ├── dinov2/                    # Modelo DINOv2 (cache)
│   ├── xgboost/                   # Modelos XGBoost por versión
│   └── loftr/                     # Pesos LoFTR (cache)
├── tests/
│   ├── unit/                      # Tests unitarios
│   │   ├── test_metrics.py
│   │   ├── test_ecc_aligner.py
│   │   └── test_dino_extractor.py
│   ├── integration/               # Tests de integración
│   │   ├── test_cascade.py
│   │   └── test_api.py
│   └── fixtures/                  # Datos de prueba
│       └── sample_blocks/
├── notebooks/                     # Jupyter notebooks exploración
│   ├── 01_eda.ipynb
│   ├── 02_alignment_experiments.ipynb
│   └── 03_model_evaluation.ipynb
├── docs/                          # Documentación
│   ├── architecture.md
│   ├── api_reference.md
│   └── deployment.md
├── scripts/                       # Scripts de utilidad
│   ├── download_tiles.py
│   ├── train_model.py
│   └── export_onnx.py
├── .dvc/                          # Configuración DVC
├── .pre-commit-config.yaml        # Hooks pre-commit
├── pyproject.toml                 # Configuración Poetry + tools
├── Dockerfile                     # Imagen de producción
├── docker-compose.yml             # Dev environment
├── dvc.yaml                       # Pipelines DVC
├── mlflow.yaml                    # Config MLflow
└── README.md                      # Documentación principal
```

---

## 6. Historias de Usuario por Sprint

### Sprint 1: Fundamentos (Semanas 1-2)
**Objetivo:** Pipeline de adquisición + alineación básica funcional.

| ID | Historia | Criterios de Aceptación |
|----|----------|------------------------|
| US-1.1 | Configuración del entorno | Poetry + Docker + pre-commit funcionando |
| US-1.2 | Cliente Google Maps API | Descarga de tiles con retry + cache |
| US-1.3 | Rasterizador de vectores | GeoJSON → PNG con georeferencia |
| US-1.4 | Alineador ECC básico | Transformación rígida con confianza |
| US-1.5 | Pipeline DVC inicial | Reproducibilidad de datos |
| US-1.6 | Tests unitarios core | Coverage ≥ 80% módulos críticos |

### Sprint 2: ML Core (Semanas 3-4)
**Objetivo:** Clasificador entrenado con métricas baseline.

| ID | Historia | Criterios de Aceptación |
|----|----------|------------------------|
| US-2.1 | Alineador LoFTR | Fallback para ECC con baja confianza |
| US-2.2 | Validador SAM | Segmentación semántica de respaldo |
| US-2.3 | Orquestador cascada | Lógica de fallback completa |
| US-2.4 | Extractor DINOv2 | Embeddings 384-dim batch processing |
| US-2.5 | Métricas geométricas | 7 métricas implementadas + tests |
| US-2.6 | Entrenamiento XGBoost | F1 ≥ 0.75 en validación |
| US-2.7 | Tracking MLflow | Experimentos versionados |

### Sprint 3: Producción (Semanas 5-6)
**Objetivo:** API desplegada + frontend funcional.

| ID | Historia | Criterios de Aceptación |
|----|----------|------------------------|
| US-3.1 | API FastAPI | Endpoints /predict, /batch, /health |
| US-3.2 | Optimización Optuna | F1 ≥ 0.80 post-HPO |
| US-3.3 | Explicabilidad SHAP | Endpoint /explain funcional |
| US-3.4 | Dockerfile producción | Imagen < 2GB, multi-stage |
| US-3.5 | Deploy Cloud Run | Latencia p95 < 3s |
| US-3.6 | Frontend MapLibre | Visualización de predicciones |
| US-3.7 | Documentación | README + API docs completos |

---

## 7. Métricas de Éxito y Umbrales

### 7.1 Métricas del Modelo
| Métrica | Umbral Mínimo | Objetivo | Descripción |
|---------|--------------|----------|-------------|
| **Recall (Discrepancias)** | ≥ 85% | ≥ 90% | No perder manzanas problemáticas |
| **Precision** | ≥ 70% | ≥ 80% | Minimizar falsos positivos |
| **F1-Score** | ≥ 0.75 | ≥ 0.80 | Balance precision/recall |
| **AUC-ROC** | ≥ 0.85 | ≥ 0.90 | Discriminación general |

### 7.2 Métricas de Rendimiento
| Métrica | Umbral | Descripción |
|---------|--------|-------------|
| **Latencia p50** | < 1.5s | Mediana de tiempo respuesta |
| **Latencia p95** | < 3.0s | Percentil 95 |
| **Throughput** | ≥ 100 manzanas/hora | En batch processing |
| **VRAM Usage** | < 7 GB | Dejar margen en RTX 4070 |

### 7.3 Métricas de Calidad
| Métrica | Umbral | Descripción |
|---------|--------|-------------|
| **Test Coverage** | ≥ 80% | Cobertura de código |
| **Type Coverage** | ≥ 90% | Cobertura de tipos (mypy) |
| **Linting Score** | 0 errors | Ruff sin errores |

---

## 8. Gestión de Riesgos

| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|--------|--------------|---------|------------|
| R1 | VRAM insuficiente para DINOv2 | Media | Alto | Usar ViT-S/14 (384MB), batch size pequeño |
| R2 | Pocos datos etiquetados (300) | Alta | Alto | Transfer learning DINOv2, data augmentation |
| R3 | ECC falla en imágenes ruidosas | Media | Medio | Cascada LoFTR → SAM como fallback |
| R4 | Latencia API > 3s | Baja | Medio | Precomputar embeddings, cache Redis |
| R5 | Costo GCP > $300 | Baja | Alto | Edge-First, billing alerts, Cloud Run min=0 |
| R6 | Google Maps API rate limits | Media | Medio | Cache agresivo, exponential backoff |
| R7 | Desbalance de clases severo | Alta | Medio | SMOTE, class_weight, threshold tuning |
| R8 | Cambios en API Google Maps | Baja | Alto | Abstracción de cliente, tests de contrato |
| R9 | Overfitting por pocos datos | Alta | Alto | Cross-validation, regularización fuerte |
| R10 | Docker image muy grande | Media | Bajo | Multi-stage build, slim base images |

---

## 9. Configuración del Entorno de Desarrollo

### 9.1 Instalación Inicial
```powershell
# Clonar repositorio
git clone https://github.com/arthuzumbo/geo-rect.git
cd geo-rect

# Instalar Poetry (si no está instalado)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Crear entorno virtual e instalar dependencias
poetry install --with dev

# Activar entorno
poetry shell

# Configurar pre-commit hooks
pre-commit install

# Verificar instalación
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 9.2 Variables de Entorno Requeridas
```bash
# .env (no commitear)
GOOGLE_MAPS_API_KEY=AIza...
GCP_PROJECT_ID=geo-rect-prod
GCP_REGION=us-central1
MLFLOW_TRACKING_URI=file:./mlruns
DVC_REMOTE=gs://geo-rect-data
```

### 9.3 Comandos de Desarrollo Frecuentes
```powershell
# Ejecutar tests
poetry run pytest tests/ -v --cov=src

# Linting y formateo
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/

# Type checking
poetry run mypy src/

# Ejecutar pipeline DVC
poetry run dvc repro

# Levantar API local
poetry run uvicorn src.api.main:app --reload --port 8000

# Entrenar modelo
poetry run python scripts/train_model.py --config configs/xgboost.yaml
```

---

## 10. Decisiones Arquitectónicas Clave (ADRs)

### ADR-001: Local-First Training
**Decisión:** Todo el entrenamiento se realiza en hardware local (RTX 4070).
**Razón:** Presupuesto limitado ($300), GPU local suficiente para modelos seleccionados.
**Consecuencias:** Cloud solo para serving, sin costos de GPU cloud.

### ADR-002: Transfer Learning con DINOv2
**Decisión:** Usar embeddings DINOv2 pre-entrenados en lugar de entrenar CNN from scratch.
**Razón:** Solo 300 manzanas etiquetadas, DINOv2 generaliza bien con few-shot.
**Consecuencias:** Dependencia de modelo externo, pero excelente calidad de features.

### ADR-003: Cascada ECC → LoFTR → SAM
**Decisión:** Pipeline de alineación con fallback progresivo.
**Razón:** Diferentes casos requieren diferentes algoritmos, optimizar tiempo/calidad.
**Consecuencias:** Mayor complejidad de código, pero mejor robustez.

### ADR-004: XGBoost + LightGBM Ensemble
**Decisión:** Ensemble de gradient boosting en lugar de red neuronal end-to-end.
**Razón:** Mejor interpretabilidad, funciona bien con features tabulares mixtas.
**Consecuencias:** Más fácil de depurar y explicar, SHAP nativo.

### ADR-005: Poetry + DVC + MLflow Stack
**Decisión:** Usar Poetry para deps, DVC para datos, MLflow para experimentos.
**Razón:** Stack moderno, reproducible, cada herramienta hace una cosa bien.
**Consecuencias:** Curva de aprendizaje, pero máxima reproducibilidad.

---

## 11. Clases de Discrepancias y Etiquetado

### 11.1 Taxonomía de Clases
```python
from enum import Enum

class DiscrepancyType(str, Enum):
    """Tipos de discrepancias detectables."""

    ALIGNED = "aligned"
    # Manzana correctamente alineada, sin cambios

    GEOMETRIC_MISALIGNMENT = "geometric_misalignment"
    # Vector desplazado, rotado o escalado respecto a realidad

    SEMANTIC_CHANGE = "semantic_change"
    # Construcciones nuevas, demoliciones, cambios de uso

    BOUNDARY_CHANGE = "boundary_change"
    # Modificación de linderos o subdivisión de parcelas

    REQUIRES_REVIEW = "requires_review"
    # Caso ambiguo que requiere revisión humana
```

### 11.2 Distribución Esperada
```
Total: 5,000 manzanas
├── aligned:                 ~60% (3,000)
├── geometric_misalignment:  ~15% (750)
├── semantic_change:         ~15% (750)
├── boundary_change:         ~5%  (250)
└── requires_review:         ~5%  (250)
```

---

## 12. Información de Contacto y Recursos

**Autor:** Arthur Zizumbo
**Fecha de Inicio:** Noviembre 2025

### Recursos Adicionales
- **Documentación Técnica:** `docs/architecture.md`
- **API Reference:** `docs/api_reference.md`
- **Guía de Contribución:** `CONTRIBUTING.md`
- **Changelog:** `CHANGELOG.md`

---

## 13. Checklist para Agentes de Código

Antes de implementar cualquier funcionalidad, verificar:

- [ ] ¿Cumple con el paradigma Local-First?
- [ ] ¿Respeta el límite de 8GB VRAM?
- [ ] ¿Tiene tests unitarios (coverage ≥ 80%)?
- [ ] ¿Usa type hints (mypy compatible)?
- [ ] ¿Sigue el estilo de Ruff?
- [ ] ¿Está documentado con docstrings?
- [ ] ¿Los datos se versionan con DVC?
- [ ] ¿Los experimentos se trackean con MLflow?
- [ ] ¿El código es reproducible con Poetry?
- [ ] ¿Se maneja el error gracefully con logs?

---

**Versión del Documento:** 2.0
**Última Actualización:** Noviembre 2025
**Para uso de:** Agentes de código IA en iteraciones de desarrollo
