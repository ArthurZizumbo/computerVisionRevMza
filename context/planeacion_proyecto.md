# Plan de Proyecto: Sistema Híbrido de Validación Geoespacial con Visión Computacional y Machine Learning

**Sistema Inteligente para Detección de Discrepancias Cartográficas mediante CV + ML**

---

**Desarrollador Principal:**
- **Arthur Zizumbo** - Full Stack Developer, ML Engineer, MLOps/SRE (Rol Único)

**Detalles del Proyecto:**
- **Duración:** 6 semanas (3 sprints de 2 semanas)
- **Dedicación:** 40 horas/semana
- **Metodología:** Scrum adaptado para desarrollo unipersonal con MLOps
- **Presupuesto GCP:** $300 USD total
- **Hardware Local:** NVIDIA RTX 4070 Laptop (8GB VRAM), 22 núcleos CPU
- **Versión del Documento:** 1.0
- **Fecha:** Noviembre 2025

---

## Tabla de Contenidos

- [Plan de Proyecto: Sistema Híbrido de Validación Geoespacial con Visión Computacional y Machine Learning](#plan-de-proyecto-sistema-híbrido-de-validación-geoespacial-con-visión-computacional-y-machine-learning)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [1. Resumen Ejecutivo](#1-resumen-ejecutivo)
    - [Problemática Central](#problemática-central)
    - [Solución Propuesta: Pipeline Híbrido CV + ML](#solución-propuesta-pipeline-híbrido-cv--ml)
    - [Innovación Técnica](#innovación-técnica)
    - [Estrategia de Ejecución](#estrategia-de-ejecución)
    - [Resultados Esperados](#resultados-esperados)
    - [Impacto Operativo](#impacto-operativo)
  - [2. Introducción y Contexto del Problema](#2-introducción-y-contexto-del-problema)
    - [2.1 La Crisis de la "Disonancia Cartográfica"](#21-la-crisis-de-la-disonancia-cartográfica)
      - [2.1.1 Naturaleza del Problema de Registro Vector-Raster](#211-naturaleza-del-problema-de-registro-vector-raster)
      - [2.1.2 Dimensión del Problema](#212-dimensión-del-problema)
    - [2.2 Limitaciones de Enfoques Tradicionales](#22-limitaciones-de-enfoques-tradicionales)
      - [2.2.1 Validación Manual: Inviable a Escala](#221-validación-manual-inviable-a-escala)
      - [2.2.2 Modelos de ML "Naive": Aprenden el Ruido](#222-modelos-de-ml-naive-aprenden-el-ruido)
    - [2.3 Justificación de la Solución Híbrida](#23-justificación-de-la-solución-híbrida)
  - [3. Objetivos del Proyecto](#3-objetivos-del-proyecto)
    - [3.1 Objetivo General](#31-objetivo-general)
    - [3.2 Objetivos Específicos](#32-objetivos-específicos)
      - [Técnicos - Fase 1: Ingeniería de Datos](#técnicos---fase-1-ingeniería-de-datos)
      - [Técnicos - Fase 2: Visión Computacional](#técnicos---fase-2-visión-computacional)
      - [Técnicos - Fase 3: Machine Learning](#técnicos---fase-3-machine-learning)
      - [Técnicos - Fase 4: Deployment y Operacionalización](#técnicos---fase-4-deployment-y-operacionalización)
      - [MLOps y Calidad](#mlops-y-calidad)
      - [Financieros y Operativos](#financieros-y-operativos)
    - [3.3 Criterios de Éxito](#33-criterios-de-éxito)
  - [4. Marco Teórico y Estado del Arte (2022-2025)](#4-marco-teórico-y-estado-del-arte-2022-2025)
    - [4.1 Alineación Geométrica y Registro de Imágenes](#41-alineación-geométrica-y-registro-de-imágenes)
      - [4.1.1 Métodos Clásicos: Vigencia y Limitaciones](#411-métodos-clásicos-vigencia-y-limitaciones)
      - [4.1.2 Deep Learning para Registro: El Salto Cualitativo](#412-deep-learning-para-registro-el-salto-cualitativo)
      - [4.1.3 Estrategia Recomendada: Cascada Adaptativa](#413-estrategia-recomendada-cascada-adaptativa)
    - [4.2 Detección de Cambios y Clasificación con Pocos Datos](#42-detección-de-cambios-y-clasificación-con-pocos-datos)
      - [4.2.1 El Desafío del "Few-Shot Learning"](#421-el-desafío-del-few-shot-learning)
      - [4.2.2 Modelos de Ensamblaje para Datos Tabulares](#422-modelos-de-ensamblaje-para-datos-tabulares)
    - [4.3 Métricas Geométricas: Fundamentos Matemáticos](#43-métricas-geométricas-fundamentos-matemáticos)
      - [4.3.1 Intersection over Union (IoU)](#431-intersection-over-union-iou)
      - [4.3.2 Distancia de Hausdorff](#432-distancia-de-hausdorff)
      - [4.3.3 Diferencia Simétrica de Área](#433-diferencia-simétrica-de-área)
    - [4.4 Arquitecturas de Deployment: Local-First MLOps](#44-arquitecturas-de-deployment-local-first-mlops)
      - [4.4.1 El Paradigma "Edge MLOps"](#441-el-paradigma-edge-mlops)
      - [4.4.2 DVC + MLflow: Reproducibilidad sin Vendor Lock-In](#442-dvc--mlflow-reproducibilidad-sin-vendor-lock-in)
    - [4.5 IA Generativa para Interfaces Conversacionales](#45-ia-generativa-para-interfaces-conversacionales)
      - [4.5.1 LLMs Locales: Ollama + Llama 3.2](#451-llms-locales-ollama--llama-32)
      - [4.5.2 Prompt Engineering para Análisis Geoespacial](#452-prompt-engineering-para-análisis-geoespacial)
  - [5. Stack Tecnológico](#5-stack-tecnológico)
    - [5.1 Lenguajes y Frameworks Core](#51-lenguajes-y-frameworks-core)
    - [5.2 Procesamiento Geoespacial](#52-procesamiento-geoespacial)
    - [5.3 Visión Computacional](#53-visión-computacional)
    - [5.4 Machine Learning](#54-machine-learning)
    - [5.5 MLOps](#55-mlops)
    - [5.6 Backend y API](#56-backend-y-api)
    - [5.7 Frontend](#57-frontend)
    - [5.8 IA Generativa](#58-ia-generativa)
    - [5.9 Google Cloud Platform](#59-google-cloud-platform)
  - [6. Arquitectura de la Solución](#6-arquitectura-de-la-solución)
    - [6.1 Visión General](#61-visión-general)
    - [6.2 Flujo de Datos End-to-End](#62-flujo-de-datos-end-to-end)
    - [6.3 Decisiones Arquitectónicas Clave](#63-decisiones-arquitectónicas-clave)
      - [6.3.1 ¿Por qué Dagster y no Airflow?](#631-por-qué-dagster-y-no-airflow)
      - [6.3.2 ¿Por qué DuckDB y no PostgreSQL?](#632-por-qué-duckdb-y-no-postgresql)
      - [6.3.3 ¿Por qué XGBoost y no una CNN end-to-end?](#633-por-qué-xgboost-y-no-una-cnn-end-to-end)
  - [7. Metodología: Scrum Adaptado Unipersonal](#7-metodología-scrum-adaptado-unipersonal)
    - [7.1 Adaptaciones para Desarrollo Individual](#71-adaptaciones-para-desarrollo-individual)
    - [7.2 Definición de "Done"](#72-definición-de-done)
    - [7.3 Estimación de Esfuerzo](#73-estimación-de-esfuerzo)
  - [8. Plan de Ejecución: Sprint 1 - Fundamentos y Adquisición de Datos](#8-plan-de-ejecución-sprint-1---fundamentos-y-adquisición-de-datos)
    - [8.1 Épica 1: Infraestructura y Setup (8 puntos)](#81-épica-1-infraestructura-y-setup-8-puntos)
      - [US-001: Configuración del Entorno Local Robusto (3 puntos)](#us-001-configuración-del-entorno-local-robusto-3-puntos)
      - [US-002: Inicialización de DVC y GCS (2 puntos)](#us-002-inicialización-de-dvc-y-gcs-2-puntos)
      - [US-003: Configuración de MLflow Tracking (3 puntos)](#us-003-configuración-de-mlflow-tracking-3-puntos)
    - [8.2 Épica 2: Adquisición de Datos (12 puntos)](#82-épica-2-adquisición-de-datos-12-puntos)
      - [US-004: Pipeline de Descarga Inteligente con Google Maps API (5 puntos)](#us-004-pipeline-de-descarga-inteligente-con-google-maps-api-5-puntos)
      - [US-005: Rasterización de Geometrías Vectoriales (4 puntos)](#us-005-rasterización-de-geometrías-vectoriales-4-puntos)
      - [US-006: Cálculo de Métricas Geométricas Base (3 puntos)](#us-006-cálculo-de-métricas-geométricas-base-3-puntos)
    - [8.3 Épica 3: Motor de Alineación CV (12 puntos)](#83-épica-3-motor-de-alineación-cv-12-puntos)
      - [US-007: Implementación de Alineación con ECC (5 puntos)](#us-007-implementación-de-alineación-con-ecc-5-puntos)
      - [US-008: Implementación de Alineación con LoFTR (5 puntos)](#us-008-implementación-de-alineación-con-loftr-5-puntos)
      - [US-009: Pipeline de Alineación en Cascada (2 puntos)](#us-009-pipeline-de-alineación-en-cascada-2-puntos)
      - [US-0010: Validación de Integridad Vectorial con SAM (5 puntos)](#us-0010-validación-de-integridad-vectorial-con-sam-5-puntos)
  - [9. Plan de Ejecución: Sprint 2 - Machine Learning y Clasificación](#9-plan-de-ejecución-sprint-2---machine-learning-y-clasificación)
    - [9.1 Épica 4: Feature Engineering (10 puntos)](#91-épica-4-feature-engineering-10-puntos)
      - [US-011: Extracción de Embeddings con DINOv2 (5 puntos)](#us-011-extracción-de-embeddings-con-dinov2-5-puntos)
      - [US-012: Cálculo de Métricas Post-Alineación (3 puntos)](#us-012-cálculo-de-métricas-post-alineación-3-puntos)
      - [US-013: Construcción del Dataset de Entrenamiento (2 puntos)](#us-013-construcción-del-dataset-de-entrenamiento-2-puntos)
    - [9.2 Épica 5: Modelado y Entrenamiento (15 puntos)](#92-épica-5-modelado-y-entrenamiento-15-puntos)
      - [US-014: Entrenamiento de Baseline con XGBoost (5 puntos)](#us-014-entrenamiento-de-baseline-con-xgboost-5-puntos)
      - [US-015: Entrenamiento de Modelo Alternativo con LightGBM (3 puntos)](#us-015-entrenamiento-de-modelo-alternativo-con-lightgbm-3-puntos)
      - [US-016: Ensemble y Calibración (4 puntos)](#us-016-ensemble-y-calibración-4-puntos)
      - [US-017: Explicabilidad con SHAP (3 puntos)](#us-017-explicabilidad-con-shap-3-puntos)
    - [9.3 Épica 6: Orquestación con Dagster (7 puntos)](#93-épica-6-orquestación-con-dagster-7-puntos)
      - [US-018: Definición de Assets en Dagster (4 puntos)](#us-018-definición-de-assets-en-dagster-4-puntos)
      - [US-019: Configuración de Schedules y Sensors (3 puntos)](#us-019-configuración-de-schedules-y-sensors-3-puntos)
      - [US-020: Loop de Active Learning y Uncertainty Sampling (5 puntos)](#us-020-loop-de-active-learning-y-uncertainty-sampling-5-puntos)
  - [10. Plan de Ejecución: Sprint 3 - Deployment y Visualización](#10-plan-de-ejecución-sprint-3---deployment-y-visualización)
    - [10.1 Épica 7: Backend API (10 puntos)](#101-épica-7-backend-api-10-puntos)
      - [US-021: Desarrollo de API con FastAPI (5 puntos)](#us-021-desarrollo-de-api-con-fastapi-5-puntos)
      - [US-022: Testing y Optimización de API (3 puntos)](#us-022-testing-y-optimización-de-api-3-puntos)
      - [US-023: Containerización de API (2 puntos)](#us-023-containerización-de-api-2-puntos)
    - [10.2 Épica 8: Frontend Geoespacial (12 puntos)](#102-épica-8-frontend-geoespacial-12-puntos)
      - [US-024: Setup de Proyecto Nuxt 4 (2 puntos)](#us-024-setup-de-proyecto-nuxt-4-2-puntos)
      - [US-025: Visualización de Mapa Interactivo (5 puntos)](#us-025-visualización-de-mapa-interactivo-5-puntos)
      - [US-026: Panel de Métricas y Filtros (3 puntos)](#us-026-panel-de-métricas-y-filtros-3-puntos)
      - [US-027: Integración con API Backend (2 puntos)](#us-027-integración-con-api-backend-2-puntos)
    - [10.3 Épica 9: Copiloto Conversacional (6 puntos)](#103-épica-9-copiloto-conversacional-6-puntos)
      - [US-028: Integración de Ollama con Llama 3.2 (3 puntos)](#us-028-integración-de-ollama-con-llama-32-3-puntos)
      - [US-029: Interfaz de Chat en Frontend (3 puntos)](#us-029-interfaz-de-chat-en-frontend-3-puntos)
    - [10.4 Épica 10: Deployment en GCP (4 puntos)](#104-épica-10-deployment-en-gcp-4-puntos)
      - [US-030: Deployment de API en Cloud Run (2 puntos)](#us-030-deployment-de-api-en-cloud-run-2-puntos)
      - [US-031: Deployment de Frontend en Cloud Run (2 puntos)](#us-031-deployment-de-frontend-en-cloud-run-2-puntos)
  - [11. Roles y Responsabilidades](#11-roles-y-responsabilidades)
  - [12. Gestión de Riesgos](#12-gestión-de-riesgos)
    - [12.1 Matriz de Riesgos](#121-matriz-de-riesgos)
    - [12.2 Plan de Contingencia](#122-plan-de-contingencia)
  - [13. Análisis Financiero y FinOps](#13-análisis-financiero-y-finops)
    - [13.1 Desglose Detallado de Costos GCP](#131-desglose-detallado-de-costos-gcp)
    - [13.2 Estrategias de Optimización de Costos](#132-estrategias-de-optimización-de-costos)
  - [14. Plan de Entregables](#14-plan-de-entregables)
    - [14.1 Entregables por Sprint](#141-entregables-por-sprint)
    - [14.2 Documentación Final](#142-documentación-final)
  - [15. Conclusiones](#15-conclusiones)
    - [15.1 Viabilidad Técnica](#151-viabilidad-técnica)
    - [15.2 Viabilidad Económica](#152-viabilidad-económica)
    - [15.3 Viabilidad Operativa](#153-viabilidad-operativa)
    - [15.4 Impacto Esperado](#154-impacto-esperado)
    - [15.5 Recomendaciones Finales](#155-recomendaciones-finales)
    - [15.6 Próximos Pasos Inmediatos](#156-próximos-pasos-inmediatos)
  - [16. Referencias](#16-referencias)
    - [Artículos Científicos y Papers](#artículos-científicos-y-papers)
    - [Documentación Técnica y Recursos](#documentación-técnica-y-recursos)

---

## 1. Resumen Ejecutivo


El presente documento constituye un plan de ejecución técnica integral para resolver la problemática crítica de **discrepancia geoespacial** identificada en la Dirección de Cartografía, donde la actualización paralela de dos bases vectoriales catastrales ha resultado en inconsistencias geométricas y semánticas que afectan aproximadamente **5,000 manzanas urbanas**.

### Problemática Central

La situación presenta un desafío triple:

1. **Desfase Geométrico:** Las geometrías vectoriales no coinciden espacialmente con la realidad física observable en imágenes satelitales (Google Maps), presentando desplazamientos que pueden alcanzar decenas de metros.

2. **Cambios Semánticos:** Existen modificaciones reales en el territorio (construcciones, demoliciones, cambios de uso de suelo) que deben identificarse y clasificarse.

3. **Datos Limitados:** Solo se dispone de 300 muestras etiquetadas manualmente de un universo de 5,000 casos, configurando un escenario de "Few-Shot Learning" extremo.

### Solución Propuesta: Pipeline Híbrido CV + ML

Basándose en investigación reciente (2022-2025) en visión computacional geoespacial y aprendizaje automático con datos limitados, se valida y refina la arquitectura de tres fases propuesta:

**Fase 1 - Adquisición de Datos:**
- Script Python + Google Maps Static API para generar imágenes comparativas
- Sistema de caché inteligente para optimizar costos (< $15 USD en API calls)
- Rasterización de geometrías vectoriales con georreferenciación precisa

**Fase 2 - Corrección Geométrica (Computer Vision):**
- Alineación automática mediante algoritmos clásicos (ECC) y modernos (LoFTR)
- Detección de desfases severos que invalidan el análisis semántico
- Corrección de transformaciones afines y no lineales

**Fase 3 - Clasificación Semántica (Machine Learning):**
- Transfer Learning con modelos pre-entrenados (DINOv2, ResNet50)
- Clasificación híbrida: embeddings visuales + 7 métricas geométricas
- Modelo ensemble (XGBoost + LightGBM) optimizado para pocos datos

### Innovación Técnica

A diferencia de enfoques tradicionales que intentan resolver ambos problemas simultáneamente (resultando en modelos que aprenden el ruido posicional), esta arquitectura **desacoplada** trata el desfase geométrico como un problema determinista de pre-procesamiento y la detección de cambios como un problema estocástico de clasificación, maximizando la precisión de ambos componentes.

### Estrategia de Ejecución

El proyecto aprovecha una arquitectura **"Local-First, Cloud-Deploy"** que ejecuta todo el cómputo intensivo (entrenamiento de modelos, procesamiento de imágenes, inferencia con LLMs) en la GPU RTX 4070 local, reservando Google Cloud Platform exclusivamente para:
- Almacenamiento de artefactos (DVC remote en Cloud Storage)
- Deployment de la aplicación web final (Cloud Run con scale-to-zero)
- Servicios ligeros de orquestación

Esta estrategia mantiene el presupuesto de $300 USD con un margen de seguridad del 85%, permitiendo experimentación sin restricciones financieras.

### Resultados Esperados

| Métrica | Objetivo | Justificación |
|---------|----------|---------------|
| **Precisión de Alineación** | IoU > 0.75 post-corrección | Estándar catastral internacional |
| **Recall de Detección de Desfases** | > 95% | Crítico: falsos negativos invalidan análisis ML |
| **F1-Score Clasificación ML** | > 0.80 | Supera baseline de clasificación manual |
| **Throughput del Sistema** | 5,000 manzanas en < 8 horas | Viable para producción operativa |
| **Costo Total GCP** | < $50 USD | 83% bajo presupuesto, margen para escalado |
| **Reproducibilidad** | 100% con DVC + Docker | Auditable y replicable |

### Impacto Operativo

El sistema final permitirá a la Dirección de Cartografía:
- Reducir el tiempo de validación manual de 5,000 manzanas de ~3 meses a ~1 semana
- Identificar automáticamente casos que requieren revisión humana (Human-in-the-Loop)
- Generar reportes geoespaciales interactivos con explicabilidad de decisiones
- Establecer un pipeline reproducible para futuras actualizaciones cartográficas

---

## 2. Introducción y Contexto del Problema

### 2.1 La Crisis de la "Disonancia Cartográfica"

La actualización cartográfica es un proceso crítico para instituciones gubernamentales que gestionan catastros, planificación urbana y administración territorial. En el caso específico de la Dirección de Cartografía, la decisión de actualizar dos bases geoespaciales en paralelo —motivada por la necesidad de mejorar la calidad cartográfica en plazos ajustados— ha resultado en una pérdida de integridad referencial espacial.


#### 2.1.1 Naturaleza del Problema de Registro Vector-Raster

El problema técnico subyacente se conoce en la literatura geoespacial como **"Vector-Raster Misalignment"** o problema de registro multimodal. Investigaciones recientes demuestran que este fenómeno no es uniforme ni predecible:

- **Errores Locales No Lineales:** A diferencia de un simple desplazamiento global (que podría corregirse con una traslación), los errores varían espacialmente debido a digitalizaciones históricas heterogéneas, diferencias en sistemas de coordenadas (ITRF vs WGS84), y distorsiones topográficas acumuladas (Chen et al., 2023).

- **Dependencia del Contexto Urbano:** Zonas densamente urbanizadas presentan mayor complejidad de alineación debido a oclusiones (sombras de edificios), cambios temporales (construcciones nuevas), y variabilidad en la calidad de digitalización (Hughes et al., 2024).

- **Impacto en Análisis Downstream:** Estudios de teledetección urbana confirman que intentar entrenar modelos de detección de cambios sobre datos no alineados resulta en tasas de falsos positivos superiores al 40%, haciendo el sistema inútil para producción (Zhang et al., 2023).

#### 2.1.2 Dimensión del Problema

**Universo de Datos:**
- 5,000 manzanas urbanas afectadas
- Área geográfica: zona metropolitana con densidad variable (urbana, suburbana, periurbana)
- Periodo de actualización: 2023-2024
- Sistemas de referencia: múltiples (requiere homologación)

**Datos Etiquetados Disponibles:**
- 300 manzanas clasificadas manualmente por equipo de cartografía
- Taxonomía de clasificación:
  - **Bandera:** NO CAMBIAR (geometría correcta), CAMBIAR (requiere actualización), VERIFICAR (caso ambiguo)
  - **Tipo de Cambio:** TOTAL (reemplazo completo), PARCIAL (ajuste menor), SIN CAMBIO

**Métricas Geométricas Calculadas (7 dimensiones):**

1. **dif_simetrica_area (m²):** Área de la diferencia simétrica entre polígonos (A ∪ B - A ∩ B)
2. **distancia_centroides (m):** Distancia euclidiana entre centroides geométricos
3. **iou (Intersection over Union):** Ratio de superposición (0-1)
4. **distancia_hausdorff (m):** Máxima distancia mínima entre conjuntos de puntos
5. **dif_complejidad:** Diferencia en número de vértices
6. **dif_compacidad:** Cambio en circularidad (4π × Area/Perimetro²)
7. **cambio_area_pct (%):** Diferencia porcentual de área

Estas métricas, derivadas de la geometría computacional y ampliamente utilizadas en evaluación de segmentación semántica (Rezatofighi et al., 2019), constituyen el espacio de características para el modelo de clasificación.

### 2.2 Limitaciones de Enfoques Tradicionales

#### 2.2.1 Validación Manual: Inviable a Escala

El proceso actual de validación manual requiere que un cartógrafo:
1. Cargue la geometría en un SIG (QGIS, ArcGIS)
2. Superponga capas de imágenes satelitales
3. Evalúe visualmente la coincidencia
4. Registre la decisión en una hoja de cálculo

**Tiempo estimado:** 15-20 minutos por manzana
**Costo temporal para 5,000 manzanas:** 1,250-1,667 horas (≈ 7-9 meses de trabajo de una persona)

#### 2.2.2 Modelos de ML "Naive": Aprenden el Ruido

Intentos previos de aplicar modelos de clasificación directamente sobre las métricas geométricas sin corregir el desfase han resultado en:
- Modelos que clasifican como "CAMBIAR" casos donde solo existe desplazamiento posicional
- Incapacidad para distinguir entre error cartográfico y cambio real del territorio
- Overfitting severo debido al dataset pequeño (300 muestras)

### 2.3 Justificación de la Solución Híbrida

La arquitectura propuesta de tres fases se fundamenta en principios establecidos de visión computacional y aprendizaje automático:

**Principio 1: Separación de Preocupaciones**
La alineación geométrica es un problema de optimización determinista (encontrar la transformación que maximiza la correlación entre imágenes), mientras que la detección de cambios es un problema de inferencia estocástica (clasificar patrones visuales). Mezclar ambos en un solo modelo introduce confusión de objetivos (Goodfellow et al., 2016).

**Principio 2: Pre-procesamiento Inteligente**
En visión computacional, la calidad del pre-procesamiento determina el límite superior de performance del modelo downstream. Un modelo SOTA entrenado sobre datos mal alineados nunca superará a un modelo simple entrenado sobre datos correctamente alineados (He et al., 2022).

**Principio 3: Transfer Learning para Few-Shot Scenarios**
Con solo 300 muestras, entrenar redes neuronales profundas desde cero es inviable. La estrategia validada es utilizar modelos pre-entrenados en datasets masivos (ImageNet, COCO) y realizar fine-tuning o extracción de características (Zhai et al., 2022).

**Principio 4: Adopción de Foundation Models Geoespaciales (2023-2025)** La arquitectura integra modelos fundacionales de última generación para mitigar la escasez de datos:

- DINOv2 (Meta AI): Utilizado para la extracción de características visuales robustas ("zero-shot") sin necesidad de fine-tuning costoso.

- SAM (Segment Anything Model): Empleado como oráculo de validación para generar máscaras de segmentación "ground truth" y verificar la calidad de los vectores catastrales existentes.

- Estrategia Edge-First: Todo el cómputo pesado de estos modelos se ejecuta localmente (RTX 4070), eliminando la dependencia de GPUs en la nube.
---

## 3. Objetivos del Proyecto

### 3.1 Objetivo General

Desarrollar e implementar un sistema automatizado de validación cartográfica que combine técnicas de visión computacional para corrección geométrica y modelos de machine learning para clasificación semántica, permitiendo procesar 5,000 manzanas urbanas con precisión superior al 80% en clasificación y costo computacional inferior a $50 USD en infraestructura cloud.

### 3.2 Objetivos Específicos

#### Técnicos - Fase 1: Ingeniería de Datos

1. **OE-1.1:** Implementar un pipeline de adquisición de imágenes satelitales mediante Google Maps Static API con sistema de caché persistente que garantice costo < $20 USD para 5,000 solicitudes.

2. **OE-1.2:** Desarrollar módulo de rasterización de geometrías vectoriales (GeoJSON/Shapefile) a máscaras binarias con preservación de georreferenciación y resolución espacial consistente.

3. **OE-1.3:** Establecer infraestructura de versionado de datos con DVC + Google Cloud Storage, garantizando reproducibilidad del 100% de los experimentos.

#### Técnicos - Fase 2: Visión Computacional

4. **OE-2.1:** Implementar pipeline de alineación geométrica en cascada:
   - Nivel 1: Enhanced Correlation Coefficient (ECC) para casos simples
   - Nivel 2: LoFTR (Local Feature Transformer) para casos complejos
   - Nivel 3: Marcado de casos irrecuperables para revisión manual

5. **OE-2.2:** Alcanzar métricas de alineación:
   - IoU promedio post-corrección > 0.75
   - Recall de detección de desfases severos > 95%
   - Reducción de distancia_centroides en > 70% de los casos

6. **OE-2.3:** Generar dataset de imágenes alineadas con metadatos de confianza de alineación para filtrado en fase ML.

#### Técnicos - Fase 3: Machine Learning

7. **OE-3.1:** Extraer embeddings visuales de imágenes alineadas usando modelos pre-entrenados:
   - DINOv2 (Meta AI, 2023) para características visuales robustas
   - ResNet50 (baseline) para comparación

8. **OE-3.2:** Entrenar modelo de clasificación híbrido que combine:
   - Embeddings visuales (512-1024 dimensiones)
   - 7 métricas geométricas calculadas
   - Target: clasificación en 3 clases (NO CAMBIAR, CAMBIAR, VERIFICAR)

9. **OE-3.3:** Alcanzar métricas de clasificación:
   - F1-Score macro > 0.80
   - Precision en clase "CAMBIAR" > 0.85 (minimizar falsos positivos)
   - Recall en clase "VERIFICAR" > 0.90 (capturar casos ambiguos)

#### Técnicos - Fase 4: Deployment y Operacionalización

10. **OE-4.1:** Desarrollar API REST con FastAPI que exponga endpoints:
    - `/predict`: Clasificación de manzana individual
    - `/batch`: Procesamiento batch de múltiples manzanas
    - `/explain`: Explicabilidad de decisiones (SHAP values)

11. **OE-4.2:** Implementar interfaz web interactiva (Nuxt 4 + MapLibre) con:
    - Visualización geoespacial de resultados
    - Panel de métricas por manzana
    - Filtros por tipo de clasificación y nivel de confianza

12. **OE-4.3:** Integrar copiloto conversacional con Ollama (Llama 3.2) para consultas en lenguaje natural sobre resultados.

#### MLOps y Calidad

13. **OE-5.1:** Establecer pipeline de CI/CD con GitHub Actions:
    - Testing automatizado (pytest, coverage > 70%)
    - Linting y formateo (Ruff, Black)
    - Build y push de imágenes Docker

14. **OE-5.2:** Implementar monitoreo de drift con Evidently AI:
    - Data drift en features de entrada
    - Prediction drift en distribución de clasificaciones
    - Alertas automáticas ante degradación

15. **OE-5.3:** Documentar arquitectura y decisiones técnicas en formato académico con diagramas de arquitectura (Mermaid.js, C4 model).

#### Financieros y Operativos

16. **OE-6.1:** Mantener gasto total en GCP < $50 USD mediante:
    - Ejecución local de entrenamiento e inferencia pesada
    - Scale-to-zero en Cloud Run
    - Lifecycle policies en Cloud Storage

17. **OE-6.2:** Lograr throughput de procesamiento:
    - 5,000 manzanas procesadas en < 8 horas (incluyendo descarga, alineación, clasificación)
    - Latencia de API < 500ms p95 para predicción individual

### 3.3 Criterios de Éxito

El proyecto será considerado exitoso si cumple con los siguientes criterios cuantificables:

| Dimensión | Métrica | Meta | Método de Medición |
|-----------|---------|------|-------------------|
| **Precisión CV** | IoU post-alineación | > 0.75 | Promedio sobre conjunto de validación (n=100) |
| **Robustez CV** | Recall desfases severos | > 95% | Detección de casos con distancia_centroides > 20m |
| **Precisión ML** | F1-Score macro | > 0.80 | Validación cruzada 5-fold sobre 300 muestras |
| **Precisión ML** | Precision clase CAMBIAR | > 0.85 | Matriz de confusión en test set |
| **Eficiencia** | Throughput total | 5,000 en < 8h | Benchmark end-to-end en hardware local |
| **Costo** | Gasto GCP | < $50 USD | Facturación GCP al cierre del proyecto |
| **Calidad Código** | Cobertura tests | > 70% | pytest-cov report |
| **Reproducibilidad** | Recreación resultados | 100% | DVC repro + Docker build exitosos |
| **Documentación** | Completitud técnica | 15/15 secciones | Checklist de documentación |

---

## 4. Marco Teórico y Estado del Arte (2022-2025)

Esta sección presenta una revisión exhaustiva de la literatura científica y técnica reciente que fundamenta las decisiones arquitectónicas del proyecto. Se han consultado 20 fuentes de alto impacto publicadas entre 2022-2025.

### 4.1 Alineación Geométrica y Registro de Imágenes

#### 4.1.1 Métodos Clásicos: Vigencia y Limitaciones

**Enhanced Correlation Coefficient (ECC)**

El algoritmo ECC, propuesto originalmente por Evangelidis y Psarakis (2008) y refinado en implementaciones modernas de OpenCV, sigue siendo el método de referencia para alineación de imágenes con transformaciones afines simples. Estudios recientes confirman su eficacia:

- **Performance:** Chen et al. (2023) demostraron que ECC alcanza precisión sub-píxel (< 0.5 px error) en el 78% de casos de alineación urbana cuando existe suficiente contraste de bordes.
- **Limitaciones:** Falla en escenarios con cambios de iluminación drásticos, oclusiones por vegetación, o cuando la transformación requerida es no-lineal (Hughes et al., 2024).

**SIFT y ORB: Feature Matching Tradicional**

Los detectores de características SIFT (Scale-Invariant Feature Transform) y ORB (Oriented FAST and Rotated BRIEF) han sido ampliamente utilizados en fotogrametría:

- **Ventajas:** No requieren entrenamiento, son rápidos (ORB procesa imágenes 1024x1024 en < 100ms), y funcionan bien en escenas con texturas ricas (Lowe, 2004; Rublee et al., 2011).
- **Desventajas:** Rendimiento degradado en imágenes con bajo contraste o patrones repetitivos (edificios modulares), común en cartografía urbana (DeTone et al., 2018).

#### 4.1.2 Deep Learning para Registro: El Salto Cualitativo

**LoFTR: Local Feature Transformer**

LoFTR, presentado en CVPR 2021 y ampliamente adoptado en 2023-2024, representa un cambio de paradigma en el matching de características:

- **Arquitectura:** Utiliza Transformers con mecanismos de atención cruzada para establecer correspondencias densas entre imágenes, incluso cuando la apariencia visual difiere significativamente (Sun et al., 2021).
- **Performance en Geoespacial:** Wang et al. (2024) evaluaron LoFTR en el dataset "Urban Change Detection" y reportaron una mejora del 34% en recall de correspondencias correctas vs SIFT en zonas con sombras de edificios.
- **Costo Computacional:** Requiere GPU (RTX 3060 o superior) pero procesa pares de imágenes 640x480 en ~200ms, viable para producción (implementación Kornia).

**SuperGlue: Graph Neural Networks para Matching**

SuperGlue combina detección de características (SuperPoint) con un Graph Neural Network para resolver el problema de asignación óptima:

- **Innovación:** Modela el matching como un problema de optimización en grafos, permitiendo rechazar correspondencias ambiguas (Sarlin et al., 2020).
- **Aplicación Catastral:** Li et al. (2023) lo aplicaron a alineación de mapas catastrales históricos con imágenes satelitales modernas, logrando 89% de precisión en casos donde métodos clásicos fallaban completamente.

#### 4.1.3 Estrategia Recomendada: Cascada Adaptativa

Basándose en la literatura, la estrategia óptima para el proyecto es un **pipeline en cascada**:

1. **Nivel 1 - ECC (Rápido):** Intentar alineación con ECC. Si la métrica de confianza (correlation coefficient) > 0.7, aceptar resultado.
2. **Nivel 2 - LoFTR (Robusto):** Si ECC falla, escalar a LoFTR. Aceptar si número de correspondencias (inliers) > 50.
3. **Nivel 3 - Marcado Manual:** Si ambos fallan, marcar como "REQUIERE_REVISION_MANUAL" y excluir del entrenamiento ML.

Esta estrategia balancea velocidad (ECC procesa 80% de casos en < 1s) con robustez (LoFTR rescata 15% adicional), minimizando el uso de GPU.

### 4.2 Detección de Cambios y Clasificación con Pocos Datos

#### 4.2.1 El Desafío del "Few-Shot Learning"

Con solo 300 muestras etiquetadas, el proyecto enfrenta un escenario de **extreme low-resource learning**. La literatura reciente ofrece estrategias validadas:

**Transfer Learning: La Solución Estándar**

- **Fundamento:** Modelos pre-entrenados en datasets masivos (ImageNet: 14M imágenes) han aprendido representaciones visuales generalizables que se transfieren a tareas específicas (Zhai et al., 2022).
- **Evidencia Empírica:** Kornblith et al. (2019) demostraron que features de ResNet50 pre-entrenado superan a modelos entrenados desde cero incluso con 10,000 muestras en tareas de clasificación de imágenes.

**DINOv2: Self-Supervised Learning de Meta AI**

DINOv2, lanzado en 2023, representa el estado del arte en aprendizaje auto-supervisado:

- **Innovación:** Entrenado con 142M imágenes sin etiquetas usando destilación de conocimiento, genera embeddings que capturan semántica visual sin supervisión explícita (Oquab et al., 2023).
- **Performance Zero-Shot:** En el benchmark de segmentación semántica ADE20K, DINOv2 alcanza 47.1 mIoU sin fine-tuning, superando a modelos supervisados previos (Oquab et al., 2023).
- **Aplicación al Proyecto:** Extraer embeddings de 1024 dimensiones de la capa final de DINOv2-ViT-B/14 y usarlos como features para un clasificador ligero (XGBoost).

#### 4.2.2 Modelos de Ensamblaje para Datos Tabulares

Una vez extraídos los embeddings visuales y calculadas las métricas geométricas, el problema se reduce a clasificación sobre datos tabulares de alta dimensión (1024 + 7 = 1031 features).

**XGBoost y LightGBM: Dominancia en Tabular Data**

- **Evidencia:** En competencias de Kaggle 2022-2024, modelos basados en gradient boosting (XGBoost, LightGBM, CatBoost) dominan el 80% de soluciones ganadoras en problemas tabulares (Kaggle, 2024).
- **Ventajas para Few-Shot:**
  - Robustos ante overfitting con regularización L1/L2
  - Manejan features de alta dimensión sin reducción de dimensionalidad
  - Interpretables mediante SHAP values (Lundberg & Lee, 2017)

**Estrategia de Validación:**

- **Cross-Validation Temporal:** Dado que las manzanas tienen componente espacial, usar GroupKFold con agrupación por zona geográfica para evitar data leakage (Bergstra & Bengio, 2012).
- **Hyperparameter Optimization:** Utilizar Optuna con 100 trials para optimizar learning_rate, max_depth, n_estimators (Akiba et al., 2019).

### 4.3 Métricas Geométricas: Fundamentos Matemáticos

Las 7 métricas calculadas tienen bases teóricas sólidas en geometría computacional:

#### 4.3.1 Intersection over Union (IoU)

- **Definición:** IoU = |A ∩ B| / |A ∪ B|
- **Propiedades:** Métrica de Jaccard, rango [0,1], invariante a escala
- **Uso en CV:** Métrica estándar para evaluación de segmentación semántica desde el desafío PASCAL VOC (Everingham et al., 2010)
- **Umbral Catastral:** Estudios de cartografía digital establecen IoU > 0.75 como aceptable para actualización catastral (Crommelinck et al., 2019)

#### 4.3.2 Distancia de Hausdorff

- **Definición:** dH(A,B) = max{supₐ infᵦ d(a,b), supᵦ infₐ d(b,a)}
- **Interpretación:** Máxima distancia de un punto de un conjunto al punto más cercano del otro conjunto
- **Sensibilidad:** Penaliza fuertemente outliers (un solo vértice mal digitalizado eleva la métrica)
- **Aplicación:** Detección de errores topológicos graves (antenas, errores de digitalización) (Huttenlocher et al., 1993)

#### 4.3.3 Diferencia Simétrica de Área

- **Definición:** ΔA = |A ∪ B| - |A ∩ B| = |A \ B| + |B \ A|
- **Interpretación:** Área total no compartida entre polígonos
- **Relevancia Catastral:** Directamente relacionada con errores de valoración fiscal (área incorrecta = impuesto incorrecto)

### 4.4 Arquitecturas de Deployment: Local-First MLOps

#### 4.4.1 El Paradigma "Edge MLOps"

La tendencia reciente en MLOps para proyectos con restricciones presupuestarias es el **"Edge-First Development"**:

- **Motivación:** Costos de GPU en cloud (GCP: $2.48/hora para T4, $7.02/hora para V100) hacen inviable el entrenamiento cloud para proyectos académicos o startups (Google Cloud, 2024).
- **Solución:** Entrenar localmente en hardware consumer (RTX 4070: equivalente a Tesla T4) y desplegar solo la inferencia en cloud con CPUs (Rausch et al., 2023).

**Evidencia de Viabilidad:**

- **Capacidad RTX 4070:** 8GB VRAM suficiente para fine-tuning de modelos hasta 1B parámetros con técnicas de optimización (LoRA, quantization) (Dettmers et al., 2023).
- **Performance:** Entrenamiento de ResNet50 en ImageNet subset (100K imágenes): 6 horas en RTX 4070 vs 2 horas en V100 (costo: $0 vs $14) (NVIDIA, 2023).

#### 4.4.2 DVC + MLflow: Reproducibilidad sin Vendor Lock-In

- **DVC (Data Version Control):** Sistema de versionado para datos y modelos que usa Git como backend de metadatos y storage arbitrario (GCS, S3, local) para artefactos (Kuprieiev et al., 2023).
- **MLflow:** Plataforma open-source para tracking de experimentos, registro de modelos y deployment (Zaharia et al., 2018).
- **Integración:** DVC versiona datasets, MLflow versiona experimentos, ambos se referencian mediante hashes, garantizando reproducibilidad bit-a-bit (Paleyes et al., 2022).

### 4.5 IA Generativa para Interfaces Conversacionales

#### 4.5.1 LLMs Locales: Ollama + Llama 3.2

**Ollama:**
- **Descripción:** Runtime optimizado para ejecutar LLMs localmente con quantization automática (GGUF format) (Ollama Team, 2024).
- **Performance:** Llama 3.2 3B quantizado (Q4_K_M) corre en RTX 4070 con latencia < 50 tokens/segundo, suficiente para aplicaciones interactivas (Meta AI, 2024).

**Llama 3.2:**
- **Arquitectura:** Transformer decoder-only con 3B parámetros, entrenado en 15T tokens (Meta AI, 2024).
- **Capacidades:** Razonamiento sobre datos estructurados, generación de SQL, explicación de resultados numéricos (Touvron et al., 2023).

#### 4.5.2 Prompt Engineering para Análisis Geoespacial

Estrategia de prompting para el copiloto:

```
System: Eres un analista GIS experto. Tienes acceso a una base de datos con las siguientes tablas:
- manzanas: id, geometria, clasificacion, confianza, iou, distancia_centroides, ...
- metricas: id_manzana, dif_simetrica_area, distancia_hausdorff, ...

User: ¿Cuántas manzanas tienen IoU < 0.5 y fueron clasificadas como CAMBIAR?

Assistant: SELECT COUNT(*) FROM manzanas WHERE iou < 0.5 AND clasificacion = 'CAMBIAR';
```

Esta técnica, conocida como "Text-to-SQL", ha demostrado 85% de precisión en benchmarks como Spider con modelos de 7B+ parámetros (Rajkumar et al., 2022).

---

## 5. Stack Tecnológico

### 5.1 Lenguajes y Frameworks Core

| Tecnología | Versión | Propósito | Justificación |
|------------|---------|-----------|---------------|
| **Python** | 3.11+ | Lenguaje principal | Ecosistema ML/CV más maduro, type hints mejorados |
| **Poetry** | 1.8+ | Gestión de dependencias | Lock files determinísticos, superior a pip |
| **Docker** | 24+ | Containerización | Reproducibilidad, deployment consistente |

### 5.2 Procesamiento Geoespacial

| Tecnología | Versión | Propósito | Justificación |
|------------|---------|-----------|---------------|
| **GeoPandas** | 0.14+ | Manipulación de geometrías | Integración Pandas + Shapely |
| **Shapely** | 2.0+ | Operaciones geométricas | Cálculo de métricas (IoU, Hausdorff) |
| **Rasterio** | 1.3+ | I/O de rasters | Lectura de imágenes georreferenciadas |
| **Fiona** | 1.9+ | I/O de vectores | Lectura de Shapefiles/GeoJSON |
| **PyProj** | 3.6+ | Transformaciones de CRS | Homologación de sistemas de coordenadas |

### 5.3 Visión Computacional

| Tecnología | Versión | Propósito | Justificación |
|------------|---------|-----------|---------------|
| **OpenCV** | 4.9+ | Algoritmos CV clásicos | ECC, detección de bordes, transformaciones |
| **Kornia** | 0.7+ | CV con PyTorch | Implementación LoFTR, operaciones diferenciables |
| **Pillow** | 10+ | Manipulación de imágenes | I/O, redimensionamiento |
| **scikit-image** | 0.22+ | Procesamiento de imágenes | Filtros, métricas de calidad |


### 5.4 Machine Learning

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **PyTorch** | 2.2+ | Framework DL, backend de modelos pre-entrenados |
| **Transformers (HuggingFace)** | 4.38+ | Carga de DINOv2, ViT |
| **Timm** | 0.9+ | Modelos de visión pre-entrenados (ResNet, EfficientNet) |
| **XGBoost** | 2.0+ | Clasificador principal |
| **LightGBM** | 4.3+ | Clasificador alternativo |
| **Scikit-learn** | 1.4+ | Preprocessing, métricas, pipelines |
| **Optuna** | 3.6+ | Hyperparameter optimization |
| **SHAP** | 0.44+ | Explicabilidad de modelos |

### 5.5 MLOps

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **DVC** | 3.48+ | Versionado de datos y modelos |
| **MLflow** | 2.11+ | Experiment tracking, model registry |
| **Dagster** | 1.6+ | Orquestación de pipelines |
| **Evidently AI** | 0.4+ | Monitoreo de drift |
| **pytest** | 8.0+ | Testing |
| **Black** | 24+ | Formateo de código |
| **Ruff** | 0.3+ | Linting ultrarrápido |


### 5.6 Backend y API

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **FastAPI** | 0.110+ | Framework web async |
| **Pydantic** | 2.6+ | Validación de datos |
| **Uvicorn** | 0.27+ | ASGI server |

### 5.7 Frontend

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Nuxt 4** | 4.0+ | Framework Vue.js SSR |
| **MapLibre GL JS** | 4.0+ | Visualización de mapas WebGL |
| **TailwindCSS** | 3.4+ | Styling |

### 5.8 IA Generativa

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Ollama** | 0.1+ | Runtime LLMs locales |
| **Llama 3.2** | 3B | Modelo conversacional |
| **LangChain** | 0.1+ | Framework LLM applications |

### 5.9 Google Cloud Platform

| Servicio | Propósito | Configuración FinOps |
|----------|-----------|----------------------|
| **Cloud Storage** | DVC remote, artefactos | Standard, lifecycle 30 días |
| **Cloud Run** | API deployment | Scale-to-zero, max 2 instances |
| **Maps Static API** | Imágenes satelitales | Caché agresivo |
| **Secret Manager** | Credenciales | Solo secrets críticos |


---

## 6. Arquitectura de la Solución

### 6.1 Visión General

La arquitectura sigue el paradigma **"Local-First, Cloud-Deploy"** con tres capas principales:

```
┌─────────────────────────────────────────────────────────────┐
│                  CAPA DE PRESENTACIÓN                       │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   Nuxt 4 Web     │◄────────┤  MapLibre Viewer │         │
│  │   Dashboard      │         │  (Geoespacial)   │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
└───────────┼──────────────────────────────┼──────────────────┘
            │                              │
            └──────────────┬───────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────┐
│                   CAPA DE APLICACIÓN                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │              FastAPI Backend                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │    │
│  │  │ Predict  │  │ Copilot  │  │  Monitoring  │    │    │
│  │  │ Endpoint │  │ Endpoint │  │  Endpoint    │    │    │
│  │  └────┬─────┘  └────┬─────┘  └──────┬───────┘    │    │
│  └───────┼─────────────┼────────────────┼────────────┘    │
└──────────┼─────────────┼────────────────┼─────────────────┘
           │             │                │
┌──────────▼─────────────▼────────────────▼─────────────────┐
│                  CAPA DE PROCESAMIENTO                     │
│  ┌──────────────────────────────────────────────────┐     │
│  │         Pipeline de 3 Fases (Dagster)            │     │
│  │                                                   │     │
│  │  Fase 1: Adquisición                             │     │
│  │  ┌────────────┐  ┌──────────────┐               │     │
│  │  │ Google API │─►│ Rasterizador │               │     │
│  │  │  Downloader│  │   Vectorial  │               │     │
│  │  └────────────┘  └──────────────┘               │     │
│  │                                                   │     │
│  │  Fase 2: Alineación CV                           │     │
│  │  ┌────────┐  ┌────────┐  ┌──────────┐           │     │
│  │  │  ECC   │─►│ LoFTR  │─►│ Validator│           │     │
│  │  │(OpenCV)│  │(Kornia)│  │          │           │     │
│  │  └────────┘  └────────┘  └──────────┘           │     │
│  │                                                   │     │
│  │  Fase 3: Clasificación ML                        │     │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────┐         │     │
│  │  │ DINOv2  │─►│ XGBoost │─►│ Ensemble │         │     │
│  │  │Embedder │  │Classifier│  │  Model   │         │     │
│  │  └─────────┘  └─────────┘  └──────────┘         │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    CAPA DE DATOS                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   DuckDB     │  │   Parquet    │  │    MLflow    │     │
│  │  (Analytics) │  │   (Storage)  │  │  (Tracking)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │         DVC + Git (Version Control)                │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐       │   │
│  │   │   Data   │  │  Models  │  │   Code   │       │   │
│  │   │ Versions │  │ Registry │  │ Versions │       │   │
│  │   └──────────┘  └──────────┘  └──────────┘       │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ DVC Push/Pull
                           ▼
              ┌─────────────────────────┐
              │ Google Cloud Storage    │
              │  (DVC Remote + Assets)  │
              └─────────────────────────┘
```


### 6.2 Flujo de Datos End-to-End

**Entrada:** GeoJSON con 5,000 registros conteniendo **Capa Anterior** y **Capa Nueva** + 300 etiquetas.

**Procesamiento:**

1. **Adquisición (Fase 1):**
   - Calcular bounding box unificado (union de bbox anterior y nuevo)
   - Solicitar imagen a Google Maps Static API (con caché)
   - Rasterizar **Capa Anterior** y **Capa Nueva** por separado
   - Almacenar: `data/raw/{id}_satellite.png`, `data/raw/{id}_mask_old.png`, `data/raw/{id}_mask_new.png`

2. **Alineación (Fase 2):**
   - Intentar ECC sobre **Capa Nueva**: si correlation > 0.7 → aceptar
   - Si falla, intentar LoFTR: si inliers > 50 → aceptar
   - Si ambos fallan → marcar "MANUAL_REVIEW"
   - Calcular 7 métricas geométricas post-alineación (comparando New vs Old y New vs Satélite)
   - Almacenar: `data/aligned/{id}_aligned.png`, `metrics.parquet`

3. **Clasificación (Fase 3):**
   - Extraer embeddings con DINOv2 (1024-dim) de la imagen compuesta
   - Concatenar con 7 métricas → feature vector 1031-dim
   - Predecir con XGBoost: {NO CAMBIAR, CAMBIAR, VERIFICAR}
   - Calcular confianza (probabilidad softmax)
   - Almacenar: `predictions.parquet`

**Salida:** Dashboard interactivo con mapa coloreado por clasificación + panel de métricas

### 6.3 Decisiones Arquitectónicas Clave

#### 6.3.1 ¿Por qué Dagster y no Airflow?

- **Dagster:** Gestiona "assets" (datos) con linaje automático
- **Airflow:** Gestiona "tasks" (procesos) sin contexto de datos
- **Ventaja:** Dagster permite re-materializar solo assets afectados por cambios de código

#### 6.3.2 ¿Por qué DuckDB y no PostgreSQL?

- **DuckDB:** OLAP embebido, queries SQL sobre Parquet sin ETL
- **PostgreSQL:** OLTP, requiere servidor, overhead de setup
- **Ventaja:** DuckDB procesa 1M filas en < 1s sin infraestructura

#### 6.3.3 ¿Por qué XGBoost y no una CNN end-to-end?

- **XGBoost sobre embeddings:** Aprovecha representaciones pre-entrenadas, robusto con 300 muestras
- **CNN desde cero:** Requiere 10K+ muestras, riesgo de overfitting
- **Evidencia:** Zhai et al. (2022) demuestran que transfer learning supera fine-tuning completo en regímenes de pocos datos

---

## 7. Metodología: Scrum Adaptado Unipersonal

### 7.1 Adaptaciones para Desarrollo Individual

Dado que Arthur Zizumbo es el único desarrollador, se implementan las siguientes adaptaciones:

**Ceremonias Modificadas:**

| Ceremonia Scrum | Adaptación Unipersonal | Frecuencia |
|-----------------|------------------------|------------|
| Sprint Planning | Revisión de backlog + priorización en Linear | Lunes inicio sprint |
| Daily Standup | Log escrito en Notion: "Ayer/Hoy/Bloqueadores" | Diario, 10 min |
| Sprint Review | Demo grabada en video + métricas en dashboard | Viernes final sprint |
| Sprint Retrospective | Documento de lecciones aprendidas | Viernes final sprint |

**Roles Asumidos:**

- **Product Owner:** Define prioridades basándose en impacto técnico
- **Scrum Master:** Gestiona impedimentos, ajusta plan
- **Developer:** Implementa historias de usuario
- **QA:** Escribe y ejecuta tests
- **DevOps:** Configura CI/CD y deployment

### 7.2 Definición de "Done"

Una historia de usuario se considera completa cuando:

1. ✅ Código implementado y funcional
2. ✅ Tests unitarios escritos (coverage > 70% del módulo)
3. ✅ Documentación actualizada (docstrings + README)
4. ✅ Code review automatizado pasado (Ruff, Black, pytest)
5. ✅ Commit pusheado a GitHub con mensaje descriptivo
6. ✅ Artefactos versionados con DVC (si aplica)

### 7.3 Estimación de Esfuerzo

**Sistema de Puntos de Historia:**

- 1 punto = 2 horas de trabajo
- 2 puntos = 4 horas (medio día)
- 3 puntos = 6 horas (día completo)
- 5 puntos = 10 horas (tarea compleja, 2 días)
- 8 puntos = 16 horas (épica, requiere división)

**Velocidad Esperada:**

- Dedicación: 40 horas/semana
- Capacidad por sprint (2 semanas): 80 horas = 40 puntos
- Buffer (20% para imprevistos): 32 puntos efectivos por sprint


---

## 8. Plan de Ejecución: Sprint 1 - Fundamentos y Adquisición de Datos

**Objetivo del Sprint:** Establecer infraestructura MLOps, adquirir datos de Google Maps con control de costos, e implementar pipeline de alineación geométrica básica.

**Duración:** Semanas 1-2
**Puntos Totales:** 32 puntos

### 8.1 Épica 1: Infraestructura y Setup (8 puntos)

#### US-001: Configuración del Entorno Local Robusto (3 puntos)

**Como** desarrollador
**Quiero** un entorno de desarrollo local aislado y reproducible
**Para** trabajar eficientemente con soporte GPU nativo sin overhead de virtualización

**Criterios de Aceptación:**
- [ ] Repositorio Git inicializado con estructura de proyecto
- [ ] Entorno Python 3.11 gestionado con Poetry
- [ ] Drivers NVIDIA y CUDA Toolkit instalados en host (Windows)
- [ ] PyTorch reconoce `cuda:0` nativamente
- [ ] `docker-compose.yml` para servicios auxiliares (MLflow, Dagster)
- [ ] README con instrucciones de setup (Poetry install)

**Tareas Técnicas:**
1. Crear estructura de directorios:
   ```
   geo-rect/
   ├── data/
   │   ├── raw/
   │   ├── aligned/
   │   └── predictions/
   ├── src/
   │   ├── acquisition/
   │   ├── alignment/
   │   ├── classification/
   │   └── api/
   ├── tests/
   ├── notebooks/
   ├── docker/
   └── .dvc/
   ```
2. Inicializar proyecto Poetry (`poetry init`)
3. Instalar dependencias con soporte GPU (PyTorch con CUDA 12.1)
4. Configurar docker-compose solo para MLflow y Dagster

**Estimación:** 4 horas

#### US-002: Inicialización de DVC y GCS (2 puntos)

**Como** ML Engineer
**Quiero** versionar datos y modelos en la nube
**Para** mantener trazabilidad de experimentos

**Criterios de Aceptación:**
- [ ] DVC inicializado en el repositorio
- [ ] Bucket GCS creado: `gs://geo-rect-artifacts`
- [ ] DVC remote configurado apuntando a GCS
- [ ] Credenciales GCP configuradas con Secret Manager
- [ ] `dvc push` y `dvc pull` funcionan correctamente

**Tareas Técnicas:**
1. `dvc init`
2. Crear bucket GCS con lifecycle policy (30 días)
3. `dvc remote add -d storage gs://geo-rect-artifacts`
4. Configurar service account con permisos mínimos
5. Probar ciclo completo: add → push → remove → pull

**Estimación:** 4 horas

#### US-003: Configuración de MLflow Tracking (3 puntos)

**Como** Data Scientist
**Quiero** registrar experimentos automáticamente
**Para** comparar modelos y reproducir resultados

**Criterios de Aceptación:**
- [ ] MLflow server corriendo en `localhost:5000`
- [ ] Backend store: SQLite local
- [ ] Artifact store: directorio local (sincronizado con DVC)
- [ ] Experimento "alignment-cv" creado
- [ ] Experimento "classification-ml" creado
- [ ] Script de prueba registra métricas correctamente

**Tareas Técnicas:**
1. Configurar MLflow en docker-compose
2. Crear script `src/utils/mlflow_setup.py`
3. Implementar decorador `@mlflow_track` para funciones
4. Probar logging de parámetros, métricas y artefactos

**Estimación:** 6 horas


### 8.2 Épica 2: Adquisición de Datos (12 puntos)

#### US-004: Pipeline de Descarga Inteligente con Google Maps API (5 puntos)

**Como** Data Engineer
**Quiero** descargar imágenes satelitales con caché
**Para** minimizar costos de API (< $20 USD)

**Criterios de Aceptación:**
- [ ] Script `src/acquisition/download_satellite.py` funcional
- [ ] Sistema de caché verifica existencia antes de descargar
- [ ] Manejo de errores: reintentos con backoff exponencial
- [ ] Logging detallado de costos estimados
- [ ] 5,000 imágenes descargadas en `data/raw/`
- [ ] Gasto reportado en GCP Console < $15 USD

**Tareas Técnicas:**
1. Implementar función `get_bounding_box(polygon: Polygon) -> BBox`
2. Implementar función `download_static_map(bbox, zoom=18, size=640x640)`
3. Implementar caché con hash de bbox como key
4. Configurar API key en Secret Manager
5. Implementar rate limiting (50 req/s máximo)
6. Crear dashboard de progreso con tqdm

**Código Ejemplo:**
```python
def download_with_cache(polygon_id: str, bbox: BBox) -> Path:
    cache_path = Path(f"data/raw/{polygon_id}_satellite.png")
    if cache_path.exists():
        logger.info(f"Cache hit: {polygon_id}")
        return cache_path

    url = build_static_map_url(bbox, API_KEY)
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    with open(cache_path, 'wb') as f:
        f.write(response.content)

    logger.info(f"Downloaded: {polygon_id}, Cost: $0.002")
    return cache_path
```

**Estimación:** 10 horas

#### US-005: Rasterización de Capas Vectoriales (Nueva y Anterior) (4 puntos)

**Como** GIS Developer
**Quiero** rasterizar ambas geometrías (Anterior y Nueva) en canales o archivos separados
**Para** que el modelo de visión pueda comparar los cambios visualmente contra el satélite

**Criterios de Aceptación:**
- [ ] Script `src/acquisition/rasterize_vectors.py` procesa ambas capas
- [ ] Generación de máscara "Anterior" (ej. Canal Rojo o archivo `_old.png`)
- [ ] Generación de máscara "Nueva" (ej. Canal Verde o archivo `_new.png`)
- [ ] Ambas máscaras alineadas al mismo bounding box del satélite
- [ ] 5,000 pares de máscaras generadas en `data/raw/`

**Tareas Técnicas:**
1. Modificar `rasterize_polygon` para aceptar lista de geometrías
2. Generar imagen compuesta o archivos separados por ID
3. Validar que ambas capas comparten el mismo sistema de coordenadas
4. Optimizar con multiprocessing

**Código Ejemplo:**
```python
from rasterio import features
from affine import Affine

def rasterize_dual_layers(poly_old: Polygon, poly_new: Polygon, bbox: BBox, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    transform = Affine.translation(bbox.minx, bbox.miny) * Affine.scale(
        (bbox.maxx - bbox.minx) / size[0],
        (bbox.maxy - bbox.miny) / size[1]
    )

    mask_old = features.rasterize([(poly_old, 255)], out_shape=size, transform=transform, fill=0, dtype=np.uint8)
    mask_new = features.rasterize([(poly_new, 255)], out_shape=size, transform=transform, fill=0, dtype=np.uint8)

    return mask_old, mask_new
```

**Estimación:** 8 horas

#### US-006: Cálculo de Métricas Geométricas Base (3 puntos)

**Como** Data Analyst
**Quiero** calcular las 7 métricas para todas las manzanas
**Para** tener baseline antes de alineación

**Criterios de Aceptación:**
- [ ] Script `src/metrics/calculate_metrics.py` funcional
- [ ] Métricas calculadas: IoU, Hausdorff, dif_simetrica_area, etc.
- [ ] Resultados almacenados en `data/metrics_raw.parquet`
- [ ] Estadísticas descriptivas generadas (media, std, percentiles)

**Tareas Técnicas:**
1. Implementar funciones para cada métrica usando Shapely
2. Optimizar cálculo de Hausdorff (usar scipy.spatial)
3. Paralelizar con Polars (más rápido que Pandas)
4. Generar reporte HTML con distribuciones

**Estimación:** 6 horas


### 8.3 Épica 3: Motor de Alineación CV (12 puntos)

#### US-007: Implementación de Alineación con ECC (5 puntos)

**Como** Computer Vision Engineer
**Quiero** alinear máscaras con imágenes usando ECC
**Para** corregir desplazamientos simples rápidamente

**Criterios de Aceptación:**
- [ ] Función `align_with_ecc(satellite_img, mask_img) -> (aligned_mask, confidence)`
- [ ] Detecta bordes con Canny en imagen satelital
- [ ] Calcula matriz de transformación (warp matrix)
- [ ] Aplica transformación a máscara
- [ ] Retorna confidence score (correlation coefficient)
- [ ] Procesa 1 imagen en < 2 segundos (CPU)

**Tareas Técnicas:**
1. Implementar detección de bordes optimizada
2. Configurar `cv2.findTransformECC` con parámetros óptimos:
   - Motion type: MOTION_EUCLIDEAN (traslación + rotación)
   - Iterations: 5000
   - Termination epsilon: 1e-6
3. Implementar validación de resultado (IoU pre vs post)
4. Registrar métricas en MLflow

**Código Ejemplo:**
```python
def align_with_ecc(satellite: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    # Detectar bordes en imagen satelital
    edges = cv2.Canny(satellite, 50, 150)

    # Inicializar matriz de transformación
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Criterio de terminación
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

    # Calcular transformación
    try:
        cc, warp_matrix = cv2.findTransformECC(
            mask.astype(np.float32),
            edges.astype(np.float32),
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria
        )
    except cv2.error:
        return mask, 0.0  # Falló la alineación

    # Aplicar transformación
    aligned = cv2.warpAffine(mask, warp_matrix, (mask.shape[1], mask.shape[0]))

    return aligned, cc
```

**Estimación:** 10 horas

#### US-008: Implementación de Alineación con LoFTR (5 puntos)

**Como** Deep Learning Engineer
**Quiero** usar LoFTR para casos donde ECC falla
**Para** rescatar manzanas con desfases complejos

**Criterios de Aceptación:**
- [ ] Modelo LoFTR cargado desde Kornia
- [ ] Función `align_with_loftr(satellite_img, mask_img) -> (aligned_mask, num_inliers)`
- [ ] Detecta correspondencias densas
- [ ] Estima homografía con RANSAC
- [ ] Procesa 1 imagen en < 5 segundos (GPU RTX 4070)

**Tareas Técnicas:**
1. Instalar Kornia y descargar pesos pre-entrenados
2. Implementar preprocesamiento (resize, normalización)
3. Ejecutar inferencia en GPU
4. Filtrar correspondencias con confidence > 0.5
5. Estimar homografía robusta con cv2.findHomography + RANSAC

**Código Ejemplo:**
```python
from kornia.feature import LoFTR

loftr = LoFTR(pretrained='outdoor').eval().cuda()

def align_with_loftr(satellite: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int]:
    # Convertir a tensores
    sat_tensor = torch.from_numpy(satellite).float().cuda() / 255.0
    mask_tensor = torch.from_numpy(mask).float().cuda() / 255.0

    # Inferencia
    with torch.no_grad():
        correspondences = loftr({'image0': sat_tensor, 'image1': mask_tensor})

    # Extraer puntos
    pts0 = correspondences['keypoints0'].cpu().numpy()
    pts1 = correspondences['keypoints1'].cpu().numpy()

    # Estimar homografía
    H, inliers = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)

    if H is None:
        return mask, 0

    # Aplicar transformación
    aligned = cv2.warpPerspective(mask, H, (mask.shape[1], mask.shape[0]))

    return aligned, inliers.sum()
```

**Estimación:** 10 horas

#### US-009: Pipeline de Alineación en Cascada (2 puntos)

**Como** MLOps Engineer
**Quiero** orquestar ECC → LoFTR → Manual Review
**Para** optimizar velocidad y robustez

**Criterios de Aceptación:**
- [ ] Función `align_cascade(satellite, mask) -> AlignmentResult`
- [ ] Lógica de decisión implementada:
   - Si ECC confidence > 0.7 → usar ECC
   - Else if LoFTR inliers > 50 → usar LoFTR
   - Else → marcar "MANUAL_REVIEW"
- [ ] Resultados registrados en `data/alignment_results.parquet`
- [ ] Métricas agregadas: % éxito ECC, % éxito LoFTR, % manual

**Estimación:** 4 horas

#### US-0010: Validación de Integridad Vectorial con SAM (5 puntos)
**Como** Computer Vision Engineer
**Quiero** generar una máscara de segmentación automática con SAM
**Para** descartar vectores cuya geometría difiere radicalmente de la realidad física (error topológico grave)

**Criterios de Aceptación:**

- [ ] Modelo MobileSAM o FastSAM corriendo en local (RTX 4070).
- [ ] Función validate_with_sam(satellite_img, vector_mask) -> integrity_score.
- [ ] Si IoU(Mascara_Vectorial, Mascara_SAM) < 0.4, marcar automáticamente como "ERROR_TOPOLOGICO".
- [ ] Esto filtra datos "basura" que dañarían el entrenamiento del modelo ML.

Justificación Técnica: SAM ha demostrado capacidad "zero-shot" excepcional para segmentación de estructuras, sirviendo como un juez imparcial para la calidad de los datos de entrada.

---

## 9. Plan de Ejecución: Sprint 2 - Machine Learning y Clasificación

**Objetivo del Sprint:** Entrenar modelo de clasificación híbrido (embeddings + métricas) y alcanzar F1-Score > 0.80 en conjunto de validación.

**Duración:** Semanas 3-4
**Puntos Totales:** 32 puntos

### 9.1 Épica 4: Feature Engineering (10 puntos)

#### US-011: Extracción de Embeddings con DINOv2 (5 puntos)

**Como** ML Engineer
**Quiero** extraer representaciones visuales de imágenes alineadas
**Para** capturar patrones semánticos complejos

**Criterios de Aceptación:**
- [ ] Modelo DINOv2-ViT-B/14 cargado desde HuggingFace
- [ ] Función `extract_embeddings(image_path) -> np.ndarray[1024]`
- [ ] 5,000 embeddings extraídos y almacenados en `data/embeddings.parquet`
- [ ] Procesamiento batch optimizado (batch_size=32)
- [ ] Tiempo total < 2 horas en RTX 4070

**Tareas Técnicas:**
1. Cargar modelo pre-entrenado:
   ```python
   from transformers import AutoModel, AutoImageProcessor

   model = AutoModel.from_pretrained('facebook/dinov2-base').cuda()
   processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
   ```
2. Implementar dataloader con PyTorch
3. Extraer features de la capa final (antes del clasificador)
4. Normalizar embeddings (L2 norm)
5. Guardar en formato Parquet con compresión

**Estimación:** 10 horas


#### US-012: Cálculo de Métricas Post-Alineación (3 puntos)

**Como** Data Scientist
**Quiero** recalcular las 7 métricas después de alineación
**Para** medir la mejora y usarlas como features

**Criterios de Aceptación:**
- [ ] Métricas recalculadas para imágenes alineadas
- [ ] Comparación pre vs post alineación generada
- [ ] Mejora promedio en IoU > 30%
- [ ] Resultados en `data/metrics_aligned.parquet`

**Estimación:** 6 horas

#### US-013: Construcción del Dataset de Entrenamiento (2 puntos)

**Como** Data Engineer
**Quiero** unir embeddings + métricas + etiquetas
**Para** crear dataset final de entrenamiento

**Criterios de Aceptación:**
- [ ] Dataset con 300 muestras etiquetadas
- [ ] Columnas: 1024 (embeddings) + 7 (métricas) + 1 (label)
- [ ] Split train/val/test: 60%/20%/20%
- [ ] Balanceo de clases verificado
- [ ] Guardado en `data/train_dataset.parquet`
- [ ] Data augmentation implementado: rotaciones, flips
- [ ] GroupKFold configurado con agrupación por zona geográfica



**Tareas Técnicas:**
1. Join de tablas con Polars
2. Implementar stratified split
3. Verificar no hay data leakage
4. Generar estadísticas descriptivas

**Estimación:** 4 horas

### 9.2 Épica 5: Modelado y Entrenamiento (15 puntos)

#### US-014: Entrenamiento de Baseline con XGBoost (5 puntos)

**Como** Data Scientist
**Quiero** entrenar modelo XGBoost con hyperparameter tuning
**Para** establecer baseline de clasificación

**Criterios de Aceptación:**
- [ ] Modelo XGBoost entrenado con Optuna (100 trials)
- [ ] Hiperparámetros optimizados: learning_rate, max_depth, n_estimators
- [ ] F1-Score en validación > 0.75
- [ ] Modelo guardado en `models/xgboost_v1.json`
- [ ] Experimento registrado en MLflow

**Tareas Técnicas:**
1. Definir espacio de búsqueda de hiperparámetros
2. Implementar función objetivo para Optuna
3. Entrenar con early stopping
4. Evaluar en conjunto de validación
5. Generar matriz de confusión

**Código Ejemplo:**
```python
import optuna
from xgboost import XGBClassifier

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }

    model = XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')

    return f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**Estimación:** 10 horas

#### US-015: Entrenamiento de Modelo Alternativo con LightGBM (3 puntos)

**Como** ML Engineer
**Quiero** entrenar LightGBM para comparar con XGBoost
**Para** seleccionar el mejor modelo

**Criterios de Aceptación:**
- [ ] Modelo LightGBM entrenado
- [ ] Comparación de métricas: XGBoost vs LightGBM
- [ ] Análisis de importancia de features
- [ ] Modelo guardado si supera a XGBoost

**Estimación:** 6 horas

#### US-016: Ensemble y Calibración (4 puntos)

**Como** ML Engineer
**Quiero** crear ensemble de modelos y calibrar probabilidades
**Para** maximizar F1-Score y confiabilidad

**Criterios de Aceptación:**
- [ ] Ensemble ponderado de XGBoost + LightGBM
- [ ] Pesos optimizados con validación cruzada
- [ ] Calibración de probabilidades con CalibratedClassifierCV
- [ ] F1-Score final > 0.80
- [ ] Modelo final guardado en `models/ensemble_v1.pkl`

**Tareas Técnicas:**
1. Implementar voting classifier
2. Optimizar pesos con grid search
3. Aplicar calibración isotónica
4. Evaluar en test set (no tocado hasta ahora)
5. Generar reporte de clasificación completo

**Estimación:** 8 horas

#### US-017: Explicabilidad con SHAP (3 puntos)

**Como** Data Scientist
**Quiero** generar explicaciones de predicciones
**Para** entender qué features son más importantes

**Criterios de Aceptación:**
- [ ] SHAP values calculados para conjunto de test
- [ ] Gráficos generados: summary plot, dependence plots
- [ ] Top 10 features más importantes identificadas
- [ ] Reporte HTML con explicaciones

**Tareas Técnicas:**
1. Calcular SHAP values con TreeExplainer
2. Generar visualizaciones
3. Analizar casos específicos (correctos vs incorrectos)
4. Documentar insights

**Estimación:** 6 horas

### 9.3 Épica 6: Orquestación con Dagster (7 puntos)

#### US-018: Definición de Assets en Dagster (4 puntos)

**Como** MLOps Engineer
**Quiero** definir pipeline end-to-end como assets
**Para** automatizar y reproducir el flujo completo

**Criterios de Aceptación:**
- [ ] Assets definidos: raw_images, aligned_images, embeddings, metrics, predictions
- [ ] Dependencias entre assets configuradas
- [ ] Materialización completa exitosa
- [ ] Dagster UI muestra linaje de datos

**Tareas Técnicas:**
1. Crear archivo `src/dagster_assets.py`
2. Definir cada fase como @asset
3. Configurar recursos (conexiones, configs)
4. Implementar particiones por lote (1000 manzanas/partición)

**Estimación:** 8 horas

#### US-019: Configuración de Schedules y Sensors (3 puntos)

**Como** MLOps Engineer
**Quiero** automatizar re-entrenamiento periódico
**Para** mantener modelo actualizado

**Criterios de Aceptación:**
- [ ] Schedule semanal para re-cálculo de métricas
- [ ] Sensor que detecta nuevos datos y dispara pipeline
- [ ] Alertas configuradas para fallos

**Estimación:** 6 horas


#### US-020: Loop de Active Learning y Uncertainty Sampling (5 puntos)
**Como** Data Scientist
**Quiero** identificar las manzanas donde el modelo tiene mayor incertidumbre
**Para** priorizar la revisión manual de esos casos y re-entrenar el modelo eficientemente

**Criterios de Aceptación:**
- [ ] Generar predicciones en 4,700 manzanas no etiquetadas.
- [ ] Cálculo de entropía de predicción: $H(x) = -\sum p_i \log p_i$.
- [ ] Selección del Top-50 de manzanas con mayor entropía (mayor confusión).
- [ ] Enviar a experto cartográfico para etiquetado
- [ ] Generación de un reporte "Anotación Prioritaria" para el experto cartográfico.
- [ ] Estrategia validada para expandir el dataset efectivo sin etiquetar todo aleatoriamente.
- [ ] Re-entrenar modelo con 350 muestras

---

## 10. Plan de Ejecución: Sprint 3 - Deployment y Visualización

**Objetivo del Sprint:** Desarrollar API REST, interfaz web interactiva, integrar copiloto conversacional y desplegar en GCP.

**Duración:** Semanas 5-6
**Puntos Totales:** 32 puntos

### 10.1 Épica 7: Backend API (10 puntos)

#### US-021: Desarrollo de API con FastAPI (5 puntos)

**Como** Backend Developer
**Quiero** exponer modelo como servicio REST
**Para** permitir consumo desde frontend

**Criterios de Aceptación:**
- [ ] Endpoints implementados:
  - `POST /predict`: Predicción individual
  - `POST /batch`: Predicción batch
  - `GET /explain/{id}`: Explicación SHAP
  - `GET /metrics/{id}`: Métricas de manzana
  - `GET /health`: Health check
- [ ] Validación de entrada con Pydantic
- [ ] Documentación automática en `/docs`
- [ ] Latencia p95 < 500ms

**Tareas Técnicas:**
1. Crear estructura de proyecto FastAPI
2. Implementar modelos Pydantic para request/response
3. Cargar modelo en memoria al inicio
4. Implementar endpoints con async/await
5. Agregar middleware de logging y CORS

**Código Ejemplo:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Geo-Rect API")

class PredictionRequest(BaseModel):
    polygon_id: str

class PredictionResponse(BaseModel):
    polygon_id: str
    classification: str
    confidence: float
    metrics: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Cargar imagen y extraer features
    embeddings = extract_embeddings(request.polygon_id)
    metrics = load_metrics(request.polygon_id)

    # Predecir
    features = np.concatenate([embeddings, metrics])
    prediction = model.predict_proba([features])[0]

    return PredictionResponse(
        polygon_id=request.polygon_id,
        classification=CLASSES[prediction.argmax()],
        confidence=float(prediction.max()),
        metrics=metrics_to_dict(metrics)
    )
```

**Estimación:** 10 horas


#### US-022: Testing y Optimización de API (3 puntos)

**Como** QA Engineer
**Quiero** tests automatizados para la API
**Para** garantizar calidad y performance

**Criterios de Aceptación:**
- [ ] Tests unitarios con pytest (coverage > 80%)
- [ ] Tests de integración con TestClient
- [ ] Load testing con Locust (100 req/s)
- [ ] Optimización de carga de modelo (lazy loading)

**Estimación:** 6 horas

#### US-023: Containerización de API (2 puntos)

**Como** DevOps Engineer
**Quiero** API containerizada y optimizada
**Para** deployment en Cloud Run

**Criterios de Aceptación:**
- [ ] Dockerfile multi-stage optimizado
- [ ] Imagen < 1GB
- [ ] Health check configurado
- [ ] Variables de entorno externalizadas

**Estimación:** 4 horas

### 10.2 Épica 8: Frontend Geoespacial (12 puntos)

#### US-024: Setup de Proyecto Nuxt 4 (2 puntos)

**Como** Frontend Developer
**Quiero** proyecto Nuxt 4 configurado
**Para** desarrollar interfaz moderna

**Criterios de Aceptación:**
- [ ] Proyecto Nuxt 4 inicializado
- [ ] TailwindCSS configurado
- [ ] MapLibre GL JS instalado
- [ ] Estructura de componentes definida

**Estimación:** 4 horas

#### US-025: Visualización de Mapa Interactivo (5 puntos)

**Como** Frontend Developer
**Quiero** mapa con manzanas coloreadas por clasificación
**Para** visualizar resultados geoespacialmente

**Criterios de Aceptación:**
- [ ] Mapa base con MapLibre (tiles de OpenStreetMap)
- [ ] Capa de polígonos con colores:
  - Verde: NO CAMBIAR
  - Rojo: CAMBIAR
  - Amarillo: VERIFICAR
- [ ] Click en polígono muestra panel lateral con detalles
- [ ] Filtros por clasificación y confianza
- [ ] Performance: 5,000 polígonos renderizados sin lag

**Tareas Técnicas:**
1. Configurar MapLibre con estilo base
2. Cargar GeoJSON de resultados
3. Implementar estilo dinámico basado en propiedades
4. Agregar interactividad (hover, click)
5. Optimizar con clustering para zoom bajo

**Código Ejemplo (Vue 3 Composition API):**
```vue
<script setup>
import maplibregl from 'maplibre-gl'
import { onMounted, ref } from 'vue'

const mapContainer = ref(null)
const selectedPolygon = ref(null)

onMounted(async () => {
  const map = new maplibregl.Map({
    container: mapContainer.value,
    style: 'https://demotiles.maplibre.org/style.json',
    center: [-99.1332, 19.4326], // Ciudad de México
    zoom: 12
  })

  map.on('load', async () => {
    const results = await fetch('/api/results.geojson').then(r => r.json())

    map.addSource('manzanas', {
      type: 'geojson',
      data: results
    })

    map.addLayer({
      id: 'manzanas-fill',
      type: 'fill',
      source: 'manzanas',
      paint: {
        'fill-color': [
          'match',
          ['get', 'clasificacion'],
          'NO CAMBIAR', '#10b981',
          'CAMBIAR', '#ef4444',
          'VERIFICAR', '#f59e0b',
          '#6b7280'
        ],
        'fill-opacity': 0.6
      }
    })

    map.on('click', 'manzanas-fill', (e) => {
      selectedPolygon.value = e.features[0].properties
    })
  })
})
</script>

<template>
  <div class="flex h-screen">
    <div ref="mapContainer" class="flex-1"></div>
    <div v-if="selectedPolygon" class="w-96 bg-white p-6 shadow-lg">
      <h2 class="text-2xl font-bold mb-4">Manzana {{ selectedPolygon.id }}</h2>
      <div class="space-y-2">
        <p><strong>Clasificación:</strong> {{ selectedPolygon.clasificacion }}</p>
        <p><strong>Confianza:</strong> {{ (selectedPolygon.confianza * 100).toFixed(1) }}%</p>
        <p><strong>IoU:</strong> {{ selectedPolygon.iou.toFixed(3) }}</p>
        <p><strong>Distancia Centroides:</strong> {{ selectedPolygon.distancia_centroides.toFixed(2) }}m</p>
      </div>
    </div>
  </div>
</template>
```

**Estimación:** 10 horas

#### US-026: Panel de Métricas y Filtros (3 puntos)

**Como** Frontend Developer
**Quiero** panel con estadísticas y filtros
**Para** explorar resultados interactivamente

**Criterios de Aceptación:**
- [ ] Dashboard con métricas agregadas:
  - Total manzanas por clasificación
  - Distribución de confianza
  - Métricas promedio
- [ ] Filtros funcionales:
  - Por clasificación
  - Por rango de confianza
  - Por rango de IoU
- [ ] Gráficos con Chart.js

**Estimación:** 6 horas

#### US-027: Integración con API Backend (2 puntos)

**Como** Full Stack Developer
**Quiero** frontend consumiendo API
**Para** datos en tiempo real

**Criterios de Aceptación:**
- [ ] Composables de Nuxt para llamadas API
- [ ] Manejo de estados de carga y error
- [ ] Caché de resultados en cliente

**Estimación:** 4 horas

### 10.3 Épica 9: Copiloto Conversacional (6 puntos)

#### US-028: Integración de Ollama con Llama 3.2 (3 puntos)

**Como** AI Engineer
**Quiero** LLM local respondiendo consultas
**Para** análisis conversacional de resultados

**Criterios de Aceptación:**
- [ ] Ollama corriendo localmente con Llama 3.2 3B
- [ ] Endpoint `/copilot/query` en FastAPI
- [ ] Prompt engineering para generación de SQL
- [ ] Ejecución de queries en DuckDB
- [ ] Respuesta en lenguaje natural

**Tareas Técnicas:**
1. Instalar Ollama y descargar modelo
2. Crear prompt template con esquema de BD
3. Implementar parser de SQL generado
4. Ejecutar query y formatear resultados
5. Generar respuesta natural

**Código Ejemplo:**
```python
import ollama
import duckdb

SYSTEM_PROMPT = """
Eres un analista GIS experto. Tienes acceso a una base de datos con la tabla 'manzanas':
- id (string): Identificador único
- clasificacion (string): NO CAMBIAR, CAMBIAR, VERIFICAR
- confianza (float): 0-1
- iou (float): 0-1
- distancia_centroides (float): metros

Genera SOLO la query SQL para responder la pregunta del usuario.
"""

@app.post("/copilot/query")
async def copilot_query(question: str):
    # Generar SQL con LLM
    response = ollama.chat(model='llama3.2:3b', messages=[
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question}
    ])

    sql_query = extract_sql(response['message']['content'])

    # Ejecutar query
    conn = duckdb.connect('data/results.duckdb')
    result = conn.execute(sql_query).fetchall()

    # Generar respuesta natural
    answer = ollama.chat(model='llama3.2:3b', messages=[
        {'role': 'user', 'content': f"Pregunta: {question}\nResultado SQL: {result}\nResponde en lenguaje natural."}
    ])

    return {
        'question': question,
        'sql': sql_query,
        'answer': answer['message']['content']
    }
```

**Estimación:** 6 horas

#### US-029: Interfaz de Chat en Frontend (3 puntos)

**Como** Frontend Developer
**Quiero** interfaz de chat para copiloto
**Para** consultas en lenguaje natural

**Criterios de Aceptación:**
- [ ] Componente de chat con historial
- [ ] Input de texto con autocompletado
- [ ] Visualización de respuestas con markdown
- [ ] Ejemplos de preguntas sugeridas

**Estimación:** 6 horas

### 10.4 Épica 10: Deployment en GCP (4 puntos)

#### US-030: Deployment de API en Cloud Run (2 puntos)

**Como** DevOps Engineer
**Quiero** API desplegada en Cloud Run
**Para** acceso público

**Criterios de Aceptación:**
- [ ] Imagen Docker pusheada a Container Registry
- [ ] Servicio Cloud Run creado
- [ ] Configuración: min-instances=0, max-instances=2
- [ ] URL pública funcional
- [ ] Costo proyectado < $10/mes

**Tareas Técnicas:**
1. Build y push de imagen
2. Crear servicio con gcloud CLI
3. Configurar variables de entorno
4. Configurar IAM y permisos
5. Probar endpoints públicos

**Estimación:** 4 horas

#### US-031: Deployment de Frontend en Cloud Run (2 puntos)

**Como** DevOps Engineer
**Quiero** frontend desplegado
**Para** demo pública

**Criterios de Aceptación:**
- [ ] Build de producción de Nuxt
- [ ] Dockerfile para SSR
- [ ] Deployment en Cloud Run
- [ ] Dominio personalizado (opcional)

**Estimación:** 4 horas

---

## 11. Roles y Responsabilidades

Dado que Arthur Zizumbo es el único desarrollador, asume múltiples roles:

| Rol | Responsabilidades | Tiempo Estimado |
|-----|-------------------|-----------------|
| **Product Owner** | Priorizar backlog, definir criterios de aceptación | 10% |
| **Scrum Master** | Gestionar impedimentos, facilitar ceremonias | 5% |
| **Data Engineer** | Pipelines de datos, ETL, versionado | 20% |
| **ML Engineer** | Entrenamiento de modelos, feature engineering | 25% |
| **Backend Developer** | API REST, integración de modelos | 15% |
| **Frontend Developer** | Interfaz web, visualización geoespacial | 15% |
| **DevOps/MLOps** | CI/CD, deployment, monitoreo | 10% |

---

## 12. Gestión de Riesgos


### 12.1 Matriz de Riesgos

| ID | Riesgo | Probabilidad | Impacto | Severidad | Estrategia de Mitigación |
|----|--------|--------------|---------|-----------|--------------------------|
| **R-01** | Costo excesivo en Google Maps API | Media | Alto | **ALTA** | - Implementar caché persistente obligatorio<br>- Budget alert en GCP a $50 USD<br>- Limitar a 5,000 requests exactas<br>- Usar Static Maps (no Dynamic) |
| **R-02** | Alineación fallida en zonas rurales/periféricas | Alta | Medio | **MEDIA** | - Pipeline en cascada (ECC → LoFTR)<br>- Marcar casos irrecuperables como "MANUAL_REVIEW"<br>- Excluir del entrenamiento ML (evitar ruido) |
| **R-03** | Overfitting del modelo ML (300 muestras) | Alta | Alto | **ALTA** | - Transfer learning con DINOv2 (no entrenar desde cero)<br>- Cross-validation estricta (GroupKFold)<br>- Regularización agresiva en XGBoost<br>- Monitoreo de drift con Evidently |
| **R-04** | Latencia alta en inferencia (> 500ms) | Media | Medio | **MEDIA** | - Caché de embeddings pre-calculados<br>- Batch processing para múltiples manzanas<br>- Optimización de modelo (quantization) |
| **R-05** | GPU insuficiente para LoFTR (8GB VRAM) | Baja | Alto | **MEDIA** | - Procesar imágenes en batches pequeños (batch_size=1)<br>- Usar mixed precision (FP16)<br>- Fallback a CPU si OOM |
| **R-06** | Datos de entrada de mala calidad | Media | Alto | **ALTA** | - Validación exhaustiva en ingesta<br>- Detección de outliers automática<br>- Pipeline de limpieza de datos |
| **R-07** | Desbalanceo de clases en dataset | Alta | Medio | **MEDIA** | - Usar class_weight en XGBoost<br>- SMOTE para oversampling de clase minoritaria<br>- Métrica principal: F1-Score macro (no accuracy) |
| **R-08** | Falta de tiempo para completar Sprint 3 | Media | Medio | **MEDIA** | - Priorizar MVP: API + mapa básico<br>- Copiloto conversacional como nice-to-have<br>- Deployment local si Cloud Run falla |
| **R-09** | Incompatibilidad de versiones de librerías | Baja | Bajo | **BAJA** | - Poetry lock file versionado<br>- Docker con versiones fijas<br>- Testing en CI/CD |
| **R-10** | Pérdida de datos por fallo de hardware | Baja | Alto | **MEDIA** | - DVC push diario a GCS<br>- Backup de código en GitHub<br>- Commits frecuentes |

### 12.2 Plan de Contingencia

**Si R-01 se materializa (costo > $50 USD):**
1. Detener descarga inmediatamente
2. Usar solo muestras descargadas para desarrollo
3. Considerar alternativas: Sentinel-2 (gratuito pero menor resolución)

**Si R-03 se materializa (F1-Score < 0.70):**
1. Aumentar data augmentation (rotaciones, flips)
2. Probar modelos más simples (Random Forest)
3. Solicitar más etiquetas manuales (priorizar casos ambiguos)

**Si R-08 se materializa (retraso en cronograma):**
1. Reducir scope: eliminar copiloto conversacional
2. Deployment solo local (Docker Compose)
3. Documentar roadmap para futuras iteraciones

---

## 13. Análisis Financiero y FinOps

### 13.1 Desglose Detallado de Costos GCP

| Servicio | Uso Estimado | Precio Unitario | Costo Total |
|----------|--------------|-----------------|-------------|
| **Maps Static API** | 5,000 requests | $0.002/request | $10.00 |
| **Cloud Storage** | 2GB almacenamiento × 1.5 meses | $0.020/GB/mes | $0.06 |
| **Cloud Storage** | 5GB egress (DVC pull) | $0.12/GB | $0.60 |
| **Container Registry** | 1GB imagen × 1.5 meses | $0.10/GB/mes | $0.15 |
| **Cloud Run (API)** | 100 horas × 1 vCPU × 2GB RAM | $0.00002400/vCPU-s | $8.64 |
| **Cloud Run (Frontend)** | 50 horas × 1 vCPU × 1GB RAM | $0.00002400/vCPU-s | $4.32 |
| **Secret Manager** | 3 secrets × 1.5 meses | $0.06/secret/mes | $0.27 |
| **Cloud Logging** | 5GB logs (dentro de free tier) | $0.00 | $0.00 |
| **Cloud Monitoring** | Métricas básicas (free tier) | $0.00 | $0.00 |
| **Buffer (10%)** | Imprevistos | - | $2.40 |
| **TOTAL ESTIMADO** | - | - | **$26.44** |

**Margen de seguridad:** $300 - $26.44 = **$273.56 (91% bajo presupuesto)**

### 13.2 Estrategias de Optimización de Costos

1. **Cómputo Local Primero:**
   - Entrenamiento: 100% local (RTX 4070)
   - Inferencia pesada (embeddings): 100% local
   - Cloud: solo API ligera y almacenamiento

2. **Arquitectura Edge-First Estricta (GPU Local Obligatoria):**
   - Prohibición de GPU en Cloud: Bajo ninguna circunstancia se instanciarán máquinas con GPU (T4/L4/A100) en GCP.
   - Todo el entrenamiento (XGBoost, Fine-tuning) y la inferencia pesada (DINOv2, SAM, LoFTR) deben ocurrir en la NVIDIA RTX 4070 local.
   - Artefactos estáticos: La nube solo recibe resultados procesados (CSV, Parquet, PNGs pequeños), nunca datos crudos para procesar.
   - Scale-to-Zero: El servicio de Cloud Run para la API debe configurarse con min-instances=0 para garantizar costo cero cuando no hay tráfico de usuarios.

2. **Scale-to-Zero en Cloud Run:**
   - min-instances=0 (no cobro cuando no hay tráfico)
   - max-instances=2 (limitar escalado)
   - concurrency=80 (maximizar uso de instancia)

3. **Lifecycle Policies en Storage:**
   - Mover a Nearline después de 30 días
   - Eliminar artefactos temporales después de 60 días

4. **Caché Agresivo:**
   - Imágenes satelitales: caché local permanente
   - Embeddings: pre-calcular y cachear
   - Resultados de API: caché con TTL de 1 hora

5. **Monitoreo de Presupuesto:**
   - Budget alert a $50 USD (notificación por email)
   - Budget alert a $100 USD (detener servicios automáticamente)

---

## 14. Plan de Entregables

### 14.1 Entregables por Sprint

**Sprint 1 (Semanas 1-2):**
- [ ] Repositorio GitHub con estructura completa
- [ ] Docker Compose funcional (MLflow + Dagster + Jupyter)
- [ ] 5,000 imágenes satelitales descargadas
- [ ] 5,000 máscaras vectoriales rasterizadas
- [ ] Pipeline de alineación ECC + LoFTR implementado
- [ ] Métricas geométricas calculadas (pre y post alineación)
- [ ] Reporte de alineación (% éxito, mejora en IoU)
- [ ] Datos versionados con DVC en GCS

**Sprint 2 (Semanas 3-4):**
- [ ] Embeddings DINOv2 extraídos (5,000 manzanas)
- [ ] Dataset de entrenamiento construido (300 muestras)
- [ ] Modelo XGBoost entrenado y optimizado
- [ ] Modelo LightGBM entrenado
- [ ] Ensemble final con F1-Score > 0.80
- [ ] Análisis SHAP de explicabilidad
- [ ] Pipeline Dagster end-to-end funcional
- [ ] Experimentos registrados en MLflow
- [ ] Reporte de performance del modelo

**Sprint 3 (Semanas 5-6):**
- [ ] API REST con FastAPI desplegada localmente
- [ ] Tests automatizados (coverage > 70%)
- [ ] Frontend Nuxt 4 con mapa interactivo
- [ ] Panel de métricas y filtros
- [ ] Copiloto conversacional con Ollama
- [ ] Deployment en Cloud Run (API + Frontend)
- [ ] Documentación técnica completa
- [ ] Video demo del sistema (5-10 minutos)

### 14.2 Documentación Final

**Documentos Técnicos:**
1. **README.md:** Instrucciones de instalación y uso
2. **ARCHITECTURE.md:** Diagramas y decisiones arquitectónicas
3. **API_DOCS.md:** Documentación de endpoints (auto-generada con FastAPI)
4. **MODEL_CARD.md:** Descripción del modelo, métricas, limitaciones
5. **DEPLOYMENT.md:** Guía de deployment en GCP
6. **CONTRIBUTING.md:** Guía para futuros contribuidores

**Notebooks Jupyter:**
1. `01_EDA.ipynb`: Análisis exploratorio de datos
2. `02_Alignment_Experiments.ipynb`: Comparación ECC vs LoFTR
3. `03_Model_Training.ipynb`: Entrenamiento y evaluación de modelos
4. `04_SHAP_Analysis.ipynb`: Explicabilidad

**Reportes:**
1. Reporte de alineación geométrica (PDF)
2. Reporte de performance del modelo (PDF)
3. Análisis de costos GCP (Excel/CSV)

---

## 15. Conclusiones

### 15.1 Viabilidad Técnica

El análisis exhaustivo de la literatura científica (2022-2025) y la evaluación de tecnologías disponibles confirman que la solución propuesta es **técnicamente viable y representa el estado del arte** en validación cartográfica automatizada.

**Fundamentos Validados:**

1. **Separación de Preocupaciones:** La arquitectura de tres fases (adquisición → alineación CV → clasificación ML) está respaldada por múltiples estudios que demuestran que intentar resolver ambos problemas simultáneamente resulta en performance degradado (Chen et al., 2023; Zhang et al., 2023).

2. **Transfer Learning para Few-Shot:** Con solo 300 muestras, el uso de modelos pre-entrenados (DINOv2) es la única estrategia viable, con evidencia empírica de superioridad sobre entrenamiento desde cero (Zhai et al., 2022; Oquab et al., 2023).

3. **Pipeline en Cascada:** La combinación de métodos clásicos (ECC) para casos simples y deep learning (LoFTR) para casos complejos optimiza el balance velocidad-robustez, procesando el 95% de casos exitosamente (Wang et al., 2024).

### 15.2 Viabilidad Económica

El presupuesto de $300 USD es **más que suficiente** para completar el proyecto, con un gasto estimado de $26.44 (91% bajo presupuesto). La estrategia "Local-First, Cloud-Deploy" aprovecha el hardware existente (RTX 4070) para eliminar costos de cómputo cloud, reservando GCP solo para servicios ligeros.

**Comparación con Alternativas:**

- **Cloud-Native (todo en GCP):** Costo estimado $800-1,200 USD (entrenamiento en GPU cloud)
- **Solución Propuesta:** $26.44 USD (ahorro del 97%)

### 15.3 Viabilidad Operativa

El cronograma de 6 semanas (3 sprints) es **realista** considerando:

- **Referencia histórica:** Arthur completó un proyecto similar (PlaneacionProyecto.md) en 3 meses trabajando solo, demostrando capacidad de ejecución.
- **Scope ajustado:** El proyecto se enfoca en un MVP funcional, con features avanzadas (copiloto conversacional) como opcionales.
- **Metodología probada:** Scrum adaptado para desarrollo unipersonal con ceremonias simplificadas.

### 15.4 Impacto Esperado

**Beneficios Cuantitativos:**

- **Reducción de tiempo:** De 3 meses (manual) a 1 semana (automatizado) = 92% de ahorro
- **Precisión:** F1-Score > 0.80 supera la consistencia de clasificación manual (típicamente 70-75%)
- **Escalabilidad:** Pipeline puede procesar 50,000+ manzanas sin cambios arquitectónicos

**Beneficios Cualitativos:**

- **Reproducibilidad:** DVC + Docker garantizan resultados replicables
- **Explicabilidad:** SHAP values permiten auditar decisiones del modelo
- **Extensibilidad:** Arquitectura modular facilita agregar nuevas fuentes de datos o modelos

### 15.5 Recomendaciones Finales

1. **Priorizar Sprint 1:** La calidad de la alineación geométrica determina el límite superior de performance del modelo ML. Invertir tiempo extra aquí es crítico.

2. **Monitoreo Continuo:** Implementar Evidently AI desde Sprint 2 para detectar drift tempranamente.

3. **Documentación Progresiva:** Escribir documentación mientras se desarrolla, no al final.

4. **Comunicación con Stakeholders:** Generar demos visuales al final de cada sprint para mantener alineación con la Dirección de Cartografía.

5. **Plan de Escalado:** Considerar migración a Kubernetes si el volumen de datos crece 10x (50,000+ manzanas).

### 15.6 Próximos Pasos Inmediatos

**Semana 0 (Pre-Sprint 1):**
1. Crear cuenta GCP y configurar billing alerts
2. Solicitar acceso a datos de las 5,000 manzanas (GeoJSON)
3. Obtener API key de Google Maps
4. Configurar entorno de desarrollo local
5. Crear repositorio GitHub y estructura inicial

**Inicio de Sprint 1 (Lunes Semana 1):**
1. Sprint Planning: revisar backlog y comprometer historias
2. Iniciar US-1.1: Configuración del entorno
3. Daily log: documentar progreso y bloqueadores

---

## 16. Referencias

### Artículos Científicos y Papers

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623-2631. https://doi.org/10.1145/3292500.3330701

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13(1), 281-305.

Chen, L., Zhang, H., & Wang, Y. (2023). Robust image registration for urban change detection using multi-modal remote sensing data. *ISPRS Journal of Photogrammetry and Remote Sensing*, 198, 45-62. https://doi.org/10.1016/j.isprsjprs.2023.02.008

Crommelinck, S., Bennett, R., Gerke, M., Nex, F., Yang, M. Y., & Vosselman, G. (2019). Review of automatic feature extraction from high-resolution optical sensor data for UAV-based cadastral mapping. *Remote Sensing*, 11(6), 689. https://doi.org/10.3390/rs11060689

DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). SuperPoint: Self-supervised interest point detection and description. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops*, 224-236.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems*, 36, 10088-10115.

Evangelidis, G. D., & Psarakis, E. Z. (2008). Parametric image alignment using enhanced correlation coefficient maximization. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(10), 1858-1865. https://doi.org/10.1109/TPAMI.2008.113

Everingham, M., Van Gool, L., Williams, C. K., Winn, J., & Zisserman, A. (2010). The Pascal visual object classes (VOC) challenge. *International Journal of Computer Vision*, 88(2), 303-338. https://doi.org/10.1007/s11263-009-0275-4

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 16000-16009. https://doi.org/10.1109/CVPR52688.2022.01553

Hughes, M., Chen, Z., & Li, W. (2024). Deep learning approaches for multi-temporal satellite image registration in urban environments. *Remote Sensing of Environment*, 301, 113945. https://doi.org/10.1016/j.rse.2023.113945

Huttenlocher, D. P., Klanderman, G. A., & Rucklidge, W. J. (1993). Comparing images using the Hausdorff distance. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 15(9), 850-863. https://doi.org/10.1109/34.232073

Kornblith, S., Shlens, J., & Le, Q. V. (2019). Do better ImageNet models transfer better? *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2661-2671. https://doi.org/10.1109/CVPR.2019.00277

Kuprieiev, R., Cheptsov, A., Zvonkov, D., Shcheklein, I., Seleznev, I., & Petrov, D. (2023). DVC: Data version control - Git for data & models. *Journal of Open Source Software*, 8(82), 4910. https://doi.org/10.21105/joss.04910

Li, X., Wang, S., & Zhang, Y. (2023). Historical cadastral map alignment using deep learning-based feature matching. *Cartography and Geographic Information Science*, 50(3), 245-262. https://doi.org/10.1080/15230406.2023.2180934

Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2), 91-110. https://doi.org/10.1023/B:VISI.0000029664.99615.94

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., ... & Bojanowski, P. (2023). DINOv2: Learning robust visual features without supervision. *arXiv preprint arXiv:2304.07193*. https://doi.org/10.48550/arXiv.2304.07193

Paleyes, A., Urma, R. G., & Lawrence, N. D. (2022). Challenges in deploying machine learning: A survey of case studies. *ACM Computing Surveys*, 55(6), 1-29. https://doi.org/10.1145/3533378

Rajkumar, N., Li, R., & Bahdanau, D. (2022). Evaluating the text-to-SQL capabilities of large language models. *arXiv preprint arXiv:2204.00498*. https://doi.org/10.48550/arXiv.2204.00498

Rausch, T., Lachner, C., Frangoudis, P. A., Raith, P., & Dustdar, S. (2023). Sustainable computing at the edge: A survey on energy-efficient edge AI. *ACM Computing Surveys*, 55(9), 1-39. https://doi.org/10.1145/3565803

Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., & Savarese, S. (2019). Generalized intersection over union: A metric and a loss for bounding box regression. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 658-666. https://doi.org/10.1109/CVPR.2019.00075

Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. *2011 International Conference on Computer Vision*, 2564-2571. https://doi.org/10.1109/ICCV.2011.6126544

Sarlin, P. E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020). SuperGlue: Learning feature matching with graph neural networks. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 4938-4947. https://doi.org/10.1109/CVPR42600.2020.00499

Sun, J., Shen, Z., Wang, Y., Bao, H., & Zhou, X. (2021). LoFTR: Detector-free local feature matching with transformers. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 8922-8931. https://doi.org/10.1109/CVPR46437.2021.00881

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Scialom, T. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*. https://doi.org/10.48550/arXiv.2307.09288

Wang, Y., Li, H., & Chen, X. (2024). Transformer-based image registration for urban change detection: A comparative study. *IEEE Transactions on Geoscience and Remote Sensing*, 62, 1-15. https://doi.org/10.1109/TGRS.2024.3356789

Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., ... & Stoica, I. (2018). Accelerating the machine learning lifecycle with MLflow. *IEEE Data Engineering Bulletin*, 41(4), 39-45.

Zhai, X., Kolesnikov, A., Houlsby, N., & Beyer, L. (2022). Scaling vision transformers. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 12104-12113. https://doi.org/10.1109/CVPR52688.2022.01179

Zhang, L., Wang, M., & Liu, Y. (2023). Impact of geometric misalignment on deep learning-based change detection in remote sensing. *International Journal of Applied Earth Observation and Geoinformation*, 118, 103247. https://doi.org/10.1016/j.jag.2023.103247

### Documentación Técnica y Recursos

Google Cloud. (2024). *Cloud Run pricing*. https://cloud.google.com/run/pricing

Kaggle. (2024). *State of competitive machine learning 2024*. https://www.kaggle.com/competitions

Meta AI. (2024). *Llama 3.2: Lightweight models for edge and mobile*. https://ai.meta.com/blog/llama-3-2/

NVIDIA. (2023). *GeForce RTX 4070 specifications*. https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070-family/

Ollama Team. (2024). *Ollama: Get up and running with large language models locally*. https://ollama.ai/

---

**Fin del Documento**

**Versión:** 1.0
**Fecha de Elaboración:** Noviembre 2024
**Autor:** Arthur Zizumbo
**Revisión:** Pendiente
**Próxima Actualización:** Post-Sprint 1 (Semana 2)
