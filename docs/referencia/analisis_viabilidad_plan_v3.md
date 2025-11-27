# Análisis de Viabilidad y Coherencia: Plan de Proyecto v3

Este documento presenta el análisis solicitado sobre el archivo `plan_final_v3.md`, evaluando su viabilidad técnica, coherencia metodológica y completitud.

## 1. Evaluación General
El plan es **excepcionalmente sólido, coherente y detallado**. Demuestra un entendimiento profundo no solo de las tecnologías individuales (CV, ML, Web), sino de cómo se integran en un sistema productivo bajo restricciones reales (presupuesto, hardware).

*   **Calificación General:** 9.5/10
*   **Estado:** Listo para implementación inmediata (con observaciones menores).

## 2. Viabilidad Técnica y Coherencia

### Aciertos Críticos (Puntos Fuertes)
1.  **Arquitectura Desacoplada (CV + ML):** La decisión de separar la alineación geométrica (determinista) de la detección de cambios (estocástica) es la correcta. Intentar hacer todo con una sola red neuronal (End-to-End) con solo 300 datos hubiera garantizado el fracaso.
2.  **Estrategia "Local-First":** El uso de la RTX 4070 para el "trabajo pesado" y GCP solo para servir la API es la única forma de mantener el presupuesto de $300 USD. Es una estrategia financiera impecable.
3.  **Stack Tecnológico Moderno:** El uso de **DINOv2** (SOTA en features visuales), **LoFTR** (mejor que SIFT para urbano) y **DuckDB** (analítica sin servidor) demuestra actualización tecnológica al 2024-2025.
4.  **Enfoque en Datos (Data-Centric AI):** El plan dedica el Sprint 1 entero a la calidad de los datos (alineación, limpieza), lo cual es lo que distingue a los proyectos de ML exitosos de los académicos.

### Coherencia Metodológica
El plan sigue una lógica impecable:
1.  No se puede clasificar lo que no está alineado -> **Sprint 1: Alineación**.
2.  No se puede entrenar sin features robustas (pocos datos) -> **Sprint 2: Transfer Learning**.
3.  No sirve de nada si no se puede visualizar -> **Sprint 3: Dashboard**.

## 3. ¿Qué falta? (Gaps Identificados)

Aunque el plan es muy completo, existen áreas grises que podrían bloquear la implementación si no se definen:

### A. Herramienta de Etiquetado (Human-in-the-Loop)
*   **El Problema:** La US-020 menciona un "Loop de Active Learning" donde se envían 50 manzanas difíciles a un experto.
*   **Lo que falta:** ¿Dónde las etiqueta el experto? El plan menciona un dashboard de visualización (Sprint 3), pero no especifica si tendrá capacidades de **edición/escritura** para corregir etiquetas.
*   **Recomendación:** Agregar una tarea en el Sprint 3 para habilitar un botón de "Corregir Clasificación" en el frontend que actualice la base de datos (DuckDB/Parquet).

### B. Manejo de Proyecciones (CRS)
*   **El Problema:** Se menciona "homologación de sistemas", pero no se define el CRS canónico del proyecto (ej. EPSG:32614 para metros UTM vs EPSG:4326 para lat/lon).
*   **Riesgo:** Calcular áreas y distancias (métricas geométricas) en lat/lon (grados) da resultados erróneos.
*   **Recomendación:** Definir explícitamente en la US-005 que toda geometría debe reproyectarse a un sistema proyectado local (UTM) antes de rasterizar o calcular métricas.

## 4. Nivel Necesario para Implementación

Para ejecutar este plan en 6 semanas (unipersonal), se requiere un perfil **Senior Full Stack ML Engineer** con las siguientes competencias específicas:

1.  **Nivel Experto (Senior):**
    *   **Python/MLOps:** Capacidad para configurar Docker, DVC, Poetry y GPU drivers sin perder días en "configuration hell".
    *   **Geometría Computacional:** Entender transformaciones afines, sistemas de coordenadas y operaciones vectoriales (Shapely/GeoPandas).

2.  **Nivel Intermedio (Mid-Level):**
    *   **Computer Vision:** Uso de OpenCV y Kornia. No se requiere investigar nuevos algoritmos, solo implementar los existentes.
    *   **Web Dev:** Vue/Nuxt y FastAPI. El dashboard no requiere complejidad extrema de UI, solo funcionalidad.

3.  **Nivel Básico:**
    *   **Cloud:** Desplegar en Cloud Run es sencillo si el contenedor ya funciona localmente.

**Veredicto:** El plan está diseñado *por* un experto *para* un experto. Un junior o mid-level colapsaría intentando integrar todas estas piezas (CV, Geo, Web, Cloud) en 6 semanas.

## 5. Conclusión
El plan `plan_final_v3.md` está **completo en un 95%**.
*   **¿Se puede empezar ya?** SÍ.
*   **¿Es viable?** SÍ, altamente viable bajo la arquitectura propuesta.
*   **¿Qué agregar?** Solo la definición de la interfaz de etiquetado (feedback loop) y la verificación legal de los datos.
