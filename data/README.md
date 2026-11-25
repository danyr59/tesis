# Directorio de Datos (`data/`)

Este directorio aloja los conjuntos de datos de semillas temporales y las salidas generadas por el pipeline.

---

## 📂 Archivos Esperados

- **`time_series_generation.jsonl`** / **`TS_Dataset.jsonl`**:
  Dataset de semillas de series temporales del benchmark **TSQA** utilizado por el Módulo Maestro para la base de datos vectorial ChromaDB.
  - Cada registro contiene: `id`, `metadata` (`original_domain`, `task_type`), `input` (pregunta o contexto) y `output` (serie temporal de referencia).

- **`synthetic/`**:
  Carpeta donde el pipeline exporta las series temporales generadas (`synthetic_series.csv`) y los gráficos de validación.

- **`checkpoints/`**:
  Puntos de control generados durante el entrenamiento de la red LSTM predictiva (M6).

---

> [!NOTE]
> Por motivos de tamaño, los archivos `.jsonl` voluminosos (>70 MB) están excluidos del seguimiento de Git vía `.gitignore`. Pueden descargarse desde el repositorio del benchmark TSQA o generarse con los scripts en `scripts/`.
