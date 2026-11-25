# 🔍 Base de Conocimiento RAG (ChromaDB)

Este directorio almacena la base de datos vectorial de recuperación aumentada (**Retrieval-Augmented Generation - RAG**) utilizada por el **Agente Maestro** para contextualizar la generación de series temporales con prototipos empíricos reales.

---

## 🏛️ Arquitectura del RAG

- **Colección:** `maestro_time_series` (alojada en `rag/maestro`).
- **Motor Vectorial:** [ChromaDB](https://www.trychroma.com/) en modo persistente cliente/servidor embebido (SQLite + índices HNSW).
- **Modelo de Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (dimensionalidad: 384, normalización $L_2$).
- **Corpus Fuente:** Dataset TSQA de series temporales (`data/time_series_generation.jsonl`).
- **Semillas Recuperadas ($k$):** Configurado a $k = 50$ prototipos candidatos (`RAG_TOP_K=50`). El Maestro evalúa semántica y cotas físicas de estas 50 semillas para seleccionar las óptimas.

---

## 🚀 Cómo Generar o Actualizar la Base RAG

### Opción 1: Localmente mediante CLI (GPU / MPS / CPU)
```bash
python scripts/build_rag_from_jsonl.py --dataset data/time_series_generation.jsonl --output rag/maestro
```

### Opción 2: En la nube mediante Kaggle
Ejecutar el cuaderno [`notebooks_kaggle/00_crear_rag_kaggle.ipynb`](file:///Users/daniel/source/repo/proyecto_tesis/notebooks_kaggle/00_crear_rag_kaggle.ipynb), el cual procesa el corpus en una GPU Tesla T4 y genera el archivo comprimido `rag_maestro.zip`.

---

## 📦 Gestión de Almacenamiento y Archivos Pesados

> [!NOTE]
> Una base de datos ChromaDB indexada con el corpus completo genera aproximadamente **3.3 GB** en disco (`chroma.sqlite3` ~2.9 GB más los índices vectoriales HNSW `data_level0.bin` ~590 MB).
> Debido a que GitHub impone un límite estricto de **100 MB por archivo**, la base de datos binaria cruda no se versiona en el repositorio Git tradicional.

Para su distribución sin fricción:
1. **En Kaggle:** Se adjunta directamente como dataset público (`danielrangelsanch/rag-maestro`), donde el orquestador lo monta automáticamente en `/kaggle/input/rag-maestro`.
2. **En Local:** Se puede generar en ~5 minutos con `scripts/build_rag_from_jsonl.py` o descargar la versión preconstruida desde los *Releases* o Hugging Face.
