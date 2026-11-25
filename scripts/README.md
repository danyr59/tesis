# Scripts y Utilidades del Proyecto

Este directorio contiene herramientas de preparación de datos, indexación RAG, descarga de modelos y entrenamiento local CLI para los modelos docentes.

---

## 🛠️ Herramientas de Datos y RAG

| Script | Descripción | Uso Típico |
| :--- | :--- | :--- |
| **`download_assets.py`** | **Inicialización de Assets desde Kaggle:** Lee `.env` para descargar automáticamente la base RAG ChromaDB y el adaptador LoRA del Maestro configurado (`qwen`, `llama`, `gemma`) desde los repositorios oficiales de Kaggle. | `python scripts/download_assets.py` |
| **`build_rag_from_jsonl.py`** | **Indexador Oficial:** Construye la colección vectorial en ChromaDB a partir del corpus de semillas TSQA (`data/time_series_generation.jsonl`) con soporte para GPU/MPS y procesamiento por lotes dinámico. | `python scripts/build_rag_from_jsonl.py` |
| **`extraer_rag.py`** | Script de prueba para consultar e inspeccionar interactivamente las semillas recuperadas por similitud semántica. | `python scripts/extraer_rag.py` |
| **`setup_models.py`** | Descarga y valida los pesos de los modelos base requeridos desde Hugging Face. | `python scripts/setup_models.py` |

---

## 🧠 Entrenamiento y Fine-Tuning CLI (`finetune_maestro.py`)

Permite ejecutar el fine-tuning supervisado (SFT QLoRA 4-bit) de los Modelos Maestros (Qwen2.5-7B, LLaMA-3.1-8B, Gemma-2-9B o modelos personalizados) de manera directa desde la terminal sin depender de un entorno Jupyter interactivo:

```bash
# Fine-tuning estándar del maestro Qwen (1 época)
python scripts/finetune_maestro.py --model qwen --epochs 1 --batch_size 2

# Fine-tuning de LLaMA con dataset específico y salida personalizada
python scripts/finetune_maestro.py --model llama --dataset data/time_series_generation.jsonl --output_dir models/adapter_maestro_llama

# Fine-tuning rápido de prueba (500 muestras) en CPU/MPS sin cuantización 4-bit
python scripts/finetune_maestro.py --model qwen --max_samples 500 --no_4bit
```

### Argumentos CLI Disponibles:
- `--model`: Preset (`qwen`, `llama`, `gemma`) o identificador completo de Hugging Face.
- `--dataset`: Ruta al archivo `.jsonl` de entrenamiento.
- `--output_dir`: Directorio de destino para los adaptadores LoRA (default: `models/adapter_maestro_{familia}`).
- `--epochs`: Número de épocas de optimización (default: 1).
- `--batch_size`: Tamaño del lote por dispositivo (default: 2).
- `--grad_accum`: Pasos de acumulación de gradiente (default: 4).
- `--lr`: Tasa de aprendizaje (default: `2e-4`).
- `--max_seq_length`: Longitud máxima de secuencia en tokens (default: 768).
- `--lora_r` / `--lora_alpha`: Rango y escalado LoRA (default: 16 y 32).
- `--max_samples`: Limita el número de muestras (ideal para validación rápida).
- `--no_4bit`: Desactiva cuantización 4-bit para entornos sin CUDA (Apple Silicon / CPU).
- `--push_to_hub`: Repositorio de Hugging Face para publicar el adaptador automáticamente tras entrenar.
