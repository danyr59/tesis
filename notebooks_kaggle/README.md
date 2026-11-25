# Cuadernos de Kaggle — Experimentación y Entrenamiento

Este directorio contiene los Jupyter Notebooks optimizados para ejecutarse en **Kaggle** sobre aceleradores GPU gratuitos (**NVIDIA Tesla T4 x2** o **NVIDIA Tesla P100 de 16 GB**).

---

## 📋 Catálogo de Cuadernos

| Cuaderno | Propósito | Hardware Recomendado | Tiempo Estimado |
| :--- | :--- | :--- | :--- |
| **`00_crear_rag_kaggle.ipynb`** | **Indexación Vectorial (RAG):** Procesa el corpus TSQA y genera la base de datos ChromaDB persistente (`all-mpnet-base-v2`) comprimiéndola para descarga o conexión como dataset. | GPU T4 / P100 o CPU | ~10-15 min |
| **`01_finetuning_maestro_qwen_kaggle.ipynb`** | Fine-Tuning QLoRA (4-bit) de `Qwen/Qwen2.5-7B-Instruct` como Agente Maestro sobre corpus TSQA. | GPU T4 / P100 (16 GB) | ~4 - 5 horas (3 épocas) |
| **`02_finetuning_maestro_llama_kaggle.ipynb`** | Fine-Tuning QLoRA (4-bit) de `meta-llama/Llama-3.1-8B-Instruct` como Agente Maestro. | GPU T4 / P100 (16 GB) | ~4.5 - 5.5 horas (3 épocas) |
| **`03_finetuning_maestro_gemma_kaggle.ipynb`** | Fine-Tuning QLoRA (4-bit) de `google/gemma-2-9b-it` como Agente Maestro. | GPU T4 / P100 (16 GB) | ~5 - 6 horas (3 épocas) |
| **`orquestador_experimentos_kaggle.ipynb`** | **Orquestador Principal:** Ejecuta de forma secuencial la matriz completa de 54 experimentos ($9 \text{ escenarios} \times 6 \text{ perfiles}$), con purga forzada de tensores entre el Maestro y los Estudiantes. | GPU T4 x2 / P100 (16 GB) | ~4.5 - 6 horas (por bloque) |

---

## ⚙️ Configuración Previa en Kaggle

Para ejecutar cualquiera de estos cuadernos en Kaggle:

1. **Crear o Importar el Notebook:**
   - En Kaggle, ve a **Code** $\to$ **New Notebook** $\to$ **File** $\to$ **Import Notebook** y sube el archivo `.ipynb` correspondiente.

2. **Habilitar Acelerador y Acceso a Internet:**
   - En el panel derecho (*Notebook options*):
     - **Accelerator:** Seleccionar `GPU T4 x2` o `GPU P100`.
     - **Internet:** Activar (`Internet on`).

3. **Configurar Secrets (Credenciales):**
   - En el menú superior de Kaggle: **Add-ons** $\to$ **Secrets**.
   - Añadir los siguientes secretos:
     - `HF_TOKEN`: Tu token de Hugging Face (con permisos de lectura para modelos con gating como LLaMA-3.1 o Gemma-2).
     - `GITHUB_TOKEN` *(opcional)*: Token personal de GitHub si el repositorio es privado.

4. **Persistencia de Resultados:**
   - Todos los resultados, gráficos y modelos entrenados se guardan automáticamente en `/kaggle/working/`, listos para ser descargados directamente desde la interfaz de Kaggle.
