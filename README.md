# Arquitectura Maestro-Estudiante para la Generación de Series Temporales Sintéticas

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20T4%20%2F%20P100-20BEFF.svg)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Este repositorio contiene la implementación oficial del proyecto de tesis:
**«Generación de Series Temporales Sintéticas mediante Destilación Maestro-Estudiante con RAG y Validación Numérica»**.

El sistema propone un marco neuro-simbólico multi-agente que desacopla la semántica morfológica de la escala física absoluta, permitiendo a Modelos de Lenguaje Pequeños (SLMs de 0.5B a 3B) generar series temporales largas, continuas y de alta fidelidad con **consumo estrictamente acotado en GPU de consumo ($\le 7.20$\,GB de VRAM)** y **cero violaciones de cotas físicas ($0.00\%$)**, superando las limitaciones de colapso de modo en GANs y la necesidad de supercómputo en LLMs multimodales gigantescos.

---

## 🏛️ Arquitectura del Sistema

El flujo de trabajo desacopla el razonamiento cognitivo de la proyección determinista mediante una orquestación secuencial con purga de tensores en memoria:

```
[Entrada en Lenguaje Natural] (CLI o Query abierta)
              │
              ▼
    ┌──────────────────────────────────────────────┐
    │  1. AGENTE MAESTRO (TeacherAgent + RAG TSQA) │
    │  • Interpretación de dominio y escala física │
    │  • Clasificación de perfil temporal (P1..P6) │
    │  • Deducción de cotas analíticas [ymin, ymax]│
    │  • Filtrado cognitivo de 50 semillas RAG     │
    └──────────────────────────────────────────────┘
              │
              ▼ (Descarga de tensores, gc.collect, empty_cache)
    ┌──────────────────────────────────────────────┐
    │  2. AGENTE ESTUDIANTE (StudentAgent + QLoRA) │
    │  • Destilación ligera supervisada (SFT)      │
    │  • Inferencia morfológica normalizada en [0,1]│
    │  • Estructuración en formato JSON estricto    │
    └──────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────────────────────┐
    │  3. DESACOPLAMIENTO ESCALAR DETERMINISTA     │
    │  • Proyección analítica: y_t = ymin + y~_t·Δy│
    │  • Extrapolación estocástica multiserie      │
    │  • Generación continua de T=8.000 puntos     │
    └──────────────────────────────────────────────┘
              │
              ▼ (Exportación: synthetic_series.csv)
    ┌──────────────────────────────────────────────┐
    │  4. VALIDACIÓN NUMÉRICA Y DOWNSTREAM (M6-M8) │
    │  • M6: Modelado dinámico causal con LSTM     │
    │  • M7: Métricas predictivas (RMSE, MAE, R²)  │
    │  • M8: Tasa de alucinación semántica H_rate  │
    └──────────────────────────────────────────────┘
```

---

## 📊 Comparación Cuantitativa con el Estado del Arte

Resultados empíricos consolidados frente a los principales enfoques de la literatura científica:

| Enfoque / Trabajo | Hardware Mínimo | Datos Previos ($N_{\text{datos}}$) | Longitud Máx. ($T$) | Tasa de Inconsistencia | Retención Estructural | Desempeño Downstream |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **TimeGAN** *(Yoon et al., 2019)* | GPU dedicada ($>12$\,GB) | $>10.000$ series reales | $T \le 100$ | Colapso modal estocástico | Generador directo continuo | $R^2 \approx 0.60 - 0.70$ (LSTM) |
| **SDForger** *(IBM Research, 2025)* | NVIDIA V100 / A100 ($100$\,GB) | Cientos de series tabulares | $T \le 500$ | $3.87\%$ a $58.82\%$ NaNs | Filtro $L_2$ (norma $19.06$ vs $1.71$) | RMSE $0.48 - 0.62$ (TTM) |
| **Montes Trujillo** *(UC3M, 2025)* | GPU local / Ollama ($\ge 16$\,GB) | $0$ (prompts estáticos) | $T \le 60$ | $>93\%$ falla estricta en 1B | Curación forzada en $>51\%$ puntos | No implementado |
| **ChatTS** *(Tsinghua / ByteDance, 2025)* | Clúster $8\times$ NVIDIA A800 | $1.34 \times 10^5$ pares sintéticos | $T \le 1.024$ | Desalineación de parches locales | Proyección latente MLP (5 capas) | Precisión relativa $0.78$ |
| **Nuestra Propuesta (Esta Tesis)** | **GPU Consumo ($\le 7.20$\,GB)** | **50 semillas RAG** *(abiertas)* | **$T = 8.000$** ($160.000$\,pts/lote) | **$0.00\%$ violaciones cota** | **$100\%$ JSON válido directo** | **MAE $\le 1.05$\,lpm**, **$R^2 > 0.982$** |

*Nota: $\mathcal{H}_{\text{rate}} \le 0.26\%$ denota la tasa de alucinación semántica.*

---

## 📁 Estructura del Repositorio

```
proyecto_tesis/
├── pipeline/                 # Núcleo del framework neuro-simbólico
│   ├── config/               # Configuraciones, perfiles P1-P6 y settings
│   ├── teacher_agent/        # Agente Maestro (Razonamiento LoRA + RAG ChromaDB)
│   ├── student_agent/        # Agente Estudiante (Inferencia ligera QLoRA 4-bit)
│   ├── numeric_pipeline.py   # Desacoplamiento escalar y orquestación M6-M8
│   ├── training/             # M6: Red recurrente LSTM para modelado predictivo
│   ├── prediction/           # M7: Inferencia causal y métricas (RMSE, MAE, R²)
│   ├── evaluation/           # M8: Detector de alucinaciones (H_rate) y Juez
│   ├── utils/                # Carga de modelos, cuantización y logging
│   └── main.py               # Punto de entrada ejecutable CLI
│
├── notebooks_kaggle/         # Cuadernos Jupyter para ejecución en Kaggle (GPU)
│   ├── 00_crear_rag_kaggle.ipynb              # Construcción e indexación del RAG en Kaggle
│   ├── 01_finetuning_maestro_qwen_kaggle.ipynb
│   ├── 02_finetuning_maestro_llama_kaggle.ipynb
│   ├── 03_finetuning_maestro_gemma_kaggle.ipynb
│   ├── orquestador_experimentos_kaggle.ipynb  # Matriz completa de 54 experimentos
│   └── README.md                              # Guía paso a paso de uso en Kaggle
│
├── scripts/                  # Scripts auxiliares y herramientas CLI
│   ├── download_assets.py                     # Descarga automatizada de RAG y Adaptadores LoRA de Kaggle
│   ├── finetune_maestro.py                    # Fine-tuning CLI supervisado (QLoRA 4-bit)
│   ├── build_rag_from_jsonl.py                # Construcción del índice vectorial ChromaDB
│   ├── extraer_rag.py                         # Inspección de semillas del corpus TSQA
│   ├── setup_models.py                        # Descarga y verificación de checkpoints HF
│   └── README.md
│
├── resultados/               # Banco integral de resultados y trazas forenses
│   ├── E1/ ... E9/                            # 54 ejecuciones factoriales completas
│   │   ├── P1_ESTABLE ... P6_MULTIVARIADO     # config_generacion.json, metricas, csv, plots
│   ├── reproducibilidad/                      # Réplicas estocásticas inter-corrida (E1, E3, E5)
│   ├── logit_kd/                              # Ablación Hinton Knowledge Distillation (E1)
│   ├── resumen_general_todos_escenarios.csv   # Matriz factorial consolidada
│   └── README.md                              # Guía exhaustiva de auditoría de resultados
│
├── rag/                      # Base de conocimiento vectorial RAG
│   ├── README.md                              # Documentación del índice vectorial ChromaDB
│   └── .gitkeep
│
├── data/                     # Directorio de datos y salidas sintéticas
│   ├── README.md
│   └── .gitkeep
│
├── requirements.txt          # Dependencias de Python auditadas
├── .env.example              # Plantilla de variables de entorno
├── .gitignore                # Reglas de exclusión de Git
└── README.md                 # Documentación principal del repositorio
```

---

## 🧠 Catálogo de Modelos Docentes y Recomendaciones de Selección

El marco neuro-simbólico fue validado rigurosamente sobre tres familias fundacionales de última generación. Para facilitar la experimentación sin costo de cómputo previo, se disponen de **adaptadores LoRA preentrenados y curados** alojados en repositorios públicos oficiales en Kaggle:

| Familia Docente | Modelo Maestro Oficial | Parámetros | Ventana de Contexto | Adaptador Preentrenado en Kaggle | Estudiante Recomendado |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **Qwen 2.5** | `Qwen/Qwen2.5-7B-Instruct` | 7.61B | 128k tokens | [`danielrangelsanch/maestro-qwen-adapter`](https://www.kaggle.com/datasets/danielrangelsanch/maestro-qwen-adapter) | `Qwen/Qwen2.5-1.5B-Instruct` |
| **LLaMA 3.1** | `meta-llama/Llama-3.1-8B-Instruct` | 8.03B | 128k tokens | [`danielrangelsanch/maestro-llama-adapter`](https://www.kaggle.com/datasets/danielrangelsanch/maestro-llama-adapter) | `meta-llama/Llama-3.2-1B-Instruct` |
| **Gemma 2** | `google/gemma-2-9b-it` | 9.24B | 8k tokens (*SWA*) | [`danielrangelsanch/maestro-gemma-adapter`](https://www.kaggle.com/datasets/danielrangelsanch/maestro-gemma-adapter) | `google/gemma-2-2b-it` |

---

### ⚙️ Guía de Dimensionamiento y Presupuesto de Hardware (VRAM)

Si decides utilizar modelos distintos a los predeterminados, selecciona los tamaños siguiendo esta matriz de compatibilidad según tu hardware disponible:

| Hardware / VRAM | Modelo Docente Máximo | Modelo Estudiante Recomendado | Modo de Cuantización | Consumo VRAM Pico |
| :--- | :---: | :---: | :---: | :---: |
| **GPU de Consumo (8 GB)** | $\le 7\text{B}$ / $8\text{B}$ parámetros | $\le 1.5\text{B}$ parámetros | 4-bit NF4 (BitsAndBytes) | $\approx 5.8 - 7.2\text{ GB}$ |
| **GPU Media (12 - 16 GB)** | $\le 8\text{B}$ / $9\text{B}$ parámetros | $\le 3\text{B}$ parámetros | 4-bit NF4 o FP16 | $\approx 8.5 - 11.0\text{ GB}$ |
| **GPU Alta / Servidor ($\ge 24\text{ GB}$)** | $\le 14\text{B}$ parámetros | $\le 7\text{B}$ parámetros | FP16 / BF16 nativo | $\approx 16.0 - 22.0\text{ GB}$ |
| **Apple Silicon (MPS $\ge 16\text{ GB}$)** | $\le 7\text{B}$ / $8\text{B}$ parámetros | $\le 1.5\text{B}$ parámetros | FP32 / FP16 unificado | $\approx 10.0\text{ GB}$ RAM |
| **Solo CPU ($\ge 32\text{ GB}$ RAM)** | $\le 7\text{B}$ parámetros | $\le 1.5\text{B}$ parámetros | FP32 (Inferencia lenta) | $\approx 18.0\text{ GB}$ RAM |

> [!IMPORTANT]
> **¿Por qué la cota recomendada es $\le 7\text{B}-8\text{B}$ para el Maestro y $\le 1.5\text{B}-3\text{B}$ para el Estudiante?**
> 1. **Presupuesto térmico y memoria:** Permite la orquestación secuencial completa en una única GPU comercial de 8 GB (como NVIDIA RTX 3070/4060 o Kaggle T4) sin desbordamiento de memoria (*Out of Memory*).
> 2. **Rendimiento morfológico:** Los modelos de 7B/8B Instruct poseen capacidad semántica óptima para interpretar dominios físicos y cotas analíticas sin necesidad de recurrir a modelos de 70B.
> 3. **Velocidad de destilación:** El estudiante ligero (1B-1.5B) aprende los patrones temporales en menos de 3 minutos de inferencia QLoRA.

---

### ⚠️ Advertencia Crítica: Modelos de Razonamiento (Chain-of-Thought)

> [!WARNING]
> **NO utilizar modelos de razonamiento como DeepSeek-R1, QwQ-32B o variantes similares con pensamiento implícito.**
> 
> **Razón técnica:** Estos modelos emiten cadenas no estructuradas de pensamiento reflexivo delimitadas por etiquetas `<think> ... </think>` antes de dar su respuesta final. En nuestra arquitectura:
> - **Rompe la extracción JSON:** El parser sintáctico del Agente Maestro espera un objeto JSON directo con las cotas físicas $[y_{\min}, y_{\max}]$ y el perfil temporal. El preámbulo de pensamiento corrompe la deserialización (`json.loads`).
> - **Agotamiento de contexto:** Los tokens de pensamiento consumen la ventana de contexto de generación, provocando truncamiento de la serie morfológica.
> 
> **Recomendación estricta:** Emplear únicamente modelos convencionales de alineación instructiva que finalicen en **`Instruct`** (Qwen, LLaMA, Mistral) o **`-it`** (Gemma).

---

### 🎓 Cómo Hacer Fine-Tuning a un Nuevo Modelo Docente (Paso a Paso)

Si deseas utilizar un modelo docente que **no** pertenece a los 3 preevaluados (por ejemplo `mistralai/Mistral-7B-Instruct-v0.3`, `microsoft/Phi-3-mini-4k-instruct` o entrenar tu propia variante de Qwen/LLaMA):

#### Opción A: Entrenamiento Local por CLI (`scripts/finetune_maestro.py`)

El script implementa optimización supervisada eficiente mediante **QLoRA en 4 bits (NF4 con doble cuantización)**, requiriendo únicamente $\approx 6\text{ GB}$ de VRAM:

```bash
# Ejemplo: Fine-tuning del modelo docente con el dataset TSQA
python scripts/finetune_maestro.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset data/time_series_generation.jsonl \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4 \
    --max_seq_length 1024 \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir models/teacher_adapters/nuevo_maestro
```

**Parámetros clave explicados:**
- `--model`: Identificador del modelo en Hugging Face Hub (o ruta local a los pesos).
- `--dataset`: Ruta al archivo `.jsonl` con los pares instrucción-serie de entrenamiento (por defecto busca en `data/`).
- `--epochs`: Número de épocas de ajuste supervisado (recomendado: 3 a 5).
- `--batch_size` y `--grad_accum`: Tamaño de lote por paso (2) y pasos de acumulación de gradiente (4), resultando en un *batch size* efectivo de $2 \times 4 = 8$.
- `--lora_r` y `--lora_alpha`: Rango intrínseco de LoRA ($r=16$) y factor de escala ($\alpha=32$).
- `--output_dir`: Directorio donde se guardarán los pesos del adaptador (`adapter_model.safetensors` y `adapter_config.json`).
- `--push_to_hub` *(opcional)*: Sube automáticamente el adaptador entrenado a tu cuenta de Hugging Face Hub.

#### Opción B: Entrenamiento en la Nube (Kaggle con GPU T4)

Si deseas reentrenar los maestros en la nube o adaptar una nueva arquitectura, dispones de los cuadernos preparados en [`notebooks_kaggle/`](notebooks_kaggle/):
- **`01_finetuning_maestro_qwen_kaggle.ipynb`**
- **`02_finetuning_maestro_llama_kaggle.ipynb`**
- **`03_finetuning_maestro_gemma_kaggle.ipynb`**

> [!TIP]
> **Duración y Costo Computacional (4 a 6 horas por Maestro):**
> En Kaggle con acelerador NVIDIA Tesla T4/P100, el ciclo completo de fine-tuning QLoRA sobre el corpus TSQA toma **entre 4 y 6 horas por modelo docente**.
> **Por esta razón, se recomienda fuertemente utilizar los adaptadores oficiales ya preentrenados y alojados en KaggleHub** (descargables en segundos mediante `python scripts/download_assets.py`), reservando el reentrenamiento únicamente para experimentaciones con nuevas arquitecturas docentes.

#### Cómo Vincular el Nuevo Modelo al Pipeline en `.env`

Una vez generado el adaptador, simplemente actualiza tu archivo `.env`:

```env
# Configuración del nuevo docente
TEACHER_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
TEACHER_MODEL_PATH_ADAPTER=./models/teacher_adapters/nuevo_maestro

# Configuración del estudiante discente
STUDENT_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
```

El pipeline detectará automáticamente la ruta del adaptador y cargará los pesos durante la etapa del Agente Maestro sin requerir cambios en el código fuente.

---

## 🚀 Guía de Instalación y Ejecución Local

### 1. Requisitos de Hardware y Software
- **Python:** 3.10 o superior.
- **GPU:** NVIDIA (>= 8 GB VRAM recomendada) o Apple Silicon (MPS) o CPU.
- **CUDA:** 11.8 o 12.1+ (si se usa GPU NVIDIA con cuantización BitsAndBytes).

### 2. Clonar el Repositorio y Crear el Entorno Virtual
```bash
git clone https://github.com/danielrangelsanch/proyecto_tesis.git
cd proyecto_tesis

python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuración de Variables de Entorno
Copia la plantilla y configura tus credenciales:
```bash
cp .env.example .env
```
Edita `.env` e introduce tu token de Hugging Face (`HF_TOKEN`) para descargar los modelos base.

### 4. Inicialización Automática de Assets (RAG y Adaptadores LoRA)
El script `scripts/download_assets.py` lee directamente tu configuración en `.env` (variables `TEACHER_MODEL_ID` y `RAG_DB_PATH`) y descarga automáticamente sin necesidad de pasar argumentos:
- Verifica si la base RAG ya existe en `rag/maestro`; si no existe, la descarga desde Kaggle (`danielrangelsanch/rag-maestro`).
- Verifica si el adaptador LoRA del Maestro configurado en `.env` existe; si falta y pertenece al catálogo oficial, lo descarga e instala en su ruta.
```bash
# Inicialización automática guiada por .env (Recomendado)
python scripts/download_assets.py

# Opcional: Descargar los 3 adaptadores del catálogo al mismo tiempo
python scripts/download_assets.py --all-adapters
```

### 5. Indexación del Corpus RAG (Opcional)
Si prefieres construir la base vectorial localmente desde cero a partir del archivo JSONL de semillas abiertas de TSQA:
```bash
python scripts/build_rag_from_jsonl.py
```

### 6. Ejecutar el Pipeline de Generación
Ejecución interactiva o mediante consulta declarativa:
```bash
# Modo directo con requerimiento clínico (Perfil P1)
python pipeline/main.py --query "Necesito generar lecturas de frecuencia cardíaca de un paciente en reposo monitoreado en una clínica, con bajo nivel de ruido y valores estables alrededor de 70 lpm."

# Modo interactivo (el Maestro solicitará los parámetros faltantes)
python pipeline/main.py
```

### 7. Ejecutar la Validación Numérica (M6–M8)
Una vez generadas las series (`data/synthetic/synthetic_series.csv`), ejecuta la validación downstream con LSTM:
```bash
python pipeline/numeric_pipeline.py
```

---

## ☁️ Experimentación en la Nube (Kaggle)

Para reproducir los experimentos completos sin consumir recursos locales:

1. Importa cualquiera de los cuadernos ubicados en [`notebooks_kaggle/`](notebooks_kaggle/):
   - **`01_finetuning_maestro_qwen_kaggle.ipynb`**, **`02_..._llama_...`**, **`03_..._gemma_...`**: Para entrenar los adaptadores LoRA de los Maestros (Qwen-2.5-7B, LLaMA-3.1-8B o Gemma-2-9B).
   - **`orquestador_experimentos_kaggle.ipynb`**: Para reproducir de forma autónoma la matriz completa de **54 ejecuciones experimentales** ($9 \text{ escenarios} \times 6 \text{ perfiles}$).
2. Habilita el acelerador **GPU T4 x2** o **GPU P100** e **Internet On**.
3. Añade tu `HF_TOKEN` en **Add-ons $\to$ Secrets**.
4. Consulta las instrucciones detalladas en [`notebooks_kaggle/README.md`](notebooks_kaggle/README.md).

---

## 📈 Perfiles Temporales Formales Evaluados

El catálogo en [`pipeline/config/profiles.py`](pipeline/config/profiles.py) modela 6 perfiles con dinámicas físicas contrastantes:

- **$P_1$ (Estable):** Régimen estacionario con ruido blanco gaussiano acotado (ej. pulso basal en reposo).
- **$P_2$ (Estacional Puro):** Periodicidad armónica sinusoidal con estacionalidad fija (ej. afluencia de transporte público).
- **$P_3$ (Tendencia + Estacionalidad):** Crecimiento secular monótono superpuesto con oscilaciones estacionales (ej. ventas retail).
- **$P_4$ (Tendencia + Ruido Estocástico):** Deriva no estacionaria con variabilidad browniana y volatilidad local (ej. cotizaciones bursátiles).
- **$P_5$ (Dinámica Compleja):** Multi-régimen con transiciones abruptas y ciclos no lineales (ej. tráfico de ancho de banda en servidores).
- **$P_6$ (Multivariado Acoplado):** Series bivariadas con dependencia cruzada y escalado desacoplado por canal (ej. presión arterial sistólica y diastólica).

---

## 📜 Licencia y Cita Académica

Este proyecto está distribuido bajo la licencia MIT. Si utilizas este código o los resultados en tu investigación, por favor cita:

```bibtex
@mastersthesis{rangel2026series,
  author  = {Daniel Rangel Sánchez},
  title   = {Generación de Series Temporales Sintéticas mediante Destilación Maestro-Estudiante con RAG y Validación Numérica},
  school  = {Facultad de Ingeniería, Universidad de Los Andes / Universidad Internacional de Valencia},
  year    = {2026}
}
```
