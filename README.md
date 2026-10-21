# Proyecto de Tesis: Generación de Series Temporales Sintéticas mediante Destilación Maestro-Estudiante con RAG y Validación Numérica

Este proyecto de tesis implementa una arquitectura híbrida neuro-simbólica multi-agente para la **generación autónoma de series temporales sintéticas de alta fidelidad**. El sistema integra un agente **Maestro experto** con adaptadores LoRA y acceso a memoria vectorial (RAG), un agente **Estudiante adaptativo** con destilación *on-the-fly*, desacoplamiento morfológico-escalar con diferenciación estocástica multiserie, y un pipeline numérico de validación empírica basado en Redes Neuronales Recurrentes (LSTM), detección estadística de alucinaciones y evaluación cualitativa con LLM como juez.

---

## 🚀 Arquitectura y Flujo de Ejecución Secuencial

Para optimizar recursos de cómputo y memoria VRAM (garantizando compatibilidad en entornos con límites de GPU como Kaggle o Colab) y asegurar el rigor metodológico, el pipeline ejecuta un flujo secuencial estructurado en las siguientes fases:

```
[Usuario CLI]
      │
      ▼
[1. TeacherAgent (Maestro LoRA)]
      ├── Interpretación del Requerimiento (Variables, Escala, Perfil Temporal P1..P5)
      ├── Consulta RAG TSQA (ChromaDB Vector Store)
      └── Filtrado Cognitivo por Lotes de Semillas (Diversidad & Relevancia)
      │
      ▼
[2. Descarga y Purga de Memoria VRAM/GPU] (del teacher, gc.collect, empty_cache)
      │
      ▼
[3. StudentAgent (Estudiante)]
      ├── Razonamiento de Hiperparámetros (Épocas, Batch Size, LR)
      ├── Destilación LoRA On-the-fly (Fine-Tuning con semillas del Maestro)
      ├── Generación de Semillas Iniciales (10 puntos morfológicos guiados por EPIC)
      └── Desacoplamiento Escalar y Extrapolación Estocástica Multiserie (extend_series)
      │
      ▼ (Exportación tabular: synthetic_series.csv)
[4. Pipeline Numérico Autónomo]
      ├── M6: Entrenamiento de Red Neuronal Recurrente (LinearRNN LSTM)
      ├── M7: Inferencia y Métricas de Rendimiento (RMSE, MAE, MAPE, R²)
      └── M8: Evaluación Dual (Estudiante como Juez + Detector de Alucinaciones)
```

### 1. Entrada del Usuario e Interpretación (Maestro)
- El usuario ingresa su requerimiento vía CLI (`pipeline/main.py`), interactiva o mediante el argumento `--query`.
- El **Maestro** (`pipeline/teacher_agent/agent.py`), respaldado por adaptadores LoRA especializados en razonamiento de series temporales (`models/teacher_adapters`), analiza la solicitud.
- Extrae de forma estructurada:
  - **Dominio y Tema:** Ej. Retail, Medicina, Finanzas, Transporte.
  - **Variable Estricta:** Respeta nombres explícitos sin alucinar columnas colaterales.
  - **Escala Objetivo (`target_value`):** Extrae la magnitud solicitada (ej. 70 lpm, 2500 USD, 15000 pasajeros).
  - **Tipo de Serie:** `univariada` o `multivariada`.
  - **Perfil Matemático (`profile_id`):** Clasifica en uno de los 5 perfiles formales (`P1_ESTABLE` a `P5_COMPLEJO`).
- Si faltan parámetros clave o la confianza es baja, el Maestro abre un diálogo interactivo para clarificar.

### 2. Recuperación RAG y Filtrado Cognitivo (ChromaDB)
- El Maestro formula un plan de búsqueda y consulta la base de conocimiento vectorial **TSQA** indexada en ChromaDB (`rag/maestro`).
- Realiza una evaluación por lotes (`_evaluate_documents_in_batches`) evaluando la coherencia semántica, física y la diversidad estadística de las semillas encontradas.
- Produce un **Dataset Semilla** validado acompañado de un **Resumen de Calidad** justificativo.

### 3. Descarga Secuencial de Memoria (VRAM / GPU)
- Para evitar sobrecarga de memoria o excepciones de OOM (*Out-Of-Memory*) por la coexistencia simultánea de múltiples modelos en la GPU, el pipeline elimina y purga al Maestro (`del teacher.llm`, `del teacher.embedder`, `gc.collect()`, `empty_cache()`) antes de instanciar al Estudiante.

### 4. Destilación Adaptativa *On-the-fly* (Estudiante)
- Se inicializa el modelo liviano del **Estudiante** (`models/student`).
- El Estudiante razona sobre las semillas entregadas por el Maestro y decide dinámicamente sus propios hiperparámetros de entrenamiento (épocas, batch size, learning rate) para evitar sobreajuste.
- Aplica un ajuste fino ligero con PEFT/LoRA (`models/student_lora/adapter`).

### 5. Generación de Semillas y Extrapolación Estocástica Multiserie
- **Generación Morfológica:** El Estudiante fine-tuneado genera los primeros 10 valores (semilla) de la serie guiado por el perfil matemático y los ejemplos morfológicos EPIC.
- **Desacoplamiento Escalar y Extrapolación (`extend_series`):**
  - **Desacoplamiento:** Separa la dinámica de la señal (forma/autocorrelación) de la magnitud solicitada, proyectándola exactamente sobre el `target_value` del usuario.
  - **Diferenciación Inter-Serie:** Cada serie temporal se extrapola a su longitud objetivo (ej. 400 puntos) asignando una semilla aleatoria individual (`series_seed = 42 + idx`) y un micro-desplazamiento fisiológico/físico independiente, garantizando que cada serie represente un sujeto, sensor o tienda distinto y no una réplica idéntica.
- **Exportación Tabular:** Los datos se guardan en `data/synthetic/synthetic_series.csv` en formato long tabular (`series_id`, `timestamp`, `variable`).

### 6. Pipeline Numérico Autónomo (M6 → M7 → M8)
- **M6 — Entrenamiento RNN (LinearRNN LSTM/GRU):**
  - Carga el CSV generado y aplica split temporal secuencial (70% Train, 15% Val, 15% Test).
  - Construye ventanas deslizantes (ej. `seq_len=60`, `horizon=1`) y entrena una red recurrente profunda de 2 capas con regularización *Dropout* y *Early Stopping*.
  - Guarda el checkpoint en `data/checkpoints/rnn_final.pt`.
- **M7 — Predicción y Métricas de Rendimiento:**
  - Realiza inferencia sobre el conjunto de test y calcula métricas estándar: $MSE$, $RMSE$, $MAE$, $MAPE$ y coeficiente de determinación $R^2$.
  - Genera automáticamente un gráfico comparativo en `data/synthetic/predictions_plot.png`.
- **M8 — Evaluación Dual (Juez LLM + Detector de Alucinaciones):**
  - **Estudiante como Juez:** Evalúa la consistencia, plausibilidad física y asigna un veredicto formal (`APROBADO` / `RECHAZADO`).
  - **Detector de Alucinaciones:** Compara las predicciones frente a la referencia real utilizando cuatro analizadores:
    1. *Alucinaciones de Rango:* Valores fuera de límites físicos.
    2. *Alucinaciones Temporales:* Rupturas de estacionalidad o tendencia.
    3. *Alucinaciones de Correlación:* Discrepancias en relaciones multivariadas.
    4. *Alucinaciones Estadísticas:* Test de bondad de ajuste Kolmogorov-Smirnov (`ks_test`).

---

## 📊 Catálogo de Perfiles Matemáticos (`pipeline/config/profiles.py`)

El pipeline soporta 5 perfiles formales de series temporales:

| Perfil | Identificador | Tendencia | Estacionalidad | Nivel de Ruido | Casos de Uso Típicos |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P1** | `P1_ESTABLE` | Plana (sin tendencia) | Ninguna | Muy bajo ($\le 2\%$) | Sensores médicos en reposo (ej. ECG, frecuencia cardíaca), voltaje regulado. |
| **P2** | `P2_ESTACIONAL` | Plana (sin tendencia) | Semanal (7 días) | Moderado ($\le 5\%$) | Tráfico urbano, afluencia en estaciones de metro, ventas base estacionales. |
| **P3** | `P3_TENDENCIA_ESTACIONAL` | Creciente sostenida | Semanal (7 días) | Moderado + Anomalías | Comercio electrónico en expansión, ventas de retail, demanda energética. |
| **P4** | `P4_TENDENCIA_RUIDO` | Fluctuante / Descendente | Ninguna | Medio a Alto | Mercados financieros, cotización de acciones, criptoactivos volátiles. |
| **P5** | `P5_COMPLEJO` | Cambio de régimen (3 fases)| Fuerte / Multiciclo | Alto + Picos extremos | Cargas de servidores cloud, tráfico web bajo eventos DDoS, telemetría industrial. |

---

## 📚 Origen de Datasets y Referencias

El proyecto se fundamenta en tres fuentes de datos y marcos metodológicos:

| Recurso / Dataset | Archivo Local | Origen / Enlace | Función en el Proyecto |
| :--- | :--- | :--- | :--- |
| **Time-Series Reasoning** | `data/TS_Dataset.jsonl` (8,787 registros) | Hugging Face: [`mikeam/time-series-reasoning`](https://huggingface.co/datasets/mikeam/time-series-reasoning) (Mike Merrill) | **Fine-Tuning LoRA del Maestro:** Pares instrucción-serie con escenarios del mundo real, explicaciones contextuales y generadores en NumPy. Entrenó los adaptadores en `models/teacher_adapters`. |
| **TSQA (Time Series Question Answering)** | `data/time_series_generation.jsonl` (~192,800 registros) | Hugging Face: [`Time-MQA/TSQA`](https://huggingface.co/datasets/Time-MQA/TSQA) (TSQA Dataset) | Benchmark **TSQA** / Kaggle Web Traffic & Yahoo Server Metrics | **Base de Conocimiento RAG:** Indexado en la base vectorial ChromaDB (`rag/maestro`). Provee al Maestro de semillas empíricas para contextualizar la generación. |


### Ficha Técnica del Fine-Tuning del Maestro
Los adaptadores LoRA del Maestro (`models/teacher_adapters`) fueron entrenados en **Kaggle GPU** a partir de `data/TS_Dataset.jsonl`:
- **Hiperparámetros:** 2 épocas, batch size por dispositivo de 1, `gradient_accumulation_steps=4` (batch size efectivo = 4).
- **Optimizador y Scheduler:** `paged_adamw_8bit` con tasa de aprendizaje `2e-4` y scheduler `cosine` (10 pasos de warmup).
- **LoRA:** Rank $r=16$, $\alpha=32$, dropout $0.05$ en módulos de atención y MLP (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Checkpoint final:** 4,394 pasos (`checkpoint-4394`).

---

## 🛠 Configuración del Entorno (`.env`)

El pipeline parametriza rutas y modelos a través de variables de entorno:

```env
# Modelos base en Hugging Face
TEACHER_MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
STUDENT_MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
HF_TOKEN=tu_token_huggingface_aqui

# Rutas locales de modelos y adaptadores
TEACHER_MODEL_PATH=./models/profesor
TEACHER_MODEL_PATH_ADAPTERS=./models/teacher_adapters
STUDENT_MODEL_PATH=./models/student
STUDENT_LORA_PATH=./models/student_lora

# Configuración del dataset sintético
SYNTHETIC_NUM_SERIES=20
SYNTHETIC_POINTS_PER_SERIES=400
SYNTHETIC_CHUNK_SERIES=2
SYNTHETIC_CHUNK_POINTS=80

# Control de longitud de generación
LLM_MAX_NEW_TOKENS=2048
```

**Dimensiones del Dataset Sintético:**
`Total filas = SYNTHETIC_NUM_SERIES × SYNTHETIC_POINTS_PER_SERIES` (por defecto: $20 \times 400 = 8,000$ filas CSV), garantizando datos suficientes para entrenar ventanas temporales de longitud 60 en M6.

---

## 📂 Estructura de Directorios

```text
proyecto_tesis/
├── pipeline/
│   ├── main.py                # Entrada principal (CLI) del pipeline
│   ├── numeric_pipeline.py    # Orquestador del pipeline numérico (M6 → M7 → M8)
│   ├── teacher_agent/         # Agente Maestro: Interpretación, RAG ChromaDB, Filtrado cognitivo
│   │   └── agent.py
│   ├── student_agent/         # Agente Estudiante: Destilación LoRA, Generación, Extrapolación
│   │   ├── agent.py
│   │   └── fine_tuner.py
│   ├── training/              # Módulo 6 (M6): Entrenamiento RNN Lineal (LSTM/GRU)
│   │   ├── model.py
│   │   ├── trainer.py
│   │   └── dataset.py
│   ├── prediction/            # Módulo 7 (M7): Predictor e inferencia de series
│   │   └── predictor.py
│   ├── evaluation/            # Módulo 8 (M8): Estudiante como Juez y Detector de Alucinaciones
│   │   ├── judge.py
│   │   └── hallucination.py
│   ├── config/                # Configuraciones y Catálogos
│   │   ├── profiles.py        # Catálogo formal de perfiles matemáticos (P1..P5)
│   │   ├── settings.py        # Configuración global y rutas
│   │   └── pipeline_config.py # Configuración de hiperparámetros RNN y métricas
│   └── utils/                 # Extrapolación, series_utils, LLMClient, logger
│       ├── series_utils.py    # Algoritmo de extrapolación estocástica multiserie
│       └── llm_client.py      # Cliente unificado de inferencia LLM con rescate JSON
├── data/
│   ├── TS_Dataset.jsonl       # Dataset de fine-tuning del Maestro (8,787 registros)
│   ├── time_series_generation.jsonl # Dataset para la base de conocimiento RAG (~192k registros)
│   ├── checkpoints/           # Checkpoints del modelo LSTM entrenado (rnn_final.pt)
│   └── synthetic/             # Salida del dataset sintético (synthetic_series.csv y plots)
├── rag/
│   └── maestro/               # Persistencia local de ChromaDB (Vector Store RAG)
├── models/
│   ├── profesor/              # Modelo base del Maestro
│   ├── teacher_adapters/      # Adaptadores LoRA preentrenados del Maestro
│   ├── student/               # Modelo base del Estudiante
│   └── student_lora/          # Adaptadores LoRA destilados del Estudiante
├── scripts/
│   ├── setup_models.py        # Descarga de modelos base desde Hugging Face
│   └── build_rag_from_jsonl.py# Construcción e indexación del RAG en ChromaDB
└── requirements.txt           # Dependencias (PyTorch, Transformers, PEFT, ChromaDB, etc.)
```

---

## 💻 Inicio Rápido y Ejemplos de Ejecución

### 1. Instalación y Configuración Inicial
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar modelos base (configurar previamente HF_TOKEN en .env)
python scripts/setup_models.py

# 3. Construir base de datos vectorial ChromaDB (si no existe rag/maestro)
python scripts/build_rag_from_jsonl.py
```

### 2. Ejecución según Perfil Matemático

#### Perfil P1_ESTABLE (Salud / Frecuencia Cardíaca):
```bash
python -m pipeline.main --query "Necesito generar lecturas de frecuencia cardíaca de un paciente en reposo monitoreado en una clínica. Dominio: Medicina y salud. Variable estricta: Únicamente 'Frecuencia_Cardiaca'. Perfil matemático: P1_ESTABLE, valores constantes con bajo nivel de ruido. Escala: mantenerse estable alrededor de 70 lpm durante todo el periodo."
```

#### Perfil P2_ESTACIONAL (Transporte / Metro):
```bash
python -m pipeline.main --query "Simular el flujo diario de pasajeros en una estación de metro durante un año completo. Dominio: Transporte urbano. Variable estricta: Únicamente 'Pasajeros'. Perfil matemático: P2_ESTACIONAL, con ciclos semanales marcados (mayor afluencia en días laborales y menor en fines de semana), sin tendencia global. Escala: promedio de 15000 pasajeros diarios."
```

#### Perfil P3_TENDENCIA_ESTACIONAL (Retail / Ventas):
```bash
python -m pipeline.main --query "Simular las ventas diarias de una plataforma de comercio electrónico en expansión durante un año. Dominio: Retail y Ventas. Variable estricta: Únicamente 'Ventas_Diarias'. Perfil matemático: P3_TENDENCIA_ESTACIONAL, con tendencia creciente sostenida y ciclos semanales marcados por mayor consumo en fines de semana. Escala: promedio inicial de 2500 USD creciendo hacia 4000 USD."
```

#### Perfil P4_TENDENCIA_RUIDO (Finanzas / Cotizaciones):
```bash
python -m pipeline.main --query "Simular la cotización diaria de una acción tecnológica durante un ciclo correctivo de mercado. Dominio: Finanzas y Mercados. Variable estricta: Únicamente 'Precio_Accion'. Perfil matemático: P4_TENDENCIA_RUIDO, con tendencia fluctuante descendente y alto nivel de ruido o volatilidad, sin componentes estacionales. Escala: cotizando alrededor de 120 USD."
```

#### Perfil P5_COMPLEJO (Servidores / Telecomunicaciones):
```bash
python -m pipeline.main --query "Monitorear el consumo de ancho de banda en un centro de datos empresarial. Dominio: Telecomunicaciones y Servidores. Variable estricta: Únicamente 'Ancho_Banda_Gbps'. Perfil matemático: P5_COMPLEJO, con fuerte estacionalidad por horario laboral, alto ruido de tráfico y picos anómalos esporádicos por sobrecarga. Escala: promedio de 45 Gbps."
```
