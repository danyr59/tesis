# Pipeline de Mantenimiento Predictivo con Destilación Pedagógica

Sistema modular de 9 módulos que integra la metodología pedagógica **IOA** (Identifier-Organizer-Adapter) con capacidades de series temporales para predicción de fallas en equipos industriales.

## Arquitectura

```
Consulta NL ──► M0 Orquestador
                 │
                 ├── M1 Ingestión        (Chunking + Embeddings + ChromaDB)
                 ├── M2 Recuperación     (Búsqueda semántica)
                 ├── M3 Contexto         (Mensaje enriquecido RAG)
                 ├── M4 Mentor (IOA)     (Diagnóstico + Organización + Adaptación)
                 ├── M5 Síntesis         (Generación datos sintéticos)
                 ├── M6 Entrenamiento    (RNN Lineal: LSTM/GRU)
                 ├── M7 Predicción       (Métricas: RMSE, MAE, R², MAPE)
                 └── M8 Evaluación       (Juez + Detección de alucinaciones)
```

## Inicio Rápido

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo (con datos placeholder)
python -m pipeline.main --query "Predecir fallas en bomba hidráulica"

# Ejecutar con documentos de dominio
python -m pipeline.main \
    --data-dir ./data/raw \
    --query "Analizar patrón de degradación del compresor" \
    --num-synthetic 1000
```

## Uso Programático

```python
from pipeline.main import TesisPipeline
import numpy as np

# Crear pipeline
pipeline = TesisPipeline()

# Ingestar documentos del dominio
pipeline.ingest("./data/raw")

# Ejecutar con datos de sensores reales
sensor_data = np.loadtxt("./data/sensor_readings.csv", delimiter=",", skiprows=1)
result = pipeline.run(
    query="Predecir fallas en bomba hidráulica",
    real_data=sensor_data,
    feature_names=["temperatura", "vibracion", "presion", "rpm"],
    num_synthetic=500,
)

print(result.summary())
```

## Estructura del Proyecto

```
proyecto_tesis/
├── IOA/                     # Framework pedagógico (intacto)
├── ChatTS/                  # Series temporales (referencia)
├── pipeline/                # Pipeline unificado (NUEVO)
│   ├── main.py              # M0: Orquestador
│   ├── config/              # Configuración centralizada
│   ├── ingestion/           # M1: Chunker + Embedder + VectorStore
│   ├── retrieval/           # M2: Retriever
│   ├── context_builder/     # M3: ContextBuilder
│   ├── mentor/              # M4: Wrapper IOA
│   ├── synthesis/           # M5: SeedAnalyzer + PromptEngineer + Generator
│   ├── training/            # M6: LinearRNN + Dataset + Trainer
│   ├── prediction/          # M7: Predictor
│   ├── evaluation/          # M8: Judge + HallucinationDetector
│   └── utils/               # Logger
├── data/                    # Datos
│   ├── raw/                 # Documentos originales
│   ├── processed/           # Chunks procesados
│   └── synthetic/           # Datos sintéticos
├── outputs/                 # Resultados
└── requirements.txt
```

## Módulos

| # | Módulo | Descripción |
|---|--------|-------------|
| 0 | Orquestador | Recibe consulta NL y coordina el flujo |
| 1 | Ingestión | Segmenta, embebe y almacena documentos en ChromaDB |
| 2 | Recuperación | Busca fragmentos por similitud semántica |
| 3 | Contexto | Construye mensaje enriquecido (RAG) |
| 4 | Mentor (IOA) | Diagnóstica gaps, organiza curriculum, adapta conocimiento |
| 5 | Síntesis | Genera datos sintéticos (estadístico/LLM) |
| 6 | Entrenamiento | Entrena RNN lineal (LSTM/GRU) con early stopping |
| 7 | Predicción | Genera predicciones y calcula métricas |
| 8 | Evaluación | Juez de calidad + detección de alucinaciones numéricas |
