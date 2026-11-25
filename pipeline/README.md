# Módulo Central del Pipeline (`pipeline/`)

Este paquete implementa la arquitectura neuro-simbólica multi-agente de generación de series temporales sintéticas y su validación matemática y predictiva.

---

## 🏗️ Estructura de Módulos

```
pipeline/
├── config/                  # Configuraciones y perfiles matemáticos
│   ├── pipeline_config.py   # Configuración de hiperparámetros (RNN, alucinaciones)
│   ├── profiles.py          # Catálogo formal de perfiles temporales P1 a P6
│   └── settings.py          # Gestión centralizada de variables de entorno (.env)
│
├── teacher_agent/           # Agente Maestro (Razonamiento y RAG)
│   └── agent.py             # Deducción de parámetros físicos, consulta RAG y filtrado cognitivo
│
├── student_agent/           # Agente Estudiante (Generación Ligera)
│   ├── agent.py             # Inferencia QLoRA de semillas morfológicas (0 a 1 en JSON)
│   └── fine_tuner.py        # Destilación supervisada ligera on-the-fly
│
├── numeric_pipeline.py      # Orquestador del desacoplamiento escalar y validación M6-M8
│
├── training/                # M6: Modelado Causal Recurrente
│   ├── dataset.py           # Construcción de ventanas deslizantes
│   ├── model.py             # Red Neuronal Recurrente (LSTM/GRU)
│   └── trainer.py           # Bucle de entrenamiento y checkpoints
│
├── prediction/              # M7: Predicción Downstream y Métricas
│   └── predictor.py         # Inferencia causal, cálculo de RMSE, MAE, MAPE, R² y gráficos
│
├── evaluation/              # M8: Evaluación Dual y Detección de Alucinaciones
│   ├── hallucination.py     # HallucinationDetector (cálculo de H_rate y cotas físicas)
│   └── judge.py             # Estudiante como Juez cualitativo de consistencia
│
├── utils/                   # Utilidades del Sistema
│   ├── llm_client.py        # Carga defensiva y cuantización (4-bit BitsAndBytes / LoRA)
│   ├── logger.py            # Logging estructurado con formato enriquecido
│   └── series_utils.py      # Desacoplamiento escalar y estadísticas de semillas
│
└── main.py                  # Punto de entrada CLI ejecutable
```

---

## ⚡ Flujo de Ejecución Modular

1. **Entrada y Razonamiento (Maestro):**
   El `TeacherAgent` analiza la consulta del usuario en lenguaje natural, deduce las cotas físicas ($y_{\min}, y_{\max}$), la frecuencia temporal y el perfil morfológico ($P_1$ a $P_6$), recuperando además las semillas afines desde ChromaDB.

2. **Purga de Memoria:**
   Se ejecuta una descarga forzada de tensores y modelos del Maestro de la VRAM (`gc.collect()` y `empty_cache()`), garantizando que el Estudiante opere en un espacio de memoria limpio ($\le 7.20$ GB).

3. **Destilación y Generación (Estudiante):**
   El `StudentAgent` genera la semilla morfológica normalizada en $[0, 1]$ estructurada en formato JSON estricto.

4. **Desacoplamiento Escalar Determinista:**
   Una función analítica proyecta la forma normalizada hacia la escala física real mediante la transformación afín $y_t = y_{\min} + \tilde{y}_t \cdot (y_{\max} - y_{\min})$, erradicando matemáticamente el riesgo de NaNs o violaciones de cota.

5. **Validación Downstream (M6-M8):**
   Se entrena una red LSTM para verificar la preservación de la dinámica causal, computando $R^2$, MAE, RMSE y la tasa de alucinación semántica $\mathcal{H}_{\text{rate}}$.
