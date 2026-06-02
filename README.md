# Proyecto de Tesis: Generación de Series Temporales Sintéticas mediante Destilación Maestro-Estudiante con RAG

Este proyecto de tesis implementa una arquitectura avanzada de agentes LLM (Maestro-Estudiante) interactuando con una base de datos vectorial (RAG) para la **generación autónoma de datos sintéticos de series temporales**, con el objetivo de entrenar posteriormente arquitecturas predictivas (Redes Neuronales Recurrentes M6).

## 🚀 Arquitectura y Flujo de Ejecución (6 Pasos)

El núcleo del sistema es un bucle interactivo y conversacional orquestado por el Agente Estudiante, que sigue exactamente este flujo de 6 pasos:

### 1. Entrada del Usuario (CLI)
El sistema inicia de forma interactiva a través de la consola (`pipeline/main.py`), donde el usuario ingresa su requerimiento inicial (ej. *"Necesito series temporales de temperatura en Mérida"*).

### 2. Análisis, Razonamiento y Estructuración (Estudiante)
Un LLM liviano (configurable en `.env`) toma el mando:
- **Interpretación Semántica:** Identifica sinónimos y ambigüedades.
- **Identificación de Tipo de Serie:** Razona automáticamente si se requieren datos **univariados** o **multivariados**.
- **Diálogo Interactivo:** Si faltan parámetros clave (dominio, frecuencia, localización, variables) o su confianza es `< 80%`, el Estudiante **pausa y dialoga con el usuario vía consola** para completarlos.
- **Prompt Estructurado:** Consolida todo en un JSON justificando sus inferencias y enviando una estrategia sugerida al Maestro.

### 3. Búsqueda, Evaluación y Selección (Maestro + RAG)
Un LLM experto (configurable en `.env`) que tiene acceso a un RAG de series temporales (ChromaDB):
- **Búsqueda Fija:** Ejecuta consultas en la base vectorial, extrayendo siempre un **top_k de 200 ítems**.
- **Filtro Cognitivo (Sin Hardcode):** El Maestro lee un resumen de los metadatos y razona si las series son coherentes con el requerimiento del estudiante, y si la longitud o la diversidad son adecuadas.
- **Refinamiento:** Si la calidad es baja o hay menos de 3 ejemplos válidos, relaja los parámetros de forma autónoma y vuelve a buscar.
- **Salida:** Envía un "Dataset Semilla" validado y un "Resumen de Calidad" justificado al Estudiante.

### 4. Destilación (Fine-Tuning Inteligente)
El Estudiante recibe las semillas y razona sobre cómo debe entrenarse:
- Decide dinámicamente los hiperparámetros (épocas, batch size, learning rate) dependiendo de si hay muchos o pocos datos (para evitar sobreajuste).
- Explica sus decisiones de forma explícita en consola (`verbose`).
- Entrena sobre su modelo base utilizando técnicas ligeras (PEFT/LoRA).

### 5. Ciclo de Retroalimentación (Iteraciones)
- El Estudiante lee el Resumen de Calidad del Maestro y evalúa si la diversidad/calidad de su entrenamiento es buena.
- Si determina que la calidad es baja (ej. demasiada similitud), formula una acción correctiva y **pide más datos al Maestro**.
- Mantiene un registro de las estrategias ya usadas en este ciclo para no repetir errores. 
- Este bucle se ejecuta un máximo de **3 iteraciones** (configurable en settings).

### 6. Generación de Dataset (Estructura HuggingFace/RNN)
Una vez el Estudiante está conforme, el LLM procede a generar numéricamente las series sintéticas solicitadas, respetando la estacionalidad y características.
- **Output Robusto:** Convierte los datos a un CSV de "formato largo" (long format tabular: `series_id`, `time_step`, `var1`, `var2`...).
- Los datos se guardan en `data/synthetic/synthetic_series.csv` de manera independiente, dejándolos listos para que herramientas como Pandas, PyTorch o los siguientes módulos de la tesis los consuman directamente sin limpieza adicional.

---

## 🛠 Modelos y Configuración Global

**Cero Datos "Quemados" (Hardcodeados):**
- La lógica no está atada a "Llama 1B" ni "Llama 3B". Los modelos (en función de $N$ y $n$ parámetros) se inyectan dinámicamente.
- Todos los umbrales lógicos, iteraciones y rutas residen centralizados en `pipeline/config/settings.py`.

El archivo `.env` debe configurarse de esta manera:
```env
TEACHER_MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
STUDENT_MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
HF_TOKEN=tu_token_aqui
```

---

## 📂 Estructura de Directorios

```text
proyecto_tesis/
├── pipeline/
│   ├── main.py                # Entrada (CLI) del pipeline
│   ├── config/
│   │   └── settings.py        # Configuración global del proyecto
│   ├── student_agent/         # Módulo: LLM Liviano (Razonamiento, LoRA, Generación .csv)
│   ├── teacher_agent/         # Módulo: LLM Experto (RAG ChromaDB, Filtrado, Evaluación)
│   ├── training/              # Módulo 6 (M6): Entrenamiento de red RNN Lineal (LSTM/GRU)
│   ├── prediction/            # Módulo 7 (M7): Predicción de series
│   ├── evaluation/            # Módulo 8 (M8): Métricas, RMSE, Detección de Alucinaciones
│   └── utils/                 # Utilidades generales y Logger
├── data/
│   └── synthetic/             # Destino final: Dataframes .csv listos para Entrenamiento (M6)
├── rag/
│   └── maestro/               # Persistencia local de ChromaDB (Vector Store)
├── models/                    # Guardado local de modelos LLM descargados
└── requirements.txt           # Dependencias (transformers, peft, chromadb, etc.)
```

---

## 💻 Inicio Rápido

Para iniciar el ciclo interactivo de agentes de forma autónoma:

```bash
# 1. Instalar requerimientos y librerías clave
pip install -r requirements.txt

# 2. Ejecutar CLI de forma interactiva (Te preguntará qué necesitas)
python -m pipeline.main

# 3. O ejecutar pasándole la consulta directo como parámetro
python -m pipeline.main --query "necesito series temporales de temperatura para la ciudad de Mérida, Venezuela"
```
