# 📁 Banco Integral de Resultados Experimentales y Trazas Forenses

Este directorio contiene el registro empírico y forense exhaustivo de la totalidad de las pruebas experimentales llevadas a cabo en la investigación:
- **Las 54 ejecuciones factoriales:** $9\text{ Escenarios} \times 6\text{ Perfiles Temporales}$ ($E_1$ a $E_9 \times P_1$ a $P_6$).
- **Estudio de Reproducibilidad Estocástica Independiente:** Réplicas inter-corrida sobre $E_1-P_2$, $E_3-P_5$ y $E_5-P_3$.
- **Ablación Metodológica de Paradigma:** Destilación canónica por Logits de Hinton ($\mathcal{L}_{\text{KD}}$) frente a SFT QLoRA en $E_1$ ($P_2$, $P_3$, $P_4$).

Cada carpeta contiene la traza científica completa que permite auditar, reproducir y mapear cada decisión del Agente Maestro, las semillas RAG recuperadas, los datos sintéticos generados y su evaluación predictiva downstream.

---

## 🗂️ Estructura del Directorio

```
resultados/
├── E1/ ... E9/                               # Escenarios factoriales principales
│   ├── resumen_E*.csv                       # Tabla síntesis de las 6 métricas del escenario
│   ├── P1_ESTABLE/                          # Perfil 1: Frecuencia cardíaca (Estable)
│   │   ├── config_generacion.json           # Traza del Maestro: prompt, directivas JSON y 50 semillas RAG
│   │   ├── metricas.json                    # Telemetría downstream, Juez, H_rate, VRAM y tiempos
│   │   ├── synthetic_series.csv             # Las 8.000 observaciones numéricas generadas en crudo
│   │   ├── grafica_prediccion.png           # Visualización del desempeño predictivo LSTM
│   │   ├── grafica_serie_completa_400p.png  # Gráfica de alta resolución de la dinámica temporal
│   │   └── rnn_checkpoint.pt                # Checkpoint PyTorch del predictor recurrente
│   ├── P2_ESTACIONAL/                       # Perfil 2: Ciclos semanales / Bitcoin USD
│   ├── P3_TEND_ESTAC/                       # Perfil 3: Sensores IoT y consumo eléctrico
│   ├── P4_TEND_RUIDO/                       # Perfil 4: Tráfico probabilístico estocástico
│   ├── P5_COMPLEJO/                         # Perfil 5: Atractor caótico no lineal de Lorenz
│   └── P6_MULTIVARIADO/                     # Perfil 6: Vector tridimensional de variables físicas
│
├── reproducibilidad/                        # Auditoría de estabilidad estocástica
│   ├── E1/P2_ESTACIONAL/                    # Réplica independiente E1-P2 (Qwen-Qwen)
│   ├── E3/P5_COMPLEJO/                      # Réplica independiente E3-P5 (Qwen-Gemma)
│   └── E5/P3_TEND_ESTAC/                    # Réplica independiente E5-P3 (LLaMA-Qwen)
│
├── logit_kd/                                # Ablación de Hinton Knowledge Distillation
│   └── E1/                                  # Escenario homólogo E1 (Qwen-Qwen)
│       ├── P2_ESTACIONAL/
│       ├── P3_TEND_ESTAC/
│       ├── P4_TEND_RUIDO/
│       └── resumen_E1.csv
│
├── resumen_general_todos_escenarios.csv     # Matriz consolidada de las 54 ejecuciones
└── README.md                                # Esta guía de auditoría
```

---

## 🔬 Anatomía de los Archivos de Cada Experimento

Dentro de cada combinación `E{i}/P{j}/` se alojan 6 artefactos complementarios:

### 1. `config_generacion.json` (Trazabilidad Cognitiva y Semillas RAG)
Almacena el registro de auditoría del **Agente Maestro**:
- `query`: Instrucción cualitativa en lenguaje natural proporcionada por el usuario.
- `structured_prompt`: Directiva cognitiva generada por el Maestro (dominio, perfil matemático asignado, cotas analíticas estimadas $[y_{\min}, y_{\max}]$).
- `metadata_summary`: Resumen estadístico deducido para condicionar la morfología.
- `seed_dataset`: **Las 50 semillas empíricas reales** recuperadas desde ChromaDB (dataset TSQA) ordenadas por similitud semántica con sus valores numéricos y metadatos.

### 2. `metricas.json` (Evaluación Cuantitativa y Telemetría)
Registro exhaustivo de desempeño del ciclo completo:
- `downstream`: Coeficiente $R^2$, MAPE (%), RMSE y MAE alcanzados por la red recurrente LSTM. En perfiles multivariados ($P_6$), incluye desglose individual por variable física (`per_feature`).
- `juez`: Veredicto del Agente Juez (`aprobado`: true/false), confianza analítica y score de alucinación.
- `alucinaciones`: Tasa de alucinación semántica $\mathcal{H}_{\text{rate}}$ y desglose de violaciones de cota.
- `eficiencia`: Tiempo total de ejecución en segundos y consumo pico de memoria de video (`vram_pico_gb`).

### 3. `synthetic_series.csv` (Datos Numéricos Generados)
Archivo en formato tabular estándar con las observaciones numéricas sintéticas (20 series independientes de 400 puntos temporales cada una = **8.000 filas de datos**). Permite importar la serie directamente en Pandas, R, MATLAB o Excel.

### 4. Gráficas de Evidencia Visual
- `grafica_prediccion.png`: Comparación visual entre la serie sintética real y la predicción multi-paso generada por la red LSTM entrenada sobre ella.
- `grafica_serie_completa_400p.png`: Trazado completo de la dinámica temporal a lo largo de los 400 pasos de tiempo continuos.

### 5. `rnn_checkpoint.pt` (Modelo Predictivo Downstream)
Pesos binarios serializados de la red recurrente LSTM entrenada de forma causal sobre la serie sintética generada, permitiendo reanudar o auditar el modelado sin re-entrenar.

---

## 🗺️ Mapa de los 9 Escenarios Experimentales

| Escenario | Par Docente $\rightarrow$ Discente | Tipo Arquitectónico | Propósito Científico |
| :---: | :---: | :---: | :--- |
| **$E_1$** | Qwen2.5-7B $\rightarrow$ Qwen2.5-1.5B | Intra-Familia | Línea base homogénea de alta densidad simbólica. |
| **$E_2$** | Qwen2.5-7B $\rightarrow$ LLaMA-3.2-1B | Inter-Familia (*Cross*) | Transferencia跨-familia hacia modelo de ultra-baja latencia. |
| **$E_3$** | Qwen2.5-7B $\rightarrow$ Gemma-2-2B | Inter-Familia (*Cross*) | Transferencia hacia arquitectura de atención por ventana deslizante. |
| **$E_4$** | LLaMA-3.1-8B $\rightarrow$ LLaMA-3.2-1B | Intra-Familia | Línea base homogénea de la familia LLaMA (Meta AI). |
| **$E_5$** | LLaMA-3.1-8B $\rightarrow$ Qwen2.5-1.5B | Inter-Familia (*Cross*) | Robustez ante disparidad de tokenizadores (128k vs 151k tokens). |
| **$E_6$** | LLaMA-3.1-8B $\rightarrow$ Gemma-2-2B | Inter-Familia (*Cross*) | Transferencia inter-arquitectura de modelos occidentales. |
| **$E_7$** | Gemma-2-9B $\rightarrow$ Gemma-2-2B | Intra-Familia | Línea base homogénea de Google con *logit soft-capping*. |
| **$E_8$** | Gemma-2-9B $\rightarrow$ Qwen2.5-1.5B | Inter-Familia (*Cross*) | Transferencia desde Gemma hacia modelo compacto de Alibaba. |
| **$E_9$** | Gemma-2-9B $\rightarrow$ LLaMA-3.2-1B | Inter-Familia (*Cross*) | Transferencia desde Gemma hacia modelo compacto de Meta. |
