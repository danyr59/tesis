# Guía de Escalabilidad: Panel de Control del Pipeline

Este documento describe cómo transicionar el proyecto desde un prototipo de alta velocidad (etapa actual) hacia una generación de alta fidelidad para la defensa de la tesis.

**Archivo Central de Configuración:** `pipeline/config/pipeline_config.py`

---

## 1. Módulo 4: Maestro & Mentor (Diagnóstico)
**Archivo:** `pipeline/config/pipeline_config.py` → Clase `PipelineConfig` y `SeedConfig`

| Parámetro | Atributo en Código | Valor Desarrollo | Tesis (Final) | Impacto |
| :--- | :--- | :--- | :--- | :--- |
| `max_taxonomy_modules` | `config.max_taxonomy_modules` | 5 | **15 - 20** | Mayor detalle en el desglose de fallas. |
| `num_probes` | `config.seed_factory.num_probes` | 25 | **50 - 100** | Diagnóstico más riguroso del gap de conocimiento. |
| `teacher_model` | `config.llm.teacher_model` | llama3.2 | **llama3.1:8b** | Razonamiento pedagógico superior. |

---

## 2. Módulo 5: Síntesis (Generación Masiva)
**Archivo:** `pipeline/config/pipeline_config.py` → Clase `SynthesisConfig`

| Parámetro | Atributo en Código | Valor Desarrollo | Tesis (Final) | Impacto |
| :--- | :--- | :--- | :--- | :--- |
| `num_synthetic` | `config.synthesis.num_synthetic` | 100 | **500 - 1000** | Soporte estadístico robusto para la RNN. |
| `batch_size` | `config.synthesis.batch_size` | 5 | **10 - 15** | Velocidad de generación (requiere más VRAM). |
| `temperature` | `config.synthesis.temperature` | 0.85 | **0.7 - 0.9** | Balance entre creatividad y realismo físico. |
| `diversity_threshold`| `config.synthesis.diversity_threshold`| 0.5 | **0.7** | Mayor exigencia en la unicidad de datos. |

---

## 3. Módulo 6: Entrenamiento (Predicción RNN)
**Archivo:** `pipeline/config/pipeline_config.py` → Clase `RNNConfig`

| Parámetro | Atributo en Código | Valor Desarrollo | Tesis (Final) | Impacto |
| :--- | :--- | :--- | :--- | :--- |
| `max_epochs` | `config.rnn.max_epochs` | 100 | **500 - 1000** | Captura de patrones sutiles a largo plazo. |
| `patience` | `config.rnn.patience` | 10 | **25 - 50** | Evita el early stopping prematuro. |
| `hidden_size` | `config.rnn.hidden_size` | 64 | **128 - 256** | Mayor capacidad de memoria para la RNN. |

---

## 4. Módulo 8: Evaluación (Juez y Alucinaciones)
**Archivo:** `pipeline/config/pipeline_config.py` → Clase `HallucinationConfig`

| Parámetro | Atributo en Código | Valor Desarrollo | Tesis (Final) | Impacto |
| :--- | :--- | :--- | :--- | :--- |
| `min_r2` | `config.hallucination.min_r2` | 0.5 | **0.85+** | Exigencia de precisión en la predicción. |
| `max_mape` | `config.hallucination.max_mape` | 20.0 | **5.0 - 10.0** | Margen de error porcentual permitido. |
| `approval_threshold` | `config.hallucination.approval_threshold` | 0.7 | **0.9** | Confianza mínima global para aprobar. |
| `z_score_threshold` | `config.hallucination.z_score_threshold` | 3.0 | **2.0 - 2.5** | Mayor sensibilidad a valores atípicos. |

---

## 5. Otros Parámetros de Calidad
- **Módulo 2 (Retrieval):** `RetrievalConfig.top_k` (Prototipo: 5 | Final: **10-15**). Mejora el contexto inyectado.

---

## Instrucciones para el Cambio
Para aplicar los cambios de "Alta Fidelidad", simplemente edita los valores por defecto en `pipeline/config/pipeline_config.py` o pasa un objeto `PipelineConfig` modificado al instanciar `TesisPipeline`.

> [!IMPORTANT]
> Aumentar los valores a la columna "Tesis (Final)" puede incrementar el tiempo de ejecución de minutos a horas, dependiendo de tu hardware local (GPU/VRAM).

---

## 6. Glosario de Parámetros Clave

Para tu defensa de tesis, puedes usar estas definiciones para justificar tus decisiones de diseño:

### Conceptos Pedagógicos (IOA)
*   **max_taxonomy_modules**: Define la **granularidad** del mapa de conocimiento. Un valor alto obliga al Maestro a identificar fallas en niveles muy específicos de la materia.
*   **num_probes (max_probes)**: Es la **rigurosidad del diagnóstico**. Determina cuántas pruebas debe superar el Estudiante para que el Maestro valide su conocimiento. A mayor número, diagnóstico más preciso pero más costoso en tiempo.
*   **tau_mastery**: (En `ioa_config`) El nivel de "maestría" que el Estudiante debe alcanzar respecto al Maestro para considerar que el conocimiento fue transferido.

### Conceptos de Calidad de Datos
*   **diversity_threshold**: El filtro de **anti-repetición**. Si un lote de datos generados es muy similar entre sí, el sistema lo descarta. Un valor de 0.7 significa que al menos el 70% de los datos deben ser únicos.
*   **mandatory_jitter**: Simulación de **ruido real de sensor**. Evita que el modelo aprenda datos matemáticamente perfectos (lo cual es una alucinación en el mundo real) inyectando variaciones de micro-precisión.

### Conceptos de Validación (Juez)
*   **z_score_threshold**: Sensibilidad a **anomalías**. Define cuántas desviaciones estándar puede alejarse un dato del promedio real antes de ser marcado como "alucinación numérica".
*   **approval_threshold**: El **filtro de confianza final**. Es el promedio de todas las pruebas de calidad. Si el pipeline no confía al menos en un 90% (0.9) en los resultados, el Juez marcará el proceso como "No Aprobado".

