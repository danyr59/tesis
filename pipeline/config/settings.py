import os


class Settings:
  # RAG (índice TSQA en ChromaDB)
  RAG_TOP_K = 50
  RAG_COLLECTION_NAME = "time_series_generation"
  RAG_DB_PATH = "./rag/maestro"
  RAG_DATASET_NAME = "TSQA"

  # Iteración Maestro / Estudiante
  MAX_ITERATIONS = 3
  MAX_TEACHER_REFINEMENTS = 5
  MIN_CONFIDENCE_THRESHOLD = 0.8
  MIN_SEED_EXAMPLES = 3
  MIN_SERIES_LENGTH = 10

  # ── Generación sintética (NÚMERO FIJO — ajusta aquí o en .env) ─────────────
  # Cantidad de series y puntos por serie que SIEMPRE genera el Estudiante.
  # No se infieren desde la consulta del usuario.
  #
  # Recomendado para RNN (ventana M6 = 60 pasos):
  #   • Mínimo práctico:  15 series × 300 puntos  → 4 500 filas (~4 400 ventanas)
  #   • Valor por defecto: 20 series × 400 puntos  → 8 000 filas (~7 900 ventanas)
  #   • Más robusto:      30 series × 500 puntos  → 15 000 filas
  #
  # Regla rápida: filas_totales = SYNTHETIC_NUM_SERIES * SYNTHETIC_POINTS_PER_SERIES
  # Debe ser >> 60 y preferiblemente > 3 000 filas para entrenar/validar con estabilidad.
  _DEFAULT_NUM_SERIES = 20
  _DEFAULT_POINTS_PER_SERIES = 400

  SYNTHETIC_NUM_SERIES = int(
    os.getenv("SYNTHETIC_NUM_SERIES", str(_DEFAULT_NUM_SERIES))
  )
  SYNTHETIC_POINTS_PER_SERIES = int(
    os.getenv("SYNTHETIC_POINTS_PER_SERIES", str(_DEFAULT_POINTS_PER_SERIES))
  )
  # Tamaño de cada llamada al LLM (interno; no cambia el total generado)
  SYNTHETIC_CHUNK_SERIES = int(os.getenv("SYNTHETIC_CHUNK_SERIES", "2"))
  SYNTHETIC_CHUNK_POINTS = int(os.getenv("SYNTHETIC_CHUNK_POINTS", "80"))

  # LLM
  LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "2048"))

  # LoRA / destilación
  STUDENT_LORA_PATH = os.getenv("STUDENT_LORA_PATH", "./models/student_lora")
  LORA_R = int(os.getenv("LORA_R", "8"))
  LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
  LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

  # Paths de salida
  SYNTHETIC_DATA_DIR = "./data/synthetic"
  SYNTHETIC_CSV_NAME = "synthetic_series.csv"

  @classmethod
  def get_teacher_model_id(cls):
    return os.getenv("TEACHER_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")

  @classmethod
  def get_student_model_id(cls):
    return os.getenv("STUDENT_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")

  @classmethod
  def get_teacher_model_path(cls):
    return os.getenv("TEACHER_MODEL_PATH", "./models/profesor")

  @classmethod
  def get_student_model_path(cls):
    return os.getenv("STUDENT_MODEL_PATH", "./models/student")

  @classmethod
  def synthetic_csv_path(cls) -> str:
    return os.path.join(cls.SYNTHETIC_DATA_DIR, cls.SYNTHETIC_CSV_NAME)

  @classmethod
  def synthetic_generation_summary(cls) -> str:
    total = cls.SYNTHETIC_NUM_SERIES * cls.SYNTHETIC_POINTS_PER_SERIES
    return (
      f"{cls.SYNTHETIC_NUM_SERIES} series × {cls.SYNTHETIC_POINTS_PER_SERIES} puntos "
      f"= {total} filas CSV"
    )
