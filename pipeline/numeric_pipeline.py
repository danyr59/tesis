"""Orquestación M6 → M7 → M8 tras generar series sintéticas."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.config.pipeline_config import NumericPipelineConfig, RNNConfig
from pipeline.config.settings import Settings
from pipeline.evaluation.hallucination import HallucinationDetector
from pipeline.evaluation.judge import Judge
from pipeline.prediction.predictor import Predictor
from pipeline.training.trainer import RNNTrainer
from pipeline.utils.logger import get_logger
from pipeline.utils.series_utils import seeds_statistics

logger = get_logger("numeric_pipeline")


def load_synthetic_matrix(csv_path: str) -> Tuple[np.ndarray, List[str]]:
  df = pd.read_csv(csv_path)
  feature_cols = [c for c in df.columns if c not in ("series_id", "timestamp")]
  if not feature_cols:
    raise ValueError("El CSV sintético no tiene columnas de variables.")

  blocks = []
  for _, group in df.groupby("series_id", sort=True):
    sub = group.sort_values("timestamp")
    blocks.append(sub[feature_cols].astype(float).values)

  data = np.vstack(blocks)
  return data, feature_cols


def run_numeric_pipeline(
  csv_path: Optional[str] = None,
  seed_dataset: Optional[List[Dict[str, Any]]] = None,
  domain: str = "general",
) -> Dict[str, Any]:
  csv_path = csv_path or Settings.synthetic_csv_path()
  if not Path(csv_path).exists():
    raise FileNotFoundError(f"No existe el CSV sintético: {csv_path}")

  logger.info("=" * 60)
  logger.info("M6 — Entrenamiento RNN (LSTM/GRU)")
  logger.info("=" * 60)

  data, feature_names = load_synthetic_matrix(csv_path)
  n_features = data.shape[1]

  cfg = NumericPipelineConfig()
  rnn_cfg = cfg.rnn
  rnn_cfg.input_size = n_features
  rnn_cfg.output_size = n_features

  min_rows = rnn_cfg.sequence_length + rnn_cfg.prediction_horizon + 20
  if len(data) < min_rows:
    raise ValueError(
      f"Datos insuficientes para RNN: {len(data)} filas; se requieren al menos {min_rows}. "
      f"Aumente SYNTHETIC_NUM_SERIES o SYNTHETIC_POINTS_PER_SERIES en .env."
    )

  trainer = RNNTrainer(config=rnn_cfg)
  history = trainer.train_from_data(data, checkpoint_dir=cfg.checkpoint_dir)
  trainer.save_model(cfg.model_output_path)

  logger.info("=" * 60)
  logger.info("M7 — Predicción y métricas (RMSE, MAE, MAPE, R²)")
  logger.info("=" * 60)

  predictor = Predictor(trainer=trainer)
  result = predictor.predict_from_data(data)
  predictor.plot_predictions(result, output_path=cfg.plot_output_path)

  logger.info("=" * 60)
  logger.info("M8 — Estudiante como juez + detección de alucinaciones")
  logger.info("=" * 60)

  domain_constraints = _domain_constraints_for(domain)
  judge = Judge(config=cfg.hallucination, domain_constraints=domain_constraints)
  judgment = judge.evaluate(result, ground_truth=data, feature_names=feature_names)

  detector = HallucinationDetector(config=cfg.hallucination)
  # Usar los datos de la serie (ground truth) como referencia estadística para evitar
  # falsos positivos causados por discrepancia de escala con las semillas crudas del RAG
  detector.set_reference(data, feature_names=feature_names)

  hall_report = detector.detect(result.predictions, feature_names=feature_names)

  print("\n" + result.summary())
  print("\n" + judgment.summary())
  print("\n" + hall_report.summary())

  return {
    "history": history,
    "prediction": result.to_dict(),
    "judgment": {
      "approved": judgment.approved,
      "confidence": judgment.confidence_score,
      "hallucination_score": judgment.hallucination_score,
    },
    "hallucination_rate": hall_report.hallucination_rate,
    "model_path": cfg.model_output_path,
    "plot_path": cfg.plot_output_path,
  }


def _domain_constraints_for(domain: str) -> Dict[str, Any]:
  d = domain.lower()
  if "clima" in d or "temperatura" in d:
    return {
      "temperatura": {"min": -40, "max": 55},
      "temperatura (°C)": {"min": -40, "max": 55},
      "humedad (%)": {"min": 0, "max": 100},
    }
  return {}
