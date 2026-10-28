#!/usr/bin/env python3
"""Script rápido para ejecutar la evaluación (M7 + M8) sobre el modelo ya entrenado.

No requiere reentrenar la RNN; carga directamente data/checkpoints/rnn_final.pt.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Agregar raíz del proyecto al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config.pipeline_config import NumericPipelineConfig
from pipeline.config.settings import Settings
from pipeline.evaluation.hallucination import HallucinationDetector
from pipeline.evaluation.judge import Judge
from pipeline.prediction.predictor import Predictor
from pipeline.numeric_pipeline import load_synthetic_matrix


def main():
    csv_path = Settings.synthetic_csv_path()
    checkpoint_path = "data/checkpoints/rnn_final.pt"

    if not Path(csv_path).exists():
        print(f"Error: no existe el dataset en {csv_path}")
        sys.exit(1)

    if not Path(checkpoint_path).exists():
        print(f"Error: no existe el checkpoint en {checkpoint_path}")
        print("Ejecute primero el entrenamiento completo con: python -m pipeline.numeric_pipeline")
        sys.exit(1)

    print("=" * 65)
    print(" EVALUACIÓN M7 (PREDICTOR) + M8 (JUEZ & DETECTOR ALUCINACIONES)")
    print("=" * 65)
    print(f"📂 Dataset:    {csv_path}")
    print(f"🧠 Checkpoint: {checkpoint_path} (sin reentrenar)")
    print("-" * 65)

    data, feature_names = load_synthetic_matrix(csv_path)

    # 1. Cargar Predictor y evaluar
    predictor = Predictor.from_checkpoint(checkpoint_path)
    result = predictor.predict_from_data(data)

    # 2. Juez (Consistencia y Plausibilidad)
    cfg = NumericPipelineConfig()
    judge = Judge(config=cfg.hallucination)
    judgment = judge.evaluate(result, ground_truth=data, feature_names=feature_names)

    # 3. Detector Estadístico Multidimensional de Alucinaciones
    detector = HallucinationDetector(config=cfg.hallucination)
    detector.set_reference(data, feature_names=feature_names)
    hall_report = detector.detect(result.predictions, feature_names=feature_names)

    # 4. Mostrar resumen consolidado
    print("\n" + "=" * 65)
    print("RESUMEN DE MÉTRICAS OBTENIDAS:")
    print("=" * 65)
    print(result.summary())
    print("\n" + judgment.summary())
    print("\n" + hall_report.summary())


if __name__ == "__main__":
    main()
