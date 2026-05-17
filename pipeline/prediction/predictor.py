"""
Módulo 7 — Predictor: Prueba y predicción.

Carga el modelo entrenado (M6), genera predicciones sobre el dataset
de prueba, y calcula las métricas de rendimiento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.logger import get_logger
from ..config.pipeline_config import RNNConfig
from ..training.model import LinearRNN
from ..training.dataset import (
    TimeSeriesDataset,
    create_dataloaders,
    normalize_data,
    denormalize_data,
)
from ..training.trainer import RNNTrainer

logger = get_logger("m7.predictor")


@dataclass
class PredictionResult:
    """Resultado completo de predicción."""

    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    actuals: np.ndarray = field(default_factory=lambda: np.array([]))

    # Métricas
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    r2: float = 0.0

    # Metadata
    num_samples: int = 0
    model_type: str = ""

    def summary(self) -> str:
        return (
            f"Predicción — {self.num_samples} muestras ({self.model_type})\n"
            f"  MSE:  {self.mse:.6f}\n"
            f"  RMSE: {self.rmse:.6f}\n"
            f"  MAE:  {self.mae:.6f}\n"
            f"  MAPE: {self.mape:.2f}%\n"
            f"  R²:   {self.r2:.4f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "num_samples": self.num_samples,
            "model_type": self.model_type,
        }


class Predictor:
    """
    Genera predicciones con el modelo RNN entrenado y calcula métricas.
    """

    def __init__(
        self,
        model: Optional[LinearRNN] = None,
        trainer: Optional[RNNTrainer] = None,
        device: Optional[str] = None,
    ):
        if trainer is not None:
            self.model = trainer.model
            self.device = trainer.device
            self._means = getattr(trainer, "_means", None)
            self._stds = getattr(trainer, "_stds", None)
        elif model is not None:
            self.model = model
            self.device = (
                torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
            )
            self._means = None
            self._stds = None
        else:
            raise ValueError("Debe proporcionar un model o un trainer")

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, filepath: str) -> "Predictor":
        """Carga predictor desde un checkpoint guardado."""
        trainer = RNNTrainer.load_model(filepath)
        return cls(trainer=trainer)

    # ── Predicción principal ─────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> PredictionResult:
        """
        Genera predicciones sobre un DataLoader de prueba.

        Args:
            test_loader: DataLoader con datos de test.

        Returns:
            PredictionResult con predicciones, actuals y métricas.
        """
        self.model.eval()
        all_preds: List[torch.Tensor] = []
        all_actuals: List[torch.Tensor] = []

        for x, y in test_loader:
            x = x.to(self.device)
            pred, _ = self.model(x)
            all_preds.append(pred.cpu())
            all_actuals.append(y.cpu())

        preds = torch.cat(all_preds, dim=0).numpy()
        actuals = torch.cat(all_actuals, dim=0).numpy()

        # Desnormalizar si corresponde
        if self._means is not None and self._stds is not None:
            preds = denormalize_data(preds, self._means, self._stds)
            actuals = denormalize_data(actuals, self._means, self._stds)

        # Calcular métricas
        metrics = self._compute_metrics(preds, actuals)

        result = PredictionResult(
            predictions=preds,
            actuals=actuals,
            num_samples=len(preds),
            model_type=self.model.config.model_type,
            **metrics,
        )

        logger.info(result.summary())
        return result

    def predict_from_data(
        self,
        data: np.ndarray,
        target_col: Optional[int] = None,
    ) -> PredictionResult:
        """
        Conveniencia: predicción directa desde un array numpy.

        Args:
            data: Array temporal (n_timestamps, n_features).
            target_col: Columna target.
        """
        # Normalizar
        norm_data, means, stds = normalize_data(data)
        self._means = means
        self._stds = stds

        config = self.model.config
        _, _, test_loader = create_dataloaders(
            norm_data, config, target_col=target_col,
        )

        return self.predict(test_loader)

    @torch.no_grad()
    def predict_next(
        self, sequence: np.ndarray, steps: int = 1
    ) -> np.ndarray:
        """
        Predicción de los próximos `steps` valores a partir de una secuencia.

        Args:
            sequence: Secuencia de entrada (seq_len, n_features).
            steps: Pasos hacia adelante.

        Returns:
            Array de predicciones (steps, output_size).
        """
        self.model.eval()

        if sequence.ndim == 1:
            sequence = sequence.reshape(-1, 1)

        # Normalizar si tenemos stats
        if self._means is not None:
            sequence = (sequence - self._means) / self._stds

        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        preds = self.model.predict_sequence(x, steps=steps)
        result = preds.squeeze(0).cpu().numpy()

        # Desnormalizar
        if self._means is not None:
            result = denormalize_data(result, self._means, self._stds)

        return result

    # ── Métricas ─────────────────────────────────────────────────────────

    def _compute_metrics(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> Dict[str, float]:
        """Calcula todas las métricas de rendimiento."""
        # Aplanar si es necesario
        preds = predictions.flatten()
        acts = actuals.flatten()

        # MSE
        mse = float(np.mean((preds - acts) ** 2))

        # RMSE
        rmse = float(np.sqrt(mse))

        # MAE
        mae = float(np.mean(np.abs(preds - acts)))

        # MAPE (con protección contra div/0)
        nonzero_mask = acts != 0
        if nonzero_mask.any():
            mape = float(
                np.mean(np.abs((acts[nonzero_mask] - preds[nonzero_mask]) / acts[nonzero_mask])) * 100
            )
        else:
            mape = 0.0

        # R²
        ss_res = np.sum((acts - preds) ** 2)
        ss_tot = np.sum((acts - np.mean(acts)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
        }

    # ── Visualización ────────────────────────────────────────────────────

    def plot_predictions(
        self,
        result: PredictionResult,
        output_path: Optional[str] = None,
        title: str = "Predicciones vs Reales",
    ) -> None:
        """
        Genera gráfico de predicciones vs valores reales.

        Args:
            result: Resultado de predicción.
            output_path: Ruta para guardar el gráfico (None = show).
            title: Título del gráfico.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            preds = result.predictions.flatten()
            acts = result.actuals.flatten()

            fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

            # Panel superior: predicciones vs reales
            ax1 = axes[0]
            ax1.plot(acts, label="Real", alpha=0.8, linewidth=1)
            ax1.plot(preds, label="Predicción", alpha=0.8, linewidth=1)
            ax1.set_title(title, fontsize=14)
            ax1.set_ylabel("Valor")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Panel inferior: error
            ax2 = axes[1]
            errors = preds - acts
            ax2.fill_between(
                range(len(errors)), errors, alpha=0.4, color="red", label="Error"
            )
            ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
            ax2.set_xlabel("Paso temporal")
            ax2.set_ylabel("Error")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Texto con métricas
            metrics_text = (
                f"RMSE={result.rmse:.4f}  MAE={result.mae:.4f}  "
                f"R²={result.r2:.4f}  MAPE={result.mape:.1f}%"
            )
            fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=10, style="italic")

            plt.tight_layout(rect=[0, 0.03, 1, 1])

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                logger.info(f"Gráfico guardado en {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib no instalado para visualización")
