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

    # Métricas (macro-promedio o de la variable única)
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    r2: float = 0.0

    # Desglose por variable en series multivariadas
    per_feature: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Metadata
    num_samples: int = 0
    model_type: str = ""

    def summary(self) -> str:
        lines = [
            f"Predicción — {self.num_samples} muestras ({self.model_type})",
            f"  MSE:  {self.mse:.6f}",
            f"  RMSE: {self.rmse:.6f}",
            f"  MAE:  {self.mae:.6f}",
            f"  MAPE: {self.mape:.2f}%",
            f"  R²:   {self.r2:.4f}"
        ]
        if self.per_feature:
            lines.append("  Desglose por variable:")
            for feat, m in self.per_feature.items():
                lines.append(
                    f"    • {feat:20s}: R²={m['r2']:.4f} | RMSE={m['rmse']:.4f} | MAE={m['mae']:.4f} | MAPE={m['mape']:.2f}%"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "num_samples": self.num_samples,
            "model_type": self.model_type,
        }
        if self.per_feature:
            d["per_feature"] = self.per_feature
        return d


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
    def predict(
        self,
        test_loader: DataLoader,
        feature_names: Optional[List[str]] = None,
    ) -> PredictionResult:
        """
        Genera predicciones sobre un DataLoader de prueba.

        Args:
            test_loader: DataLoader con datos de test.
            feature_names: Nombres de las variables en los datos.

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

        # Calcular métricas (por variable y promedios macro)
        metrics = self._compute_metrics(preds, actuals, feature_names=feature_names)

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
        feature_names: Optional[List[str]] = None,
    ) -> PredictionResult:
        """
        Conveniencia: predicción directa desde un array numpy.

        Args:
            data: Array temporal (n_timestamps, n_features).
            target_col: Columna target.
            feature_names: Nombres de las features para reportar métricas individuales.
        """
        # Normalizar
        norm_data, means, stds = normalize_data(data)
        self._means = means
        self._stds = stds

        config = self.model.config
        _, _, test_loader = create_dataloaders(
            norm_data, config, target_col=target_col,
        )

        return self.predict(test_loader, feature_names=feature_names)

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
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Calcula todas las métricas de rendimiento por variable y promedios macro."""
        # Asegurar 2D: (N, n_features)
        if predictions.ndim == 1:
            preds_2d = predictions.reshape(-1, 1)
            acts_2d = actuals.reshape(-1, 1)
        else:
            preds_2d = predictions
            acts_2d = actuals

        n_features = preds_2d.shape[1]
        if feature_names is None or len(feature_names) != n_features:
            feature_names = [f"var_{i}" for i in range(n_features)]

        per_feature: Dict[str, Dict[str, float]] = {}
        mses, rmses, maes, mapes, r2s = [], [], [], [], []

        for i, name in enumerate(feature_names):
            p = preds_2d[:, i]
            a = acts_2d[:, i]

            mse_i = float(np.mean((p - a) ** 2))
            rmse_i = float(np.sqrt(mse_i))
            mae_i = float(np.mean(np.abs(p - a)))

            nonzero_mask = a != 0
            if nonzero_mask.any():
                mape_i = float(
                    np.mean(np.abs((a[nonzero_mask] - p[nonzero_mask]) / a[nonzero_mask])) * 100
                )
            else:
                mape_i = 0.0

            ss_res = np.sum((a - p) ** 2)
            ss_tot = np.sum((a - np.mean(a)) ** 2)
            r2_i = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            per_feature[name] = {
                "mse": mse_i,
                "rmse": rmse_i,
                "mae": mae_i,
                "mape": mape_i,
                "r2": r2_i,
            }
            mses.append(mse_i)
            rmses.append(rmse_i)
            maes.append(mae_i)
            mapes.append(mape_i)
            r2s.append(r2_i)

        res: Dict[str, Any] = {
            "mse": float(np.mean(mses)),
            "rmse": float(np.mean(rmses)),
            "mae": float(np.mean(maes)),
            "mape": float(np.mean(mapes)),
            "r2": float(np.mean(r2s)),
        }
        if n_features > 1:
            res["per_feature"] = per_feature

        return res

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

            preds = result.predictions
            acts = result.actuals

            if preds.ndim == 1 or preds.shape[1] == 1:
                # Caso univariado estándar
                preds_1d = preds.flatten()
                acts_1d = acts.flatten()

                fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

                ax1 = axes[0]
                ax1.plot(acts_1d, label="Real", alpha=0.8, linewidth=1)
                ax1.plot(preds_1d, label="Predicción", alpha=0.8, linewidth=1)
                ax1.set_title(title, fontsize=14)
                ax1.set_ylabel("Valor")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                ax2 = axes[1]
                errors = preds_1d - acts_1d
                ax2.fill_between(
                    range(len(errors)), errors, alpha=0.4, color="red", label="Error"
                )
                ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
                ax2.set_xlabel("Paso temporal")
                ax2.set_ylabel("Error")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                metrics_text = (
                    f"RMSE={result.rmse:.4f}  MAE={result.mae:.4f}  "
                    f"MAPE={result.mape:.2f}%  R²={result.r2:.4f}"
                )
                fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=10, style="italic")
            else:
                # Caso multivariado: un panel individual por cada variable
                n_feats = preds.shape[1]
                names = list(result.per_feature.keys()) if result.per_feature else [f"Var_{i+1}" for i in range(n_feats)]
                fig, axes = plt.subplots(n_feats, 1, figsize=(14, 4 * n_feats), sharex=True)
                if n_feats == 1:
                    axes = [axes]

                for j in range(n_feats):
                    ax = axes[j]
                    col_name = names[j]
                    feat_metrics = result.per_feature.get(col_name, {})
                    r2_str = f" | R²={feat_metrics['r2']:.4f} | MAE={feat_metrics['mae']:.4f}" if feat_metrics else ""
                    ax.plot(acts[:, j], label=f"Real ({col_name})", alpha=0.8, linewidth=1)
                    ax.plot(preds[:, j], label=f"Predicción ({col_name})", alpha=0.8, linestyle="--", linewidth=1)
                    ax.set_title(f"{col_name}{r2_str}", fontsize=12)
                    ax.set_ylabel(col_name)
                    ax.legend(loc="upper right")
                    ax.grid(True, alpha=0.3)

                axes[-1].set_xlabel("Paso temporal")
                metrics_text = (
                    f"Promedio Macro — RMSE={result.rmse:.4f}  MAE={result.mae:.4f}  "
                    f"MAPE={result.mape:.2f}%  R²={result.r2:.4f}"
                )
                fig.suptitle(title, fontsize=14)
                fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=10, style="italic")

            plt.tight_layout(rect=[0, 0.03, 1, 0.97] if preds.ndim > 1 and preds.shape[1] > 1 else [0, 0.03, 1, 1])

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                logger.info(f"Gráfico guardado en {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib no instalado para visualización")
