"""
Módulo 6 — TimeSeriesDataset: Dataset PyTorch para series temporales.

Crea ventanas deslizantes (sliding windows) para entrenamiento supervisado:
  input:  [x_t, x_{t+1}, ..., x_{t+seq_len-1}]
  target: [x_{t+seq_len}, ..., x_{t+seq_len+horizon-1}]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..utils.logger import get_logger
from ..config.pipeline_config import RNNConfig

logger = get_logger("m6.dataset")


class TimeSeriesDataset(Dataset):
    """
    Dataset para series temporales con ventanas deslizantes.
    """

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        target_col: Optional[int] = None,
    ):
        """
        Args:
            data: Array de shape (n_timestamps, n_features).
            sequence_length: Longitud de la ventana de entrada.
            prediction_horizon: Pasos a predecir.
            target_col: Índice de la columna target (None = todas).
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.data = torch.FloatTensor(data)
        self.seq_len = sequence_length
        self.horizon = prediction_horizon
        self.target_col = target_col
        self.n_samples = len(data) - sequence_length - prediction_horizon + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"Datos insuficientes: {len(data)} timestamps para "
                f"seq_len={sequence_length}, horizon={prediction_horizon}"
            )

        logger.info(
            f"Dataset creado: {self.n_samples} ventanas — "
            f"seq_len={sequence_length}, horizon={prediction_horizon}, "
            f"features={data.shape[1]}"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna:
            x: (seq_len, n_features) — secuencia de entrada
            y: (prediction_horizon, target_features) — target
        """
        x = self.data[idx : idx + self.seq_len]
        y_start = idx + self.seq_len
        y_end = y_start + self.horizon
        y = self.data[y_start:y_end]

        if self.target_col is not None:
            y = y[:, self.target_col : self.target_col + 1]

        # Si horizon == 1, aplanar el target
        if self.horizon == 1:
            y = y.squeeze(0)

        return x, y


def create_dataloaders(
    data: np.ndarray,
    config: Optional[RNNConfig] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    target_col: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para train/val/test a partir de un array numpy.

    Split temporal (no aleatorio) para respetar la causalidad.

    Args:
        data: Array (n_timestamps, n_features).
        config: Configuración de la RNN.
        train_ratio: Proporción de datos para training.
        val_ratio: Proporción de datos para validation.
        target_col: Columna target (None = todas).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    config = config or RNNConfig()

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    logger.info(
        f"Split temporal: train={len(train_data)}, "
        f"val={len(val_data)}, test={len(test_data)}"
    )

    train_ds = TimeSeriesDataset(
        train_data,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
        target_col=target_col,
    )
    val_ds = TimeSeriesDataset(
        val_data,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
        target_col=target_col,
    )
    test_ds = TimeSeriesDataset(
        test_data,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
        target_col=target_col,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


def normalize_data(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normaliza datos con z-score.

    Returns:
        (normalized_data, means, stds)
    """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0] = 1.0  # Evitar div/0
    normalized = (data - means) / stds
    return normalized, means, stds


def denormalize_data(
    data: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    """Desnormaliza datos."""
    return data * stds + means
