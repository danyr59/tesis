"""
Módulo 6 — RNNTrainer: Loop de entrenamiento con early stopping.

Ejecuta el entrenamiento progresivo de la RNN lineal sobre datos
sintéticos (M5) + datos reales del dominio.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..utils.logger import get_logger
from ..config.pipeline_config import RNNConfig
from .model import LinearRNN
from .dataset import TimeSeriesDataset, create_dataloaders, normalize_data

logger = get_logger("m6.trainer")


class RNNTrainer:
    """
    Entrenador de la RNN lineal con:
    - Early stopping
    - Learning rate scheduling
    - Checkpointing
    - Métricas de entrenamiento (MSE, MAE, MAPE)
    """

    def __init__(
        self,
        model: Optional[LinearRNN] = None,
        config: Optional[RNNConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or RNNConfig()

        # Device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Modelo
        self.model = model or LinearRNN(self.config)
        self.model.to(self.device)

        # Optimizador
        self.optimizer = Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Loss
        self.criterion = nn.MSELoss()

        # Historial
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "lr": [],
        }

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None

        logger.info(
            f"RNNTrainer inicializado — device={self.device}, "
            f"params={self.model.count_parameters():,}"
        )

    # ── Entrenamiento principal ──────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Loop de entrenamiento completo.

        Args:
            train_loader: DataLoader de entrenamiento.
            val_loader: DataLoader de validación.
            max_epochs: Épocas máximas (default: config.max_epochs).
            checkpoint_dir: Directorio para guardar checkpoints.

        Returns:
            Historial de métricas.
        """
        epochs = max_epochs or self.config.max_epochs
        logger.info(f"Iniciando entrenamiento — {epochs} épocas máximas")
        logger.info(self.model.summary())

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_mae = self._train_epoch(train_loader)

            # Validate
            val_loss, val_mae = self._validate(val_loader)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            # Historial
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_mae"].append(train_mae)
            self.history["val_mae"].append(val_mae)
            self.history["lr"].append(current_lr)

            # Log
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Época {epoch:3d}/{epochs} | "
                    f"Train: loss={train_loss:.6f} mae={train_mae:.6f} | "
                    f"Val: loss={val_loss:.6f} mae={val_mae:.6f} | "
                    f"LR: {current_lr:.2e}"
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                if checkpoint_dir:
                    self._save_checkpoint(checkpoint_dir, epoch)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(
                        f"Early stopping en época {epoch} "
                        f"(paciencia={self.config.patience})"
                    )
                    break

        # Restaurar mejor modelo
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        elapsed = time.time() - start_time
        logger.info(
            f"Entrenamiento completo en {elapsed:.1f}s — "
            f"Mejor val_loss: {self.best_val_loss:.6f}"
        )

        return self.history

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Una época de entrenamiento."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred, _ = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(pred - y)).item()
            n_batches += 1

        return total_loss / max(n_batches, 1), total_mae / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Validación."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred, _ = self.model(x)
            loss = self.criterion(pred, y)

            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(pred - y)).item()
            n_batches += 1

        return total_loss / max(n_batches, 1), total_mae / max(n_batches, 1)

    # ── Utilidades ────────────────────────────────────────────────────────

    def _save_checkpoint(self, directory: str, epoch: int) -> None:
        """Guarda un checkpoint del modelo."""
        ckpt_dir = Path(directory)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": self.best_val_loss,
                "config": self.config,
            },
            path,
        )
        logger.debug(f"Checkpoint guardado: {path}")

    def save_model(self, filepath: str) -> None:
        """Guarda el modelo final."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "history": self.history,
            },
            path,
        )
        logger.info(f"Modelo guardado: {path}")

    @classmethod
    def load_model(cls, filepath: str) -> "RNNTrainer":
        """Carga un modelo entrenado."""
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        config = data["config"]
        trainer = cls(config=config)
        trainer.model.load_state_dict(data["model_state_dict"])
        trainer.history = data.get("history", {})
        logger.info(f"Modelo cargado desde {filepath}")
        return trainer

    # ── Entrenamiento desde datos crudos ─────────────────────────────────

    def train_from_data(
        self,
        data: np.ndarray,
        target_col: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Conveniencia: crea los dataloaders y entrena.

        Args:
            data: Array numpy (n_timestamps, n_features).
            target_col: Columna target.
            checkpoint_dir: Directorio para checkpoints.
        """
        # Normalizar
        norm_data, self._means, self._stds = normalize_data(data)

        # Crear dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            norm_data, self.config, target_col=target_col,
        )

        return self.train(train_loader, val_loader, checkpoint_dir=checkpoint_dir)
