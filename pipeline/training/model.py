"""
Módulo 6 — LinearRNN: Arquitectura de RNN con capa de salida lineal.

Variantes disponibles:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- RNN Vanilla

Todas las variantes comparten la misma capa lineal de salida para mapear
del espacio oculto al espacio de predicción.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..utils.logger import get_logger
from ..config.pipeline_config import RNNConfig

logger = get_logger("m6.model")


class LinearRNN(nn.Module):
    """
    RNN con capa de salida lineal para predicción de series temporales.

    Arquitectura:
        input → [RNN/LSTM/GRU] → (hidden_state) → [Linear] → output
    """

    def __init__(self, config: Optional[RNNConfig] = None):
        super().__init__()
        self.config = config or RNNConfig()
        c = self.config

        # Seleccionar tipo de RNN
        rnn_cls = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn": nn.RNN,
        }.get(c.model_type.lower(), nn.LSTM)

        self.rnn = rnn_cls(
            input_size=c.input_size,
            hidden_size=c.hidden_size,
            num_layers=c.num_layers,
            dropout=c.dropout if c.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=c.bidirectional,
        )

        # Capa lineal de salida
        direction_factor = 2 if c.bidirectional else 1
        self.fc = nn.Linear(c.hidden_size * direction_factor, c.output_size)

        # Dropout antes de la capa lineal
        self.dropout = nn.Dropout(c.dropout)

        self._model_type = c.model_type
        self._num_directions = direction_factor

        logger.info(
            f"LinearRNN creada: {c.model_type.upper()} — "
            f"input={c.input_size}, hidden={c.hidden_size}, "
            f"layers={c.num_layers}, output={c.output_size}"
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Tensor de shape (batch, seq_len, input_size)
            hidden: Estado oculto previo (opcional).

        Returns:
            (output, hidden_state)
            output: shape (batch, output_size)
            hidden_state: último estado oculto de la RNN
        """
        # RNN forward
        rnn_out, hidden_state = self.rnn(x, hidden)

        # Tomar solo la salida del último paso temporal
        last_out = rnn_out[:, -1, :]  # (batch, hidden_size * directions)

        # Dropout + capa lineal
        out = self.dropout(last_out)
        out = self.fc(out)  # (batch, output_size)

        return out, hidden_state

    def predict_sequence(
        self,
        x: torch.Tensor,
        steps: int = 1,
    ) -> torch.Tensor:
        """
        Predicción autoregresiva: genera `steps` pasos futuros.

        Args:
            x: Secuencia de entrada (batch, seq_len, input_size)
            steps: Pasos a predecir hacia adelante.

        Returns:
            Tensor de predicciones (batch, steps, output_size)
        """
        self.eval()
        predictions = []

        current_input = x
        hidden = None

        with torch.no_grad():
            for _ in range(steps):
                out, hidden = self.forward(current_input, hidden)
                predictions.append(out.unsqueeze(1))

                # Usar la predicción como entrada del siguiente paso
                # Expandir para que coincida con input_size
                next_input = out.unsqueeze(1)
                if next_input.shape[-1] != self.config.input_size:
                    # Pad o slice según sea necesario
                    next_input = next_input[:, :, : self.config.input_size]

                current_input = next_input

        return torch.cat(predictions, dim=1)

    def count_parameters(self) -> int:
        """Cuenta los parámetros trainables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Resumen del modelo."""
        params = self.count_parameters()
        c = self.config
        return (
            f"LinearRNN ({c.model_type.upper()})\n"
            f"  Input size:  {c.input_size}\n"
            f"  Hidden size: {c.hidden_size}\n"
            f"  Layers:      {c.num_layers}\n"
            f"  Dropout:     {c.dropout}\n"
            f"  Bidirect:    {c.bidirectional}\n"
            f"  Output size: {c.output_size}\n"
            f"  Parameters:  {params:,}\n"
        )
