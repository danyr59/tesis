"""Configuración de los módulos numéricos M6–M8."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RNNConfig:
  model_type: str = "lstm"
  input_size: int = 1
  hidden_size: int = 64
  num_layers: int = 2
  dropout: float = 0.2
  bidirectional: bool = False
  output_size: int = 1
  sequence_length: int = 60
  prediction_horizon: int = 1
  batch_size: int = 32
  learning_rate: float = 1e-3
  max_epochs: int = 100
  patience: int = 10


@dataclass
class HallucinationConfig:
  z_score_threshold: float = 3.0
  max_step_change_pct: float = 50.0
  min_cross_correlation: float = 0.5
  confidence_level: float = 0.95
  approval_threshold: float = 0.6
  min_r2: float = 0.3
  max_mape: float = 50.0
  max_nrmse: float = 0.5
  max_corr_diff: float = 0.4
  max_mean_diff_pct: float = 25.0
  min_std_ratio: float = 0.5
  max_std_ratio: float = 2.0


@dataclass
class NumericPipelineConfig:
  rnn: RNNConfig = field(default_factory=RNNConfig)
  hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
  checkpoint_dir: str = "./data/checkpoints"
  model_output_path: str = "./data/checkpoints/rnn_final.pt"
  plot_output_path: str = "./data/synthetic/predictions_plot.png"
