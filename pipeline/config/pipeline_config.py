from dataclasses import dataclass, field
from pathlib import Path
import sys
import torch
from typing import Optional, Any

# Agregar proyecto_tesis/ al path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.ioa_engine.config.config import IOAConfig

@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: list = field(default_factory=lambda: ["\n\n", "\n", ".", " ", ""])

@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    normalize: bool = True
    max_length: int = 512

@dataclass
class VectorStoreConfig:
    collection_methodology: str = "methodology"
    collection_domain: str = "domain_knowledge"
    persist_directory: str = "./chroma_db"
    distance_metric: str = "cosine"

@dataclass
class RetrievalConfig:
    top_k: int = 5
    score_threshold: float = 0.5
    z_score_threshold: float = 3.0
    max_step_change_pct: float = 20.0
    min_cross_correlation: float = 0.7
    confidence_level: float = 0.95
    approval_threshold: float = 0.7
    min_r2: float = 0.5
    max_mape: float = 20.0
    max_nrmse: float = 0.2
    max_mean_diff_pct: float = 10.0
    min_std_ratio: float = 0.5
    max_std_ratio: float = 1.5
    max_corr_diff: float = 0.3

@dataclass
class ContextBuilderConfig:
    max_context_tokens: int = 2048
    include_metadata: bool = True
    include_history: bool = True

@dataclass
class LLMConfig:
    teacher_model: str = "./models/profesor"
    teacher_type: str = "local"
    teacher_timeout: int = 120
    student_type: str = "local"
    student_model: str = "./models/student"
    student_api_key: str = None
    student_api_base: str = None
    student_timeout: int = 120

@dataclass
class SeedFactoryConfig:
    seed_dir: str = "./data/seed"
    methodology_size: int = 5
    domain_size: int = 5

@dataclass
class SynthesisConfig:
    num_samples: int = 30
    method: str = "llm_autoregressive"
    batch_size: int = 5
    temperature: float = 0.85
    top_p: float = 0.95
    mandatory_jitter: float = 0.002
    max_tokens_per_sample: int = 128
    diversity_threshold: float = 0.5
    z_score_threshold: float = 3.0
    max_step_change_pct: float = 20.0
    min_cross_correlation: float = 0.7
    confidence_level: float = 0.95
    approval_threshold: float = 0.7
    min_r2: float = 0.5
    max_mape: float = 20.0
    max_nrmse: float = 0.2
    max_mean_diff_pct: float = 10.0
    min_std_ratio: float = 0.5
    max_std_ratio: float = 1.5
    max_corr_diff: float = 0.3

@dataclass
class RNNConfig:
    model_type: str = "LSTM"
    input_size: int = 4
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    learning_rate: float = 0.001
    max_epochs: int = 5
    batch_size: int = 16
    sequence_length: int = 24
    prediction_horizon: int = 1

@dataclass
class PredictionConfig:
    threshold: float = 0.8

@dataclass
class HallucinationConfig:
    threshold: float = 0.5
    z_score_threshold: float = 3.0
    max_step_change_pct: float = 20.0
    min_cross_correlation: float = 0.7
    confidence_level: float = 0.95
    approval_threshold: float = 0.7
    min_r2: float = 0.5
    max_mape: float = 20.0
    max_nrmse: float = 0.2
    max_mean_diff_pct: float = 10.0
    min_std_ratio: float = 0.5
    max_std_ratio: float = 1.5
    max_corr_diff: float = 0.3

@dataclass
class EvaluationConfig:
    metric: str = "accuracy"

@dataclass
class PipelineConfig:
    domain: Optional[str] = None
    query: Optional[str] = None
    output_dir: str = "./outputs"
    log_level: str = "DEBUG"
    seed: int = 42
    max_taxonomy_modules: int = 5
    
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    context: ContextBuilderConfig = field(default_factory=ContextBuilderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    seed_factory: SeedFactoryConfig = field(default_factory=SeedFactoryConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    training: RNNConfig = field(default_factory=RNNConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ioa: IOAConfig = field(default_factory=IOAConfig)

def get_default_pipeline_config():
    return PipelineConfig()
