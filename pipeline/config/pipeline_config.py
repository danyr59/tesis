"""
Configuración centralizada del pipeline.

Extiende la IOAConfig existente con parámetros específicos de:
- Ingestión y chunking
- Base de datos vectorial
- RNN lineal
- Detección de alucinaciones
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import sys

# Agregar proyecto_tesis/ al path para que IOA sea importable como paquete
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IOA.config.config import (
    IOAConfig,
    IdentifierConfig,
    OrganizerConfig,
    AdapterConfig,
    get_default_config as get_default_ioa_config,
)


# ─────────────────────────────────────────────────────────────────────────────
# Nuevas configs específicas del pipeline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChunkingConfig:
    """Configuración de segmentación de documentos (Módulo 1)."""
    chunk_size: int = 512          # tokens por chunk
    chunk_overlap: int = 64        # tokens de solapamiento
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "]
    )


@dataclass
class EmbeddingConfig:
    """Configuración de embeddings (Módulo 1)."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 64
    normalize: bool = True


@dataclass
class SeedConfig:
    """Configuración para la Fábrica de Semillas (Módulo 4)."""
    methodology_size: int = 20
    domain_size: int =5
    seed_dir: str = "./data/seed/predictive_maintenance"


@dataclass
class VectorStoreConfig:
    """Configuración de ChromaDB (Módulo 1)."""
    persist_directory: str = "./data/vectordb"
    collection_domain: str = "domain_knowledge"
    collection_methodology: str = "methodology_knowledge"
    distance_metric: str = "cosine"   # cosine | l2 | ip


@dataclass
class RetrievalConfig:
    """Configuración de recuperación (Módulo 2)."""
    top_k: int = 5
    score_threshold: float = 0.15
    rerank: bool = False


@dataclass
class ContextBuilderConfig:
    """Configuración del constructor de contexto (Módulo 3)."""
    max_context_tokens: int = 4096
    include_metadata: bool = True
    include_history: bool = True
    history_window: int = 5


@dataclass
class RNNConfig:
    """Configuración de la RNN lineal (Módulo 6)."""
    model_type: str = "lstm"        # lstm | gru | rnn
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    output_size: int = 1
    # Entrenamiento
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10              # early stopping
    sequence_length: int = 24       # ventana de entrada
    prediction_horizon: int = 1     # pasos a predecir


@dataclass
class HallucinationConfig:
    """Configuración de detección de alucinaciones (Módulo 8)."""
    # Umbrales de validación de rango
    z_score_threshold: float = 3.0
    # Correlaciones cruzadas mínimas esperadas
    min_cross_correlation: float = 0.3
    # Cambio porcentual máximo entre pasos consecutivos
    max_step_change_pct: float = 50.0
    # Nivel de confianza para tests estadísticos
    confidence_level: float = 0.95


@dataclass
class LLMConfig:
    """Configuración de los agentes LLM (Local vía Ollama por defecto)."""
    # Maestro (M4) - Mantiene el de 3 Billones (llama3.2)
    teacher_type: str = "openai"
    teacher_model: str = "llama3.2"
    teacher_api_base: str = "http://localhost:11434/v1"
    teacher_api_key: str = "ollama-local"
    
    teacher_timeout: int = 300
    student_timeout: int = 300
    student_type: str = "openai"
    student_model: str = "llama3.2:1b"
    student_api_base: str = "http://localhost:11434/v1"
    student_api_key: str = "ollama-local"


@dataclass
class PipelineConfig:
    """
    Configuración maestra del pipeline de 9 módulos.

    Compone la IOAConfig original con las configs nuevas.
    """

    # --- Config de LLMs ---
    llm: LLMConfig = field(default_factory=LLMConfig)

    # --- IOA (Módulo 4) ---
    ioa: IOAConfig = field(default_factory=get_default_ioa_config)

    # --- Ingestión (Módulo 1) ---
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    # --- Recuperación (Módulo 2) ---
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    context_builder: ContextBuilderConfig = field(
        default_factory=ContextBuilderConfig
    )

    # --- Maestro (Módulo 4) ---
    seed_factory: SeedConfig = field(default_factory=SeedConfig)

    # --- RNN (Módulo 6) ---
    rnn: RNNConfig = field(default_factory=RNNConfig)

    # --- Alucinaciones (Módulo 8) ---
    hallucination: HallucinationConfig = field(
        default_factory=HallucinationConfig
    )

    # --- Paths generales ---
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    raw_docs_dir: str = "./data/raw"
    seed: int = 42

    # --- Experiment ---
    experiment_name: str = "thesis_pipeline"
    log_level: str = "INFO"
    log_file: Optional[str] = "pipeline.log"

    # ───── helpers ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serializa la config a JSON."""
        from dataclasses import asdict

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Carga config desde JSON (simplificado)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Reconstruir IOA
        ioa = IOAConfig.from_dict(data.pop("ioa", {}))
        # Reconstruir sub-configs
        llm = LLMConfig(**data.pop("llm", {}))
        chunking = ChunkingConfig(**data.pop("chunking", {}))
        embedding = EmbeddingConfig(**data.pop("embedding", {}))
        vector_store = VectorStoreConfig(**data.pop("vector_store", {}))
        retrieval = RetrievalConfig(**data.pop("retrieval", {}))
        context_builder = ContextBuilderConfig(**data.pop("context_builder", {}))
        seed_factory = SeedConfig(**data.pop("seed_factory", {}))
        rnn = RNNConfig(**data.pop("rnn", {}))
        hallucination = HallucinationConfig(**data.pop("hallucination", {}))
        return cls(
            ioa=ioa,
            llm=llm,
            chunking=chunking,
            embedding=embedding,
            vector_store=vector_store,
            retrieval=retrieval,
            context_builder=context_builder,
            seed_factory=seed_factory,
            rnn=rnn,
            hallucination=hallucination,
            **data,
        )


def get_default_pipeline_config() -> PipelineConfig:
    """Config por defecto lista para usar."""
    return PipelineConfig()
