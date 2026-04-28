"""
Módulo 1 — Embedder: Generación de embeddings.

Convierte fragmentos de texto en vectores numéricos (embeddings)
que capturan su significado semántico.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from ..utils.logger import get_logger
from ..config.pipeline_config import EmbeddingConfig

logger = get_logger("m1.embedder")


class Embedder:
    """
    Wrapper sobre sentence-transformers para generar embeddings.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None

    # ── Lazy-load para evitar import pesado al instanciar ─────────────────

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info(f"Cargando modelo de embeddings: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)
            # Actualizar dim real
            test = self._model.encode(["test"])
            self.config.embedding_dim = test.shape[1]
            logger.info(
                f"Modelo cargado — dim={self.config.embedding_dim}"
            )
        except ImportError:
            logger.warning(
                "sentence-transformers no instalado. "
                "Instalar con: pip install sentence-transformers"
            )
            raise

    # ── API principal ──────────────────────────────────────────────────────

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts: Lista de strings a codificar.

        Returns:
            np.ndarray de shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(texts) > 100,
        )

        logger.debug(f"Codificados {len(texts)} textos → shape {embeddings.shape}")
        return np.array(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Codifica una sola consulta al espacio vectorial."""
        return self.encode([query])[0]
