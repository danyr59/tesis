"""
Módulo 2 — Retriever: Búsqueda de similitud semántica.

Cuando el usuario realiza una consulta, el sistema:
1. Codifica la consulta al mismo espacio vectorial (mismo modelo de embeddings).
2. Recupera los k fragmentos más cercanos usando similitud de coseno.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from ..utils.logger import get_logger
from ..config.pipeline_config import RetrievalConfig
from ..ingestion.vector_store import VectorStore

logger = get_logger("m2.retriever")


@dataclass
class RetrievalResult:
    """Resultado de una búsqueda de recuperación."""
    query: str
    fragments: List[Dict[str, Any]] = field(default_factory=list)
    total_found: int = 0


class Retriever:
    """
    Recupera fragmentos relevantes de la base vectorial para una consulta.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        config: Optional[RetrievalConfig] = None,
    ):
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Recupera los fragmentos más relevantes para una consulta.

        Args:
            query: Consulta en lenguaje natural.
            top_k: Cantidad de fragmentos a recuperar (default: config.top_k).
            filters: Filtros adicionales (e.g., {"source": "manual.pdf"}).

        Returns:
            RetrievalResult con los fragmentos encontrados y sus scores.
        """
        k = top_k or self.config.top_k

        logger.info(f"Recuperando top-{k} fragmentos para: '{query[:80]}...'")

        raw_results = self.vector_store.query(
            query_text=query,
            top_k=k,
            where=filters,
        )

        # Filtrar por umbral de score
        filtered = [
            r for r in raw_results
            if r["score"] >= self.config.score_threshold
        ]

        logger.info(
            f"Recuperados {len(raw_results)} fragmentos, "
            f"{len(filtered)} pasaron umbral ({self.config.score_threshold})"
        )

        result = RetrievalResult(
            query=query,
            fragments=filtered,
            total_found=len(filtered),
        )

        return result

    def retrieve_with_context(
        self,
        query: str,
        context_queries: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Recuperación enriquecida: busca con la consulta + consultas de contexto.

        Útil para multi-turn conversations donde el historial agrega semántica.

        Args:
            query: Consulta principal.
            context_queries: Consultas adicionales de contexto (historial).
            top_k: Cantidad total de fragmentos.

        Returns:
            RetrievalResult combinando resultados de todas las consultas.
        """
        k = top_k or self.config.top_k
        all_fragments: Dict[str, Dict[str, Any]] = {}

        # Consulta principal (peso mayor)
        main_result = self.retrieve(query, top_k=k)
        for frag in main_result.fragments:
            fid = frag["id"]
            frag["relevance_source"] = "query"
            all_fragments[fid] = frag

        # Consultas de contexto (peso menor)
        if context_queries:
            per_ctx_k = max(1, k // len(context_queries))
            for ctx_q in context_queries:
                ctx_result = self.retrieve(ctx_q, top_k=per_ctx_k)
                for frag in ctx_result.fragments:
                    fid = frag["id"]
                    if fid not in all_fragments:
                        frag["relevance_source"] = "context"
                        frag["score"] *= 0.8  # Penalizar contexto
                        all_fragments[fid] = frag

        # Ordenar por score y tomar top_k
        sorted_frags = sorted(
            all_fragments.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:k]

        return RetrievalResult(
            query=query,
            fragments=sorted_frags,
            total_found=len(sorted_frags),
        )
