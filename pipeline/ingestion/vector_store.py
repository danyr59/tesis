"""
Módulo 1 — VectorStore: Almacenamiento en base de datos vectorial.

Los vectores se indexan en ChromaDB para permitir búsquedas rápidas
por similitud.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from pathlib import Path

from ..utils.logger import get_logger
from ..config.pipeline_config import VectorStoreConfig
from .chunker import Chunk
from .embedder import Embedder

logger = get_logger("m1.vector_store")


class VectorStore:
    """
    Interfaz con ChromaDB (persistente, local).
    """

    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.config = config or VectorStoreConfig()
        self.embedder = embedder or Embedder()
        self._client = None
        self._collection = None

    # ── Lazy init ─────────────────────────────────────────────────────────

    def _ensure_client(self):
        if self._client is not None:
            return

        try:
            import chromadb  # type: ignore

            persist_dir = Path(self.config.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(persist_dir)
            )
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric},
            )

            logger.info(
                f"ChromaDB inicializado — colección '{self.config.collection_name}' "
                f"({self._collection.count()} documentos)"
            )

        except ImportError:
            logger.error(
                "chromadb no instalado. Instalar con: pip install chromadb"
            )
            raise

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Agrega chunks a la base vectorial.

        Returns:
            Cantidad de chunks agregados.
        """
        self._ensure_client()

        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "position": c.position,
                **({"page": c.page} if c.page is not None else {}),
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]

        # Generar embeddings
        embeddings = self.embedder.encode(texts)

        # Upsert en ChromaDB
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        logger.info(f"Agregados/actualizados {len(chunks)} chunks en VectorDB")
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Busca los k fragmentos más cercanos a una consulta.

        Returns:
            Lista de dicts con keys: id, text, score, metadata
        """
        self._ensure_client()

        query_embedding = self.embedder.encode_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        items: List[Dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            items.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "score": 1.0 - results["distances"][0][i],  # cosine → similarity
                    "metadata": results["metadatas"][0][i],
                }
            )

        return items

    def count(self) -> int:
        """Retorna la cantidad de documentos en la colección."""
        self._ensure_client()
        return self._collection.count()

    def delete_collection(self) -> None:
        """Elimina la colección completa."""
        self._ensure_client()
        self._client.delete_collection(self.config.collection_name)
        self._collection = None
        logger.info(f"Colección '{self.config.collection_name}' eliminada")

    def reset(self) -> None:
        """Reset: elimina y recrea la colección."""
        self._ensure_client()
        try:
            self._client.delete_collection(self.config.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )
        logger.info("VectorDB reseteado")
