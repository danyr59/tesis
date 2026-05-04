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
        self._collections: Dict[str, Any] = {}

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
            logger.info(f"ChromaDB conectado en: {persist_dir}")

        except ImportError:
            logger.error(
                "chromadb no instalado. Instalar con: pip install chromadb"
            )
            raise

    def _get_collection(self, name: Optional[str] = None):
        self._ensure_client()
        name = name or self.config.collection_domain
        
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": self.config.distance_metric},
            )
            logger.info(f"Colección cargada: '{name}' ({self._collections[name].count()} docs)")
            
        return self._collections[name]

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk], collection_name: Optional[str] = None) -> int:
        """
        Agrega chunks a la base vectorial.

        Returns:
            Cantidad de chunks agregados.
        """
        collection = self._get_collection(collection_name)

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
        collection.upsert(
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
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Busca los k fragmentos más cercanos a una consulta.

        Returns:
            Lista de dicts con keys: id, text, score, metadata
        """
        collection = self._get_collection(collection_name)

        query_embedding = self.embedder.encode_query(query_text)

        results = collection.query(
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

    def count(self, collection_name: Optional[str] = None) -> int:
        """Retorna la cantidad de documentos en la colección."""
        collection = self._get_collection(collection_name)
        return collection.count()

    def delete_collection(self) -> None:
        """Elimina la colección completa."""
        self._ensure_client()
        self._client.delete_collection(self.config.collection_name)
        self._collection = None
        logger.info(f"Colección '{self.config.collection_name}' eliminada")

    def reset(self, collection_name: Optional[str] = None) -> None:
        """Reset: elimina y recrea la colección."""
        self._ensure_client()
        name = collection_name or self.config.collection_domain
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        
        if name in self._collections:
            del self._collections[name]
            
        _ = self._get_collection(name)
        logger.info(f"VectorDB reseteado: {name}")
