"""
Módulo 1 — Chunker: Segmentación de documentos.

Divide documentos extensos en fragmentos (chunks) más pequeños para
mantener el contexto relevante y ajustarse a la ventana de contexto del LLM.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..utils.logger import get_logger
from ..config.pipeline_config import ChunkingConfig

logger = get_logger("m1.chunker")


@dataclass
class Chunk:
    """Un fragmento individual de un documento."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    source: str = ""
    page: Optional[int] = None
    position: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            import hashlib

            self.chunk_id = hashlib.md5(
                f"{self.source}:{self.position}:{self.text[:50]}".encode()
            ).hexdigest()[:12]


class RecursiveChunker:
    """
    Divide documentos extensos en fragmentos usando separadores jerárquicos.

    Soporta: texto plano, CSV, JSON.
    Para PDF, se espera texto pre-extraído o se usa pdfplumber si está instalado.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    # ── API principal ──────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        source: str = "unknown",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Divide un texto plano en chunks con solapamiento."""
        if not text.strip():
            return []

        chunks: List[Chunk] = []
        segments = self._split_recursive(
            text,
            self.config.chunk_size,
            self.config.separators,
        )

        for i, seg in enumerate(segments):
            meta = {"source": source, "chunk_index": i}
            if extra_metadata:
                meta.update(extra_metadata)
            chunks.append(
                Chunk(text=seg, metadata=meta, source=source, position=i)
            )

        # Agregar solapamiento
        if self.config.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        logger.info(f"Texto '{source}' → {len(chunks)} chunks")
        return chunks

    def chunk_file(self, filepath: str | Path) -> List[Chunk]:
        """Detecta el tipo de archivo y lo segmenta."""
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Archivo no encontrado: {filepath}")
            return []

        suffix = filepath.suffix.lower()
        source = filepath.name

        if suffix == ".csv":
            return self._chunk_csv(filepath, source)
        elif suffix == ".json" or suffix == ".jsonl":
            return self._chunk_json(filepath, source)
        elif suffix == ".pdf":
            return self._chunk_pdf(filepath, source)
        else:
            # Texto plano (.txt, .md, etc.)
            text = filepath.read_text(encoding="utf-8", errors="replace")
            return self.chunk_text(text, source=source)

    def chunk_directory(self, dir_path: str | Path) -> List[Chunk]:
        """Procesa todos los archivos soportados en un directorio."""
        dir_path = Path(dir_path)
        all_chunks: List[Chunk] = []
        supported = {".txt", ".md", ".csv", ".json", ".jsonl", ".pdf"}

        for fp in sorted(dir_path.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in supported:
                all_chunks.extend(self.chunk_file(fp))

        logger.info(
            f"Directorio '{dir_path}' → {len(all_chunks)} chunks totales"
        )
        return all_chunks

    # ── Internals ──────────────────────────────────────────────────────────

    def _split_recursive(
        self, text: str, max_size: int, separators: List[str]
    ) -> List[str]:
        """Split recursivo usando separadores jerárquicos."""
        if len(text) <= max_size:
            return [text.strip()] if text.strip() else []

        # Buscar el mejor separador
        for sep in separators:
            parts = text.split(sep)
            if len(parts) > 1:
                segments: List[str] = []
                current = ""
                for part in parts:
                    candidate = (current + sep + part) if current else part
                    if len(candidate) <= max_size:
                        current = candidate
                    else:
                        if current.strip():
                            segments.append(current.strip())
                        current = part
                if current.strip():
                    segments.append(current.strip())
                return segments

        # Fallback: corte por caracteres
        return [
            text[i : i + max_size].strip()
            for i in range(0, len(text), max_size)
            if text[i : i + max_size].strip()
        ]

    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Agrega solapamiento entre chunks consecutivos."""
        overlap = self.config.chunk_overlap
        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            tail = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
            chunks[i].text = tail + " " + chunks[i].text
        return chunks

    # ── Formatos especiales ───────────────────────────────────────────────

    def _chunk_csv(self, filepath: Path, source: str) -> List[Chunk]:
        """Cada fila o grupo de filas del CSV es un chunk."""
        text = filepath.read_text(encoding="utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return []

        header = rows[0]
        chunks: List[Chunk] = []
        batch: List[str] = []

        for i, row in enumerate(rows[1:], start=1):
            line = ", ".join(f"{h}: {v}" for h, v in zip(header, row))
            batch.append(line)

            if len("\n".join(batch)) >= self.config.chunk_size:
                chunks.append(
                    Chunk(
                        text="\n".join(batch),
                        source=source,
                        position=len(chunks),
                        metadata={"format": "csv", "rows": f"{i-len(batch)+1}-{i}"},
                    )
                )
                batch = []

        if batch:
            chunks.append(
                Chunk(
                    text="\n".join(batch),
                    source=source,
                    position=len(chunks),
                    metadata={"format": "csv"},
                )
            )

        logger.info(f"CSV '{source}' → {len(chunks)} chunks")
        return chunks

    def _chunk_json(self, filepath: Path, source: str) -> List[Chunk]:
        """Convierte cada objeto/registro JSON en un chunk."""
        text = filepath.read_text(encoding="utf-8", errors="replace")

        if filepath.suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            records = data if isinstance(data, list) else [data]

        chunks: List[Chunk] = []
        for i, rec in enumerate(records):
            chunks.append(
                Chunk(
                    text=json.dumps(rec, ensure_ascii=False, indent=1),
                    source=source,
                    position=i,
                    metadata={"format": "json", "record_index": i},
                )
            )
        return chunks

    def _chunk_pdf(self, filepath: Path, source: str) -> List[Chunk]:
        """Extrae texto de PDF y lo segmenta."""
        try:
            import pdfplumber  # type: ignore

            pages_text: List[str] = []
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    pages_text.append(t)

            all_chunks: List[Chunk] = []
            for page_num, page_text in enumerate(pages_text, start=1):
                page_chunks = self.chunk_text(
                    page_text,
                    source=source,
                    extra_metadata={"page": page_num, "format": "pdf"},
                )
                for c in page_chunks:
                    c.page = page_num
                all_chunks.extend(page_chunks)

            return all_chunks

        except ImportError:
            logger.warning(
                "pdfplumber no instalado. Instalar con: pip install pdfplumber"
            )
            return []
