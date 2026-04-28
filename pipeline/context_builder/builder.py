"""
Módulo 3 — ContextBuilder: Preparación del envío.

Paso final de la cadena RAG: integra la información recuperada
construyendo un "mensaje enriquecido" que incluye:
- Consulta original del usuario
- Fragmentos de información recuperados como contexto explícito
- Datos históricos / sesión anterior
- Metadata de los documentos fuente
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from ..utils.logger import get_logger
from ..config.pipeline_config import ContextBuilderConfig
from ..retrieval.retriever import RetrievalResult

logger = get_logger("m3.builder")


@dataclass
class EnrichedMessage:
    """Mensaje enriquecido listo para enviar al LLM maestro."""

    system_prompt: str = ""
    user_prompt: str = ""
    context_fragments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_full_prompt(self) -> str:
        """Retorna el prompt completo formateado para el LLM."""
        parts: List[str] = []

        if self.system_prompt:
            parts.append(f"[SISTEMA]\n{self.system_prompt}\n")

        if self.context_fragments:
            ctx = "\n---\n".join(self.context_fragments)
            parts.append(
                f"[CONTEXTO DEL DOMINIO]\n"
                f"Los siguientes fragmentos provienen de la base de conocimiento:\n\n"
                f"{ctx}\n"
            )

        parts.append(f"[CONSULTA DEL USUARIO]\n{self.user_prompt}")

        return "\n\n".join(parts)


class ContextBuilder:
    """
    Construye el mensaje enriquecido combinando la consulta original,
    los fragmentos recuperados, y el contexto histórico.
    """

    def __init__(self, config: Optional[ContextBuilderConfig] = None):
        self.config = config or ContextBuilderConfig()
        self._history: List[Dict[str, str]] = []

    def build(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        domain_context: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> EnrichedMessage:
        """
        Construye el mensaje enriquecido.

        Args:
            query: Consulta original del usuario.
            retrieval_result: Resultado de la recuperación (M2).
            domain_context: Contexto adicional del dominio (opcional).
            extra_data: Datos extra que se quieren incluir.

        Returns:
            EnrichedMessage listo para enviar al maestro.
        """
        logger.info(
            f"Construyendo mensaje enriquecido con "
            f"{retrieval_result.total_found} fragmentos"
        )

        # 1. System prompt base orientado al dominio
        system_prompt = self._build_system_prompt(domain_context)

        # 2. Fragmentos de contexto
        context_fragments = self._format_fragments(
            retrieval_result.fragments
        )

        # 3. Incluir historial si corresponde
        if self.config.include_history and self._history:
            history_text = self._format_history()
            context_fragments.insert(0, history_text)

        # 4. Truncar al máximo de tokens permitido
        context_fragments = self._truncate_context(context_fragments)

        # 5. Metadata
        metadata: Dict[str, Any] = {
            "num_fragments": len(context_fragments),
            "sources": list(
                {
                    f.get("metadata", {}).get("source", "unknown")
                    for f in retrieval_result.fragments
                }
            ),
        }
        if extra_data:
            metadata["extra"] = extra_data

        # 6. Guardar en historial
        self._add_to_history(query)

        msg = EnrichedMessage(
            system_prompt=system_prompt,
            user_prompt=query,
            context_fragments=context_fragments,
            metadata=metadata,
        )

        logger.info(
            f"Mensaje construido — {len(context_fragments)} fragmentos de contexto"
        )
        return msg

    # ── Internals ──────────────────────────────────────────────────────────

    def _build_system_prompt(self, domain_context: Optional[str]) -> str:
        base = (
            "Eres un sistema experto en mantenimiento predictivo. "
            "Tu tarea es analizar datos del dominio, diagnosticar vacíos "
            "de conocimiento, y construir un plan pedagógico para transferir "
            "el know-how al modelo estudiante. "
            "Usa la información del contexto para fundamentar tus respuestas."
        )
        if domain_context:
            base += f"\n\nContexto adicional del dominio:\n{domain_context}"
        return base

    def _format_fragments(
        self, fragments: List[Dict[str, Any]]
    ) -> List[str]:
        """Formatea los fragmentos recuperados con su metadata."""
        formatted: List[str] = []
        for i, frag in enumerate(fragments, 1):
            source = frag.get("metadata", {}).get("source", "desconocido")
            score = frag.get("score", 0.0)
            text = frag.get("text", "")

            header = f"[Fragmento {i} | Fuente: {source} | Relevancia: {score:.2f}]"
            formatted.append(f"{header}\n{text}")

        return formatted

    def _format_history(self) -> str:
        """Formatea el historial de consultas anteriores."""
        window = self._history[-self.config.history_window :]
        lines = [
            f"- Turno {i+1}: {h['query']}" for i, h in enumerate(window)
        ]
        return "[HISTORIAL DE CONSULTAS]\n" + "\n".join(lines)

    def _truncate_context(self, fragments: List[str]) -> List[str]:
        """Trunca fragmentos para no exceder el máximo de tokens."""
        max_chars = self.config.max_context_tokens * 4  # ~4 chars/token aprox
        total = 0
        result: List[str] = []

        for frag in fragments:
            if total + len(frag) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    result.append(frag[:remaining] + "...")
                break
            result.append(frag)
            total += len(frag)

        return result

    def _add_to_history(self, query: str) -> None:
        self._history.append({"query": query})

    def clear_history(self) -> None:
        """Limpia el historial de consultas."""
        self._history.clear()
