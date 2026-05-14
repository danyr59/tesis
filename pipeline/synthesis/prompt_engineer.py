"""
Módulo 5 — Prompt Engineer: Generación de prompts técnicos para síntesis.
"""
import logging
from typing import List, Dict, Any, Optional

class PromptEngineer:
    """
    Diseña y optimiza prompts para que el LLM genere datos sintéticos realistas.
    """
    def __init__(self, domain: str = None):
        if domain is None:
            raise ValueError("ERROR: El PromptEngineer requiere un dominio explícito.")
        self.domain = domain
        self.logger = logging.getLogger("m5.prompt_engineer")

    def create_generation_prompt(self, seeds: List[Any], context: str = "") -> str:
        """Prompt genérico para síntesis."""
        return (
            f"Como experto en {self.domain}, tu tarea es generar nuevos registros de datos.\n"
            f"Contexto técnico: {context}\n"
            f"Usa estas muestras como referencia: {seeds}\n"
            f"Genera solo los valores numéricos separados por comas."
        )

    def autoregressive_prompt(
        self,
        seeds: List[List[float]],
        feature_names: List[str] = None,
        guidelines: str = "",
        num_continue: int = 5,
        profile: Any = None
    ) -> str:
        """Prompt específico para generación autoregresiva."""
        features_str = ", ".join(feature_names) if feature_names else "sensores"
        seeds_str = "\n".join([", ".join(map(str, s)) for s in seeds])
        
        prompt = (
            f"Eres un sistema experto en {self.domain}.\n"
            f"Tu objetivo es continuar la serie temporal de estos registros: [{features_str}].\n\n"
            f"REGLAS DE DOMINIO:\n{guidelines}\n\n"
            f"DATOS RECIENTES (Semillas):\n{seeds_str}\n\n"
            f"TAREA:\nGenera exactamente {num_continue} nuevos registros siguiendo la tendencia física.\n"
            f"Formato: CSV puro, sin encabezados, solo números."
        )
        return prompt

    def few_shot_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """Prompt para few-shot learning."""
        return f"Genera datos similares a estos ejemplos: {examples}"
