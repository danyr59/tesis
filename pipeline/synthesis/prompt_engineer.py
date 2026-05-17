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
        """Prompt ultra-estricto para modelos de 8B."""
        features_str = ", ".join(feature_names) if feature_names else "sensores"
        seeds_str = "\n".join([", ".join(map(str, s)) for s in seeds])
        
        prompt = (
            f"SYSTEM: Solo genera datos numéricos. NO hables. NO expliques.\n"
            f"DOMINIO: {self.domain}\n"
            f"COLUMNAS: [{features_str}]\n"
            f"REGLAS FÍSICAS: {guidelines}\n\n"
            f"DATOS DE ENTRADA (Últimas lecturas):\n{seeds_str}\n\n"
            f"TAREA:\nGenera {num_continue} líneas adicionales de datos siguiendo la tendencia.\n"
            f"REQUERIMIENTO TÉCNICO:\n"
            f"- Formato: JSON list de listas.\n"
            f"- Ejemplo de salida: [[25.4, 3.2, 10.1, 1800.5], [26.1, 3.5, 10.2, 1810.0]]\n"
            f"- NO incluyas texto antes ni después del JSON.\n"
            f"- NO incluyas encabezados.\n"
            f"RESPUESTA:"
        )
        return prompt

    def few_shot_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """Prompt para few-shot learning."""
        return f"Genera datos similares a estos ejemplos: {examples}"
