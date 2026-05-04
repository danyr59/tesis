"""
Módulo 5 — PromptEngineer: Diseño de prompts para generación sintética.

Dos enfoques principales:
1. Generación Directa: pedir al modelo que cree registros nuevos basados en descripción.
2. Auto-regresiva con Contexto: proveer lista parcial y pedir que la continúe.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from .seed_analyzer import DatasetProfile, SeedSelection

logger = get_logger("m5.prompt_engineer")


class PromptEngineer:
    """Diseña prompts especializados para generación de datos sintéticos."""

    def __init__(self, domain: str = "predictive_maintenance"):
        self.domain = domain

    # ── Generación Directa ────────────────────────────────────────────────

    def direct_generation_prompt(
        self,
        profile: DatasetProfile,
        num_samples: int = 100,
        constraints: Optional[List[str]] = None,
        guidelines: str = "",
    ) -> str:
        """
        Prompt para generar registros nuevos a partir de una descripción.

        Ejemplo: "Genera 100 registros de sensores de vibración con
        las siguientes distribuciones..."
        """
        profile_desc = profile.to_prompt_description()
        extra_constraints = ""
        if constraints:
            extra_constraints = "\n".join(f"  - {c}" for c in constraints)
            extra_constraints = f"\n\nRestricciones adicionales:\n{extra_constraints}"

        guidelines_section = ""
        if guidelines:
            guidelines_section = f"\n\nKNOW-HOW DESTILADO POR EL MAESTRO (Sigue estas reglas lógicas):\n{guidelines}"

        return f"""Eres un generador experto de datos sintéticos para {self.domain}.

TAREA: Genera exactamente {num_samples} registros sintéticos que sigan las distribuciones
y restricciones del dataset original descrito abajo.

PERFIL DEL DATASET ORIGINAL:
{profile_desc}
{extra_constraints}
{guidelines_section}

INSTRUCCIONES:
1. Cada registro debe respetar las distribuciones estadísticas (medias, desviaciones, rangos).
2. Mantener las correlaciones entre variables.
3. Introducir variabilidad realista (no copiar valores exactos).
4. Los datos deben ser físicamente plausibles para {self.domain}.
REGLA DE FORMATO CRÍTICA:
- RESPONDE ÚNICAMENTE CON EL JSON.
- NO INCLUYAS TEXTO EXPLICATIVO, INTRODUCCIONES NI CONCLUSIONES.
- EL RESULTADO DEBE SER UN ARRAY JSON VÁLIDO.

FORMATO DE SALIDA:
```json
[
  {{"feature_1": valor, "feature_2": valor, ...}},
  ...
]
```

Genera los {num_samples} registros siguiendo estrictamente los principios de excelencia y el formato JSON:"""

    # ── Auto-regresiva con Contexto ──────────────────────────────────────

    def autoregressive_prompt(
        self,
        existing_data: List[Dict[str, Any]],
        num_continue: int = 50,
        profile: Optional[DatasetProfile] = None,
    ) -> str:
        """
        Prompt que provee una lista parcial y pide continuación coherente.

        Se proporciona una lista parcial de datos y se pide al modelo
        que la continúe, manteniendo la coherencia lógica.
        """
        # Mostrar hasta 10 ejemplos como contexto
        context_data = existing_data[:10]
        context_json = "\n".join(
            f"  {i+1}. {row}" for i, row in enumerate(context_data)
        )

        profile_hint = ""
        if profile:
            profile_hint = f"\n\nREFERENCIA ESTADÍSTICA:\n{profile.to_prompt_description()}"

        return f"""Eres un generador de datos para {self.domain}.

CONTEXTO: Tienes los siguientes datos reales de sensores/equipos:
{context_json}
{profile_hint}

TAREA: Continúa esta secuencia generando {num_continue} registros adicionales que:
1. Mantengan la coherencia temporal y lógica con los datos existentes.
2. Sigan los mismos patrones y distribuciones.
3. Introduzcan variabilidad natural (tendencias, estacionalidad, ruido).
4. Sean físicamente plausibles.

FORMATO: JSON array con los nuevos registros (sin incluir los existentes).

Genera los {num_continue} registros nuevos:"""

    # ── Few-shot con semillas ────────────────────────────────────────────

    def few_shot_prompt(
        self,
        seeds: SeedSelection,
        profile: DatasetProfile,
        num_generate: int = 50,
        task_description: Optional[str] = None,
    ) -> str:
        """
        Prompt con ejemplos few-shot seleccionados como semillas.
        """
        seed_examples = "\n".join(
            f"  Ejemplo {i+1}: {s['values']}"
            for i, s in enumerate(seeds.seeds)
        )

        task = task_description or (
            f"dados de sensores para {self.domain} "
            f"(vibración, temperatura, presión, etc.)"
        )

        return f"""Eres un generador experto de datos sintéticos.

TAREA: Genera {num_generate} registros sintéticos de {task}.

EJEMPLOS REALES (semillas):
{seed_examples}

PERFIL ESTADÍSTICO:
{profile.to_prompt_description()}

INSTRUCCIONES:
1. Usa los ejemplos como referencia de formato y distribución.
2. NO copies los ejemplos; genera variaciones realistas.
3. Mantén las correlaciones entre features.
4. Cada registro debe ser temporal y físicamente plausible.
5. Formato de salida: JSON array.

Genera los {num_generate} registros:"""

    # ── Prompt para series temporales ────────────────────────────────────

    def time_series_prompt(
        self,
        profile: DatasetProfile,
        sequence_length: int = 100,
        num_series: int = 5,
        scenario: str = "operación normal",
        guidelines: str = "",
    ) -> str:
        """
        Prompt especializado para generar series temporales sintéticas
        de mantenimiento predictivo.
        """
        return f"""Eres un generador de series temporales para {self.domain}.

ESCENARIO: {scenario}

TAREA: Genera {num_series} series temporales, cada una con {sequence_length} pasos
de tiempo, que representen lecturas de sensores en equipos industriales.

PERFIL DE REFERENCIA:
{profile.to_prompt_description()}

KNOW-HOW DESTILADO POR EL MAESTRO (Reglas lógicas a seguir):
{guidelines if guidelines else "Mantener coherencia física del dominio."}

ESCENARIOS A INCLUIR:
1. Operación normal estable (valores dentro de rangos típicos)
2. Degradación gradual (drift lento hacia valores anómalos)
3. Evento de falla incipiente (cambio abrupto seguido de recuperación parcial)
4. Mantenimiento correctivo (reset a valores nominales)
5. Condiciones de estrés (valores altos pero dentro de tolerancia)

FORMATO: JSON con estructura:
```json
{{
  "series": [
    {{
      "scenario": "nombre",
      "data": [[t1_f1, t1_f2, ...], [t2_f1, t2_f2, ...], ...]
    }}
  ]
}}
```

Genera las {num_series} series:"""

    # ── Prompt de validación ─────────────────────────────────────────────

    def validation_prompt(
        self,
        synthetic_data: List[Dict[str, Any]],
        profile: DatasetProfile,
    ) -> str:
        """
        Prompt para que el LLM valide datos sintéticos generados.
        """
        sample = synthetic_data[:5]

        return f"""Eres un validador experto de datos sintéticos para {self.domain}.

DATOS SINTÉTICOS GENERADOS (muestra de {len(sample)}):
{sample}

PERFIL DEL DATASET ORIGINAL:
{profile.to_prompt_description()}

TAREA: Evalúa la calidad de los datos sintéticos:
1. ¿Las distribuciones son similares al original?
2. ¿Se mantienen las correlaciones entre variables?
3. ¿Los valores son físicamente plausibles?
4. ¿Hay valores fuera de rango o imposibles?
5. ¿Hay patrones repetitivos sospechosos?

Responde con un JSON:
{{
  "quality_score": 0.0-1.0,
  "issues": ["lista de problemas detectados"],
  "recommendations": ["sugerencias de mejora"],
  "pass": true/false
}}"""
