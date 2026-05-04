"""
Módulo 5 — SyntheticGenerator: Estrategias de generación de datos sintéticos.

Tras asimilar las lecciones del mentor, el aprendiz inicia la producción masiva.

Estrategias:
1. In-context Learning: usar la capacidad del modelo para aprender de ejemplos
   proporcionados en el prompt sin modificar sus pesos.
2. Fine-tuning (futuro): entrenar una versión más pequeña del modelo con datos
   reales específicos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.logger import get_logger
from .seed_analyzer import SeedAnalyzer, DatasetProfile, SeedSelection
from .prompt_engineer import PromptEngineer

logger = get_logger("m5.generator")

# Importar utilidades IOA de validación
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "IOA"))

try:
    from data.data_utils import parse_llm_json_response
except ImportError:
    # Fallback local si IOA no está disponible
    def parse_llm_json_response(response: str) -> List[Dict[str, Any]]:
        import re

        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        try:
            data = json.loads(response)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", response)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            return []


class SyntheticGenerator:
    """
    Genera datos sintéticos usando el LLM maestro con varias estrategias.
    """

    def __init__(
        self,
        domain: str = "predictive_maintenance",
        random_seed: int = 42,
    ):
        self.domain = domain
        self.analyzer = SeedAnalyzer(random_seed=random_seed)
        self.prompt_engineer = PromptEngineer(domain=domain)

        # Stats
        self.stats = {
            "total_generated": 0,
            "valid": 0,
            "invalid": 0,
        }

    # ── API Principal ────────────────────────────────────────────────────

    def generate_from_data(
        self,
        real_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        num_synthetic: int = 100,
        method: str = "statistical",
        llm_client: Any = None,
        guidelines: str = "",
    ) -> np.ndarray:
        """
        Genera datos sintéticos a partir de datos reales.

        Args:
            real_data: Dataset original (n_samples, n_features).
            feature_names: Nombres de las features.
            num_synthetic: Cantidad de muestras a generar.
            method: Método de generación:
                - "statistical": Muestreo estadístico (sin LLM).
                - "llm_direct": Generación directa vía LLM.
                - "llm_autoregressive": Continuación autoregresiva vía LLM.
                - "llm_fewshot": Few-shot con semillas vía LLM.
            llm_client: Cliente LLM (requerido para métodos llm_*).

        Returns:
            np.ndarray de shape (num_synthetic, n_features).
        """
        logger.info(
            f"Generando {num_synthetic} muestras sintéticas — método: {method}"
        )

        # 1. Analizar datos reales
        profile = self.analyzer.profile_dataset(real_data, feature_names)

        # 2. Generar según método
        if method == "statistical":
            synthetic = self._generate_statistical(real_data, profile, num_synthetic)
        elif method.startswith("llm_") and llm_client is not None:
            synthetic = self._generate_via_llm(
                real_data, profile, num_synthetic, method, llm_client, guidelines
            )
        else:
            logger.warning(
                f"Método '{method}' requiere llm_client. Fallback a estadístico."
            )
            synthetic = self._generate_statistical(real_data, profile, num_synthetic)

        # 3. Post-procesamiento
        synthetic = self._postprocess(synthetic, profile)

        self.stats["total_generated"] += len(synthetic)
        logger.info(f"Generadas {len(synthetic)} muestras sintéticas válidas")

        return synthetic

    # ── Generación Estadística (sin LLM) ─────────────────────────────────

    def _generate_statistical(
        self,
        real_data: np.ndarray,
        profile: DatasetProfile,
        num_samples: int,
    ) -> np.ndarray:
        """
        Generación estadística: muestreo de distribución multivariante.

        Preserva correlaciones usando descomposición de Cholesky.
        """
        if real_data.ndim == 1:
            real_data = real_data.reshape(-1, 1)

        n_features = real_data.shape[1]

        # Calcular media y covarianza
        mean = np.mean(real_data, axis=0)
        cov = np.cov(real_data.T)

        if n_features == 1:
            cov = np.array([[cov]]) if cov.ndim == 0 else cov.reshape(1, 1)

        # Asegurar que la covarianza sea definida positiva
        cov += np.eye(n_features) * 1e-6

        # Generar muestras
        try:
            synthetic = np.random.multivariate_normal(mean, cov, size=num_samples)
        except np.linalg.LinAlgError:
            # Fallback: generar feature por feature
            synthetic = np.column_stack([
                np.random.normal(mean[i], np.sqrt(max(cov[i, i], 1e-8)), num_samples)
                for i in range(n_features)
            ])

        # Agregar ruido para variabilidad adicional
        noise_scale = 0.02
        noise = np.random.normal(0, noise_scale, synthetic.shape) * np.std(real_data, axis=0)
        synthetic += noise

        logger.info(f"Generación estadística: {num_samples} muestras")
        return synthetic

    # ── Generación vía LLM ───────────────────────────────────────────────

    def _generate_via_llm(
        self,
        real_data: np.ndarray,
        profile: DatasetProfile,
        num_samples: int,
        method: str,
        llm_client: Any,
        guidelines: str = "",
    ) -> np.ndarray:
        """Genera datos usando el LLM con diferentes estrategias de prompting."""

        # Seleccionar semillas
        seeds = self.analyzer.select_seeds(real_data, num_seeds=5, method="diverse")
        logger.info(f"Semillas seleccionadas del CSV para guiar la generación:")
        for i, seed in enumerate(seeds.seeds):
            logger.info(f"  Semilla {i+1}: {seed['values']}")

        # Generar en lotes pequeños para evitar timeouts y asegurar calidad
        all_rows = []
        batch_size = 10
        max_attempts = (num_samples // batch_size) + 2
        attempts = 0
        
        logger.info(f"Iniciando generación iterativa: {num_samples} registros en lotes de {batch_size}")
        
        while len(all_rows) < num_samples and attempts < max_attempts:
            attempts += 1
            remaining = num_samples - len(all_rows)
            current_batch = min(remaining, batch_size)
            
            # Construir prompt según método
            if method == "llm_direct":
                prompt = self.prompt_engineer.direct_generation_prompt(
                    profile, num_samples=current_batch, guidelines=guidelines
                )
            elif method == "llm_autoregressive":
                # Usar los últimos datos generados como contexto si existen
                context_source = real_data if not all_rows else np.array(all_rows[-10:])
                existing = [dict(zip(profile.feature_names, row.tolist())) for row in context_source[:10]]
                prompt = self.prompt_engineer.autoregressive_prompt(
                    existing, num_continue=current_batch, profile=profile
                )
            else: # fewshot o default
                prompt = self.prompt_engineer.few_shot_prompt(
                    seeds, profile, num_generate=current_batch
                )

            # Llamar al LLM
            try:
                response = llm_client.generate(
                    prompt=prompt,
                    max_tokens=2048,
                    temperature=0.8, # Un poco más de creatividad
                )

                parsed = parse_llm_json_response(response)
                if parsed:
                    batch_rows = []
                    for record in parsed:
                        if isinstance(record, dict):
                            row = [record.get(name, 0.0) for name in profile.feature_names]
                            batch_rows.append([float(v) for v in row])
                    
                    all_rows.extend(batch_rows)
                    logger.info(f"  [Progreso] {len(all_rows)}/{num_samples} registros generados...")
                else:
                    logger.warning(f"  [Aviso] El lote {attempts} no devolvió JSON válido.")
                    logger.warning(f"  [RAW RESPONSE DEBUG]: {response[:300]}...")
                    # Reintentar con temperatura más baja si falla mucho
                    attempts += 0.5 # No penalizar tanto el intento si falló el formato

            except Exception as e:
                logger.error(f"Error en intento {attempts}: {e}")
                if len(all_rows) > 0: break
                else: return self._generate_statistical(real_data, profile, num_samples)

        # Convertir a numpy
        if not all_rows:
            return self._generate_statistical(real_data, profile, num_samples)
            
        synthetic = np.array(all_rows, dtype=np.float64)
        
        # Si faltan algunos, completar con estadístico
        if len(synthetic) < num_samples:
            logger.info(f"Completando {num_samples - len(synthetic)} registros con método estadístico")
            extra = self._generate_statistical(real_data, profile, num_samples - len(synthetic))
            synthetic = np.vstack([synthetic, extra])

        return synthetic[:num_samples]

    # ── Post-procesamiento ───────────────────────────────────────────────

    def _postprocess(
        self, synthetic: np.ndarray, profile: DatasetProfile
    ) -> np.ndarray:
        """Valida y limpia los datos sintéticos generados."""
        if synthetic.ndim == 1:
            synthetic = synthetic.reshape(-1, 1)

        n_features = synthetic.shape[1]

        # Clip a rangos válidos
        for i, name in enumerate(profile.feature_names[:n_features]):
            fmin = profile.mins.get(name)
            fmax = profile.maxs.get(name)
            if fmin is not None and fmax is not None:
                margin = (fmax - fmin) * 0.1  # 10% de margen
                synthetic[:, i] = np.clip(
                    synthetic[:, i], fmin - margin, fmax + margin
                )

        # Eliminar filas con NaN
        mask = ~np.any(np.isnan(synthetic), axis=1)
        synthetic = synthetic[mask]

        valid = len(synthetic)
        self.stats["valid"] += valid
        logger.debug(f"Post-procesamiento: {valid} muestras válidas")

        return synthetic

    # ── Utilidades ────────────────────────────────────────────────────────

    def save_synthetic(
        self,
        data: np.ndarray,
        filepath: str | Path,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Guarda datos sintéticos a CSV."""
        import csv

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if feature_names is None and self.analyzer.profile:
            feature_names = self.analyzer.profile.feature_names

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if feature_names:
                writer.writerow(feature_names)
            writer.writerows(data.tolist())

        logger.info(f"Datos sintéticos guardados en {filepath}")

    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()
