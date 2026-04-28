"""
Módulo 5 — SeedAnalyzer: Análisis de metadatos y muestreo de semillas.

Antes de interactuar con el LLM, es necesario definir qué se quiere replicar:
- Análisis de metadatos: distribuciones, tipos de datos, restricciones del dataset original.
- Selección de ejemplos (Few-shot): muestras reales ("semillas") como contexto.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger("m5.seed_analyzer")


@dataclass
class DatasetProfile:
    """Perfil estadístico de un dataset."""

    num_samples: int = 0
    num_features: int = 0
    feature_names: List[str] = field(default_factory=list)
    feature_types: Dict[str, str] = field(default_factory=dict)

    # Estadísticas por feature
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)
    mins: Dict[str, float] = field(default_factory=dict)
    maxs: Dict[str, float] = field(default_factory=dict)
    medians: Dict[str, float] = field(default_factory=dict)

    # Correlaciones
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Restricciones detectadas
    constraints: List[str] = field(default_factory=list)

    def to_prompt_description(self) -> str:
        """Genera una descripción textual del perfil para usar en prompts."""
        lines = [
            f"Dataset con {self.num_samples} muestras y {self.num_features} features.",
            "",
            "Features y estadísticas:",
        ]
        for feat in self.feature_names:
            ftype = self.feature_types.get(feat, "numérico")
            mean = self.means.get(feat, 0)
            std = self.stds.get(feat, 0)
            fmin = self.mins.get(feat, 0)
            fmax = self.maxs.get(feat, 0)
            lines.append(
                f"  - {feat} ({ftype}): media={mean:.4f}, std={std:.4f}, "
                f"rango=[{fmin:.4f}, {fmax:.4f}]"
            )

        if self.constraints:
            lines.append("")
            lines.append("Restricciones detectadas:")
            for c in self.constraints:
                lines.append(f"  - {c}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "num_features": self.num_features,
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "means": self.means,
            "stds": self.stds,
            "mins": self.mins,
            "maxs": self.maxs,
            "constraints": self.constraints,
        }


@dataclass
class SeedSelection:
    """Resultado de selección de semillas."""

    seeds: List[Dict[str, Any]] = field(default_factory=list)
    selection_method: str = "stratified"
    num_selected: int = 0


class SeedAnalyzer:
    """
    Analiza el dataset original e identifica las mejores semillas para
    generación de datos sintéticos.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self._profile: Optional[DatasetProfile] = None

    # ── Análisis de metadatos ─────────────────────────────────────────────

    def profile_dataset(self, data: np.ndarray, feature_names: Optional[List[str]] = None) -> DatasetProfile:
        """
        Analiza distribuciones, tipos de datos y restricciones.

        Args:
            data: Array numpy de shape (n_samples, n_features).
            feature_names: Nombres de las features (opcional).

        Returns:
            DatasetProfile con estadísticas completas.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        profile = DatasetProfile(
            num_samples=n_samples,
            num_features=n_features,
            feature_names=feature_names,
        )

        for i, name in enumerate(feature_names):
            col = data[:, i]
            valid = col[~np.isnan(col)]

            if len(valid) == 0:
                continue

            profile.feature_types[name] = self._detect_type(valid)
            profile.means[name] = float(np.mean(valid))
            profile.stds[name] = float(np.std(valid))
            profile.mins[name] = float(np.min(valid))
            profile.maxs[name] = float(np.max(valid))
            profile.medians[name] = float(np.median(valid))

        # Correlaciones entre features
        if n_features > 1:
            try:
                corr_matrix = np.corrcoef(data.T)
                for i, name_i in enumerate(feature_names):
                    profile.correlations[name_i] = {}
                    for j, name_j in enumerate(feature_names):
                        if i != j and not np.isnan(corr_matrix[i, j]):
                            profile.correlations[name_i][name_j] = float(
                                corr_matrix[i, j]
                            )
            except Exception:
                pass

        # Detectar restricciones
        profile.constraints = self._detect_constraints(data, feature_names, profile)

        self._profile = profile
        logger.info(
            f"Perfil del dataset: {n_samples} muestras, {n_features} features"
        )
        return profile

    def profile_from_csv(self, filepath: str | Path) -> DatasetProfile:
        """Analiza un dataset desde un archivo CSV."""
        import csv

        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in reader]

        # Convertir a numpy (solo columnas numéricas)
        numeric_cols: List[int] = []
        numeric_names: List[str] = []

        for i, name in enumerate(header):
            try:
                float(rows[0][i])
                numeric_cols.append(i)
                numeric_names.append(name)
            except (ValueError, IndexError):
                pass

        if not numeric_cols:
            logger.warning("No se encontraron columnas numéricas en el CSV")
            return DatasetProfile()

        data = np.array(
            [[float(row[c]) for c in numeric_cols] for row in rows if len(row) > max(numeric_cols)],
            dtype=np.float64,
        )

        return self.profile_dataset(data, numeric_names)

    # ── Selección de semillas (Few-shot) ─────────────────────────────────

    def select_seeds(
        self,
        data: np.ndarray,
        num_seeds: int = 5,
        method: str = "stratified",
    ) -> SeedSelection:
        """
        Selecciona muestras reales para usar como contexto few-shot.

        Args:
            data: Dataset completo.
            num_seeds: Cantidad de semillas a seleccionar.
            method: Método de selección:
                - "random": Aleatorio simple.
                - "stratified": Muestreo estratificado por distribución.
                - "diverse": Maximizar diversidad (k-medoids simplificado).

        Returns:
            SeedSelection con las semillas seleccionadas.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n = len(data)
        num_seeds = min(num_seeds, n)

        if method == "random":
            indices = np.random.choice(n, size=num_seeds, replace=False)
        elif method == "stratified":
            indices = self._stratified_sample(data, num_seeds)
        elif method == "diverse":
            indices = self._diverse_sample(data, num_seeds)
        else:
            indices = np.random.choice(n, size=num_seeds, replace=False)

        seeds = [
            {"index": int(idx), "values": data[idx].tolist()}
            for idx in indices
        ]

        selection = SeedSelection(
            seeds=seeds,
            selection_method=method,
            num_selected=len(seeds),
        )

        logger.info(
            f"Seleccionadas {len(seeds)} semillas con método '{method}'"
        )
        return selection

    # ── Internals ─────────────────────────────────────────────────────────

    def _detect_type(self, col: np.ndarray) -> str:
        """Detecta si una columna es entera, float, binaria, etc."""
        if np.all(np.isin(col, [0, 1])):
            return "binario"
        if np.all(col == col.astype(int)):
            unique = len(np.unique(col))
            if unique <= 10:
                return "categórico"
            return "entero"
        return "continuo"

    def _detect_constraints(
        self,
        data: np.ndarray,
        names: List[str],
        profile: DatasetProfile,
    ) -> List[str]:
        """Detecta restricciones del dataset (rangos, dependencias, etc.)."""
        constraints: List[str] = []

        for i, name in enumerate(names):
            col = data[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue

            # Restricción de no-negatividad
            if np.all(valid >= 0):
                constraints.append(f"{name} es siempre >= 0")

            # Restricción de rango [0, 1]
            if np.all(valid >= 0) and np.all(valid <= 1):
                constraints.append(f"{name} está en rango [0, 1]")

            # Restricción de monotonía parcial
            diffs = np.diff(valid)
            if len(diffs) > 10:
                pct_increase = np.sum(diffs > 0) / len(diffs)
                if pct_increase > 0.9:
                    constraints.append(
                        f"{name} es predominantemente creciente"
                    )
                elif pct_increase < 0.1:
                    constraints.append(
                        f"{name} es predominantemente decreciente"
                    )

        # Correlaciones fuertes
        for name_i, corrs in profile.correlations.items():
            for name_j, corr_val in corrs.items():
                if abs(corr_val) > 0.8:
                    direction = "positiva" if corr_val > 0 else "negativa"
                    constraints.append(
                        f"Correlación {direction} fuerte entre {name_i} y {name_j} "
                        f"(r={corr_val:.2f})"
                    )

        return list(set(constraints))  # Deduplicar

    def _stratified_sample(self, data: np.ndarray, n: int) -> np.ndarray:
        """Muestreo estratificado por cuantiles de la primera feature."""
        col = data[:, 0]
        quantiles = np.linspace(0, 100, n + 1)
        bins = np.percentile(col, quantiles)
        indices: List[int] = []

        for i in range(len(bins) - 1):
            mask = (col >= bins[i]) & (col <= bins[i + 1])
            candidates = np.where(mask)[0]
            if len(candidates) > 0:
                idx = np.random.choice(candidates)
                indices.append(idx)

        return np.array(indices[:n])

    def _diverse_sample(self, data: np.ndarray, n: int) -> np.ndarray:
        """Selección diversa: greedy farthest-first traversal."""
        from numpy.linalg import norm

        # Normalizar
        data_norm = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

        indices = [np.random.randint(len(data))]

        for _ in range(n - 1):
            selected = data_norm[indices]
            dists = np.min(
                [norm(data_norm - s, axis=1) for s in selected], axis=0
            )
            dists[indices] = -1
            indices.append(int(np.argmax(dists)))

        return np.array(indices)

    @property
    def profile(self) -> Optional[DatasetProfile]:
        return self._profile
