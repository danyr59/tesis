"""
Módulo 8 — HallucinationDetector: Detección de alucinaciones numéricas.

Detecta predicciones "alucinadas" — resultados que parecen plausibles
pero no están fundamentados en los datos reales.

Estrategias de detección:
1. Validación de rango: valores fuera de la distribución esperada.
2. Consistencia temporal: cambios abruptos no justificados.
3. Correlaciones cruzadas: rupturas en relaciones entre variables.
4. Comparación estadística: tests contra distribución real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger
from ..config.pipeline_config import HallucinationConfig

logger = get_logger("m8.hallucination")


@dataclass
class Hallucination:
    """Una alucinación detectada."""

    timestamp_index: int
    feature_name: str = ""
    predicted_value: float = 0.0
    expected_range: Tuple[float, float] = (0.0, 0.0)
    detection_method: str = ""
    confidence: float = 0.0  # 0-1, qué tan seguros estamos de que es alucinación
    description: str = ""


@dataclass
class HallucinationReport:
    """Reporte completo de detección de alucinaciones."""

    total_predictions: int = 0
    hallucinations: List[Hallucination] = field(default_factory=list)
    hallucination_rate: float = 0.0

    # Desglose por método
    range_violations: int = 0
    temporal_violations: int = 0
    correlation_violations: int = 0
    statistical_violations: int = 0

    # Veredicto
    is_clean: bool = True

    def summary(self) -> str:
        status = "[LIMPIO]" if self.is_clean else "[ALUCINACIONES DETECTADAS]"
        lines = [
            f"=== REPORTE DE ALUCINACIONES: {status} ===",
            f"Total predicciones:    {self.total_predictions}",
            f"Alucinaciones:         {len(self.hallucinations)}",
            f"Tasa de alucinación:   {self.hallucination_rate:.2%}",
            "",
            "Desglose por tipo:",
            f"  Rango:               {self.range_violations}",
            f"  Temporal:            {self.temporal_violations}",
            f"  Correlación:         {self.correlation_violations}",
            f"  Estadístico:         {self.statistical_violations}",
        ]

        if self.hallucinations:
            lines.append(f"\nPrimeras 5 alucinaciones:")
            for h in self.hallucinations[:5]:
                lines.append(
                    f"  [{h.detection_method}] t={h.timestamp_index}: "
                    f"{h.feature_name}={h.predicted_value:.4f} "
                    f"(esperado: {h.expected_range})"
                )

        return "\n".join(lines)


class HallucinationDetector:
    """
    Detecta alucinaciones numéricas en las predicciones del modelo.
    """

    def __init__(self, config: Optional[HallucinationConfig] = None):
        self.config = config or HallucinationConfig()
        self._reference_stats: Optional[Dict[str, Any]] = None

    def set_reference(
        self,
        real_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Establece los datos reales como referencia para la detección.

        Args:
            real_data: Datos reales (n_samples, n_features).
            feature_names: Nombres de las features.
        """
        if real_data.ndim == 1:
            real_data = real_data.reshape(-1, 1)

        n_features = real_data.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        self._reference_stats = {
            "data": real_data,
            "names": feature_names,
            "means": np.nanmean(real_data, axis=0),
            "stds": np.nanstd(real_data, axis=0),
            "mins": np.nanmin(real_data, axis=0),
            "maxs": np.nanmax(real_data, axis=0),
            "medians": np.nanmedian(real_data, axis=0),
        }

        # Calcular correlaciones
        if n_features > 1:
            try:
                self._reference_stats["correlations"] = np.corrcoef(real_data.T)
            except Exception:
                self._reference_stats["correlations"] = np.eye(n_features)

        # Calcular diffs temporales típicos
        diffs = np.diff(real_data, axis=0)
        self._reference_stats["diff_means"] = np.nanmean(np.abs(diffs), axis=0)
        self._reference_stats["diff_stds"] = np.nanstd(np.abs(diffs), axis=0)

        logger.info(
            f"Referencia establecida: {real_data.shape[0]} muestras, "
            f"{n_features} features"
        )

    def detect(
        self,
        predictions: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> HallucinationReport:
        """
        Ejecuta todos los detectores de alucinaciones.

        Args:
            predictions: Predicciones del modelo (n_samples, n_features).
            feature_names: Nombres de las features.

        Returns:
            HallucinationReport con todas las alucinaciones detectadas.
        """
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        if self._reference_stats is None:
            logger.warning(
                "No se ha establecido referencia. "
                "Llame a set_reference() primero."
            )
            return HallucinationReport(total_predictions=len(predictions))

        names = feature_names or self._reference_stats["names"]
        all_hallucinations: List[Hallucination] = []

        # 1. Validación de rango (z-score)
        range_h = self._detect_range_violations(predictions, names)
        all_hallucinations.extend(range_h)

        # 2. Consistencia temporal
        temporal_h = self._detect_temporal_violations(predictions, names)
        all_hallucinations.extend(temporal_h)

        # 3. Correlaciones cruzadas
        corr_h = self._detect_correlation_violations(predictions, names)
        all_hallucinations.extend(corr_h)

        # 4. Tests estadísticos
        stat_h = self._detect_statistical_violations(predictions, names)
        all_hallucinations.extend(stat_h)

        # Compilar reporte
        total = predictions.shape[0] * predictions.shape[1]
        report = HallucinationReport(
            total_predictions=total,
            hallucinations=all_hallucinations,
            hallucination_rate=len(all_hallucinations) / max(total, 1),
            range_violations=len(range_h),
            temporal_violations=len(temporal_h),
            correlation_violations=len(corr_h),
            statistical_violations=len(stat_h),
            is_clean=len(all_hallucinations) == 0,
        )

        logger.info(report.summary())
        return report

    # ── Detectores individuales ──────────────────────────────────────────

    def _detect_range_violations(
        self, preds: np.ndarray, names: List[str]
    ) -> List[Hallucination]:
        """
        Detecta valores fuera de rango usando z-score.

        Un valor es sospechoso si su z-score respecto a la distribución
        real excede el umbral configurado.
        """
        hallucinations: List[Hallucination] = []
        stats = self._reference_stats
        #for j in range(min(preds.shape[1], len(names))):
        for j in range(min(preds.shape[1], len(names), len(stats["means"]))):
            mean = stats["means"][j]
            std = max(stats["stds"][j], 1e-8)
            fmin = stats["mins"][j]
            fmax = stats["maxs"][j]

            for i in range(preds.shape[0]):
                val = preds[i, j]
                z = abs((val - mean) / std)

                if z > self.config.z_score_threshold:
                    hallucinations.append(
                        Hallucination(
                            timestamp_index=i,
                            feature_name=names[j],
                            predicted_value=float(val),
                            expected_range=(float(fmin), float(fmax)),
                            detection_method="range_zscore",
                            confidence=min(1.0, z / (self.config.z_score_threshold * 2)),
                            description=(
                                f"z-score={z:.2f} excede umbral "
                                f"{self.config.z_score_threshold}"
                            ),
                        )
                    )

        return hallucinations

    def _detect_temporal_violations(
        self, preds: np.ndarray, names: List[str]
    ) -> List[Hallucination]:
        """
        Detecta cambios abruptos no justificados entre pasos consecutivos.
        """
        hallucinations: List[Hallucination] = []
        stats = self._reference_stats

        if preds.shape[0] < 2:
            return []

        diffs = np.abs(np.diff(preds, axis=0))
        max_change_pct = self.config.max_step_change_pct / 100.0

        for j in range(min(preds.shape[1], len(names), len(stats["means"]))):
            ref_mean = max(abs(stats["means"][j]), 1e-8)

            for i in range(len(diffs)):
                change_pct = diffs[i, j] / ref_mean * 100

                if change_pct > self.config.max_step_change_pct:
                    hallucinations.append(
                        Hallucination(
                            timestamp_index=i + 1,
                            feature_name=names[j],
                            predicted_value=float(preds[i + 1, j]),
                            expected_range=(
                                float(preds[i, j] - ref_mean * max_change_pct),
                                float(preds[i, j] + ref_mean * max_change_pct),
                            ),
                            detection_method="temporal_jump",
                            confidence=min(1.0, change_pct / (self.config.max_step_change_pct * 2)),
                            description=(
                                f"Cambio abrupto de {change_pct:.1f}% entre "
                                f"t={i} y t={i+1}"
                            ),
                        )
                    )

        return hallucinations

    def _detect_correlation_violations(
        self, preds: np.ndarray, names: List[str]
    ) -> List[Hallucination]:
        """
        Detecta rupturas en correlaciones esperadas entre variables.

        Si temperatura y vibración están históricamente correlacionadas
        (r=0.8), pero en las predicciones la correlación cae a 0.1,
        indica una alucinación de la estructura relacional.
        """
        hallucinations: List[Hallucination] = []

        if preds.shape[1] < 2 or preds.shape[0] < 5:
            return []

        ref_corr = self._reference_stats.get("correlations")
        if ref_corr is None:
            return []

        try:
            pred_corr = np.corrcoef(preds.T)
        except Exception:
            return []

        n = min(preds.shape[1], len(names), ref_corr.shape[0])

        for i in range(n):
            for j in range(i + 1, n):
                ref_r = ref_corr[i, j]
                pred_r = pred_corr[i, j]

                # Solo verificar correlaciones fuertes en referencia
                if abs(ref_r) < self.config.min_cross_correlation:
                    continue

                diff = abs(ref_r - pred_r)
                if diff > 0.4:
                    hallucinations.append(
                        Hallucination(
                            timestamp_index=-1,  # Aplica a toda la serie
                            feature_name=f"{names[i]}<->{names[j]}",
                            predicted_value=float(pred_r),
                            expected_range=(
                                float(ref_r - 0.2),
                                float(ref_r + 0.2),
                            ),
                            detection_method="correlation_break",
                            confidence=min(1.0, diff / 0.8),
                            description=(
                                f"Correlación {names[i]}-{names[j]}: "
                                f"esperada={ref_r:.2f}, observada={pred_r:.2f}"
                            ),
                        )
                    )

        return hallucinations

    def _detect_statistical_violations(
        self, preds: np.ndarray, names: List[str]
    ) -> List[Hallucination]:
        """
        Tests estadísticos para comparar distribuciones.

        Usa Kolmogorov-Smirnov test para comparar la distribución
        de predicciones vs datos reales.
        """
        hallucinations: List[Hallucination] = []
        ref_data = self._reference_stats["data"]
        alpha = 1.0 - self.config.confidence_level

        for j in range(min(preds.shape[1], len(names), ref_data.shape[1])):
            ref_col = ref_data[:, j]
            pred_col = preds[:, j]

            # KS test manual (sin scipy)
            ks_stat = self._ks_statistic(ref_col, pred_col)

            # Umbral aproximado de KS
            n1, n2 = len(ref_col), len(pred_col)
            c_alpha = np.sqrt(-0.5 * np.log(alpha / 2))
            ks_critical = c_alpha * np.sqrt((n1 + n2) / (n1 * n2))

            if ks_stat > ks_critical:
                hallucinations.append(
                    Hallucination(
                        timestamp_index=-1,
                        feature_name=names[j],
                        predicted_value=float(ks_stat),
                        expected_range=(0.0, float(ks_critical)),
                        detection_method="ks_test",
                        confidence=min(1.0, ks_stat / (ks_critical * 2)),
                        description=(
                            f"KS test rechaza H0: distribución de {names[j]} "
                            f"difiere (stat={ks_stat:.4f} > critical={ks_critical:.4f})"
                        ),
                    )
                )

        return hallucinations

    @staticmethod
    def _ks_statistic(sample1: np.ndarray, sample2: np.ndarray) -> float:
        """Calcula el estadístico Kolmogorov-Smirnov manualmente."""
        combined = np.sort(np.concatenate([sample1, sample2]))
        n1, n2 = len(sample1), len(sample2)

        cdf1 = np.searchsorted(np.sort(sample1), combined, side="right") / n1
        cdf2 = np.searchsorted(np.sort(sample2), combined, side="right") / n2

        return float(np.max(np.abs(cdf1 - cdf2)))
