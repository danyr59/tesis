"""
Módulo 8 — Judge: Contrastación con fuente de verdad.

El "Estudiante como Juez" contrasta las predicciones del modelo
entrenado (M7) con la fuente de verdad original (M1).

Funciones:
1. Verificar consistencia física de las predicciones.
2. Detectar inconsistencias lógicas entre variables correlacionadas.
3. Evaluar la confianza global del pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..config.pipeline_config import HallucinationConfig
from ..utils.logger import get_logger
from ..prediction.predictor import PredictionResult

logger = get_logger("m8.judge")


@dataclass
class JudgmentIssue:
    """Problema detectado durante la evaluación."""

    severity: str = "warning"  # info | warning | critical
    category: str = ""
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgmentResult:
    """Resultado completo de la evaluación del juez."""

    # Veredicto
    approved: bool = True
    confidence_score: float = 1.0

    # Métricas de predicción (del M7)
    prediction_metrics: Dict[str, float] = field(default_factory=dict)

    # Problemas detectados
    issues: List[JudgmentIssue] = field(default_factory=list)

    # Estadísticas
    consistency_score: float = 1.0
    physical_plausibility: float = 1.0
    hallucination_score: float = 0.0  # 0 = sin alucinaciones

    def summary(self) -> str:
        status = "✅ APROBADO" if self.approved else "❌ RECHAZADO"
        lines = [
            f"=== VEREDICTO DEL JUEZ: {status} ===",
            f"Confianza global:      {self.confidence_score:.2%}",
            f"Consistencia:          {self.consistency_score:.2%}",
            f"Plausibilidad física:  {self.physical_plausibility:.2%}",
            f"Score alucinaciones:   {self.hallucination_score:.2%}",
        ]

        if self.issues:
            lines.append(f"\nProblemas detectados ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(f"  [{issue.severity.upper()}] {issue.description}")

        return "\n".join(lines)


class Judge:
    """
    Evalúa la calidad de las predicciones contrastándolas con
    la fuente de verdad y las restricciones del dominio.
    """

    def __init__(
        self,
        config: Optional[HallucinationConfig] = None,
        domain_constraints: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            config: Configuración de umbrales de alucinación y juicio.
            domain_constraints: Restricciones físicas del dominio.
        """
        self.config = config or HallucinationConfig()
        self.domain_constraints = domain_constraints or {}

    def evaluate(
        self,
        prediction_result: PredictionResult,
        ground_truth: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> JudgmentResult:
        """
        Evaluación completa de las predicciones.

        Args:
            prediction_result: Resultado del predictor (M7).
            ground_truth: Datos reales de referencia adicionales (opcional).
            feature_names: Nombres de las features.

        Returns:
            JudgmentResult con el veredicto.
        """
        logger.info("=" * 60)
        logger.info("EVALUACIÓN — Estudiante como Juez")
        logger.info("=" * 60)

        issues: List[JudgmentIssue] = []
        scores: List[float] = []

        preds = prediction_result.predictions
        actuals = prediction_result.actuals

        # 1. Evaluar métricas de predicción
        metric_score = self._evaluate_metrics(prediction_result, issues)
        scores.append(metric_score)

        # 2. Verificar plausibilidad física
        physical_score = self._check_physical_plausibility(
            preds, feature_names, issues
        )
        scores.append(physical_score)

        # 3. Verificar consistencia entre variables
        consistency_score = self._check_consistency(
            preds, actuals, feature_names, issues
        )
        scores.append(consistency_score)

        # 4. Verificar distribuciones
        distribution_score = self._check_distribution(
            preds, actuals, issues
        )
        scores.append(distribution_score)

        # Calcular confianza global
        confidence = float(np.mean(scores))
        approved = confidence >= self.config.approval_threshold

        # Calcular hallucination score (inverso de calidad)
        hallucination_score = max(0.0, 1.0 - confidence)

        result = JudgmentResult(
            approved=approved,
            confidence_score=confidence,
            prediction_metrics=prediction_result.to_dict(),
            issues=issues,
            consistency_score=consistency_score,
            physical_plausibility=physical_score,
            hallucination_score=hallucination_score,
        )

        logger.info(result.summary())
        return result

    # ── Evaluadores individuales ─────────────────────────────────────────

    def _evaluate_metrics(
        self, pred_result: PredictionResult, issues: List[JudgmentIssue]
    ) -> float:
        """Evalúa si las métricas de predicción son aceptables."""
        score = 1.0

        # R² muy bajo indica mal modelo
        if pred_result.r2 < self.config.min_r2:
            issues.append(
                JudgmentIssue(
                    severity="critical",
                    category="metrics",
                    description=f"R² muy bajo: {pred_result.r2:.4f} (umbral: {self.config.min_r2})",
                )
            )
            score *= 0.5

        # MAPE alto
        if pred_result.mape > self.config.max_mape:
            issues.append(
                JudgmentIssue(
                    severity="warning",
                    category="metrics",
                    description=f"MAPE alto: {pred_result.mape:.1f}% (umbral: {self.config.max_mape}%)",
                )
            )
            score *= 0.7

        # RMSE relativo
        if pred_result.actuals.size > 0:
            range_val = np.ptp(pred_result.actuals)
            if range_val > 0:
                nrmse = pred_result.rmse / range_val
                if nrmse > self.config.max_nrmse:
                    issues.append(
                        JudgmentIssue(
                            severity="warning",
                            category="metrics",
                            description=f"NRMSE alto: {nrmse:.4f} (umbral: {self.config.max_nrmse})",
                        )
                    )
                    score *= 0.8

        return score

    def _check_physical_plausibility(
        self,
        predictions: np.ndarray,
        feature_names: Optional[List[str]],
        issues: List[JudgmentIssue],
    ) -> float:
        """Verifica que los valores predichos sean físicamente plausibles."""
        if not self.domain_constraints or feature_names is None:
            return 1.0

        score = 1.0
        preds = predictions.reshape(-1, predictions.shape[-1]) if predictions.ndim > 1 else predictions.reshape(-1, 1)

        for i, name in enumerate(feature_names):
            if i >= preds.shape[1]:
                break

            constraint = self.domain_constraints.get(name, {})
            col = preds[:, i]

            fmin = constraint.get("min")
            fmax = constraint.get("max")

            if fmin is not None:
                violations = np.sum(col < fmin)
                if violations > 0:
                    pct = violations / len(col) * 100
                    issues.append(
                        JudgmentIssue(
                            severity="warning" if pct < 5 else "critical",
                            category="physical",
                            description=(
                                f"{name}: {violations} valores ({pct:.1f}%) por debajo del "
                                f"mínimo físico ({fmin})"
                            ),
                        )
                    )
                    score *= max(0.5, 1 - pct / 100)

            if fmax is not None:
                violations = np.sum(col > fmax)
                if violations > 0:
                    pct = violations / len(col) * 100
                    issues.append(
                        JudgmentIssue(
                            severity="warning" if pct < 5 else "critical",
                            category="physical",
                            description=(
                                f"{name}: {violations} valores ({pct:.1f}%) por encima del "
                                f"máximo físico ({fmax})"
                            ),
                        )
                    )
                    score *= max(0.5, 1 - pct / 100)

        return score

    def _check_consistency(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        feature_names: Optional[List[str]],
        issues: List[JudgmentIssue],
    ) -> float:
        """Verifica consistencia entre variables predichas."""
        if predictions.ndim == 1 or predictions.shape[-1] < 2:
            return 1.0

        score = 1.0
        preds = predictions.reshape(-1, predictions.shape[-1])
        acts = actuals.reshape(-1, actuals.shape[-1])

        # Comparar correlaciones
        try:
            pred_corr = np.corrcoef(preds.T)
            act_corr = np.corrcoef(acts.T)

            corr_diff = np.abs(pred_corr - act_corr)
            max_diff = np.max(corr_diff)

            if max_diff > self.config.max_corr_diff:
                issues.append(
                    JudgmentIssue(
                        severity="warning",
                        category="consistency",
                        description=(
                            f"Estructura de correlaciones alterada "
                            f"(diferencia máx: {max_diff:.2f}, umbral: {self.config.max_corr_diff})"
                        ),
                    )
                )
                score *= 0.7
        except Exception:
            pass

        return score

    def _check_distribution(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        issues: List[JudgmentIssue],
    ) -> float:
        """Compara distribuciones de predicciones vs actuals."""
        preds = predictions.flatten()
        acts = actuals.flatten()

        score = 1.0

        # Comparar media
        mean_diff_pct = abs(np.mean(preds) - np.mean(acts)) / (abs(np.mean(acts)) + 1e-8) * 100
        if mean_diff_pct > self.config.max_mean_diff_pct:
            issues.append(
                JudgmentIssue(
                    severity="warning",
                    category="distribution",
                    description=f"Media de predicciones difiere {mean_diff_pct:.1f}% del real (umbral: {self.config.max_mean_diff_pct}%)",
                )
            )
            score *= 0.8

        # Comparar varianza
        std_ratio = np.std(preds) / (np.std(acts) + 1e-8)
        if std_ratio < self.config.min_std_ratio or std_ratio > self.config.max_std_ratio:
            issues.append(
                JudgmentIssue(
                    severity="warning",
                    category="distribution",
                    description=(
                        f"Varianza de predicciones anormal "
                        f"(ratio: {std_ratio:.2f}, esperado ~1.0)"
                    ),
                )
            )
            score *= 0.7

        return score
