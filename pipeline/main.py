"""
Módulo 0 — Estudiante Orquestador: Punto de entrada del pipeline.

Recibe la consulta en lenguaje natural, la procesa/analiza, y orquesta
el flujo completo a través de los 9 módulos:

    M0 (Orquestador)
     │
     ├─ M1 Ingestión       ← documentos del dominio
     ├─ M2 Recuperación    ← buscar fragmentos relevantes
     ├─ M3 Contexto        ← construir mensaje enriquecido
     ├─ M4 Mentor (IOA)    ← diagnóstico + organización + adaptación
     ├─ M5 Síntesis        ← generar datos sintéticos
     ├─ M6 Entrenamiento   ← entrenar RNN lineal
     ├─ M7 Predicción      ← generar y evaluar predicciones
     └─ M8 Evaluación      ← juez + detección de alucinaciones
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Asegurar que la raíz del proyecto está en el PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from pipeline.utils.logger import get_logger, setup_logging
from pipeline.config.pipeline_config import PipelineConfig, get_default_pipeline_config

# Módulos
from pipeline.ingestion import RecursiveChunker, Embedder, VectorStore
from pipeline.retrieval import Retriever
from pipeline.context_builder import ContextBuilder
from pipeline.mentor import Mentor
from pipeline.synthesis import SeedAnalyzer, PromptEngineer, SyntheticGenerator
from pipeline.training import LinearRNN, TimeSeriesDataset, RNNTrainer
from pipeline.prediction import Predictor
from pipeline.evaluation import Judge, HallucinationDetector

logger = get_logger("m0.orchestrator")


@dataclass
class PipelineResult:
    """Resultado completo de una ejecución del pipeline."""

    query: str = ""
    timestamp: float = 0.0
    elapsed_seconds: float = 0.0

    # Resultados por módulo
    m1_chunks_ingested: int = 0
    m2_fragments_retrieved: int = 0
    m3_context_built: bool = False
    m4_mentor_completed: bool = False
    m5_synthetic_samples: int = 0
    m6_training_completed: bool = False
    m6_best_val_loss: float = float("inf")
    m7_prediction_metrics: Dict[str, float] = field(default_factory=dict)
    m8_approved: bool = False
    m8_confidence: float = 0.0
    m8_hallucination_rate: float = 0.0

    # Errores
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║         PIPELINE DE MANTENIMIENTO PREDICTIVO            ║",
            "║         con Destilación Pedagógica (IOA)                ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
            f"  Consulta: {self.query[:80]}",
            f"  Tiempo total: {self.elapsed_seconds:.1f}s",
            "",
            "  ── Resultados por módulo ──",
            f"  M1 Ingestión:     {self.m1_chunks_ingested} chunks",
            f"  M2 Recuperación:  {self.m2_fragments_retrieved} fragmentos",
            f"  M3 Contexto:      {'✅' if self.m3_context_built else '❌'}",
            f"  M4 Mentor (IOA):  {'✅' if self.m4_mentor_completed else '❌'}",
            f"  M5 Síntesis:      {self.m5_synthetic_samples} muestras",
            f"  M6 Entrenamiento: {'✅' if self.m6_training_completed else '❌'} "
            f"(val_loss={self.m6_best_val_loss:.6f})",
            f"  M7 Predicción:    R²={self.m7_prediction_metrics.get('r2', 0):.4f}",
            f"  M8 Evaluación:    {'✅ APROBADO' if self.m8_approved else '❌ RECHAZADO'} "
            f"(conf={self.m8_confidence:.2%}, halluc={self.m8_hallucination_rate:.2%})",
        ]

        if self.errors:
            lines.append("")
            lines.append(f"  ⚠ Errores ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")

        return "\n".join(lines)


class TesisPipeline:
    """
    Orquestador central del pipeline de 9 módulos.

    Uso:
        pipeline = TesisPipeline()
        pipeline.ingest("./data/raw")
        result = pipeline.run("Predecir fallas en bomba hidráulica")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_default_pipeline_config()

        # Configurar logging
        setup_logging(
            level=self.config.log_level,
            log_file=self.config.log_file,
        )

        logger.info("Inicializando Pipeline de Mantenimiento Predictivo")

        # ── Instanciar módulos ───────────────────────────────────────────
        # M1: Ingestión
        self.chunker = RecursiveChunker(self.config.chunking)
        self.embedder = Embedder(self.config.embedding)
        self.vector_store = VectorStore(
            self.config.vector_store, self.embedder
        )

        # M2: Recuperación
        self.retriever = Retriever(self.vector_store, self.config.retrieval)

        # M3: Contexto
        self.context_builder = ContextBuilder(self.config.context_builder)

        # ── Configurar Clientes LLM ──────────────────────────────────────
        from IOA.utils.llm_client import create_teacher_client
        
        self.teacher_client = create_teacher_client(
            teacher_type=self.config.llm.teacher_type,
            api_key=self.config.llm.teacher_api_key,
            api_base=self.config.llm.teacher_api_base,
            model_name=self.config.llm.teacher_model,
            timeout=self.config.llm.teacher_timeout,
        )
        
        self.student_client = create_teacher_client(
            teacher_type=self.config.llm.student_type,
            api_key=self.config.llm.student_api_key,
            api_base=self.config.llm.student_api_base,
            model_name=self.config.llm.student_model,
            timeout=self.config.llm.student_timeout,
        )

        # M4: Mentor (IOA) - Usa el Maestro
        self.mentor = Mentor(
            ioa_config=self.config.ioa, 
            teacher_client=self.teacher_client
        )

        # M5: Síntesis - Lo hace el Estudiante
        self.synthetic_generator = SyntheticGenerator(
            domain="predictive_maintenance",
            random_seed=self.config.seed,
        )

        # M6: Entrenamiento (se crea al entrenar)
        self._trainer: Optional[RNNTrainer] = None

        # M7: Predicción (se crea al predecir)
        self._predictor: Optional[Predictor] = None

        # M8: Evaluación
        self.judge = Judge()
        self.hallucination_detector = HallucinationDetector(
            self.config.hallucination
        )

        # Estado
        self._ingested = False
        self._trained = False

        logger.info("Pipeline inicializado correctamente")

    # ══════════════════════════════════════════════════════════════════════
    # M1: INGESTIÓN
    # ══════════════════════════════════════════════════════════════════════

    def ingest(
        self,
        source: str | Path,
        reset: bool = False,
    ) -> int:
        """
        Ingesta documentos del dominio en la base vectorial.

        Args:
            source: Ruta a un archivo o directorio con documentos.
            reset: Si True, elimina datos previos antes de ingestar.

        Returns:
            Cantidad de chunks ingestados.
        """
        logger.info("═" * 60)
        logger.info("M1: INGESTIÓN — Preparación del conocimiento del dominio")
        logger.info("═" * 60)

        source = Path(source)

        if reset:
            self.vector_store.reset()

        # Segmentar
        if source.is_dir():
            chunks = self.chunker.chunk_directory(source)
        elif source.is_file():
            chunks = self.chunker.chunk_file(source)
        else:
            logger.error(f"Fuente no encontrada: {source}")
            return 0

        if not chunks:
            logger.warning("No se generaron chunks de los documentos.")
            return 0

        # Almacenar en VectorDB
        count = self.vector_store.add_chunks(chunks)
        self._ingested = True

        logger.info(f"M1 completo: {count} chunks ingestados")
        return count

    # ══════════════════════════════════════════════════════════════════════
    # M0: PIPELINE COMPLETO
    # ══════════════════════════════════════════════════════════════════════

    def run(
        self,
        query: str,
        real_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        num_synthetic: int = 500,
        domain_constraints: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Ejecuta el pipeline completo M0 → M8.

        Args:
            query: Consulta en lenguaje natural.
            real_data: Datos reales del dominio (n_timestamps, n_features).
                       Si None, se generan datos placeholder.
            feature_names: Nombres de las features.
            num_synthetic: Cantidad de muestras sintéticas a generar.
            domain_constraints: Restricciones físicas del dominio.

        Returns:
            PipelineResult con resultados de cada módulo.
        """
        start_time = time.time()
        result = PipelineResult(query=query, timestamp=start_time)

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  M0: ESTUDIANTE ORQUESTADOR — Pipeline Completo          ║")
        logger.info("╚" + "═" * 58 + "╝")
        logger.info(f"Consulta: {query}")

        try:
            # ── M2: Recuperación ─────────────────────────────────────────
            logger.info("\n═══ M2: RECUPERACIÓN ═══")
            logger.info(f"  [INPUT] Consulta del usuario: '{query}'")
            if self._ingested:
                retrieval_result = self.retriever.retrieve(query)
                result.m2_fragments_retrieved = retrieval_result.total_found
                logger.info(f"  [OUTPUT] {retrieval_result.total_found} fragmentos de conocimiento recuperados.")
            else:
                logger.info("Sin datos ingestados — saltando recuperación")
                from .retrieval.retriever import RetrievalResult
                retrieval_result = RetrievalResult(query=query)

            # ── M3: Contexto ─────────────────────────────────────────────
            logger.info("\n═══ M3: PREPARACIÓN DEL ENVÍO ═══")
            logger.info(f"  [INPUT] {len(retrieval_result.fragments) if hasattr(retrieval_result, 'fragments') else 0} fragmentos + Consulta")
            enriched_message = self.context_builder.build(
                query=query,
                retrieval_result=retrieval_result,
            )
            result.m3_context_built = True
            logger.info(f"  [OUTPUT] Mensaje enriquecido construido.")

            # ── M4: Mentor (IOA) ─────────────────────────────────────────
            logger.info("\n═══ M4: KNOW-HOW DEL MAESTRO ═══")
            logger.info(f"  [INPUT] Contexto técnico del dominio")
            try:
                mentor_result = self.mentor.run(
                    domain="predictive_maintenance",
                    enriched_message=enriched_message,
                )
                result.m4_mentor_completed = True
                logger.info(f"  [OUTPUT] Guías de síntesis generadas.")
            except Exception as e:
                logger.error(f"  [ERROR] M4 falló: {e}")
                result.errors.append(f"M4: {e}")

            # ── M5: Síntesis ─────────────────────────────────────────────
            logger.info("\n═══ M5: GENERACIÓN DE DATOS SINTÉTICOS ═══")
            
            # Obtener guías del Mentor (M4) para instruir al Estudiante
            synthesis_guidelines = ""
            if result.m4_mentor_completed and "mentor_result" in locals():
                synthesis_guidelines = mentor_result.synthesis_guidelines
                if synthesis_guidelines:
                    logger.info("Inyectando directrices del Maestro en el generador Estudiante")

            logger.info(f"  [INPUT] Semillas del CSV + Guías del Maestro")
            if real_data is not None:
                synthetic_data = self.synthetic_generator.generate_from_data(
                    real_data=real_data,
                    feature_names=feature_names,
                    num_synthetic=num_synthetic,
                    method="llm_direct",
                    llm_client=self.student_client,
                    guidelines=synthesis_guidelines
                )
            else:
                logger.info("Sin datos reales — generando datos placeholder")
                synthetic_data = self._generate_placeholder_data(num_synthetic)
                if feature_names is None:
                    feature_names = [
                        "temperatura",
                        "vibracion",
                        "presion",
                        "rpm",
                    ]

            result.m5_synthetic_samples = len(synthetic_data)
            logger.info(f"  [OUTPUT] {len(synthetic_data)} nuevas muestras generadas por el Estudiante.")

            # Guardar dataset sintético para revisión del usuario
            synthetic_path = Path(self.config.output_dir) / "synthetic_data.csv"
            self.synthetic_generator.save_synthetic(
                synthetic_data, 
                synthetic_path,
                feature_names=feature_names
            )
            logger.info(f"  [OUTPUT] Dataset sintético guardado en {synthetic_path}")

            # Combinar datos reales + sintéticos para entrenar
            if real_data is not None:
                training_data = np.vstack([real_data, synthetic_data])
            else:
                training_data = synthetic_data

            # ── M6: Entrenamiento ────────────────────────────────────────
            logger.info("\n═══ M6: ENTRENAMIENTO RNN LINEAL ═══")
            logger.info(f"  [INPUT] Dataset combinado: {training_data.shape[0]} filas, {training_data.shape[1]} variables.")
            try:
                rnn_config = self.config.rnn
                rnn_config.input_size = training_data.shape[1]
                rnn_config.output_size = 1

                self._trainer = RNNTrainer(config=rnn_config)
                history = self._trainer.train_from_data(
                    training_data,
                    target_col=0,
                    checkpoint_dir=str(Path(self.config.output_dir) / "checkpoints"),
                )

                result.m6_training_completed = True
                result.m6_best_val_loss = self._trainer.best_val_loss
                logger.info(f"  [OUTPUT] Modelo entrenado. Mejor pérdida de validación: {result.m6_best_val_loss:.4f}")

            except Exception as e:
                logger.error(f"  [ERROR] M6 falló: {e}")
                result.errors.append(f"M6: {e}")

            # ── M7: Predicción ───────────────────────────────────────────
            logger.info("\n═══ M7: PRUEBA Y PREDICCIÓN ═══")
            if self._trainer and result.m6_training_completed:
                logger.info(f"  [INPUT] Datos de prueba (últimos registros)")
                try:
                    self._predictor = Predictor(trainer=self._trainer)
                    pred_result = self._predictor.predict_from_data(
                        training_data, target_col=0
                    )
                    result.m7_prediction_metrics = pred_result.to_dict()

                    # Guardar gráfico
                    self._predictor.plot_predictions(
                        pred_result,
                        output_path=str(Path(self.config.output_dir) / "predictions.png"),
                    )
                    logger.info(f"  [OUTPUT] Gráfico de predicción generado en {self.config.output_dir}/predictions.png")

                except Exception as e:
                    logger.error(f"  [ERROR] M7 falló: {e}")
                    result.errors.append(f"M7: {e}")

            # ── M8: Evaluación ───────────────────────────────────────────
            logger.info("\n═══ M8: EVALUACIÓN — ESTUDIANTE COMO JUEZ ═══")
            if self._predictor and 'pred_result' in locals():
                logger.info(f"  [INPUT] Predicciones vs Restricciones del Dominio")
                try:
                    if domain_constraints:
                        self.judge.domain_constraints = domain_constraints
                    judgment = self.judge.evaluate(
                        pred_result, feature_names=feature_names
                    )
                    result.m8_approved = judgment.approved
                    result.m8_confidence = judgment.confidence_score

                    # Hallucination detector
                    self.hallucination_detector.set_reference(
                        training_data, feature_names
                    )
                    halluc_report = self.hallucination_detector.detect(
                        pred_result.predictions, feature_names
                    )
                    result.m8_hallucination_rate = halluc_report.hallucination_rate
                    logger.info(f"  [OUTPUT] Tasa de alucinación: {result.m8_hallucination_rate:.2%}")
                    logger.info(f"  [OUTPUT] Confianza del Juez: {result.m8_confidence:.2f}")

                except Exception as e:
                    logger.error(f"  [ERROR] M8 falló: {e}")
                    result.errors.append(f"M8: {e}")

        except Exception as e:
            logger.error(f"Error fatal en pipeline: {e}")
            result.errors.append(f"Fatal: {e}")

        result.elapsed_seconds = time.time() - start_time
        logger.info("\n" + result.summary())

        return result

    # ══════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ══════════════════════════════════════════════════════════════════════

    def _generate_placeholder_data(self, num_samples: int) -> np.ndarray:
        """
        Genera datos placeholder de mantenimiento predictivo.

        4 features:
        - Temperatura (60-120 °C con drift gradual)
        - Vibración (2-15 mm/s correlacionada con temperatura)
        - Presión (3-8 bar con ruido)
        - RPM (1400-1600 con variaciones)
        """
        np.random.seed(self.config.seed)
        t = np.linspace(0, 10, num_samples)

        # Temperatura: tendencia + estacionalidad + ruido
        temp = 80 + 10 * np.sin(2 * np.pi * t / 3) + 5 * t / 10 + np.random.normal(0, 2, num_samples)

        # Vibración: correlacionada con temperatura
        vibration = 5 + 0.1 * (temp - 80) + np.random.normal(0, 0.5, num_samples)
        vibration = np.clip(vibration, 2, 15)

        # Presión: cuasi-estable con ruido
        pressure = 5.5 + 0.5 * np.sin(2 * np.pi * t / 5) + np.random.normal(0, 0.3, num_samples)
        pressure = np.clip(pressure, 3, 8)

        # RPM: variaciones operativas
        rpm = 1500 + 30 * np.sin(2 * np.pi * t / 2) + np.random.normal(0, 10, num_samples)
        rpm = np.clip(rpm, 1400, 1600)

        data = np.column_stack([temp, vibration, pressure, rpm])
        logger.info(f"Datos placeholder generados: {data.shape}")
        return data

    def save_model(self, filepath: str = "outputs/model_final.pt") -> None:
        """Guarda el modelo entrenado."""
        if self._trainer:
            self._trainer.save_model(filepath)
        else:
            logger.warning("No hay modelo entrenado para guardar")

    def load_model(self, filepath: str) -> None:
        """Carga un modelo previamente entrenado."""
        self._trainer = RNNTrainer.load_model(filepath)
        self._predictor = Predictor(trainer=self._trainer)
        self._trained = True
        logger.info(f"Modelo cargado desde {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Punto de entrada por CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline de Mantenimiento Predictivo con Destilación Pedagógica"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Predecir fallas en bomba hidráulica basado en datos de sensores",
        help="Consulta en lenguaje natural",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw",
        help="Directorio con documentos del dominio",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=500,
        help="Muestras sintéticas a generar",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Configurar
    config = get_default_pipeline_config()
    config.output_dir = args.output_dir
    config.log_level = args.log_level

    # Crear pipeline
    pipeline = TesisPipeline(config=config)
    # Ingestar documentos si existen (Para M1)
    data_dir = Path(args.data_dir)
    if data_dir.exists() and any(data_dir.iterdir()):
        pipeline.ingest(data_dir)

    # Cargar CSV si existe para usarlo como real_data en M5
    import pandas as pd
    import numpy as np
    
    real_data = None
    feature_names = None
    csv_file = data_dir / "sensores_bomba.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        feature_names = df.columns.tolist()
        real_data = df.values

    # Ejecutar pipeline
    result = pipeline.run(
        query=args.query,
        real_data=real_data,
        feature_names=feature_names,
        num_synthetic=args.num_synthetic,
    )

    print("\n" + result.summary())

    # Guardar modelo
    pipeline.save_model(str(Path(args.output_dir) / "model_final.pt"))


if __name__ == "__main__":
    main()
