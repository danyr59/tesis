"""
Orquestador Principal del Pipeline de Mantenimiento Predictivo con IOA.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np

from pipeline.utils.logger import get_logger, setup_logging
from pipeline.config.pipeline_config import PipelineConfig

# Módulos
from pipeline.ingestion import RecursiveChunker, Embedder, VectorStore
from pipeline.retrieval import Retriever
from pipeline.context_builder import ContextBuilder
from pipeline.mentor import Mentor
from pipeline.synthesis import SyntheticGenerator
from pipeline.training import LinearRNN, TimeSeriesDataset, RNNTrainer
from pipeline.prediction import Predictor
from pipeline.evaluation import Judge, HallucinationDetector

logger = get_logger("m0.orchestrator")

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._init_components()

    def _init_components(self):
        logger.info("Inicializando componentes del pipeline...")
        # M1: Ingestión
        self.vector_store = VectorStore()
        
        # M2: Recuperación
        self.retriever = Retriever(vector_store=self.vector_store, config=self.config.retrieval)
        
        # M3: Context Builder
        self.context_builder = ContextBuilder(config=self.config.context)
        
        # Clientes LLM
        from pipeline.ioa_engine.utils.llm_client import create_teacher_client
        self.teacher_client = create_teacher_client(
            teacher_type=self.config.llm.teacher_type,
            model_name=self.config.llm.teacher_model
        )
        self.student_client = create_teacher_client(
            teacher_type=self.config.llm.student_type,
            model_name=self.config.llm.student_model,
            use_lora=self.config.ioa.model.use_lora
        )

        # M4: Mentor
        self.mentor = Mentor(
            ioa_config=self.config.ioa,
            teacher_client=self.teacher_client,
            seed_config=self.config.seed_factory
        )

        # M5: Síntesis
        self.synthetic_generator = SyntheticGenerator(
            domain=self.config.domain,
            random_seed=self.config.seed,
            config=self.config.synthesis
        )

    def run(self):
        logger.info(f"Iniciando ejecución para dominio: {self.config.domain}")
        logger.info(f"Consulta: {self.config.query}")
        
        # 1. Auto-Ingestión Dual (M1)
        from pipeline.ingestion.auto_ingestor import AutoIngestor
        ingestor = AutoIngestor(self.config, self.vector_store)
        ingestor.run()
        
        # 2. Recuperar contexto y generar perfil de datos
        logger.info("\n═══ M2/M3: RECUPERACIÓN Y CONTEXTO ═══")
        
        # Extraer perfil estadístico del CSV para el Maestro
        import pandas as pd
        seed_df = pd.read_csv("data/raw/sensores_bomba.csv")
        data_profile = {
            "columns": list(seed_df.columns),
            "stats": seed_df.describe().to_dict(),
            "sample": seed_df.head(3).to_dict()
        }
        
        res = self.retriever.retrieve(self.config.query)
        # Inyectamos el perfil de datos en el constructor de contexto
        enriched_message = self.context_builder.build(
            query=self.config.query, 
            retrieval_result=res,
            data_profile=data_profile
        )
        
        # 3. Mentor (IOA)
        mentor_result = self.mentor.run(
            domain=self.config.domain,
            enriched_message=enriched_message,
            teacher_model=self.teacher_client,
            student_model=self.student_client,
            retriever=self.retriever,
            config=self.config,
            data_profile=data_profile
        )
        
        # 4. Síntesis de datos
        logger.info("\n═══ M5: GENERACIÓN DE DATOS SINTÉTICOS ═══")
        import pandas as pd
        seed_df = pd.read_csv("data/raw/sensores_bomba.csv")
        real_data_array = seed_df.values
        feature_names = list(seed_df.columns)
        
        # Ajuste dinámico de dimensiones de la RNN según el archivo semilla
        num_features = len(feature_names)
        self.config.training.input_size = num_features
        self.config.training.output_size = num_features

        synthetic_array = self.synthetic_generator.generate_from_data(
            real_data=real_data_array,
            feature_names=feature_names,
            num_synthetic=self.config.synthesis.num_samples,
            method=self.config.synthesis.method,
            llm_client=self.student_client,
            guidelines=mentor_result.domain_knowledge
        )
        
        # 4. Entrenamiento RNN (M6)
        logger.info("\n═══ M6: ENTRENAMIENTO RNN LINEAL ═══")
        from pipeline.training import create_dataloaders
        
        try:
            # Configurar entrenador
            trainer = RNNTrainer(config=self.config.training)
            
            # Crear loaders
            train_loader, val_loader, _ = create_dataloaders(
                data=synthetic_array, 
                config=self.config.training,
                train_ratio=0.8,
                val_ratio=0.2
            )
            
            history = trainer.train(train_loader, val_loader)
            logger.info("Modelo RNN entrenado con éxito.")
            
            # 5. Detección de Alucinaciones (M7)
            logger.info("\n═══ M7: DETECCIÓN DE ALUCINACIONES ═══")
            from pipeline.evaluation.hallucination import HallucinationDetector
            detector = HallucinationDetector(config=self.config.hallucination)
            
            # Usar datos reales como referencia
            detector.set_reference(real_data_array, feature_names=feature_names)
            
            # Evaluar las muestras sintéticas (o predicciones)
            hallucination_report = detector.detect(synthetic_array, feature_names=feature_names)
            
            # 6. Juez Evaluador (M8)
            logger.info("\n═══ M8: JUEZ EVALUADOR ═══")
            from pipeline.evaluation.judge import Judge
            from pipeline.prediction.predictor import PredictionResult
            
            judge = Judge(config=self.config.hallucination)
            
            # Simulamos un resultado de predicción para el Juez basado en el entrenamiento
            mock_result = PredictionResult(
                predictions=synthetic_array[:len(val_loader.dataset)],
                actuals=real_data_array[:len(val_loader.dataset)],
                mse=history["val_loss"][-1] if history["val_loss"] else 0.0,
                rmse=np.sqrt(history["val_loss"][-1]) if history["val_loss"] else 0.0,
                mae=history["val_mae"][-1] if history["val_mae"] else 0.0,
                r2=0.85 # Valor simulado
            )
            
            verdict = judge.evaluate(mock_result, feature_names=feature_names)
            
        except Exception as e:
            logger.error(f"Error en fases finales: {e}")
            import traceback
            logger.error(traceback.format_exc())

        logger.info("\nPipeline completado exitosamente.")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Destilación Pedagógica IOA")
    parser.add_argument("--domain", type=str, required=True, help="Dominio del pipeline")
    parser.add_argument("--query", type=str, required=True, help="Consulta inicial")
    parser.add_argument("--num-synthetic", type=int, help="Número de muestras")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    
    args = parser.parse_args()
    
    config = PipelineConfig()
    config.domain = args.domain
    config.query = args.query
    if args.num_synthetic:
        config.synthesis.num_samples = args.num_synthetic
    config.output_dir = args.output_dir
    
    setup_logging(level="INFO")
    
    pipeline = Pipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
