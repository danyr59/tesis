"""
Módulo 4 — Mentor: Generar el Know-how en el Maestro.

El mentor despliega la metodología pedagógica IOA (Identifier-Organizer-Adapter)
para estructurar el conocimiento que será transferido.
"""

from __future__ import annotations

import sys
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..context_builder.builder import EnrichedMessage
from .seed_factory import SeedFactory

logger = get_logger("m4.mentor")

# ── Importar módulos IOA ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ..ioa_engine.config.config import (
    IOAConfig,
    IdentifierConfig,
    OrganizerConfig,
    AdapterConfig,
)
from ..ioa_engine.modules.identifier import KnowledgeIdentifier
from ..ioa_engine.modules.organizer import KnowledgeOrganizer, Curriculum, CurriculumStage
from ..ioa_engine.modules.adapter import KnowledgeAdapter
from ..ioa_engine.utils.llm_client import LLMClient, create_teacher_client, LocalLLMClient
from ..ioa_engine.utils.graph_utils import KnowledgeDependencyGraph, KnowledgeNode
from ..ioa_engine.data.seed_data import SeedDataset, SeedDataItem, load_seed_data


@dataclass
class MentorResult:
    """Resultado completo del proceso pedagógico del Mentor."""
    target_modules: List[str] = field(default_factory=list)
    performance_gaps: Dict[str, float] = field(default_factory=dict)
    dependency_graph: Optional[KnowledgeDependencyGraph] = None
    curriculum: Optional[Curriculum] = None
    num_stages: int = 0
    domain_knowledge: str = ""
    synthesis_guidelines: str = ""
    domain_context_used: bool = False


class Mentor:
    """
    Wrapper sobre el framework IOA que inyecta contexto de dominio.
    """

    def __init__(
        self,
        ioa_config: Optional[IOAConfig] = None,
        teacher_client: Optional[LLMClient] = None,
        seed_dataset: Optional[SeedDataset] = None,
        seed_config: Optional[Any] = None,
        max_taxonomy_modules: int = 5
    ):
        self.ioa_config = ioa_config or IOAConfig()
        self.teacher_client = teacher_client
        self.seed_dataset = seed_dataset
        self.seed_config = seed_config
        self.max_taxonomy_modules = max_taxonomy_modules
        self._identifier = None
        self._organizer = None
        self._adapter = None
        self._dependency_graph = None
        self._curriculum = None

    def setup_teacher(self, teacher_type: str = "local", model_name: Optional[str] = "./models/teacher", **kwargs) -> None:
        try:
            self.teacher_client = create_teacher_client(teacher_type=teacher_type, model_name=model_name, **kwargs)
            logger.info(f"Maestro configurado: {model_name}")
        except Exception as e:
            logger.warning(f"Error maestro: {e}")

    def setup_seed_data(self, data_dir: str = "./data/seed") -> None:
        self.seed_dataset = load_seed_data(data_dir=data_dir)

    def _ensure_identifier(self):
        if self._identifier is None:
            if self.seed_dataset is None: self.setup_seed_data()
            self._identifier = KnowledgeIdentifier(config=self.ioa_config.identifier, teacher_client=self.teacher_client, seed_dataset=self.seed_dataset)
        return self._identifier

    def _ensure_adapter(self):
        if self._adapter is None:
            self._adapter = KnowledgeAdapter(config=self.ioa_config.adapter, teacher_client=self.teacher_client)
        return self._adapter

    def diagnose(self, domain: Optional[str] = None, enriched_message=None, teacher_model=None, student_model=None):
        if domain is None:
            raise ValueError("El parámetro 'domain' es obligatorio para el diagnóstico del Mentor.")
        
        logger.info(f"DIAGNÓSTICO — Evaluando comprensión del aprendiz en dominio: {domain}")
        identifier = self._ensure_identifier()
        target_modules, dep_graph = identifier.identify(domain=domain, teacher_model=teacher_model, student_model=student_model, context=enriched_message.to_full_prompt() if enriched_message else None)
        self._dependency_graph = dep_graph
        return target_modules, dep_graph

    def organize(self, target_modules, domain: Optional[str] = None):
        if domain is None:
            raise ValueError("El parámetro 'domain' es obligatorio para la organización del currículum.")
            
        logger.info(f"ORGANIZACIÓN — Diseñando plan de estudios para: {domain}")
        self._organizer = KnowledgeOrganizer(config=self.ioa_config.organizer, dependency_graph=self._dependency_graph)
        self._curriculum = self._organizer.organize(target_modules=target_modules, domain=domain)
        return self._curriculum

    def adapt_knowledge(self, curriculum, domain: Optional[str] = None, enriched_message=None, methodology_context=""):
        # El dominio aquí se usa para dar contexto al prompt de adaptación
        adapted_data = {}
        for stage in curriculum.stages:
            adaptation_prompt = f"Genera lección técnica para {stage.stage_id} basada en el dominio {domain}. Contexto: {methodology_context[:100]}..."
            try:
                response = self.teacher_client.generate(adaptation_prompt)
                adapted_data[stage.stage_id] = {"know_how": response}
            except Exception as e:
                logger.warning(f"Error adaptación: {e}")
        return adapted_data

    def run(self, domain: Optional[str] = None, enriched_message=None, teacher_model=None, student_model=None, retriever=None, config=None) -> MentorResult:
        if domain is None:
            raise ValueError("ERROR CRÍTICO: Debe especificar un dominio (ej. 'stock_prices', 'predictive_maintenance') para ejecutar el Mentor.")
            
        logger.info(f"MENTOR RUN — Iniciando Pipeline Pedagógico en dominio: {domain}")
        
        # 1. Seeds factory
        methodology_context = ""
        domain_context = ""
        if retriever and config:
            res = retriever.retrieve("Reglas de síntesis", collection_name=config.vector_store.collection_methodology)
            methodology_context = "\n".join([f["text"] for f in res.fragments])
            if enriched_message:
                domain_context = "\n".join([f for f in enriched_message.context_fragments])

        if config and self.teacher_client:
            factory = SeedFactory(seed_dir=config.seed_factory.seed_dir)
            factory.ensure_seeds(config.seed_factory.methodology_size, config.seed_factory.domain_size, self.teacher_client, methodology_context, domain_context)
            self._inject_persistent_seeds(factory)

        # 2. Diagnose & Organize
        target_modules, dep_graph = self.diagnose(domain, enriched_message, teacher_model, student_model)
        curriculum = self.organize(target_modules, domain)

        # 3. Distillation (Loop de Maestría)
        if student_model and self.teacher_client:
            logger.info("INICIANDO DESTILACIÓN PARAMÉTRICA...")
            from ..ioa_engine.training.trainer import IOATrainer, train_curriculum
            
            # Desempaquetar el modelo de PyTorch del wrapper LLMClient para evitar hardware errors
            actual_model = None
            actual_tokenizer = None
            
            if hasattr(student_model, "client"):
                inner = student_model.client
                if hasattr(inner, "model"):
                    actual_model = inner.model
                    actual_tokenizer = getattr(inner, "tokenizer", None)
            
            if actual_model is None and hasattr(student_model, "model"):
                actual_model = student_model.model
                actual_tokenizer = getattr(student_model, "tokenizer", None)
            
            if actual_model is None:
                actual_model = student_model

            trainer = IOATrainer(
                config=self.ioa_config.training,
                model=actual_model,
                tokenizer=actual_tokenizer,
                output_dir=str(_PROJECT_ROOT / "outputs/distillation")
            )
            train_curriculum(trainer=trainer, curriculum=curriculum, synthetic_data={}, eval_data={}, teacher_scores={}, adapter=self._ensure_adapter())

        # 4. Final Adaptation
        adapted_data = self.adapt_knowledge(curriculum, domain, enriched_message, methodology_context)
        
        return MentorResult(
            target_modules=target_modules,
            performance_gaps=self._identifier.performance_gaps if self._identifier else {},
            dependency_graph=dep_graph,
            curriculum=curriculum,
            num_stages=len(curriculum.stages),
            domain_knowledge=f"Know-how pedagógico generado para {domain}",
            synthesis_guidelines=f"Guías de dominio {domain} inyectadas en el generador",
            domain_context_used=(enriched_message is not None)
        )

    def _inject_persistent_seeds(self, factory):
        meth_items = factory.load_seed_data(factory.methodology_file)
        dom_items = factory.load_seed_data(factory.domain_file)
        all_items = meth_items + dom_items
        dynamic_items = [SeedDataItem(item_id=f"s_{i}", domain="unknown", knowledge_modules=["gen"], input_text=it["input"], output_text=it["output"]) for i, it in enumerate(all_items)]
        if dynamic_items:
            self.seed_dataset = SeedDataset(items=dynamic_items)
            if self._identifier: self._identifier.seed_dataset = self.seed_dataset
