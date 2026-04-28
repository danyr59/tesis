"""
Módulo 4 — Mentor: Generar el Know-how en el Maestro.

El mentor despliega la metodología pedagógica IOA (Identifier-Organizer-Adapter)
para estructurar el conocimiento que será transferido.

Este macro-algoritmo docente sigue tres pasos:

1. **Diagnóstico** (← IOA KnowledgeIdentifier):
   Evalúa la comprensión actual del aprendiz sobre los datos reales usando
   información recibida e histórica para identificar vacíos de lógica.

2. **Organización** (← IOA KnowledgeOrganizer):
   Diseña un plan de estudios que ordena el conocimiento de menor a mayor
   complejidad, respetando las dependencias naturales del conocimiento.

3. **Adaptación** (← IOA KnowledgeAdapter):
   Destila el conocimiento al estudiante según plan de estudio, orientado a:
   - Conocimiento sobre generación de datos sintéticos
   - Datos del dominio (mantenimiento predictivo)

La diferencia con IOA vanilla: se inyecta contexto de dominio (fragmentos
del Módulo 3) en los prompts del Identifier/Organizer/Adapter para
especializarlos en el dominio de mantenimiento predictivo.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..context_builder.builder import EnrichedMessage

logger = get_logger("m4.mentor")

# ── Importar módulos IOA ──────────────────────────────────────────────────
# IOA se importa como paquete normal (proyecto_tesis/ ya está en sys.path
# gracias a pipeline_config.py que se importa primero).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IOA.config.config import (
    IOAConfig,
    IdentifierConfig,
    OrganizerConfig,
    AdapterConfig,
)
from IOA.modules.identifier import KnowledgeIdentifier
from IOA.modules.organizer import KnowledgeOrganizer, Curriculum, CurriculumStage
from IOA.modules.adapter import KnowledgeAdapter
from IOA.utils.llm_client import LLMClient, create_teacher_client
from IOA.utils.graph_utils import KnowledgeDependencyGraph, KnowledgeNode
from IOA.data.seed_data import SeedDataset, SeedDataItem, load_seed_data


# ─────────────────────────────────────────────────────────────────────────────
# Resultado del Mentor
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MentorResult:
    """Resultado completo del proceso pedagógico del Mentor."""

    # Diagnóstico
    target_modules: List[str] = field(default_factory=list)
    performance_gaps: Dict[str, float] = field(default_factory=dict)
    dependency_graph: Optional[KnowledgeDependencyGraph] = None

    # Organización
    curriculum: Optional[Curriculum] = None
    num_stages: int = 0

    # Adaptación
    domain_knowledge: str = ""
    synthesis_guidelines: str = ""

    # Metadata
    domain_context_used: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Mentor
# ─────────────────────────────────────────────────────────────────────────────


class Mentor:
    """
    Wrapper sobre el framework IOA (Identifier-Organizer-Adapter) que
    inyecta contexto de dominio de mantenimiento predictivo.

    Este módulo reutiliza la lógica interna de IOA prácticamente sin cambios.
    """

    def __init__(
        self,
        ioa_config: Optional[IOAConfig] = None,
        teacher_client: Optional[LLMClient] = None,
        seed_dataset: Optional[SeedDataset] = None,
    ):
        self.ioa_config = ioa_config or IOAConfig()
        self.teacher_client = teacher_client
        self.seed_dataset = seed_dataset

        # Componentes IOA (se crean bajo demanda)
        self._identifier: Optional[KnowledgeIdentifier] = None
        self._organizer: Optional[KnowledgeOrganizer] = None
        self._adapter: Optional[KnowledgeAdapter] = None

        # Estado
        self._dependency_graph: Optional[KnowledgeDependencyGraph] = None
        self._curriculum: Optional[Curriculum] = None

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup_teacher(
        self,
        teacher_type: str = "openai",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Configura el cliente LLM maestro."""
        try:
            self.teacher_client = create_teacher_client(
                teacher_type=teacher_type,
                api_key=api_key,
                api_base=api_base,
                model_name=model_name,
            )
            logger.info(f"Maestro configurado: {teacher_type}/{model_name}")
        except Exception as e:
            logger.warning(f"No se pudo crear el cliente LLM maestro: {e}")
            logger.info("El Mentor operará en modo sin LLM (fallback)")

    def setup_seed_data(self, data_dir: str = "./data/seed") -> None:
        """Carga datos semilla para el Identifier."""
        self.seed_dataset = load_seed_data(data_dir=data_dir)
        logger.info(f"Datos semilla cargados: {len(self.seed_dataset)} items")

    def _ensure_identifier(self) -> KnowledgeIdentifier:
        if self._identifier is None:
            if self.seed_dataset is None:
                self.setup_seed_data()
            self._identifier = KnowledgeIdentifier(
                config=self.ioa_config.identifier,
                teacher_client=self.teacher_client,
                seed_dataset=self.seed_dataset,
            )
        return self._identifier

    def _ensure_adapter(self) -> KnowledgeAdapter:
        if self._adapter is None:
            self._adapter = KnowledgeAdapter(
                config=self.ioa_config.adapter,
                teacher_client=self.teacher_client,
            )
        return self._adapter

    # ── Paso 1: Diagnóstico ──────────────────────────────────────────────

    def diagnose(
        self,
        domain: str = "predictive_maintenance",
        enriched_message: Optional[EnrichedMessage] = None,
        teacher_model: Any = None,
        student_model: Any = None,
    ) -> Tuple[List[str], KnowledgeDependencyGraph]:
        """
        Evalúa la comprensión actual del aprendiz e identifica vacíos.

        Reutiliza IOA KnowledgeIdentifier.identify() inyectando contexto
        de dominio en la descomposición del conocimiento.

        Args:
            domain: Dominio target (default para mantenimiento predictivo).
            enriched_message: Contexto enriquecido del M3 (opcional).
            teacher_model: Modelo del maestro.
            student_model: Modelo del estudiante.

        Returns:
            (target_modules, dependency_graph)
        """
        logger.info("=" * 60)
        logger.info("DIAGNÓSTICO — Evaluando comprensión del aprendiz")
        logger.info("=" * 60)

        identifier = self._ensure_identifier()

        # Inyectar contexto de dominio si disponible
        if enriched_message and enriched_message.context_fragments:
            self._inject_domain_context(identifier, enriched_message)

        # Ejecutar identificación IOA
        target_modules, dep_graph = identifier.identify(
            domain=domain,
            teacher_model=teacher_model,
            student_model=student_model,
        )

        self._dependency_graph = dep_graph

        logger.info(
            f"Diagnóstico completo: {len(target_modules)} módulos target identificados"
        )

        return target_modules, dep_graph

    # ── Paso 2: Organización ─────────────────────────────────────────────

    def organize(
        self,
        target_modules: List[str],
        domain: str = "predictive_maintenance",
    ) -> Curriculum:
        """
        Diseña un plan de estudios de menor a mayor complejidad.

        Reutiliza IOA KnowledgeOrganizer.organize() con el grafo de
        dependencias construido en el diagnóstico.

        Args:
            target_modules: Módulos a enseñar (del diagnóstico).
            domain: Nombre del dominio.

        Returns:
            Curriculum organizado con etapas y prerequisitos.
        """
        logger.info("=" * 60)
        logger.info("ORGANIZACIÓN — Diseñando plan de estudios")
        logger.info("=" * 60)

        if self._dependency_graph is None:
            logger.warning(
                "No hay grafo de dependencias. Ejecute diagnose() primero."
            )
            # Crear grafo vacío como fallback
            self._dependency_graph = KnowledgeDependencyGraph()

        self._organizer = KnowledgeOrganizer(
            config=self.ioa_config.organizer,
            dependency_graph=self._dependency_graph,
        )

        self._curriculum = self._organizer.organize(
            target_modules=target_modules,
            domain=domain,
        )

        logger.info(self._organizer.get_curriculum_summary())
        logger.info(
            f"Plan de estudios: {len(self._curriculum.stages)} etapas"
        )

        return self._curriculum

    # ── Paso 3: Adaptación ───────────────────────────────────────────────

    def adapt(
        self,
        curriculum: Optional[Curriculum] = None,
        domain: str = "predictive_maintenance",
        enriched_message: Optional[EnrichedMessage] = None,
    ) -> Dict[str, Any]:
        """
        Destila el conocimiento al estudiante según el plan de estudio.

        Reutiliza IOA KnowledgeAdapter con dos ejes:
        - Conocimiento sobre generación de datos sintéticos
        - Datos del dominio (mantenimiento predictivo)

        Args:
            curriculum: Plan de estudios (default: el del paso anterior).
            domain: Dominio.
            enriched_message: Contexto enriquecido del M3.

        Returns:
            Dict con datos adaptados por etapa del curriculum.
        """
        logger.info("=" * 60)
        logger.info("ADAPTACIÓN — Destilando conocimiento al estudiante")
        logger.info("=" * 60)

        curriculum = curriculum or self._curriculum
        if curriculum is None:
            logger.error("No hay curriculum. Ejecute organize() primero.")
            return {}

        adapter = self._ensure_adapter()
        adapted_data: Dict[str, Any] = {}

        for stage in curriculum.stages:
            logger.info(f"Adaptando etapa {stage.stage_id}...")

            # Obtener semillas para esta etapa
            seed_items: List[SeedDataItem] = []
            if self.seed_dataset:
                for module in stage.modules:
                    items = self.seed_dataset.get_train_items_by_module(module)
                    seed_items.extend(items)

            # Generar datos adaptados
            synthetic_items = adapter.adapt_for_stage(
                stage_id=stage.stage_id,
                knowledge_modules=stage.modules,
                prerequisites=list(stage.prerequisites),
                seed_items=seed_items,
                domain=domain,
            )

            adapted_data[stage.stage_id] = {
                "synthetic_items": synthetic_items,
                "modules": stage.modules,
                "num_items": len(synthetic_items),
            }

            logger.info(
                f"  Etapa {stage.stage_id}: {len(synthetic_items)} items adaptados"
            )

        stats = adapter.get_stats()
        logger.info(f"Adaptación completa — Stats: {stats}")

        return adapted_data

    # ── Pipeline completo ────────────────────────────────────────────────

    def run(
        self,
        domain: str = "predictive_maintenance",
        enriched_message: Optional[EnrichedMessage] = None,
        teacher_model: Any = None,
        student_model: Any = None,
    ) -> MentorResult:
        """
        Ejecuta el pipeline pedagógico completo: Diagnóstico → Organización → Adaptación.

        Args:
            domain: Dominio target.
            enriched_message: Contexto enriquecido del M3.
            teacher_model: Modelo maestro.
            student_model: Modelo estudiante.

        Returns:
            MentorResult con todos los resultados.
        """
        logger.info("#" * 60)
        logger.info("# MENTOR — Pipeline Pedagógico IOA")
        logger.info("#" * 60)

        # 1. Diagnóstico
        target_modules, dep_graph = self.diagnose(
            domain=domain,
            enriched_message=enriched_message,
            teacher_model=teacher_model,
            student_model=student_model,
        )

        # 2. Organización
        curriculum = self.organize(
            target_modules=target_modules,
            domain=domain,
        )

        # 3. Adaptación
        adapted_data = self.adapt(
            curriculum=curriculum,
            domain=domain,
            enriched_message=enriched_message,
        )

        # Compilar directrices de síntesis desde el conocimiento adaptado
        guidelines = []
        for stage_id, data in adapted_data.items():
            for item in data.get("synthetic_items", []):
                # Extraer la lógica destilada (output_text o final_answer)
                if hasattr(item, "output_text") and item.output_text:
                    guidelines.append(f"- {item.output_text}")
                elif hasattr(item, "final_answer") and item.final_answer:
                    guidelines.append(f"- {item.problem}: {item.final_answer}")
        
        synthesis_guidelines = "\n".join(list(set(guidelines))) # Deduplicar
        
        # Compilar resultado
        result = MentorResult(
            target_modules=target_modules,
            performance_gaps=(
                self._identifier.performance_gaps
                if self._identifier
                else {}
            ),
            dependency_graph=dep_graph,
            curriculum=curriculum,
            num_stages=len(curriculum.stages),
            domain_knowledge=synthesis_guidelines, # Guardamos aquí el conocimiento destilado
            synthesis_guidelines=synthesis_guidelines,
            domain_context_used=(enriched_message is not None),
        )

        logger.info("#" * 60)
        logger.info("# MENTOR COMPLETO")
        logger.info(
            f"# Módulos target: {len(target_modules)}, "
            f"Etapas: {result.num_stages}"
        )
        logger.info("#" * 60)

        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    def _generate_probes_from_fragments(
        self, 
        fragments: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Usa el Maestro para inventar un examen (Pregunta/Respuesta) 
        basado en los fragmentos de RAG.
        """
        if not self.teacher_client:
            return [("Analiza el fragmento", "Basado en el contexto...")] * len(fragments)
            
        probes = []
        logger.info(f"Inventando examen dinámico (sondeo) basado en {len(fragments)} fragmentos...")
        
        for fragment in fragments:
            prompt = f"""Basado en este fragmento de conocimiento técnico sobre mantenimiento predictivo:
"{fragment}"

TAREA: Genera una pregunta de examen técnica y específica cuya respuesta se encuentre en el texto.
REGLA: La pregunta debe ser directa y la respuesta debe ser breve (máximo 2 frases).

FORMATO DE RESPUESTA: Responde únicamente con un objeto JSON:
{{"pregunta": "...", "respuesta_correcta": "..."}}
"""
            try:
                # Usar temperatura baja para consistencia en el examen
                response = self.teacher_client.generate(prompt=prompt, temperature=0.1)
                
                import json
                import re
                # Extraer JSON de la respuesta
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    probes.append((data["pregunta"], data["respuesta_correcta"]))
                else:
                    # Fallback si no hay JSON
                    probes.append((f"Analiza técnicamente: {fragment[:100]}...", "Respuesta pendiente"))
            except Exception as e:
                logger.warning(f"Error generando pregunta de sondeo: {e}")
                probes.append((f"Analiza técnicamente: {fragment[:50]}...", "Error"))
                
        return probes

    def _inject_domain_context(
        self,
        identifier: KnowledgeIdentifier,
        enriched_message: EnrichedMessage,
    ) -> None:
        """
        Inyecta contexto de dominio en los prompts del Identifier y 
        reemplaza el seed dataset estático por uno construido al vuelo.
        """
        logger.debug("Construyendo SeedDataset dinámico con examen inventado")
        
        from IOA.data.seed_data import SeedDataset, SeedDataItem
        import random
        
        # --- NUEVO: Inventar el examen antes de crear los items ---
        fragments = enriched_message.context_fragments
        probes = self._generate_probes_from_fragments(fragments)
        # ----------------------------------------------------------

        dynamic_items = []
        domain_name = "predictive_maintenance"
        
        logger.info("--- EXAMEN INVENTADO POR EL MAESTRO ---")
        for i, (question, answer) in enumerate(probes):
            # Mapear correctamente al fragmento original (ahora 1:1)
            fragment_idx = i
            if fragment_idx >= len(fragments):
                fragment_idx = len(fragments) - 1
                
            item = SeedDataItem(
                item_id=f"rag_frag_{i}",
                domain=domain_name,
                knowledge_modules=[], 
                input_text=question,
                output_text=answer,
                difficulty="intermediate",
                metadata={"source": "rag_retrieval", "original_fragment": fragments[fragment_idx]}
            )
            dynamic_items.append(item)
            logger.info(f"Pregunta {i+1}: {question}")
            logger.info(f"Respuesta esperada: {answer}\n")
        logger.info("---------------------------------------")
            
        if not dynamic_items:
            logger.warning("No hay fragmentos de contexto para inyectar.")
            return
            
        # Construir dataset
        dyn_dataset = SeedDataset(items=dynamic_items)
        
        # Asignar índices de train/val artificialmente para que funcione el M4
        all_indices = list(range(len(dynamic_items)))
        random.shuffle(all_indices)
        split_idx = max(1, int(len(all_indices) * 0.8)) # Al menos 1 en val si hay muy pocos
        
        if len(all_indices) == 1:
            dyn_dataset.train_indices = all_indices
            dyn_dataset.val_indices = all_indices
        else:
            dyn_dataset.train_indices = all_indices[:split_idx]
            dyn_dataset.val_indices = all_indices[split_idx:]
            
        dyn_dataset.domain_index[domain_name] = all_indices
        
        # Reemplazar el dataset en el mentor y el identifier
        self.seed_dataset = dyn_dataset
        identifier.seed_dataset = dyn_dataset
        
        ctx_summary = f"{len(enriched_message.context_fragments)} fragmentos reales"
        logger.info(f"Contexto de dominio inyectado como SeedDataset: {ctx_summary}")

    @property
    def curriculum(self) -> Optional[Curriculum]:
        return self._curriculum

    @property
    def dependency_graph(self) -> Optional[KnowledgeDependencyGraph]:
        return self._dependency_graph
