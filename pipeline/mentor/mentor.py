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
from .seed_factory import SeedFactory

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

        # Inyectar contexto de dominio SOLO si no tenemos semillas persistentes ya cargadas
        if enriched_message and enriched_message.context_fragments:
            if not self.seed_dataset or len(self.seed_dataset) < 10:
                logger.info("No hay suficientes semillas persistentes. Generando contexto dinámico...")
                self._inject_domain_context(identifier, enriched_message)
            else:
                logger.info(f"Usando {len(self.seed_dataset)} semillas persistentes de la Fábrica.")

        # Ejecutar identificación IOA
        target_modules, dep_graph = identifier.identify(
            domain=domain,
            teacher_model=teacher_model,
            student_model=student_model,
            context=enriched_message.to_full_prompt() if enriched_message else None
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
        retriever: Optional[Any] = None,
        config: Optional[Any] = None,
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

        # 1. Obtener contextos para la Fábrica de Semillas
        methodology_context = ""
        domain_context = ""
        if retriever and config:
            # Contexto metodológico (Libro UCM)
            method_results = retriever.retrieve(
                query="Reglas de oro para síntesis de series temporales, autocorrelación y realismo físico",
                top_k=5,
                collection_name=config.vector_store.collection_methodology
            )
            methodology_context = "\n".join([f"- {f['text']}" for f in method_results.fragments])
            
            # Contexto de dominio (CSV/Manual)
            if enriched_message:
                domain_context = "\n".join([f"- {f}" for f in enriched_message.context_fragments])

        # 2. Asegurar Semillas Persistentes (Fábrica)
        if config and self.teacher_client:
            factory = SeedFactory(seed_dir=config.seed_factory.seed_dir)
            factory.ensure_seeds(
                target_methodology=config.seed_factory.methodology_size,
                target_domain=config.seed_factory.domain_size,
                teacher_client=self.teacher_client,
                methodology_context=methodology_context,
                domain_context=domain_context
            )
            # Inyectar las semillas cargadas en el Identifier antes del diagnóstico
            self._inject_persistent_seeds(factory, enriched_message)

        # 3. Diagnóstico (Ahora con 800+ items si se configuró así)
        target_modules, dep_graph = self.diagnose(
            domain=domain,
            enriched_message=enriched_message,
            teacher_model=teacher_model,
            student_model=student_model,
        )

        # 4. Organización
        curriculum = self.organize(
            target_modules=target_modules,
            domain=domain,
        )

        # 5. Adaptación (Usando semillas + Sabiduría metodológica)
        adapted_data = self.adapt(
            curriculum=curriculum,
            domain=domain,
            enriched_message=enriched_message,
            methodology_context=methodology_context
        )

        # 5. Compilar directrices de síntesis
        guidelines = []
        for stage_id, data in adapted_data.items():
            if "know_how" in data:
                guidelines.append(f"- REGLA: {data['know_how']}")
            if "ejemplo_ideal" in data:
                reasoning = data.get("justificacion_cientifica", "Sin justificación adicional.")
                guidelines.append(f"- EJEMPLO IDEAL: {data['ejemplo_ideal']}\n  - RAZONAMIENTO: {reasoning}")
        
        raw_guidelines = "\n".join(list(set(guidelines)))
            
        # 5. Refinamiento Final (Combinar Dominio + Metodología)
        synthesis_guidelines = raw_guidelines
        if methodology_context and self.teacher_client:
            logger.info("Refinando guías con sabiduría metodológica...")
            refinement_prompt = f"""Eres un Profesor Experto en Estadística y Física.
            
CONOCIMIENTO TÉCNICO (Dominio):
{raw_guidelines}

CONOCIMIENTO METODOLÓGICO (Tu Cerebro):
{methodology_context}

TAREA: Fusiona ambos conocimientos en un "Protocolo de Generación de Excelencia" para tu alumno. 
Enséñale no solo qué generar, sino CÓMO generarlo estadísticamente para que sea físicamente plausible.
Usa un tono ejecutivo y directo."""
            
            try:
                synthesis_guidelines = self.teacher_client.generate(refinement_prompt, max_tokens=1024)
                logger.info("Protocolo de excelencia generado con éxito.")
            except Exception as e:
                logger.warning(f"Error en refinamiento: {e}. Usando guías base.")
        elif methodology_context:
            synthesis_guidelines = f"METODOLOGÍA:\n{methodology_context}\n\nREGLAS TÉCNICAS:\n{raw_guidelines}"
        
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

    def adapt(
        self,
        curriculum: Curriculum,
        domain: str,
        enriched_message: Optional[EnrichedMessage] = None,
        methodology_context: str = "",
    ) -> Dict[str, Any]:
        """
        Fase de Adaptación: El Maestro genera lecciones y ejemplos ideales.
        
        Si las semillas reales son insuficientes, el Maestro crea 'Ejemplos Prototípicos'
        basados en su Cerebro Metodológico.
        """
        adapted_data = {}
        if not curriculum or not curriculum.stages:
            return adapted_data

        for stage in curriculum.stages:
            logger.info(f"Adaptando etapa: {stage.stage_id}")
            
            # Prompt para que el Maestro cree la lección y un ejemplo ideal
            adaptation_prompt = f"""Eres un Maestro de Excelencia en Series Temporales.
            
CONOCIMIENTO EXPERTO (Tu biblioteca):
{methodology_context}

OBJETIVO DE LA ETAPA: {stage.stage_id}
- Módulos: {', '.join(stage.modules)}

TAREA:
1. Genera una instrucción técnica (KNOW-HOW) para el alumno sobre este tema.
2. Si la semilla real no es suficiente para demostrar el concepto, GENERA UN EJEMPLO IDEAL (Prototipo).

FORMATO DE RESPUESTA (JSON):
{{
  "know_how": "Regla de oro sobre el tema...",
  "ejemplo_ideal": [0.1, 0.2, 0.5, 0.8, ...],
  "justificacion_cientifica": "RAZONAMIENTO PROFUNDO: Conecta una regla específica de tu biblioteca experta con el ejemplo ideal. Explica por qué, bajo las leyes de la física o estadística, este patrón es el correcto y cómo se cumple la ley en este caso específico."
}}"""
            
            try:
                response = self.teacher_client.generate(adaptation_prompt)
                import json
                import re
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    adapted_data[stage.stage_id] = json.loads(match.group())
            except Exception as e:
                logger.warning(f"Error en adaptación de etapa {stage.stage_id}: {e}")
                
        return adapted_data

    # ── Helpers ───────────────────────────────────────────────────────────

    def _generate_probes_from_fragments(
        self, 
        fragments: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Usa el Maestro para inventar un examen (Pregunta/Respuesta) 
        basado en los fragmentos de RAG.
        """
        if not self.teacher_client or not fragments:
            return [("Analiza el fragmento", "Basado en el contexto...")] * 5
            
        logger.info(f"Inventando examen dinámico (sondeo) basado en {len(fragments)} fragmentos...")
        
        context_text = "\n---\n".join(fragments)
        prompt = f"""Eres un Profesor de Nivel Doctorado en Ingeniería y Estadística. 
        
CONTEXTO TÉCNICO:
{context_text}

TAREA: Inventa un examen de sondeo EXTREMADAMENTE DIFÍCIL de 5 preguntas.
REGLAS DE ORO PARA EL MAESTRO "TIRANO":
1. PROHIBIDO preguntar cosas que se puedan copiar y pegar del texto.
2. EXIGE INFERENCIA: Pide al alumno que prediga el futuro de la serie temporal basado en la tendencia actual.
3. EXIGE CÁLCULO: Pide cálculos matemáticos precisos sobre tasas de cambio o promedios ponderados.
4. EXIGE CAUSALIDAD: Pide que explique la física profunda (termodinámica, fricción) detrás de un cambio de datos.
5. El objetivo es encontrar los límites del conocimiento del alumno.

FORMATO DE RESPUESTA: Responde ÚNICAMENTE con un JSON array de objetos:
[
  {{"pregunta": "...", "respuesta_correcta": "..."}},
  ...
]"""

        probes = []
        try:
            # Usar temperatura media para diversidad
            response = self.teacher_client.generate(prompt=prompt, temperature=0.5)
            
            # Usar el robusto parse_llm_json_response
            from ..synthesis.generator import parse_llm_json_response
            probes_data = parse_llm_json_response(response)
            
            for item in probes_data:
                q = item.get("pregunta") or item.get("input")
                a = item.get("respuesta") or item.get("output") or item.get("respuesta_correcta")
                if q and a:
                    probes.append((q, a))
            
            # Fallback si el JSON falla
            if not probes:
                probes.append((f"Analiza la relación entre temperatura y vibración", "Respuesta en contexto"))
                
        except Exception as e:
            logger.warning(f"Error generando examen: {e}")
            probes = [("Error en generación", "Error")] * 5
                
        return probes[:5]

    def _inject_persistent_seeds(
        self,
        factory: SeedFactory,
        enriched_message: Optional[EnrichedMessage] = None,
    ) -> None:
        """
        Carga las semillas de los archivos persistentes y las inyecta en el Identifier.
        Esto reemplaza la generación dinámica por una basada en dataset de gran escala.
        """
        logger.info("Inyectando semillas persistentes de la Fábrica en el Identifier...")
        
        from IOA.data.seed_data import SeedDataset, SeedDataItem
        import random
        
        # Cargar metodología y dominio
        meth_items = factory.load_seed_data(factory.methodology_file)
        dom_items = factory.load_seed_data(factory.domain_file)
        all_raw_items = meth_items + dom_items
        
        dynamic_items = []
        for i, item in enumerate(all_raw_items):
            di = SeedDataItem(
                item_id=f"seed_{i}",
                domain=item.get("domain", "predictive_maintenance"),
                knowledge_modules=[item.get("module", "general")],
                input_text=item.get("input", ""),
                output_text=item.get("output", ""),
                difficulty="intermediate",
                metadata={"type": "persistent_seed"}
            )
            dynamic_items.append(di)
            
        if not dynamic_items:
            logger.warning("No se encontraron semillas persistentes. Fallback a dinámico.")
            if enriched_message:
                self._inject_domain_context(self._identifier, enriched_message)
            return
            
        # Construir dataset
        dyn_dataset = SeedDataset(items=dynamic_items)
        
        # Asignar índices (80/20)
        all_indices = list(range(len(dynamic_items)))
        random.shuffle(all_indices)
        split_idx = int(len(all_indices) * 0.8)
        
        dyn_dataset.train_indices = all_indices[:split_idx]
        dyn_dataset.val_indices = all_indices[split_idx:]
        
        # Mapear dominios
        for di in dynamic_items:
            if di.domain not in dyn_dataset.domain_index:
                dyn_dataset.domain_index[di.domain] = []
            # Esto es una simplificación, en realidad necesitamos el índice real
        
        # Reemplazar en los módulos IOA
        self.seed_dataset = dyn_dataset
        if self._identifier:
            self._identifier.seed_dataset = dyn_dataset
            
        logger.info(f"Dataset de sondeo activado con {len(dynamic_items)} items.")

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
