import json
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from pipeline.config.settings import Settings
from pipeline.utils.llm_client import LLMClient
from pipeline.utils.series_utils import (
  diversity_score,
  parse_time_series_metadata,
  series_length_from_metadata,
)


class TeacherAgent:
  """Maestro: RAG TSQA (Chroma en rag/maestro) + razonamiento LLM."""

  REFINEMENT_STRATEGIES = [
    "relax_location",
    "broaden_domain",
    "adjust_frequency",
    "use_student_suggested",
    "complement_diversity",
  ]

  def __init__(self, llm: Optional[LLMClient] = None):
    print(f"[{self.__class__.__name__}] Inicializando (RAG: {Settings.RAG_DATASET_NAME})...")
    self.chroma_client = chromadb.PersistentClient(path=Settings.RAG_DB_PATH)
    self.collection = self.chroma_client.get_collection(Settings.RAG_COLLECTION_NAME)
    self.embedder = SentenceTransformer("all-mpnet-base-v2")
    self.llm = llm or LLMClient(
      Settings.get_teacher_model_path(),
      role_name="Maestro experto en series temporales",
    )
    self.refinements_history: List[str] = []

  def process_request(self, student_prompt: dict) -> dict:
    print(f"\n[{self.__class__.__name__}] Analizando prompt estructurado del estudiante...")

    query_text = self._plan_search(student_prompt)
    print(f"[{self.__class__.__name__}] Plan de búsqueda RAG: '{query_text}'")

    filtered_seeds: List[dict] = []
    quality_summary = ""
    refinement_round = 0

    while refinement_round <= Settings.MAX_TEACHER_REFINEMENTS:
      rag_results = self._query_rag(query_text)
      filtered_seeds, quality_summary, needs_refinement = self._filter_and_reason(
        rag_results, student_prompt
      )
      filtered_seeds = self._complement_seeds_if_needed(filtered_seeds, student_prompt, rag_results)

      quality_ok = self._validate_seed_quality(filtered_seeds)
      if quality_ok and not needs_refinement:
        break

      if refinement_round >= Settings.MAX_TEACHER_REFINEMENTS:
        if not filtered_seeds:
          print(
            f"[{self.__class__.__name__}] Sin éxito tras {Settings.MAX_TEACHER_REFINEMENTS} "
            "refinamientos. Finalizando ejecución sin datos."
          )
          return {"seed_dataset": [], "quality_summary": quality_summary, "metadata_summary": {}}
        print(
          f"[{self.__class__.__name__}] Máximo de refinamientos alcanzado; "
          f"se entregan {len(filtered_seeds)} semillas disponibles."
        )
        break

      query_text = self._next_refinement_query(student_prompt, refinement_round)
      refinement_round += 1

    print(f"[{self.__class__.__name__}] Selección justificada. Resumen: {quality_summary}")

    return {
      "seed_dataset": filtered_seeds,
      "quality_summary": quality_summary,
      "metadata_summary": {
        "count": len(filtered_seeds),
        "refinements_applied": list(self.refinements_history),
        "domain": student_prompt.get("values", {}).get("dominio"),
        "rag_dataset": Settings.RAG_DATASET_NAME,
        "diversity_score": diversity_score(filtered_seeds),
      },
    }

  def _plan_search(self, student_prompt: dict) -> str:
    prompt = f"""Eres el Maestro, experto en series temporales con acceso al RAG del dataset {Settings.RAG_DATASET_NAME}.
Analiza el prompt estructurado del Estudiante:
{json.dumps(student_prompt, ensure_ascii=False)}

1. Revisa la estrategia de búsqueda sugerida (suggested_search).
2. Elabora un plan breve y devuelve ÚNICAMENTE JSON:
{{
  "plan": "descripción corta del plan",
  "search_query": "palabras clave exactas para el motor vectorial"
}}
"""
    response = self.llm.generate_json(prompt)
    plan = response.get("plan", "")
    if plan:
      print(f"[{self.__class__.__name__}] Plan: {plan}")
    return response.get("search_query") or student_prompt.get(
      "suggested_search", "series temporales"
    )

  def _query_rag(self, text: str) -> dict:
    print(
      f"[{self.__class__.__name__}] Consultando RAG {Settings.RAG_DATASET_NAME} "
      f"(top_k={Settings.RAG_TOP_K})..."
    )
    embedding = self.embedder.encode(text).tolist()
    return self.collection.query(
      query_embeddings=[embedding],
      n_results=Settings.RAG_TOP_K,
    )

  def _build_metadata_compact(
    self, rag_results: dict, student_prompt: dict
  ) -> List[dict]:
    values = student_prompt.get("values", {})
    target_domain = values.get("dominio", "")
    target_loc = values.get("localizacion", "")
    target_freq = values.get("frecuencia", "")

    compact = []
    for i, meta in enumerate(rag_results["metadatas"][0]):
      domain = meta.get("domain", "")
      stext = str(meta.get("search_text", ""))[:120]
      compact.append(
        {
          "idx": i,
          "domain": domain,
          "task_type": meta.get("task_type", ""),
          "length": series_length_from_metadata(meta),
          "location_hint": target_loc.lower() in stext.lower() if target_loc else None,
          "frequency_hint": target_freq.lower() in stext.lower() if target_freq else None,
          "domain_match": target_domain.lower() in domain.lower() if target_domain else None,
        }
      )
    return compact

  def _filter_and_reason(
    self, rag_results: dict, student_prompt: dict
  ) -> Tuple[List[dict], str, bool]:
    metadata_compact = self._build_metadata_compact(rag_results, student_prompt)
    n_docs = len(rag_results["ids"][0])

    prompt = f"""Eres el Maestro, experto en series temporales (RAG {Settings.RAG_DATASET_NAME}).
Solicitud del Estudiante:
{json.dumps(student_prompt, ensure_ascii=False)}

Recuperaste {n_docs} documentos. Metadatos resumidos:
{json.dumps(metadata_compact[:200], ensure_ascii=False)}

Analiza: similitud semántica, dominio, frecuencia, localización, longitud y diversidad.
Puedes aceptar series cortas (<{Settings.MIN_SERIES_LENGTH} puntos) si aportan conocimiento.
Evalúa coherencia con el requerimiento.

Devuelve ÚNICAMENTE JSON:
{{
  "accepted_indexes": [0, 2],
  "needs_refinement": false,
  "quality_summary": "justificación de selección, diversidad y uso de series cortas",
  "complement_needed": false
}}
"""
    response = self.llm.generate_json(prompt, max_new_tokens=1536)
    accepted_indexes = response.get("accepted_indexes", [])
    if not accepted_indexes and n_docs:
      accepted_indexes = list(range(min(8, n_docs)))

    filtered_seeds = self._indexes_to_seeds(accepted_indexes, rag_results)
    return (
      filtered_seeds,
      response.get("quality_summary", "Calidad moderada."),
      bool(response.get("needs_refinement", False)),
    )

  def _indexes_to_seeds(self, indexes: List[int], rag_results: dict) -> List[dict]:
    filtered = []
    for idx in indexes:
      if idx >= len(rag_results["ids"][0]):
        continue
      meta = rag_results["metadatas"][0][idx]
      filtered.append(
        {
          "id": rag_results["ids"][0][idx],
          "metadata": meta,
          "series": parse_time_series_metadata(meta.get("time_series")),
        }
      )
    return filtered

  def _complement_seeds_if_needed(
    self,
    seeds: List[dict],
    student_prompt: dict,
    rag_results: dict,
  ) -> List[dict]:
    if len(seeds) >= Settings.MIN_SEED_EXAMPLES and diversity_score(seeds) >= 0.25:
      return seeds

    initial_count = len(seeds)
    extra_indexes = list(range(min(20, len(rag_results["ids"][0]))))
    existing_ids = {s["id"] for s in seeds}
    for idx in extra_indexes:
      doc_id = rag_results["ids"][0][idx]
      if doc_id in existing_ids:
        continue
      meta = rag_results["metadatas"][0][idx]
      seeds.append(
        {
          "id": doc_id,
          "metadata": meta,
          "series": parse_time_series_metadata(meta.get("time_series")),
        }
      )
      existing_ids.add(doc_id)
      if len(seeds) >= Settings.MIN_SEED_EXAMPLES + 2:
        break

    if len(seeds) > initial_count:
      self._record_refinement("complemento_automatico_semillas")
    return seeds

  def _validate_seed_quality(self, seeds: List[dict]) -> bool:
    if len(seeds) < Settings.MIN_SEED_EXAMPLES:
      return False
    long_enough = sum(
      1
      for s in seeds
      if series_length_from_metadata(s.get("metadata", {})) >= Settings.MIN_SERIES_LENGTH
      or len(str(s.get("metadata", {}).get("search_text", ""))) > 30
    )
    if long_enough < Settings.MIN_SEED_EXAMPLES:
      return False
    return True

  def _next_refinement_query(self, student_prompt: dict, round_idx: int) -> str:
    strategies = [s for s in self.REFINEMENT_STRATEGIES if s not in self.refinements_history]
    if not strategies:
      strategies = self.REFINEMENT_STRATEGIES
    strategy = strategies[round_idx % len(strategies)]
    self._record_refinement(strategy)
    values = student_prompt.get("values", {})

    if strategy == "relax_location":
      return values.get("dominio", "series temporales")
    if strategy == "broaden_domain":
      return "series temporales diversas dominio general"
    if strategy == "adjust_frequency":
      return f"{values.get('dominio', '')} {values.get('frecuencia', '')} series temporales".strip()
    if strategy == "use_student_suggested":
      return student_prompt.get("suggested_search", "series temporales")
    return f"{student_prompt.get('suggested_search', '')} complemento diversidad anomalías"

  def _record_refinement(self, action: str) -> None:
    if action not in self.refinements_history:
      self.refinements_history.append(action)
    print(f"[{self.__class__.__name__}] Refinamiento: {action}")
