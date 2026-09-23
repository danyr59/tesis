import json
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from pipeline.config.settings import Settings
from pipeline.utils.llm_client import LLMClient
from pipeline.utils.series_utils import (
  diversity_score,
  parse_time_series_metadata,
  parser_search_query,
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
    self.collection = self.chroma_client.get_collection(
        Settings.RAG_COLLECTION_NAME)
    self.embedder = SentenceTransformer("all-mpnet-base-v2")
    self.llm = llm or LLMClient(
      Settings.get_teacher_model_path(),
      role_name="Maestro experto en series temporales",
    )
    self.llm.load_adapter(Settings.get_teacher_model_path_adapter())
    self.refinements_history: List[str] = []


  

  def process_request(self, student_prompt: dict) -> dict:
    print(
        f"\n[{self.__class__.__name__}] Analizando prompt estructurado del estudiante...")

    plan, query_text, strategy = self._plan_search(student_prompt)

    query_text = parser_search_query(query_text)

    self._record_refinement(strategy)

    print(f"[{self.__class__.__name__}] Plan de búsqueda RAG: '{plan}'\n Busqueda suguerida: {query_text}")

    filtered_seeds: List[dict] = []
    quality_summary = ""
    refinement_round = 0

    while refinement_round <= Settings.MAX_TEACHER_REFINEMENTS:
      rag_results = self._query_rag(query_text)
      filtered_seeds, quality_summary, needs_refinement = self._filter_and_reason(
        rag_results, student_prompt
      )
      
      if not needs_refinement:
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

      query_text = self._next_refinement_query(
          student_prompt, refinement_round)
      refinement_round += 1

    print(
        f"[{self.__class__.__name__}] Selección justificada. Resumen: {quality_summary}")

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

  def _plan_search(self, student_prompt: dict) -> tuple[str, str, str]:
    prompt = f"""Eres el Maestro, experto en series temporales con acceso al RAG del dataset {Settings.RAG_DATASET_NAME}.
Analiza el prompt estructurado del Estudiante:
{json.dumps(student_prompt, ensure_ascii=False)}

1. Revisa la estrategia de búsqueda sugerida (suggested_search).
2. Elabora un plan breve y devuelve ÚNICAMENTE JSON
3. Escoge una de estas estrategias '{json.dumps(TeacherAgent.REFINEMENT_STRATEGIES)}' y devuelvela solo una de esa lista, que va acorde con el tipo de busqueda que suguieres:
{{
  "plan": "descripción corta del plan",
  "search_query": "palabras clave exactas para el motor vectorial, incluye el dominio (IMPORTANTE), las variables sobre las cual se quiere informacion, y el lugar a operar. Se conciso ",
  "strategy": "Escoge una de: {', '.join(TeacherAgent.REFINEMENT_STRATEGIES)}" 
}}
"""
    response = self.llm.generate_json(prompt)
    plan = response.get("plan", "")
    if plan:
      print(f"[{self.__class__.__name__}] Plan: {plan}")
    search_query =response.get("search_query") or student_prompt.get(
      "suggested_search", "series temporales"
    )
    strategy = response.get("strategy","")
    
    return plan, search_query, strategy

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

  def _build_metadata_compact(self, rag_results: dict, student_prompt: dict) -> List[dict]:
    import re
    compact = []
    for i, meta in enumerate(rag_results["metadatas"][0]):
        # Las bases de datos TSQA a veces incrustan los números crudos dentro de 'question' o 'response'.
        # Usamos regex para eliminar arreglos gigantes de números (ej. "[1.2, 3.4, ...]") que causan OOM.
        q_text = str(meta.get("question", ""))
        a_text = str(meta.get("response", ""))
        q_clean = re.sub(r'\[\s*[-\d.,\s]+\]', '[ARREGLO DE NUMEROS REMOVIDO]', q_text)
        a_clean = re.sub(r'\[\s*[-\d.,\s]+\]', '[ARREGLO DE NUMEROS REMOVIDO]', a_text)
        
        # TRUNCADO ABSOLUTO DE SEGURIDAD
        if len(q_clean) > 2000:
            q_clean = q_clean[:2000] + "... [TRUNCADO POR SEGURIDAD]"
        if len(a_clean) > 2000:
            a_clean = a_clean[:2000] + "... [TRUNCADO POR SEGURIDAD]"
        
        ts_clean = str(meta.get("time_series", ""))
        if len(ts_clean) > 2000:
            ts_clean = ts_clean[:2000] + "... [TRUNCADO POR SEGURIDAD]"
        
        compact.append({
            "idx": i,
            "domain": meta.get("domain", ""),
            "question": q_clean,
            "answer": a_clean,
            "task_type": meta.get("task_type", ""),
            "time_series": ts_clean
        })
    return compact


  

  def _filter_and_reason(
    self, rag_results: dict, student_prompt: dict
  ) -> Tuple[List[dict], str, bool]:
    metadata_compact = self._build_metadata_compact(rag_results, student_prompt)
    n_docs = len(rag_results["ids"][0])

    BATCH_SIZE = 3
    accepted_indexes_global = []
    quality_summary_acum = "No hay selecciones previas."
    refinement_votes = []

    for i in range(0, n_docs, BATCH_SIZE):
        batch_indices = list(range(i, min(i + BATCH_SIZE, n_docs)))
        if not batch_indices:
            continue

        # --- Construir metadatos del lote con índices relativos ---
        batch_metadata = []
        for rel_idx, global_idx in enumerate(batch_indices):
            meta_copy = metadata_compact[global_idx].copy()
            meta_copy["idx"] = rel_idx
            
            # TRUNCAR time_series para evitar el error de 78 GB OOM!
            ts_str = str(meta_copy.get("time_series", ""))
            if len(ts_str) > 2000:
                meta_copy["time_series"] = ts_str[:2000] + "... (truncado)"
            
            batch_metadata.append(meta_copy)

        # --- Resumen de los últimos 5 aceptados (incluye frecuencia) ---
        accepted_summary = []
        for g_idx in accepted_indexes_global[-5:]:
            m = metadata_compact[g_idx]
            
            ts_str_acc = str(m.get("time_series", ""))
            if len(ts_str_acc) > 2000:
                ts_str_acc = ts_str_acc[:2000] + "... (truncado)"
                
            accepted_summary.append({
                "domain": m.get("domain"),
                "time_series": ts_str_acc,
            })

        # --- Prompt mejorado (ejemplo de salida más claro) ---
        prompt = f"""Eres el Maestro, experto en series temporales (RAG {Settings.RAG_DATASET_NAME}).
Solicitud del Estudiante:
{json.dumps(student_prompt, ensure_ascii=False)}

Ya has aceptado previamente {len(accepted_indexes_global)} documentos.
Ejemplos de los últimos aceptados (para evaluar diversidad):
{json.dumps(accepted_summary, ensure_ascii=False) if accepted_summary else 'Ninguno aún'}.

Resumen de calidad de lo aceptado hasta ahora: {quality_summary_acum}

Ahora evalúa EXCLUSIVAMENTE este nuevo lote de {len(batch_metadata)} documentos (usa los valores "idx" relativos proporcionados abajo):
{json.dumps(batch_metadata, ensure_ascii=False)}

Analiza: similitud semántica, dominio, frecuencia, diversidad con lo ya aceptado y utilidad.
Puedes aceptar series cortas (<{Settings.MIN_SERIES_LENGTH} puntos) si aportan conocimiento nuevo.
Si este lote aporta poco o es redundante respecto al resumen anterior, puedes rechazarlo entero (lista vacía).

Devuelve ÚNICAMENTE JSON. En "accepted_relative_indexes" con los índices relativos REALES que seleccionas (NO copies el ejemplo, analiza los documentos) debes REEMPLAZAR el arreglo de ejemplo [0, 1] por los números enteros "idx" de los documentos que decidas aceptar (o déjalo vacío [] si rechazas todos):
{{
  "accepted_relative_indexes": [0, 1],
  "needs_refinement": false,
  "quality_summary": "Justifica tu selección en UNA SOLA ORACIÓN CORTA. (Debe terminar con comillas dobles)"
}}
"""
        print(f"[{self.__class__.__name__}] Evaluando lote de {len(batch_metadata)} documentos con el LLM (esto puede tardar unos segundos)...")
        print("\n[DEBUG PROMPT MAESTRO]\n" + prompt[:1500] + "\n...\n")
        # FIX: Reducimos max_new_tokens a 300. Si el modelo alucina o escribe basura, lo cortamos rápido para no saturar la RAM
        response = self.llm.generate_json(prompt, max_new_tokens=300)
        relative_accepted = response.get("accepted_relative_indexes", [])

        # Mapear índices relativos a globales
        if isinstance(relative_accepted, list):
            for rel_idx in relative_accepted:
                try:
                    rel_idx = int(rel_idx)
                    if 0 <= rel_idx < len(batch_indices):
                        global_idx = batch_indices[rel_idx]
                        if global_idx not in accepted_indexes_global:
                            accepted_indexes_global.append(global_idx)
                except (ValueError, TypeError):
                    continue

        # Actualizar resumen acumulado (siempre que se genere uno nuevo)
        new_summary = response.get("quality_summary", "")
        if new_summary:
            quality_summary_acum += "----" + new_summary

        # Registrar voto de refinamiento
        needs_ref = response.get("needs_refinement", False)
        if isinstance(needs_ref, bool):
            refinement_votes.append(needs_ref)
        elif isinstance(needs_ref, str):
            refinement_votes.append(needs_ref.lower() in ("true", "1", "yes"))
        else:
            refinement_votes.append(bool(needs_ref))

    # --- Fallback si no seleccionó nada ---
    if not accepted_indexes_global and n_docs > 0:
        accepted_indexes_global = list(range(min(5, n_docs)))

    # --- Decisión final de refinamiento (votación + calidad) ---
    total_lotes = len(refinement_votes)
    if total_lotes > 0:
        true_votes = sum(refinement_votes)
        vote_ratio = true_votes / total_lotes
        needs_refinement_by_vote = vote_ratio <= 0.4
    else:
        needs_refinement_by_vote = False

    # Evaluar calidad de las semillas seleccionadas  
    filtered_seeds = self._indexes_to_seeds(accepted_indexes_global, rag_results)
    quality_ok = self._validate_seed_quality(filtered_seeds)
    diversity_ok = diversity_score(filtered_seeds) >= 0.2 if filtered_seeds else False

    # Combinar: refinar si votación lo pide O si la calidad final es insuficiente
    needs_refinement_global = (
        needs_refinement_by_vote or
        not quality_ok or
        (not diversity_ok and len(filtered_seeds) > 0) or
        len(filtered_seeds) < Settings.MIN_SEED_EXAMPLES
    )

    return filtered_seeds, quality_summary_acum, needs_refinement_global

  def _indexes_to_seeds(self, indexes: List[int], rag_results: dict) -> List[dict]:
    filtered = []
    for idx in indexes:
      if idx >= len(rag_results["ids"][0]):
        continue
      meta = rag_results["metadatas"][0][idx]
      new_meta = meta.copy()
      time_series_data = new_meta.pop("time_series", None)
      filtered.append(
        {
          "id": rag_results["ids"][0][idx],
          "metadata": new_meta,
          "series": parse_time_series_metadata(time_series_data),
        }
      )
    return filtered

  

  def _validate_seed_quality(self, seeds: List[dict]) -> bool:
    if len(seeds) < Settings.MIN_SEED_EXAMPLES:
      return False
    long_enough = sum(
      1
      for s in seeds
      if series_length_from_metadata(s) >= Settings.MIN_SERIES_LENGTH
      
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
    #if action not in self.refinements_history:
    self.refinements_history.append(action)
    print(f"[{self.__class__.__name__}] Refinamiento: {action}")
