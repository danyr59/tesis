import ast
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
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
    adapter_path = Settings.get_teacher_model_path_adapter()
    if adapter_path and os.path.exists(adapter_path):
        self.llm.load_adapter(adapter_path)
    else:
        print(f"[{self.__class__.__name__}] Sin adaptadores previos. Usando modelo base con RAG.")
    self.refinements_history: List[str] = []
    self.num_series_target = Settings.SYNTHETIC_NUM_SERIES
    self.points_per_series_target = Settings.SYNTHETIC_POINTS_PER_SERIES


  

  def interpret_request(self, query: str) -> dict:
    print(f"\n[{self.__class__.__name__}] Interpretando requerimiento: '{query}'")
    current_query = query
    dialog_rounds = 0
    max_dialog_rounds = 8

    while dialog_rounds < max_dialog_rounds:
        dialog_rounds += 1
        prompt = self._interpretation_prompt(current_query)
        response = self.llm.generate_json(prompt, max_new_tokens=400)

        missing = response.get("missing_parameters") or []
        default_conf = 1.0 if not missing else 0.0
        confidence = float(response.get("confidence", default_conf))

        if confidence >= Settings.MIN_CONFIDENCE_THRESHOLD and not missing:
            break

        if not missing and confidence < Settings.MIN_CONFIDENCE_THRESHOLD:
            missing = ["detalles adicionales del requerimiento"]

        print(f"\n[{self.__class__.__name__}] Información insuficiente (confianza={confidence:.2f}).")
        print(f"Justificación: {response.get('justification', '')}")
        
        query_changed = False
        for param in missing:
            user_val = input(f"[{self.__class__.__name__}] Aclara el parámetro '{param}' (o deja en blanco para ignorar): ")
            if user_val.strip():
                current_query += f". {param}: {user_val}"
                query_changed = True
        
        if not query_changed:
            print(f"[{self.__class__.__name__}] No se agregaron aclaraciones. Forzando continuación...")
            break
    else:
        print(f"[{self.__class__.__name__}] Máximo de rondas de diálogo alcanzado; continuando con la mejor interpretación disponible.")
        if not response:
            response = self.llm.generate_json(self._interpretation_prompt(current_query), max_new_tokens=400)

    # Regla de consistencia: si el usuario especifica explícitamente un perfil (ej. P3_TENDENCIA_ESTACIONAL), respetarlo
    q_lower = current_query.lower()
    vals = response.get("values", {})
    m_profile = re.search(r"\b(P[1-5]_[A-Z_]+)\b", current_query, re.IGNORECASE)
    if m_profile:
        vals["profile_id"] = m_profile.group(1).upper()
    elif ("estable" in q_lower or "constante" in q_lower) and vals.get("profile_id") != "P1_ESTABLE":
        vals["profile_id"] = "P1_ESTABLE"

    # Regla de consistencia: si el usuario especifica variables estrictas o únicas, respetarlas sin alucinar
    m_strict_var = re.search(r"(?:variables?\s+estrictas?|únicamente)\s*:?\s*(?:únicamente\s*)?['\"]?([a-zA-Z0-9_]+)['\"]?", current_query, re.IGNORECASE)
    if m_strict_var:
        vals["variables"] = [m_strict_var.group(1)]

    # Regla de consistencia: tipo_serie debe ser coherente con la cantidad de variables
    var_list = vals.get("variables", [])
    if isinstance(var_list, list):
        if len(var_list) <= 1:
            vals["tipo_serie"] = "univariada"
        elif "multi" not in vals.get("tipo_serie", "").lower():
            vals["tipo_serie"] = "multivariada"

    # Extraer y sanear target_value (Desacoplamiento de escala)
    tv = vals.get("target_value")
    if isinstance(tv, list) and len(tv) > 0:
        try:
            vals["target_value"] = float(tv[0])
        except (ValueError, TypeError):
            vals["target_value"] = None
    elif tv is not None:
        try:
            vals["target_value"] = float(tv)
        except (ValueError, TypeError):
            vals["target_value"] = None

    if vals.get("target_value") is None:
        for mp in response.get("missing_parameters", []):
            if isinstance(mp, dict) and "expected_value" in mp:
                try:
                    vals["target_value"] = float(mp["expected_value"])
                    break
                except (ValueError, TypeError):
                    pass
    if vals.get("target_value") is None:
        # Búsqueda universal de escala, promedio, magnitud o nivel en el texto del requerimiento
        patterns = [
            r'(?:escala(?:\s+inicial)?|promedio|media|alrededor|cerca|nivel(?:\s+medio)?)\s*(?:es|de|en|:)?\s*(?:promedio\s+de\s+|alrededor\s+de\s+|nivel\s+medio\s+en\s+|los\s+)?(\d+(?:[.,]\d+)?)',
            r'(\d+(?:[.,]\d+)?)\s*(?:v\b|voltios|voltaje|°c|c\b|lpm\b|latidos|pasajeros|ventas|dolares|usd|\bpor\s+ciento|%)',
        ]
        for pat in patterns:
            m_val = re.search(pat, q_lower)
            if m_val:
                try:
                    vals["target_value"] = float(m_val.group(1).replace(",", "."))
                    break
                except (ValueError, TypeError):
                    pass

    structured = {
        "values": response.get("values", {}),
        "missing_parameters": response.get("missing_parameters", []),
        "pending_questions": response.get("pending_questions", missing),
        "justification": response.get("justification", ""),
        "suggested_search": response.get("suggested_search", ""),
        "search_strategy": response.get("search_strategy", response.get("suggested_search", "")),
        "confidence": confidence,
    }
    print(f"[{self.__class__.__name__}] Prompt estructurado:\n{json.dumps(structured, indent=2, ensure_ascii=False)}")
    return structured

  def _interpretation_prompt(self, query: str) -> str:
    return f"""Eres un agente experto en extracción de datos estructurados para series temporales.
Requerimiento del usuario: "{query}"

INSTRUCCIONES ESTRICTAS:
1. Extrae el nombre exacto de la(s) variable(s) que el usuario quiere medir.
   REGLA DE ORO: Las variables NUNCA son conceptos de tiempo (Fecha, Día, Mes, Hora, Año). Si el usuario pide "Pasajeros diarios", la variable es SOLO "Pasajeros".
2. Identifica el dominio (ej. Transporte, Finanzas, Clima, Electrónica).
3. Selecciona el 'profile_id' exacto:
   - P1_ESTABLE: Valores constantes, poco ruido (ej. voltaje, reposo, consumo constante).
     * REGLA ESTRICTA: Si el usuario pide que sea 'estable', 'constante', 'fijo' o indica un nivel de operación estable (ej. 220V constante), el profile_id DEBE ser P1_ESTABLE (NUNCA P2_ESTACIONAL).
   - P2_ESTACIONAL: Patrones cíclicos repetitivos puros (ej. pasajeros metro, estacionalidad climática).
   - P3_TENDENCIA_ESTACIONAL: Ciclos que además suben o bajan con el tiempo (ej. ventas).
   - P4_TENDENCIA_RUIDO: Tendencia con alta volatilidad caótica (ej. bolsa de valores, cripto).
   - P5_COMPLEJO: Cambios de régimen drásticos (ej. ejercicio físico).
4. Extrae 'target_value': el valor numérico objetivo, escala o promedio deseado si el usuario lo menciona (ej. 220.0, 15000.0, 70.0). Debe ser un único número float o null (NUNCA una lista, NUNCA agregues comentarios // en el JSON). Si no especifica ninguno, pon null.
5. El campo "profile_id" DEBE ser reemplazado exactamente por uno de los 5 IDs de arriba.
6. DEBES reemplazar todo lo que esté entre corchetes [...] por tu propio análisis.

Devuelve ÚNICAMENTE el siguiente JSON válido (sin formato markdown ```):
{{
  "confidence": 0.9,
  "values": {{
    "tema": "[Extrae el tema principal de la consulta]",
    "dominio": "[Extrae el dominio (ej. electronica, salud, finanzas)]",
    "localizacion": "[Extrae el lugar o pon null si no aplica]",
    "frecuencia": "[Diaria, Mensual, Minuto, etc.]",
    "start_date": "2024-01-01",
    "variables": ["Voltaje"],
    "target_value": null,
    "tipo_serie": "univariada",
    "profile_id": "P1_ESTABLE",
    "num_series_configuradas": {self.num_series_target},
    "puntos_por_serie_configurados": {self.points_per_series_target}
  }},
  "missing_parameters": [],
  "pending_questions": [],
  "justification": "[Escribe aquí por qué elegiste este perfil y dominio]",
  "suggested_search": "[Escribe una búsqueda técnica relacionada]",
  "search_strategy": "[Escribe una estrategia de búsqueda]"
}}
"""

  def process_request(self, student_prompt: dict) -> dict:
    print(
        f"\n[{self.__class__.__name__}] Analizando prompt estructurado del usuario...")

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
Analiza el prompt estructurado del Usuario:
{json.dumps(student_prompt, ensure_ascii=False)}

1. Revisa la estrategia de búsqueda sugerida (suggested_search).
2. Elabora un plan breve y devuelve ÚNICAMENTE JSON
3. Escoge una de estas estrategias '{json.dumps(TeacherAgent.REFINEMENT_STRATEGIES)}' y devuelvela solo una de esa lista, que va acorde con el tipo de busqueda que suguieres:
{{
  "plan": "descripción corta del plan",
  "search_query": "palabras clave exactas para el motor vectorial, incluye el dominio (IMPORTANTE), las variables sobre las cual se quiere informacion, y el lugar a operar. Se conciso ",
  "strategy": "Escoge una de: {', '.join(TeacherAgent.REFINEMENT_STRATEGIES)}" 
}}
```json
"""
    response = self.llm.generate_json(prompt, max_new_tokens=300)
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
        # if len(ts_clean) > 2000:
        #     ts_clean = ts_clean[:2000] + "... [TRUNCADO POR SEGURIDAD]"
        
        compact.append({
            "idx": i,
            "domain": meta.get("domain", ""),
            "question": q_clean,
            "answer": a_clean,
            "task_type": meta.get("task_type", ""),
            "time_series": ts_clean
        })
    return compact

  def _summarize_time_series(self, ts_str: str) -> dict:
      try:
          try:
              data = json.loads(ts_str)
          except Exception:
              try:
                  data = ast.literal_eval(ts_str)
              except Exception:
                  return {"raw_preview": ts_str[:100] + "..."}
                  
          series = []
          if isinstance(data, dict):
              if "input" in data and isinstance(data["input"], list) and len(data["input"]) > 0:
                  first_input = data["input"][0]
                  if "serie" in first_input:
                      series = first_input["serie"]
          elif isinstance(data, list):
              series = data
              
          if not series or not isinstance(series, list):
              return {"raw_preview": ts_str[:100] + "..."}
              
          floats = [float(x) for x in series]
          if not floats:
              return {"raw_preview": "empty"}
              
          arr = np.array(floats)
          stats = {
              "length": len(floats),
              "min": round(float(np.min(arr)), 4),
              "max": round(float(np.max(arr)), 4),
              "mean": round(float(np.mean(arr)), 4)
          }
          
          if len(floats) > 10:
              indices = np.linspace(0, len(floats) - 1, 10, dtype=int)
              downsampled = [round(floats[i], 4) for i in indices]
          else:
              downsampled = [round(x, 4) for x in floats]
              
          stats["downsampled_10_pts"] = downsampled
          return stats
      except Exception:
          return {"raw_preview": ts_str[:100] + "..."}



  def _filter_and_reason(
    self, rag_results: dict, student_prompt: dict
  ) -> Tuple[List[dict], str, bool]:
    metadata_compact = self._build_metadata_compact(rag_results, student_prompt)
    n_docs = len(rag_results["ids"][0])

    BATCH_SIZE = 1
    accepted_indexes_global = []
    quality_summary_list = []
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
            
            # TRUNCAR time_series fuertemente
            ts_str = str(meta_copy.get("time_series", ""))
            
            # 1. Agregar el resumen matemático y downsampling
            meta_copy["time_series_math_summary"] = self._summarize_time_series(ts_str)
            
            # 2. Mantener la serie cruda truncada como pidió el usuario (para el lote actual)
            if len(ts_str) > 4000:
                meta_copy["time_series_raw_truncated"] = ts_str[:4000] + "... (truncado)"
            else:
                meta_copy["time_series_raw_truncated"] = ts_str
                
            # Eliminamos el viejo key largo para evitar duplicación
            if "time_series" in meta_copy:
                del meta_copy["time_series"]
            
            
            batch_metadata.append(meta_copy)

        # --- Resumen de los últimos 5 aceptados (SOLO resumen matemático, SIN crudos) ---
        accepted_summary = []
        for g_idx in accepted_indexes_global[-5:]:
            m = metadata_compact[g_idx]
            
            ts_str_acc = str(m.get("time_series", ""))
            math_summary = self._summarize_time_series(ts_str_acc)
                
            accepted_summary.append({
                "domain": m.get("domain"),
                "time_series_math_summary": math_summary,
            })

        if not quality_summary_list:
            quality_summary_str = "No hay selecciones previas."
        else:
            quality_summary_str = " ---- ".join(quality_summary_list[-5:])

        # --- Prompt mejorado (ejemplo de salida más claro) ---
        prompt = f"""Eres el Maestro, experto en series temporales (RAG {Settings.RAG_DATASET_NAME}).
Solicitud del Usuario:
{json.dumps(student_prompt, ensure_ascii=False)}

Ya has aceptado previamente {len(accepted_indexes_global)} documentos.
Ejemplos de los últimos aceptados (para evaluar diversidad):
{json.dumps(accepted_summary, ensure_ascii=False) if accepted_summary else 'Ninguno aún'}.

Resumen de calidad de lo aceptado hasta ahora: {quality_summary_str}

Ahora evalúa EXCLUSIVAMENTE este nuevo lote de {len(batch_metadata)} documentos (usa los valores "idx" relativos proporcionados abajo):
{json.dumps(batch_metadata, ensure_ascii=False)}

Analiza: similitud semántica, dominio, frecuencia, diversidad con lo ya aceptado y utilidad.
Puedes aceptar series cortas (<{Settings.MIN_SERIES_LENGTH} puntos) si aportan conocimiento nuevo.
Si este lote aporta poco o es redundante respecto al resumen anterior, puedes rechazarlo entero (lista vacía).

Devuelve ÚNICAMENTE JSON. Indica con `true` o `false` en "accept_document" si decides aceptar el documento actual para tu investigación:
{{
  "accept_document": true,
  "needs_refinement": false,
  "quality_summary": "Justifica brevemente tu selección en una sola oración concisa."
}}
"""
        print(f"[{self.__class__.__name__}] Evaluando lote de {len(batch_metadata)} documentos con el LLM (esto puede tardar unos segundos)...")
        print("\n[DEBUG PROMPT MAESTRO]\n" + prompt[:1500] + "\n...\n")
        response = self.llm.generate_json(prompt, max_new_tokens=300)
        
        accept_doc = response.get("accept_document", False)
        # Soportamos múltiples formas de decir "sí" por si el modelo alucina un string
        if str(accept_doc).strip().lower() in ["true", "si", "sí", "yes", "1", "verdadero", "aceptar"]:
            accept_doc = True
            
        if accept_doc and len(batch_indices) > 0:
            global_idx = batch_indices[0]
            if global_idx not in accepted_indexes_global:
                accepted_indexes_global.append(global_idx)

        # Actualizar resumen acumulado (siempre que se genere uno nuevo)
        new_summary = response.get("quality_summary", "")
        if new_summary:
            quality_summary_list.append(new_summary)

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
        needs_refinement_by_vote = vote_ratio >= 0.8
    else:
        needs_refinement_by_vote = False

    # Evaluar calidad de las semillas seleccionadas  
    filtered_seeds = self._indexes_to_seeds(accepted_indexes_global, rag_results)
    quality_ok = self._validate_seed_quality(filtered_seeds)

    # Combinar: refinar si votación lo pide O si la calidad/cantidad final es insuficiente
    needs_refinement_global = (
        needs_refinement_by_vote or
        not quality_ok or
        len(filtered_seeds) < Settings.MIN_SEED_EXAMPLES
    )
    final_summary = " ---- ".join(quality_summary_list) if quality_summary_list else "No hay selecciones previas."
    return filtered_seeds, final_summary, needs_refinement_global

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
