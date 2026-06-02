import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pipeline.config.settings import Settings

class TeacherAgent:
    def __init__(self):
        print(f"[{self.__class__.__name__}] Inicializando...")
        # Inicializar cliente RAG
        self.chroma_client = chromadb.PersistentClient(path=Settings.RAG_DB_PATH)
        self.collection = self.chroma_client.get_collection(Settings.RAG_COLLECTION_NAME)
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        
        model_id = Settings.get_teacher_model_id()
        print(f"[{self.__class__.__name__}] Cargando LLM Maestro: {model_id}")
        # self.llm = pipeline(
        #     "text-generation",
        #     model=model_id,
        #     model_kwargs={"torch_dtype": "auto"},
        #     device_map="auto"
        # )
        self.llm = None  # Modo simulación
        self.refinements_history = []

    def process_request(self, student_prompt: dict) -> dict:
        """
        Recibe el JSON del Estudiante y devuelve el dataset semilla evaluado.
        """
        print(f"\n[{self.__class__.__name__}] Analizando prompt estructurado del estudiante...")
        
        # 1. Analiza el prompt y hace un plan de búsqueda
        query_text = self._plan_search(student_prompt)
        print(f"[{self.__class__.__name__}] Plan de búsqueda RAG generado: '{query_text}'")
        
        # 2. Consulta al RAG (Top K fijo = 200)
        rag_results = self._query_rag(query_text)
        
        # 3 y 4. Analizar los resultados (LLM razona) y verificar si es coherente
        filtered_seeds, quality_summary, needs_refinement = self._filter_and_reason(rag_results, student_prompt)
        
        # 5. Evalúa la calidad: si el número de ejemplos es insuficiente o falta diversidad
        if needs_refinement or len(filtered_seeds) < Settings.MIN_SEED_EXAMPLES:
            print(f"[{self.__class__.__name__}] Se determinó que los datos no son coherentes o son insuficientes (<{Settings.MIN_SEED_EXAMPLES}).")
            print(f"[{self.__class__.__name__}] Refinando la búsqueda (relajando criterios)...")
            filtered_seeds, quality_summary = self.refine_search(student_prompt)
            
            if len(filtered_seeds) == 0:
                print(f"[{self.__class__.__name__}] Sigue sin éxito tras refinar. El Maestro finaliza la ejecución sin datos.")
                return {"seed_dataset": []}
            
        # 6. Justifica su selección (entregando el resumen de calidad generado)
        print(f"[{self.__class__.__name__}] Selección justificada. Resumen de calidad: {quality_summary}")
        
        # 7. Entrega al Estudiante
        return {
            "seed_dataset": filtered_seeds,
            "quality_summary": quality_summary,
            "metadata_summary": {
                "count": len(filtered_seeds),
                "refinements_applied": self.refinements_history,
                "domain": student_prompt.get("values", {}).get("dominio")
            }
        }

    def _plan_search(self, student_prompt: dict) -> str:
        prompt = f"""Eres el Maestro, un agente experto con conocimiento interno de muchos tipos de series temporales.
Analiza el siguiente prompt estructurado enviado por el agente Estudiante:
{json.dumps(student_prompt)}

Con base en esta información, haz un plan de búsqueda corto y preciso para extraer información del RAG.
¿Qué palabras clave exactas enviarías al motor de búsqueda vectorial?
Devuelve ÚNICAMENTE el string de búsqueda.
"""
        # Simulación
        return student_prompt.get("suggested_search", "series temporales")

    def _query_rag(self, text: str) -> dict:
        print(f"[{self.__class__.__name__}] Consultando el RAG (Extrayendo {Settings.RAG_TOP_K} ítems fijos)...")
        embedding = self.embedder.encode(text).tolist()
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=Settings.RAG_TOP_K
        )

    def _filter_and_reason(self, rag_results, student_prompt):
        """
        El LLM lee los fragmentos devueltos por el RAG y decide si son útiles.
        No se descarta nada programáticamente; el LLM tiene la última palabra.
        """
        # Extraemos TODOS los metadatos recuperados (hasta 200) de forma compacta para el LLM.
        # Llama 3.2 3B tiene contexto suficiente para leerlos.
        metadata_compact = []
        for i, meta in enumerate(rag_results['metadatas'][0]):
            metadata_compact.append({
                "idx": i,
                "domain": meta.get("domain", ""),
                "length": len(eval(meta.get("time_series", "[]"))) if isinstance(meta.get("time_series", "[]"), str) else 0
            })
            
        prompt = f"""Eres el Maestro, un experto en series temporales.
Has ejecutado una consulta en el RAG para atender esta solicitud del Estudiante:
{json.dumps(student_prompt)}

Has recuperado {len(rag_results['ids'][0])} documentos. Resumen de metadatos (idx, domain, length):
{json.dumps(metadata_compact)}

Acciones:
1. Analiza los resultados: similitud semántica, dominio, longitud.
2. Aunque haya series cortas (<10 puntos), evalúa si tienen conocimiento valioso. TÚ decides si las descartas o no.
3. Evalúa la diversidad del conjunto seleccionado.

Devuelve tu razonamiento ÚNICAMENTE en el siguiente formato JSON válido, sin Markdown ni texto extra:
{{
    "accepted_indexes": [0, 2, 5], 
    "needs_refinement": false,
    "quality_summary": "Justifica tu selección aquí. Menciona si aprovechaste el conocimiento de series cortas y si hay diversidad."
}}
"""
        response = self._call_llm_json(prompt)
        
        # Filtramos basados SOLO en la decisión del LLM
        accepted_indexes = response.get("accepted_indexes", list(range(min(5, len(rag_results['ids'][0])))))
        filtered_seeds = []
        for idx in accepted_indexes:
            if idx < len(rag_results['ids'][0]):
                meta = rag_results['metadatas'][0][idx]
                ts_str = meta.get('time_series', '[]')
                try:
                    ts_data = eval(ts_str) if isinstance(ts_str, str) else ts_str
                except:
                    ts_data = []
                
                filtered_seeds.append({
                    "id": rag_results['ids'][0][idx],
                    "metadata": meta,
                    "series": ts_data
                })
                    
        return filtered_seeds, response.get("quality_summary", "Calidad moderada."), response.get("needs_refinement", False)

    def refine_search(self, student_prompt):
        refinement_action = "Se relajó la localización y se amplió el dominio de búsqueda."
        if refinement_action not in self.refinements_history:
            self.refinements_history.append(refinement_action)
            
        print(f"[{self.__class__.__name__}] Refinando: {refinement_action}")
        
        query_text = student_prompt.get("values", {}).get("dominio", "series temporales")
        rag_results = self._query_rag(query_text)
        
        filtered_seeds, quality_summary, _ = self._filter_and_reason(rag_results, student_prompt)
        return filtered_seeds, quality_summary

    # --- Métodos Auxiliares para LLM ---
    def _extract_json_from_text(self, text: str) -> dict:
        """Extrae de manera robusta un bloque JSON de la respuesta del LLM usando Expresiones Regulares."""
        try:
            # Buscar todo el contenido entre el primer '{' y el último '}'
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                # Si no hay corchetes, intentar forzar el parseo asumiendo que ya es json
                return json.loads(text)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error parseando JSON: {e}")
            return {}

    def _call_llm_json(self, prompt: str) -> dict:
        """Helper para invocar al LLM y extraer un JSON de forma robusta."""
        # --- Simulación temporal (reemplazar por self.llm(prompt) real) ---
        simulated_response = '''Claro, aquí tienes mi evaluación:
        ```json
        {
            "accepted_indexes": [0, 1, 2, 3],
            "needs_refinement": false,
            "quality_summary": "He revisado los resultados y seleccionado 4 series temporales coherentes. No descarté las series cortas ya que aportan información valiosa sobre el dominio."
        }
        ```'''
        return self._extract_json_from_text(simulated_response)
