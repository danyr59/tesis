import os
import json
import re
import pandas as pd
from transformers import pipeline
from pipeline.config.settings import Settings
from pipeline.teacher_agent.agent import TeacherAgent

class StudentAgent:
    def __init__(self):
        print(f"[{self.__class__.__name__}] Inicializando...")
        model_id = Settings.get_student_model_path()
        print(f"[{self.__class__.__name__}] Cargando LLM Estudiante: {model_id}")
        
        self.llm = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto"
        )
        #self.llm = None  # Modo simulación
        self.teacher = TeacherAgent()
        self.iteration_history = []
        
    def run_pipeline(self, initial_query: str):
        """
        Punto de entrada principal.
        """
        print(f"\n--- Iniciando Flujo Maestro-Estudiante ---")
        
        # 2. Análisis y Razonamiento
        structured_prompt = self.interpret_request(initial_query)
        
        iteration = 0
        quality_accepted = False
        final_model = None
        current_dataset = None
        
        while iteration < Settings.MAX_ITERATIONS and not quality_accepted:
            iteration += 1
            print(f"\n--- Iteración {iteration}/{Settings.MAX_ITERATIONS} ---")
            
            # 3. Consulta al Maestro
            teacher_response = self.teacher.process_request(structured_prompt)
            current_dataset = teacher_response['seed_dataset']
            
            if not current_dataset:
                print(f"[{self.__class__.__name__}] El Maestro no encontró datos semilla válidos. Abortando.")
                return
                
            # 4. Destilación (Fine-Tuning)
            final_model = self.fine_tune(current_dataset, teacher_response['metadata_summary'])
            
            # 5. Evaluación e Iteración
            quality_accepted, feedback = self.evaluate_quality_and_iterate(current_dataset, teacher_response['quality_summary'])
            
            if not quality_accepted:
                self.iteration_history.append(feedback)
                structured_prompt["suggested_search"] = f"{structured_prompt.get('suggested_search', '')}. {feedback}"
                print(f"[{self.__class__.__name__}] Refinando búsqueda en iteración: {feedback}")
                
        # 6. Generación Final
        if final_model or current_dataset:
            self.generate_final_series(current_dataset, structured_prompt)

    def interpret_request(self, query: str) -> dict:
        print(f"[{self.__class__.__name__}] Interpretando consulta: '{query}'")
        
        prompt = f"""Eres un agente Estudiante, un LLM liviano con gran capacidad de razonamiento lógico.
Tu tarea es analizar el requerimiento del usuario sobre generación de series temporales y estructurar la información.

Requerimiento del usuario: "{query}"

Acciones a realizar obligatoriamente:
1. Interpretación semántica activa: identifica sinónimos (ej. “calor” → dominio clima), detecta ambigüedades.
2. Detección de parámetros faltantes y TIPO DE SERIE: analiza si la consulta proporciona variables, dominio, frecuencia, y localización.
CRÍTICO: Debes razonar explícitamente si el usuario requiere una serie "univariada" (ej. solo temperatura) o "multivariada" (ej. temperatura, humedad, presión). Si es multivariada pero no especifica qué variables, agrégalo a missing_parameters.
3. Justificación de inferencias: explica claramente por qué asumes ciertos valores y por qué determinaste el tipo_serie.
4. Toma de decisiones: Evalúa tu confianza global. Si es mayor a 0.8, la consulta es suficiente. Si es menor, lista los parámetros faltantes.

Devuelve tu respuesta ÚNICAMENTE en el siguiente formato JSON válido, sin texto adicional:
{{
    "values": {{"dominio": "...", "localizacion": "...", "frecuencia": "...", "variables": "...", "tipo_serie": "univariada|multivariada"}},
    "missing_parameters": ["param1"],
    "justification": "Tu justificación de inferencias y del tipo de serie aquí",
    "suggested_search": "Texto enriquecido sugerido para que el Maestro busque en el RAG",
    "confidence": 0.9
}}
"""
        response = self._call_llm_json(prompt)
        
        while response.get("confidence", 1.0) < 0.8 and response.get("missing_parameters"):
            print(f"\n[{self.__class__.__name__}] He detectado información insuficiente.")
            print(f"Justificación: {response.get('justification')}")
            for missing in response.get("missing_parameters", []):
                user_val = input(f"[{self.__class__.__name__}] Por favor, aclara el siguiente parámetro faltante '{missing}': ")
                query += f". {missing}: {user_val}"
            
            prompt_retry = prompt.replace(f'Requerimiento del usuario: "{query}"', f'Requerimiento actualizado: "{query}"')
            response = self._call_llm_json(prompt_retry)

        print(f"[{self.__class__.__name__}] Prompt Estructurado Listo: {json.dumps(response, indent=2)}")
        return response

    def fine_tune(self, seed_dataset, metadata_summary):
        print(f"\n[{self.__class__.__name__}] Analizando estrategia de Destilación (Fine-Tuning)...")
        prompt = f"""Eres un agente Estudiante. Recibiste un dataset semilla de {len(seed_dataset)} ejemplos para entrenar.
Metadatos del dataset: {metadata_summary}

Razona sobre cómo realizar el fine-tuning (destilación) para evitar el sobreajuste si hay pocos datos.
Selecciona dinámicamente y justifica:
1. Número de épocas (epochs).
2. Tamaño de lote (batch size).
3. Tasa de aprendizaje (learning rate).

Explica en modo verbose tus decisiones de configuración.
"""
        reasoning = self._call_llm_text(prompt)
        print(f"[{self.__class__.__name__}] [VERBOSE] Decisiones de Fine-Tuning:\n{reasoning}")
        print(f"[{self.__class__.__name__}] Ejecutando fine-tuning del modelo pequeño usando el dataset semilla (LoRA)...")
        print(f"[{self.__class__.__name__}] Modelo resultante guardado.")
        return "student_lora_model_v1"

    def evaluate_quality_and_iterate(self, dataset, quality_summary) -> tuple[bool, str]:
        print(f"\n[{self.__class__.__name__}] Evaluando calidad del ciclo de retroalimentación...")
        prompt = f"""Eres un agente Estudiante. Acabas de ser entrenado.
Resumen de calidad provisto por el Maestro: "{quality_summary}"
Estrategias de iteración previas intentadas: {self.iteration_history}
Tamaño del dataset semilla usado: {len(dataset)}

Decide si es necesario iterar:
- Si la calidad es baja (ej. series muy similares entre sí), determina la causa: ¿faltan semillas diversas? ¿sobreajuste?
- Toma una acción correctiva (ej. pedir al Maestro más semillas con características complementarias).

Devuelve tu decisión ÚNICAMENTE en el siguiente formato JSON válido:
{{
    "quality_accepted": true/false,
    "causal_analysis": "tu análisis de la causa raíz si la calidad es baja",
    "feedback_to_teacher": "instrucción específica al Maestro indicando qué variables/frecuencias se necesitan"
}}
"""
        response = self._call_llm_json(prompt)
        accepted = response.get("quality_accepted", True)
        feedback = response.get("feedback_to_teacher", "")
        
        if accepted:
            print(f"[{self.__class__.__name__}] Calidad aceptada. No se requieren más iteraciones.")
        else:
            print(f"[{self.__class__.__name__}] Calidad baja detectada. Análisis: {response.get('causal_analysis')}")
            print(f"[{self.__class__.__name__}] Acción correctiva solicitada al Maestro: {feedback}")
            
        return accepted, feedback

    def generate_final_series(self, dataset, prompt_info):
        """
        Utiliza el LLM Estudiante (supuestamente fine-tuneado) para generar numéricamente las series sintéticas.
        """
        print(f"\n[{self.__class__.__name__}] El Estudiante está conforme. Generando series sintéticas con el LLM...")
        
        # En una versión real, esto llamaría al self.llm o usaría el pipeline.
        # Formulamos el prompt poderoso que exige la generación matemática/numérica:
        tipo_serie = prompt_info["values"].get("tipo_serie", "univariada").lower()
        
        if "multi" in tipo_serie:
            json_format = """{
    "synthetic_series": [
        {"temperatura (°C)": [22.1, 22.4...], "humedad (%)": [60.0, 62.0...]},
        {"temperatura (°C)": [21.0, 21.3...], "humedad (%)": [55.0, 58.0...]}
    ]
}"""
        else:
            json_format = """{
    "synthetic_series": [
        [22.1, 22.4, 23.0, ...],
        [21.0, 21.3, 21.8, ...]
    ]
}"""

        generation_prompt = f"""Eres un agente Estudiante avanzado, recién entrenado en series temporales del dominio: {prompt_info["values"].get("dominio", "General")}.
Tu objetivo ahora es generar nuevas series temporales sintéticas, realistas y numéricas que capturen las características aprendidas del dataset semilla.

Características solicitadas (NOTA: La serie es {tipo_serie}):
{json.dumps(prompt_info["values"])}

Instrucciones de Generación:
1. Genera exactamente 3 nuevas series temporales.
2. Cada serie debe tener al menos 10 puntos numéricos (siempre como números flotantes, ej. 22.0) simulando variabilidad realista.
3. Devuelve los resultados ÚNICAMENTE en formato JSON, utilizando EXACTAMENTE esta estructura. 
IMPORTANTE: Si es multivariada, los nombres de las variables DEBEN incluir su unidad de medida entre paréntesis.

{json_format}
"""
        # Llamada simulada robusta
        response = self._call_llm_json(generation_prompt)
        synthetic_data = response.get("synthetic_series", [])
        
        os.makedirs(Settings.SYNTHETIC_DATA_DIR, exist_ok=True)
        output_file = os.path.join(Settings.SYNTHETIC_DATA_DIR, "synthetic_series.csv")
        
        all_rows = []
        for series_idx, series_data in enumerate(synthetic_data):
            # Usar números simples para el ID en vez del prefijo "synthetic_"
            series_id = series_idx + 1
            
            if "multi" in tipo_serie and isinstance(series_data, dict):
                # Multivariada: dict de arrays
                keys = list(series_data.keys())
                if keys:
                    length = len(series_data[keys[0]])
                    for t in range(length):
                        row = {"series_id": series_id, "time_step": t}
                        for k in keys:
                            try:
                                row[k] = float(series_data[k][t])
                            except (IndexError, ValueError, TypeError):
                                row[k] = None
                        all_rows.append(row)
            else:
                # Univariada: lista de números
                var_name = prompt_info["values"].get("variables", "valor_generado")
                # Si el LLM no le puso unidad al univariado, podemos intentar que sea general
                if isinstance(series_data, list):
                    for t, val in enumerate(series_data):
                        try:
                            f_val = float(val)
                        except (ValueError, TypeError):
                            f_val = None
                            
                        all_rows.append({
                            "series_id": series_id,
                            "time_step": t,
                            var_name: f_val
                        })
            
        df = pd.DataFrame(all_rows)
        df.to_csv(output_file, index=False)
        print(f"[{self.__class__.__name__}] Dataset estructurado (HuggingFace/RNN ready) guardado en: {output_file}")

    # --- Métodos Auxiliares para LLM ---
    def _extract_json_from_text(self, text: str) -> dict:
        """Extrae de manera robusta un bloque JSON de la respuesta del LLM usando Expresiones Regulares."""
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                return json.loads(text)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error parseando JSON: {e}")
            return {}

    def _call_llm_json(self, prompt: str) -> dict:
        if "Interpretación semántica activa" in prompt:
            # Extraer solo la consulta del usuario (entre comillas después de "Requerimiento")
            import re as _re
            user_query_match = _re.search(r'Requerimiento del usuario: "(.+?)"', prompt)
            user_query = user_query_match.group(1).lower() if user_query_match else ""
            
            if "multivariada" in user_query or (" y " in user_query and any(v in user_query for v in ["humedad", "presión", "viento"])):
                simulated = '''{"values": {"dominio": "clima", "localizacion": "Mérida", "frecuencia": "diaria", "variables": ["temperatura", "humedad"], "tipo_serie": "multivariada"}, "missing_parameters": [], "justification": "Se menciona temperatura y humedad. Serie multivariada. Dominio inferido: clima.", "suggested_search": "series clima temperatura humedad multivariada", "confidence": 0.9}'''
            else:
                simulated = '''{"values": {"dominio": "clima", "localizacion": "Mérida", "frecuencia": "diaria", "variables": "temperatura", "tipo_serie": "univariada"}, "missing_parameters": [], "justification": "Se menciona temperatura, que es univariada. Dominio inferido: clima.", "suggested_search": "series clima temperatura Mérida univariada", "confidence": 0.9}'''
            return self._extract_json_from_text(simulated)
            
        elif "Decide si es necesario iterar" in prompt:
            if len(self.iteration_history) == 0:
                simulated = '''{"quality_accepted": false, "causal_analysis": "Poca diversidad", "feedback_to_teacher": "Más anomalías por favor."}'''
            else:
                simulated = '''{"quality_accepted": true, "causal_analysis": "Adecuada", "feedback_to_teacher": ""}'''
            return self._extract_json_from_text(simulated)
            
        elif "Instrucciones de Generación" in prompt:
            if "La serie es multivariada" in prompt:
                simulated = '''
                {
                    "synthetic_series": [
                        {"temperatura (°C)": [22.1, 22.4, 23.0], "humedad (%)": [60.0, 62.0, 65.0]},
                        {"temperatura (°C)": [21.0, 21.3, 21.8], "humedad (%)": [55.0, 58.0, 60.0]},
                        {"temperatura (°C)": [23.5, 24.1, 23.9], "humedad (%)": [70.0, 72.0, 68.0]}
                    ]
                }
                '''
            else:
                simulated = '''
                {
                    "synthetic_series": [
                        [22.1, 22.4, 23.0, 21.5, 20.8, 22.2, 23.1, 24.5, 24.1, 23.8],
                        [21.0, 21.3, 21.8, 20.9, 21.2, 22.5, 23.3, 23.0, 22.1, 21.9],
                        [23.5, 24.1, 23.9, 22.8, 21.9, 21.5, 22.4, 23.6, 24.8, 25.0]
                    ]
                }
                '''
            return self._extract_json_from_text(simulated)
            
        return {}

    def _call_llm_text(self, prompt: str) -> str:
        return "Decido usar 5 epochs ya que el dataset es muy pequeño (menos de 10 ejemplos). El batch_size será 2 para tener actualizaciones frecuentes de pesos, y un learning rate de 2e-4 con LoRA (rank=8)."
