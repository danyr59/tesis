import gc
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from pipeline.config.settings import Settings
from pipeline.numeric_pipeline import run_numeric_pipeline
from pipeline.student_agent.fine_tuner import run_lora_fine_tune
from pipeline.teacher_agent.agent import TeacherAgent
from pipeline.utils.llm_client import LLMClient, extract_json_from_text
from pipeline.utils.series_utils import (
    extend_series,
    extract_numeric_series,
    normalize_frequency,
    seeds_statistics,
)
from pipeline.config.profiles import get_profile


class StudentAgent:
    def __init__(self):
        print(f"[{self.__class__.__name__}] Inicializando...")
        self.llm = LLMClient(
            Settings.get_student_model_path(),
            role_name="Estudiante (agente liviano con razonamiento)",
        )
        self.teacher = None
        self.iteration_history: List[Dict[str, Any]] = []
        self.user_query: str = ""
        self.best_snapshot: Optional[Dict[str, Any]] = None
        self.best_score: float = -1.0
        self.lora_adapter_path: Optional[str] = None
        self.num_series_target = int(os.getenv("SYNTHETIC_NUM_SERIES", str(Settings.SYNTHETIC_NUM_SERIES)))
        self.points_per_series_target = int(os.getenv("SYNTHETIC_POINTS_PER_SERIES", str(Settings.SYNTHETIC_POINTS_PER_SERIES)))

    @property
    def is_reasoning_model(self) -> bool:
        path = getattr(self.llm, "model_path", "") or ""
        return any(tag in str(path).lower() for tag in ["deepseek", "r1"])

    def run_generation(self, structured_prompt: dict, teacher_response: dict) -> None:
        self.num_series_target = int(os.getenv("SYNTHETIC_NUM_SERIES", str(Settings.SYNTHETIC_NUM_SERIES)))
        self.points_per_series_target = int(os.getenv("SYNTHETIC_POINTS_PER_SERIES", str(Settings.SYNTHETIC_POINTS_PER_SERIES)))

        print(f"\n--- Iniciando Flujo Estudiante ---")
        print(
            f"[{self.__class__.__name__}] Generación sintética (fija): "
            f"{Settings.synthetic_generation_summary()}"
        )

        iteration = 0
        final_model_path: Optional[str] = None
        current_dataset: Optional[List[dict]] = None
        structured_prompt_active = structured_prompt

        print(f"\n--- Iteración {iteration}/{Settings.MAX_ITERATIONS} ---")

        current_dataset = teacher_response.get("seed_dataset", [])

        if not current_dataset:
            print(
                f"[{self.__class__.__name__}] El Maestro no encontró datos semilla. "
                    "Finalizando ejecución."
                )
            return

        self._update_best_snapshot(
            current_dataset, teacher_response, final_model_path, structured_prompt_active
        )

        final_model_path = self.fine_tune(
            current_dataset, teacher_response.get(
                    "metadata_summary", {}), iteration
        )

        csv_path = self.generate_final_series(
            current_dataset,
            structured_prompt_active,
            final_model_path,
        )

        domain = structured_prompt_active.get(
            "values", {}).get("dominio", "general")
        
        run_numeric_pipeline(
            csv_path=csv_path,
            seed_dataset=current_dataset,
            domain=domain,
        )

    def interpret_request(self, query: str) -> dict:
        print(f"[{self.__class__.__name__}] Interpretando: '{query}'")
        current_query = query
        dialog_rounds = 0
        max_dialog_rounds = 8

        while dialog_rounds < max_dialog_rounds:
            dialog_rounds += 1
            prompt = self._interpretation_prompt(current_query)
            response = self.llm.generate_json(prompt)

            # Si el LLM omite 'confidence' pero no reporta 'missing_parameters',
            # asumimos que está confiado (1.0). Si reporta missing, asumimos 0.0.
            missing = response.get("missing_parameters") or []
            default_conf = 1.0 if not missing else 0.0
            confidence = float(response.get("confidence", default_conf))

            if confidence >= Settings.MIN_CONFIDENCE_THRESHOLD and not missing:
                break

            if not missing and confidence < Settings.MIN_CONFIDENCE_THRESHOLD:
                missing = ["detalles adicionales del requerimiento"]

            print(
                f"\n[{self.__class__.__name__}] Información insuficiente (confianza={confidence:.2f}).")
            print(f"Justificación: {response.get('justification', '')}")
            
            query_changed = False


    def fine_tune(self, seed_dataset: List[dict], metadata_summary: dict,  iteration: int) -> str:
        print(
            f"\n[{self.__class__.__name__}] Destilación (LoRA) con {len(seed_dataset)} semillas...")
        prompt = f"""Eres el Estudiante. Dataset semilla: {len(seed_dataset)} ejemplos.
Metadatos: {json.dumps(metadata_summary, ensure_ascii=False)}

Selecciona los valores adecuados y justifica en modo verbose:
- epochs (evitar sobreajuste si hay pocos datos)
- batch_size
- learning_rate

RECUERDA PONER LOS VALORES

"""
        ft_tokens = 1500 if self.is_reasoning_model else 1024
        reasoning = self.llm.generate_text(prompt, max_new_tokens=ft_tokens)
        
        prompt_json = f"""Vas a analizar el siguiente razonamiento:
        {reasoning} 
        luego vas a hacer lo siguiente:
        1) Seleccionar los valores del analisis
        2) construir un json con esos valores de la siguiente forma:
        Devuelve ÚNICAMENTE JSON:
        {{"epochs": 3, "batch_size": 2, "learning_rate": 0.0002}}
        Nota: SOLO COMO RESPUESTA DEBES ENTREGAR EL JSON, SEGUIR EL FORMATO, sin cosas adicionales

""" 
        values_json = self.llm.generate_text(prompt_json, max_new_tokens=ft_tokens)

        values = extract_json_from_text(values_json)


        print(f"[{self.__class__.__name__}] [VERBOSE] Fine-tuning:\n{reasoning}")
        # Usamos el modelo base (limpio) que nos da el LLMClient
        base_model = self.llm.get_base_model()
        adapter_path = run_lora_fine_tune(
            base_model=base_model,
            tokenizer=self.llm.tokenizer,
            seed_dataset=seed_dataset,
            metadata_summary=metadata_summary,
            reasoning_text=reasoning,
            continue_from_adapter=self.lora_adapter_path if iteration > 1 else None,
            hyperparams_json=values,
        )
        self.lora_adapter_path = adapter_path

        # Ahora cargamos el adaptador en el LLMClient (sustituye el modelo activo)
        self.llm.load_adapter(adapter_path)

        # adapter_path = run_lora_fine_tune(
        #     self.llm.model, self.llm.tokenizer, seed_dataset, metadata_summary, reasoning)
        # self.lora_adapter_path = adapter_path

        # try:
        #  self.llm.load_adapter(adapter_path)
        # except Exception as exc:
        #  print(f"[{self.__class__.__name__}] Aviso: no se recargó adapter en memoria: {exc}")

        print(f"[{self.__class__.__name__}] Modelo LoRA guardado en {adapter_path}")
        return adapter_path

    def evaluate_quality_and_iterate(
        self,
        dataset: List[dict],
        quality_summary: str,
        model_path: str,
    ) -> Tuple[bool, str, dict]:
        print(f"\n[{self.__class__.__name__}] Evaluación de retroalimentación...")
        prompt = f"""Eres el Estudiante recién entrenado con LoRA.
Resumen del Maestro: "{quality_summary}"
Estrategias ya intentadas (no repetir): {json.dumps(self.iteration_history, ensure_ascii=False)}
Semillas: {len(dataset)}

Decide si iterar. Analiza: diversidad, sobreajuste, épocas insuficientes.
Si falta una variable o frecuencia, indica feedback_to_teacher específico.

JSON únicamente:
{{
  "quality_accepted": true,
  "causal_analysis": "...",
  "feedback_to_teacher": "...",
  "corrective_strategy": "ampliar_dominio|cambiar_frecuencia|mas_semillas|re_train",
  "re_train_with_params": {{"epochs": 5, "batch_size": 2, "learning_rate": 0.0001}}
}}
"""
        response = self.llm.generate_json(prompt)
        accepted = bool(response.get("quality_accepted", False))
        feedback = response.get("feedback_to_teacher", "")
        payload = {
            "feedback_to_teacher": feedback,
            "causal_analysis": response.get("causal_analysis", ""),
            "corrective_strategy": response.get("corrective_strategy", ""),
            "re_train_with_params": response.get("re_train_with_params"),
            "attempted_strategies": list(self.iteration_history),
        }

        if accepted:
            print(f"[{self.__class__.__name__}] Calidad aceptada.")
        else:
            print(
                f"[{self.__class__.__name__}] Calidad insuficiente: {payload['causal_analysis']}")

        return accepted, feedback, payload

    def generate_final_series(
        self,
        dataset: List[dict],
        prompt_info: dict,
        model_path: Optional[str],
    ) -> str:
        print(f"\n[{self.__class__.__name__}] Generación sintética final...")
        tipo_serie = str(prompt_info.get("values", {}).get(
            "tipo_serie", "univariada")).lower()
        stats = seeds_statistics(dataset)
        seed_preview = []
        for s in dataset[:5]:
            nums = extract_numeric_series(s)
            seed_preview.append(nums)

        all_series: List[Any] = []
        remaining = self.num_series_target
        series_offset = 0

        while remaining > 0:
            chunk_n = min(Settings.SYNTHETIC_CHUNK_SERIES, remaining)
            chunk_points = min(Settings.SYNTHETIC_CHUNK_POINTS,
                               self.points_per_series_target)
            chunk = self._generate_chunk(
                prompt_info, tipo_serie, chunk_n, chunk_points, stats, seed_preview, series_offset
            )
            all_series.extend(chunk)
            remaining -= len(chunk)
            series_offset += len(chunk)

        all_series = self._normalize_series_length(all_series, tipo_serie, prompt_info)
        output_file = self._series_to_csv(all_series, prompt_info, tipo_serie)
        print(f"[{self.__class__.__name__}] CSV guardado: {output_file}")
        print(
            f"[{self.__class__.__name__}] Total: {len(all_series)} series, "
            f"~{sum(self._series_len(s, tipo_serie) for s in all_series)} puntos"
        )
        return output_file

    def _generate_chunk(
        self,
        prompt_info: dict,
        tipo_serie: str,
        num_series: int,
        num_points: int,
        stats: dict,
        seed_preview: list,
        series_offset: int,
    ) -> List[Any]:
        multivariate = "multi" in tipo_serie
        
        # Obtener el perfil asignado por el interprete (o fallback)
        profile_id = prompt_info.get("values", {}).get("profile_id", "P1_ESTABLE")
        profile = get_profile(profile_id)

        # Extraer el valor objetivo / magnitud deseada por el usuario (Desacoplamiento de Escala)
        target_value = prompt_info.get("values", {}).get("target_value")
        if target_value is None:
            for mp in prompt_info.get("missing_parameters", []):
                if isinstance(mp, dict) and "expected_value" in mp:
                    try:
                        target_value = float(mp["expected_value"])
                        break
                    except (ValueError, TypeError):
                        pass

        scale_guideline = ""
        if target_value is not None:
            scale_guideline = f"""
[ESCALA OBJETIVO PRIORITARIA DEL USUARIO: {target_value}]
- REGLA ESTRICTA DE DESACOPLAMIENTO DE FORMA Y ESCALA:
  Las semillas del RAG sirven ÚNICAMENTE como referencia para la dinámica temporal (tipo de variaciones, autocorrelación y ruido).
  Tus 10 valores DEBEN generarse estrictamente en la escala del usuario (alrededor de {target_value}).
  NUNCA copies magnitudes numéricas del RAG que difieran de {target_value}.
"""
        
        # Construir ejemplos EPIC estáticos para guiar el comportamiento
        epic_examples = f"""
### EJEMPLOS DE COMPORTAMIENTO (EPIC)
- Ejemplo de Tendencia Creciente: [100.1, 105.3, 110.2, 115.8, 122.4]
- Ejemplo de Estacionalidad Semanal (Sube fines de semana): [50, 52, 51, 53, 55, 80, 85, 52, 51]
- Ejemplo de Ruido Medio: [10.1, 8.5, 12.3, 9.8, 11.2, 10.5]
- Ejemplo de Anomalía (Pico): [20.1, 20.5, 21.0, 95.8, 20.9, 21.1]

NO COPIES ESTOS EJEMPLOS. Úsalos como inspiración para el comportamiento de la curva.
"""

        if multivariate:
            variables = prompt_info.get("values", {}).get("variables", "var1, var2")
            example = '{"synthetic_series": [{"var1": [1.2, 1.3, 1.4], "var2": [5.1, 5.2, 5.0]}]}'
        else:
            example = '{"synthetic_series": [[1.2, 1.3, 1.4]]}'

        prompt = f"""Eres un motor experto en la generación de series temporales de alta fidelidad. 
Tu tarea es generar los primeros 10 valores (semilla) de una serie temporal basándote en el perfil solicitado y el dominio del RAG. El sistema se encargará de extrapolar el resto matemáticamente.

[PERFIL TEMPORAL OBJETIVO: {profile_id}]
- Descripción: {profile['description']}
- Longitud exacta esperada: SOLO DEBES GENERAR 10 OBSERVACIONES por serie (Semilla inicial).
- Cantidad de series a generar: {num_series} series.
- Rango de series a generar: Serie #{series_offset + 1} a #{series_offset + num_series}.
- Tendencia global: {profile['trend_expected']}
- Estacionalidad: {profile['seasonality_expected']}
- Anomalías esperadas: {profile['expected_anomalies']}
- Nivel de ruido: {profile['noise_expected']}

{scale_guideline}
[DOMINIO (Extraído de RAG)]
Tema: {prompt_info.get('values', {}).get('tema', 'General')}
Estadísticas de referencia (Semillas): Min={stats.get('min', 0):.2f}, Max={stats.get('max', 100):.2f}, Media={stats.get('mean', 50):.2f}
Muestra de semillas: {json.dumps(seed_preview)[:500]}

{epic_examples}

[REGLAS DE FORMATO ESTRICTAS - VALUE ONLY]
1. Genera ÚNICAMENTE un JSON válido que contenga listas de números flotantes.
2. Cada lista DEBE tener exactamente 10 números.
3. NO envuelvas la respuesta en bloques de código markdown (```json).
4. NO escribas texto introductorio, explicaciones, ni comentarios. 
5. Si es univariada, devuelve las series como un arreglo plano. Si es multivariada, como un diccionario plano de arreglos.

Devuelve ÚNICAMENTE el JSON con esta estructura exacta (ESTO ES UN EJEMPLO, RECUERDA DEBES GENERAR 10 VALORES POR SERIE):
{example}
"""
        # Modelos con cadena de pensamiento (DeepSeek-R1) razonan antes de volcar el JSON,
        # por lo que requieren mayor margen de tokens (1500) para no truncarse.
        chunk_max_tokens = 1500 if self.is_reasoning_model else 500
        response = self.llm.generate_json(prompt, max_new_tokens=chunk_max_tokens)
        series = response.get("synthetic_series", [])
        
        if not series:
            # Reintento aditivo: si el modelo respondió con explicaciones en prosa sin números
            retry_prompt = (
                f"{prompt}\n\n"
                f"[RECORDATORIO DE FORMATO: ENTREGA ÚNICAMENTE EL JSON]\n"
                f"NO des explicaciones ni texto conversacional.\n"
                f"Comienza directamente con la estructura JSON solicitada:\n"
                f"{example}"
            )
            response = self.llm.generate_json(retry_prompt, max_new_tokens=chunk_max_tokens, temperature=0.2)
            series = response.get("synthetic_series", [])
        
        if series:
            # Extrapolar las semillas generadas por el LLM a su longitud final
            from pipeline.utils.series_utils import extend_series
            extended_series = []
            
            if multivariate:
                for idx, s_item in enumerate(series):
                    s_seed = 42 + series_offset + idx
                    s_rng = np.random.default_rng(s_seed)
                    if isinstance(s_item, dict):
                        ext_dict = {}
                        for var, vals in s_item.items():
                            t_mean = target_value.get(var) if isinstance(target_value, dict) else None
                            ext_dict[var] = extend_series(vals, num_points, rng=s_rng, profile_id=profile_id, target_mean=t_mean, series_seed=s_seed)
                        extended_series.append(ext_dict)
                    elif isinstance(s_item, list):
                        v_name = prompt_info.get("values", {}).get("variables", ["value"])
                        v_key = str(v_name[0]) if isinstance(v_name, list) and v_name else str(v_name)
                        t_mean = target_value.get(v_key) if isinstance(target_value, dict) else None
                        extended_series.append({
                            v_key: extend_series(s_item, num_points, rng=s_rng, profile_id=profile_id, target_mean=t_mean, series_seed=s_seed)
                        })
            else:
                for idx, s_arr in enumerate(series):
                    s_seed = 42 + series_offset + idx
                    s_rng = np.random.default_rng(s_seed)
                    if isinstance(s_arr, list):
                        extended_series.append(extend_series(s_arr, num_points, rng=s_rng, profile_id=profile_id, target_mean=target_value, series_seed=s_seed))
                    elif isinstance(s_arr, dict):
                        vals = next(iter(s_arr.values())) if s_arr else []
                        extended_series.append(extend_series(vals, num_points, rng=s_rng, profile_id=profile_id, target_mean=target_value, series_seed=s_seed))
            series = extended_series

        if not series:
            series = self._fallback_series(
                num_series, num_points, stats, multivariate, prompt_info, series_seed=42 + series_offset)
        return series

    def _fallback_series(
        self,
        n: int,
        points: int,
        stats: dict,
        multivariate: bool,
        prompt_info: dict,
        series_seed: Optional[int] = None,
    ) -> List[Any]:
        target_value = prompt_info.get("values", {}).get("target_value")
        mean = float(target_value) if target_value is not None else stats.get("mean", 50.0)
        std = stats.get("std", 1.0)
        out = []
        for i in range(n):
            s_seed = (series_seed if series_seed is not None else 42) + i
            s_rng = np.random.default_rng(s_seed)
            base = float(s_rng.normal(mean, std))
            profile_id = prompt_info.get("values", {}).get("profile_id", "P1_ESTABLE")
            seq = extend_series(
                [base + float(s_rng.normal(0, std * 0.1))
                 for _ in range(min(20, points))],
                points,
                rng=s_rng,
                profile_id=profile_id,
                target_mean=target_value,
                series_seed=s_seed,
            )
            if multivariate:
                vars_ = prompt_info.get("values", {}).get(
                    "variables", ["v1", "v2"])
                if isinstance(vars_, str):
                    vars_ = [vars_, "humedad (%)"]
                out.append(
                    {
                        f"{vars_[0]} (°C)" if "(" not in str(vars_[0]) else str(vars_[0]): seq,
                        "humedad (%)": extend_series(
                            [50.0 + float(s_rng.normal(0, 3))
                             for _ in range(min(20, points))], points, rng=s_rng, profile_id=profile_id, target_mean=target_value, series_seed=s_seed
                        ),
                    }
                )
            else:
                var = prompt_info.get("values", {}).get("variables", "valor")
                out.append(seq)
        return out

    def _normalize_series_length(self, series_list: List[Any], tipo_serie: str, prompt_info: dict) -> List[Any]:
        target = self.points_per_series_target
        normalized = []
        profile_id = prompt_info.get("values", {}).get("profile_id", "P1_ESTABLE")
        target_value = prompt_info.get("values", {}).get("target_value")
        for idx, s in enumerate(series_list[: self.num_series_target]):
            s_seed = 42 + idx
            s_rng = np.random.default_rng(s_seed)
            if "multi" in tipo_serie and isinstance(s, dict):
                keys = list(s.keys())
                if not keys:
                    continue
                lengths = [len(s[k])
                           for k in keys if isinstance(s.get(k), list)]
                cur_len = min(lengths) if lengths else 0
                new_dict = {}
                for k in keys:
                    vals = [float(x) for x in s[k][:cur_len]] if isinstance(
                        s.get(k), list) else []
                    t_mean = target_value.get(k) if isinstance(target_value, dict) else None
                    new_dict[k] = extend_series(vals, target, rng=s_rng, profile_id=profile_id, target_mean=t_mean, series_seed=s_seed)
                normalized.append(new_dict)
            elif isinstance(s, list):
                vals = [float(x) for x in s]
                normalized.append(extend_series(vals, target, rng=s_rng, profile_id=profile_id, target_mean=target_value, series_seed=s_seed))
        while len(normalized) < self.num_series_target:
            idx = len(normalized)
            fallback_seed = 42 + idx
            normalized.extend(
                self._fallback_series(
                    1,
                    target,
                    seeds_statistics([]) if not normalized else {
                        "mean": target_value if target_value is not None else 0, "std": 1, "min": 0, "max": 1},
                    "multi" in tipo_serie,
                    prompt_info,
                    series_seed=fallback_seed,
                )
            )
        return normalized[: self.num_series_target]

    def _series_len(self, s: Any, tipo_serie: str) -> int:
        if "multi" in tipo_serie and isinstance(s, dict) and s:
            k = next(iter(s))
            return len(s[k]) if isinstance(s.get(k), list) else 0
        return len(s) if isinstance(s, list) else 0

    def _series_to_csv(self, series_list: List[Any], prompt_info: dict, tipo_serie: str) -> str:
        os.makedirs(Settings.SYNTHETIC_DATA_DIR, exist_ok=True)
        output_file = Settings.synthetic_csv_path()
        rows = []
        
        start_date = prompt_info.get("values", {}).get("start_date", "2024-01-01")
        freq_raw = prompt_info.get("values", {}).get("frecuencia", "D")
        freq = normalize_frequency(freq_raw)
        
        max_len = max([self._series_len(s, tipo_serie) for s in series_list] + [1])
        try:
            date_range = pd.date_range(start=start_date, periods=max_len, freq=freq)
        except Exception:
            try:
                date_range = pd.date_range(start=start_date, periods=max_len, freq="D")
            except Exception:
                date_range = pd.date_range(start="2024-01-01", periods=max_len, freq="D")

        for series_idx, series_data in enumerate(series_list):
            series_id = series_idx + 1
            if "multi" in tipo_serie and isinstance(series_data, dict):
                keys = list(series_data.keys())
                if keys:
                    length = len(series_data[keys[0]])
                    for t in range(length):
                        timestamp = date_range[t].strftime('%Y-%m-%d %H:%M:%S')
                        row = {"series_id": series_id, "timestamp": timestamp}
                        for k in keys:
                            try:
                                row[k] = float(series_data[k][t])
                            except (IndexError, ValueError, TypeError):
                                row[k] = None
                        rows.append(row)
            elif isinstance(series_data, list):
                var_name = prompt_info.get("values", {}).get(
                    "variables", "valor_generado")
                if isinstance(var_name, list):
                    # Filtrar palabras que representen tiempo para quedarnos con la variable de valor real
                    filtro_tiempo = ["fecha", "mes", "anio", "año", "date", "time", "year", "month", "day", "timestamp"]
                    vars_validas = [v for v in var_name if str(v).lower() not in filtro_tiempo]
                    var_name = vars_validas[0] if vars_validas else var_name[-1]
                
                if "(" not in str(var_name) and "temperatura" in str(var_name).lower():
                    var_name = f"{var_name} (°C)"
                for t, val in enumerate(series_data):
                    timestamp = date_range[t].strftime('%Y-%m-%d %H:%M:%S')
                    try:
                        f_val = float(val)
                    except (ValueError, TypeError):
                        f_val = None
                    rows.append({"series_id": series_id,
                                "timestamp": timestamp, str(var_name): f_val})

        df_out = pd.DataFrame(rows)
        # Limpieza defensiva de NaNs / Infs generados por modelos pequeños o variables dispares
        num_cols = [c for c in df_out.columns if c not in ("series_id", "timestamp")]
        if num_cols:
            df_out[num_cols] = df_out[num_cols].replace([np.inf, -np.inf], np.nan)
            df_out[num_cols] = df_out[num_cols].ffill().bfill().fillna(0.0)
        df_out.to_csv(output_file, index=False)
        return output_file

    def _update_best_snapshot(
        self,
        dataset: List[dict],
        teacher_response: dict,
        model_path: Optional[str],
        structured_prompt: dict,
    ) -> None:
        meta = teacher_response.get("metadata_summary", {})
        score = float(meta.get("diversity_score", 0.5)) * len(dataset)
        if score > self.best_score:
            self.best_score = score
            self.best_snapshot = {
                "dataset": dataset,
                "teacher_response": teacher_response,
                "model_path": model_path,
                "structured_prompt": structured_prompt,
            }

    def _ask_user_early_approval(self) -> bool:
        try:
            ans = input(
                f"\n[{self.__class__.__name__}] ¿Dar visto bueno y pasar a generación sintética ahora? (s/n): "
            ).strip().lower()
        except EOFError:
            return False
        return ans in ("s", "si", "sí", "y", "yes")
