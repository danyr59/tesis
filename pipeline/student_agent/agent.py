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
from pipeline.utils.llm_client import LLMClient
from pipeline.utils.series_utils import (
    extend_series,
    extract_numeric_series,
    seeds_statistics,
)


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
        self.num_series_target = Settings.SYNTHETIC_NUM_SERIES
        self.points_per_series_target = Settings.SYNTHETIC_POINTS_PER_SERIES

    def _get_teacher(self):
        """Crea el TeacherAgent solo si no existe."""
        if self.teacher is None:
            self.teacher = TeacherAgent()
        return self.teacher

    def _release_teacher(self):
        """Libera el TeacherAgent y limpia memoria."""
        if self.teacher is not None:
            del self.teacher
            self.teacher = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def run_pipeline(self, initial_query: str) -> None:
        self.user_query = initial_query
        # Volumen fijo desde Settings (.env o pipeline/config/settings.py)
        self.num_series_target = Settings.SYNTHETIC_NUM_SERIES
        self.points_per_series_target = Settings.SYNTHETIC_POINTS_PER_SERIES

        print(f"\n--- Iniciando Flujo Maestro-Estudiante ---")
        print(
            f"[{self.__class__.__name__}] Generación sintética (fija): "
            f"{Settings.synthetic_generation_summary()}"
        )

        structured_prompt = self.interpret_request(initial_query)

        iteration = 0
        quality_accepted = False
        final_model_path: Optional[str] = None
        current_dataset: Optional[List[dict]] = None
        current_teacher_response: Optional[dict] = None
        structured_prompt_active = structured_prompt

       # while iteration < Settings.MAX_ITERATIONS and not quality_accepted:
        # iteration += 1
        print(f"\n--- Iteración {iteration}/{Settings.MAX_ITERATIONS} ---")

        teacher = self._get_teacher()
        teacher_response = self.teacher.process_request(
             structured_prompt_active)
        self._release_teacher()

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

            confidence = float(response.get("confidence", 0.0))
            missing = response.get("missing_parameters") or []

            if confidence >= Settings.MIN_CONFIDENCE_THRESHOLD and not missing:
                break

            if not missing and confidence < Settings.MIN_CONFIDENCE_THRESHOLD:
                missing = ["detalles adicionales del requerimiento"]

            print(
                f"\n[{self.__class__.__name__}] Información insuficiente (confianza={confidence:.2f}).")
            print(f"Justificación: {response.get('justification', '')}")
            for param in missing:
                user_val = input(
                    f"[{self.__class__.__name__}] Aclara el parámetro '{param}': "
                ).strip()
                if user_val:
                    current_query += f". {param}: {user_val}"
        else:
            print(
                f"[{self.__class__.__name__}] Máximo de rondas de diálogo alcanzado; "
                "continuando con la mejor interpretación disponible."
            )
            if not response:
                response = self.llm.generate_json(
                    self._interpretation_prompt(current_query))

        structured = {
            "values": response.get("values", {}),
            "missing_parameters": response.get("missing_parameters", []),
            "pending_questions": response.get("pending_questions", missing),
            "justification": response.get("justification", ""),
            "suggested_search": response.get("suggested_search", ""),
            "search_strategy": response.get(
                "search_strategy", response.get("suggested_search", "")
            ),
            "confidence": confidence,
        }
        print(f"[{self.__class__.__name__}] Prompt estructurado:\n{json.dumps(structured, indent=2, ensure_ascii=False)}")
        return structured

    def _interpretation_prompt(self, query: str) -> str:
        return f"""Eres el agente Estudiante (LLM liviano con razonamiento).
Requerimiento del usuario: "{query}"

1. Interpretación semántica activa: sinónimos (ej. calor→clima) y ambigüedades.
2. Detecta parámetros faltantes (variables, dominio, frecuencia, localización).
   Si falta algo crítico, indícalo en missing_parameters para dialogar con el usuario.
3. Justifica cada inferencia (ej. frecuencia diaria estándar en por ejemplo: clima).
4. Razona tipo_serie: univariada o multivariada.
5. Si confidence >= {Settings.MIN_CONFIDENCE_THRESHOLD}, el requerimiento es suficiente.

Devuelve ÚNICAMENTE JSON:
{{
  "values": {{
    "tema": "..."
    "dominio": "Identifica el tema, dependiendo de las variables determinar el tema",
    "localizacion": "LUGAR DONDE APLICA(si es que aplica)",
    "frecuencia": "...",
    "variables": "...",
    "tipo_serie": "univariada|multivariada",
    "num_series_configuradas": {self.num_series_target},
    "puntos_por_serie_configurados": {self.points_per_series_target}
  }},
  "missing_parameters": [],
  "pending_questions": [],
  "justification": "...",
  "suggested_search": "...",
  "search_strategy": "...",
  "confidence": 0.0 //no olvidar este valor
}}
"""

    def fine_tune(self, seed_dataset: List[dict], metadata_summary: dict,  iteration: int) -> str:
        print(
            f"\n[{self.__class__.__name__}] Destilación (LoRA) con {len(seed_dataset)} semillas...")
        prompt = f"""Eres el Estudiante. Dataset semilla: {len(seed_dataset)} ejemplos.
Metadatos: {json.dumps(metadata_summary, ensure_ascii=False)}

Selecciona y justifica en modo verbose:
- epochs (evitar sobreajuste si hay pocos datos)
- batch_size
- learning_rate

Incluye también JSON al final:
{{"epochs": 3, "batch_size": 2, "learning_rate": 0.0002}}
"""
        reasoning = self.llm.generate_text(prompt, max_new_tokens=1024)
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
            seed_preview.append(nums[:30])

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

        all_series = self._normalize_series_length(all_series, tipo_serie)
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
        example = (
            '{"synthetic_series": [{"temperatura (°C)": [1.0, 2.0], "humedad (%)": [50.0, 51.0]}]}'
            if multivariate
            else '{"synthetic_series": [[1.0, 2.0, 3.0]]}'
        )

        prompt = f"""Eres el Estudiante fine-tuneado (LoRA). Genera series sintéticas numéricas.

Dominio y requisitos:
{json.dumps(prompt_info.get("values", {}), ensure_ascii=False)}

Estadísticas de semillas TSQA: {json.dumps(stats)}
Muestra de semillas (referencia): {json.dumps(seed_preview)[:3000]}

Genera exactamente {num_series} series con exactamente {num_points} puntos cada una.
Tipo: {"multivariada con unidades en los nombres" if multivariate else "univariada"}.
Valores realistas, float, con variabilidad y estacionalidad plausible.
ID base de series: {series_offset + 1}.

Devuelve ÚNICAMENTE JSON con esta forma:
{example}
"""
        response = self.llm.generate_json(
            prompt, max_new_tokens=Settings.LLM_MAX_NEW_TOKENS)
        series = response.get("synthetic_series", [])
        if not series:
            series = self._fallback_series(
                num_series, num_points, stats, multivariate, prompt_info)
        return series

    def _fallback_series(
        self,
        n: int,
        points: int,
        stats: dict,
        multivariate: bool,
        prompt_info: dict,
    ) -> List[Any]:
        rng = np.random.default_rng(42)
        mean, std = stats["mean"], stats["std"]
        out = []
        for _ in range(n):
            base = float(rng.normal(mean, std))
            seq = extend_series(
                [base + float(rng.normal(0, std * 0.1))
                 for _ in range(min(20, points))],
                points,
                rng,
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
                            [50.0 + float(rng.normal(0, 3))
                             for _ in range(min(20, points))], points, rng
                        ),
                    }
                )
            else:
                var = prompt_info.get("values", {}).get("variables", "valor")
                out.append(seq)
        return out

    def _normalize_series_length(self, series_list: List[Any], tipo_serie: str) -> List[Any]:
        target = self.points_per_series_target
        normalized = []
        for s in series_list[: self.num_series_target]:
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
                    new_dict[k] = extend_series(vals, target)
                normalized.append(new_dict)
            elif isinstance(s, list):
                vals = [float(x) for x in s]
                normalized.append(extend_series(vals, target))
        while len(normalized) < self.num_series_target:
            normalized.extend(
                self._fallback_series(
                    1,
                    target,
                    seeds_statistics([]) if not normalized else {
                        "mean": 0, "std": 1, "min": 0, "max": 1},
                    "multi" in tipo_serie,
                    {"values": {"variables": "valor"}},
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

        for series_idx, series_data in enumerate(series_list):
            series_id = series_idx + 1
            if "multi" in tipo_serie and isinstance(series_data, dict):
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
                        rows.append(row)
            elif isinstance(series_data, list):
                var_name = prompt_info.get("values", {}).get(
                    "variables", "valor_generado")
                if isinstance(var_name, list):
                    var_name = var_name[0]
                if "(" not in str(var_name) and "temperatura" in str(var_name).lower():
                    var_name = f"{var_name} (°C)"
                for t, val in enumerate(series_data):
                    try:
                        f_val = float(val)
                    except (ValueError, TypeError):
                        f_val = None
                    rows.append({"series_id": series_id,
                                "time_step": t, str(var_name): f_val})

        pd.DataFrame(rows).to_csv(output_file, index=False)
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
