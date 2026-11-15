#!/usr/bin/env python3
"""
Script de reprocesamiento específico para el perfil P6_MULTIVARIADO.

Reutiliza los adaptadores LoRA guardados ('student_adapter') y las configuraciones
de generación ('config_generacion.json') para re-sintetizar las series multivariadas
con nombres y escalas congruentes (Presion_Sistolica y Presion_Diastolica),
re-entrenar la RNN evaluadora y regenerar los gráficos limpios de 2 paneles.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Añadir raíz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cargar variables de entorno desde .env (incluye HF_TOKEN si existe)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Matriz oficial de estudiantes por escenario
ESCENARIOS_ESTUDIANTES = {
    "E1": {"nombre": "E1_qwen_qwen", "student_id": "Qwen/Qwen2.5-1.5B-Instruct"},
    "E2": {"nombre": "E2_qwen_llama", "student_id": "meta-llama/Llama-3.2-1B-Instruct"},
    "E3": {"nombre": "E3_qwen_gemma", "student_id": "google/gemma-2-2b-it"},
    "E4": {"nombre": "E4_llama_llama", "student_id": "meta-llama/Llama-3.2-1B-Instruct"},
    "E5": {"nombre": "E5_llama_qwen", "student_id": "Qwen/Qwen2.5-1.5B-Instruct"},
    "E6": {"nombre": "E6_llama_gemma", "student_id": "google/gemma-2-2b-it"},
    "E7": {"nombre": "E7_gemma_gemma", "student_id": "google/gemma-2-2b-it"},
    "E8": {"nombre": "E8_gemma_qwen", "student_id": "Qwen/Qwen2.5-1.5B-Instruct"},
    "E9": {"nombre": "E9_gemma_llama", "student_id": "meta-llama/Llama-3.2-1B-Instruct"},
}


def purgar_vram():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()


def reprocesar_p6_escenario(
    esc_key: str,
    resultados_base_dir: Path,
    figuras_tesis_dir: Path | None = None,
) -> Dict[str, Any] | None:
    esc_info = ESCENARIOS_ESTUDIANTES.get(esc_key)
    if not esc_info:
        print(f"❌ Escenario {esc_key} no reconocido.")
        return None

    esc_folder = resultados_base_dir / f"{esc_key}_resultados"
    p6_dir = esc_folder / "P6_MULTIVARIADO"
    cfg_file = p6_dir / "config_generacion.json"
    adapter_dir = p6_dir / "student_adapter"

    if not p6_dir.exists():
        print(f"⚠️ No se encontró la carpeta: {p6_dir}")
        return None

    if not cfg_file.exists():
        print(f"⚠️ No existe {cfg_file}")
        return None

    if not adapter_dir.exists():
        print(f"⚠️ No se encontró el adaptador LoRA en {adapter_dir}")
        return None

    print("\n" + "=" * 80)
    print(f"🚀 REPROCESANDO P6 EN ESCENARIO {esc_key} ({esc_info['nombre']})")
    print(f"📂 Carpeta: {p6_dir}")
    print(f"🧠 Modelo Estudiante: {esc_info['student_id']}")
    print("=" * 80)

    # 1. Configurar variables de entorno para el estudiante del escenario
    os.environ["STUDENT_MODEL_ID"] = esc_info["student_id"]
    os.environ["STUDENT_MODEL_PATH"] = esc_info["student_id"]
    os.environ["SYNTHETIC_NUM_SERIES"] = "20"
    os.environ["SYNTHETIC_POINTS_PER_SERIES"] = "400"
    os.environ["PROFILE_ID"] = "P6_MULTIVARIADO"

    with open(cfg_file, "r", encoding="utf-8") as f:
        config_gen = json.load(f)

    structured_prompt = config_gen.get("structured_prompt", {})
    seed_dataset = config_gen.get("seed_dataset", [])
    dominio = config_gen.get("dominio", "medicina")

    # Asegurar que el structured_prompt apunte a las variables canónicas
    if "values" not in structured_prompt:
        structured_prompt["values"] = {}
    structured_prompt["values"]["tipo_serie"] = "multivariada"
    structured_prompt["values"]["profile_id"] = "P6_MULTIVARIADO"
    structured_prompt["values"]["variables"] = ["Presion_Sistolica", "Presion_Diastolica"]
    structured_prompt["values"]["target_value"] = 120.0

    t_inicio = time.time()

    # 2. Inicializar Estudiante y cargar su adaptador entrenado
    print("\n[Paso 1] Inicializando Estudiante y cargando LoRA adapter...")
    from pipeline.student_agent.agent import StudentAgent
    student = StudentAgent()

    # 3. Generar series sintéticas con el adaptador LoRA
    print("\n[Paso 2] Generando series multivariadas (Presion_Sistolica y Presion_Diastolica)...")
    csv_temp_path = student.generate_final_series(
        seed_dataset, structured_prompt, model_path=str(adapter_dir)
    )

    target_csv = p6_dir / "synthetic_series.csv"
    shutil.copy(csv_temp_path, target_csv)
    print(f"✅ CSV sintético actualizado en: {target_csv}")

    # 4. Purgar Estudiante de memoria
    print("\n[Paso 3] Purgando Estudiante de memoria...")
    if hasattr(student, "llm") and student.llm is not None:
        if hasattr(student.llm, "model"): del student.llm.model
        if hasattr(student.llm, "_base_model"): del student.llm._base_model
        if hasattr(student.llm, "tokenizer"): del student.llm.tokenizer
        del student.llm
    del student
    purgar_vram()

    # 5. Evaluación Downstream con RNN y Juez
    print("\n[Paso 4] Evaluación Downstream (M6 RNN -> M7 Predictor -> M8 Juez)...")
    from pipeline.numeric_pipeline import run_numeric_pipeline
    from pipeline.config.settings import Settings

    eval_result = run_numeric_pipeline(
        csv_path=str(target_csv),
        seed_dataset=seed_dataset,
        domain=dominio,
    )

    t_total = time.time() - t_inicio
    import torch
    vram_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # 6. Guardar gráfica de predicción
    pred_plot_src = PROJECT_ROOT / "data" / "synthetic" / "predictions_plot.png"
    if not pred_plot_src.exists():
        # Fallback si se guardó en el directorio de trabajo actual
        pred_plot_src = Path("data/synthetic/predictions_plot.png")

    target_plot = p6_dir / "grafica_prediccion.png"
    if pred_plot_src.exists():
        shutil.copy(pred_plot_src, target_plot)
        print(f"📊 Gráfica actualizada en: {target_plot}")

        # Copiar también a las figuras de la tesis si el directorio existe
        if figuras_tesis_dir and figuras_tesis_dir.exists():
            tesis_fig_target = figuras_tesis_dir / f"{esc_key}_P6.png"
            shutil.copy(pred_plot_src, tesis_fig_target)
            print(f"📄 Copiada a figuras de tesis: {tesis_fig_target}")

    # 7. Guardar métricas actualizadas
    pred_dict = eval_result.get("prediction", {})
    judgment_dict = eval_result.get("judgment", {})
    hall_dict = eval_result.get("hallucinations", {})

    metricas_data = {
        "escenario": esc_info["nombre"],
        "tipo_escenario": "Intra-Familia" if esc_key in ("E1", "E4", "E7") else "Cross-Family",
        "perfil": "P6_MULTIVARIADO",
        "teacher": config_gen.get("metadata_summary", {}).get("teacher_model", "N/A"),
        "student": esc_info["student_id"],
        "downstream": {
            "r2": float(pred_dict.get("r2", 0.0)),
            "mape": float(pred_dict.get("mape", 0.0)),
            "rmse": float(pred_dict.get("rmse", 0.0)),
            "mae": float(pred_dict.get("mae", 0.0)),
            "per_feature": pred_dict.get("per_feature", {}),
        },
        "juez": {
            "aprobado": bool(judgment_dict.get("aprobado", True)),
            "confianza": float(judgment_dict.get("confianza", 1.0)),
            "score_alucinacion": float(judgment_dict.get("score_alucinacion", 0.0)),
        },
        "alucinaciones": {
            "tasa_alucinacion": float(hall_dict.get("tasa_alucinacion", 0.0)),
            "total_predictions": int(hall_dict.get("total_predictions", 0)),
            "total_hallucinations": int(hall_dict.get("total_hallucinations", 0)),
            "is_clean": bool(hall_dict.get("is_clean", True)),
        },
        "eficiencia": {
            "tiempo_segundos": round(t_total, 2),
            "vram_pico_gb": round(vram_peak, 2),
        },
    }

    target_json = p6_dir / "metricas.json"
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(metricas_data, f, indent=2, ensure_ascii=False)
    print(f"📋 Métricas actualizadas en: {target_json}")

    purgar_vram()
    return metricas_data


def main():
    parser = argparse.ArgumentParser(description="Reprocesar P6_MULTIVARIADO desde student_adapter.")
    parser.add_argument(
        "--resultados_dir",
        type=str,
        default="/Users/daniel/Documents/resultados",
        help="Ruta base de los resultados (por defecto /Users/daniel/Documents/resultados o /kaggle/working/resultados)",
    )
    parser.add_argument(
        "--escenarios",
        type=str,
        default="E1,E2,E3,E4,E5,E6,E7,E8,E9",
        help="Lista de escenarios separados por coma (ej. 'E1,E2' o 'E1')",
    )
    parser.add_argument(
        "--figuras_tesis_dir",
        type=str,
        default=str(PROJECT_ROOT / "tesis_correcciones_profe" / "figuras" / "predicciones"),
        help="Ruta de figuras de tesis para actualizar las imágenes automáticamente",
    )
    args = parser.parse_args()

    resultados_path = Path(args.resultados_dir)
    if not resultados_path.exists():
        # Fallback a local repo o kaggle working
        if Path("/kaggle/working/resultados").exists():
            resultados_path = Path("/kaggle/working/resultados")
        elif (PROJECT_ROOT / "resultados").exists():
            resultados_path = PROJECT_ROOT / "resultados"

    figuras_path = Path(args.figuras_tesis_dir) if args.figuras_tesis_dir else None

    lista_escenarios = [e.strip() for e in args.escenarios.split(",") if e.strip()]

    print("=" * 80)
    print("INICIANDO REPROCESAMIENTO DE P6_MULTIVARIADO")
    print(f"📂 Directorio de Resultados: {resultados_path}")
    print(f"🎯 Escenarios a Procesar: {lista_escenarios}")
    print("=" * 80)

    resumen_actualizado = {}
    for esc in lista_escenarios:
        try:
            m = reprocesar_p6_escenario(esc, resultados_path, figuras_path)
            if m:
                resumen_actualizado[esc] = m
        except Exception as e:
            print(f"❌ Error reprocesando {esc}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"🎉 Proceso finalizado. Escenarios actualizados exitosamente: {list(resumen_actualizado.keys())}")
    print("=" * 80)


if __name__ == "__main__":
    main()
