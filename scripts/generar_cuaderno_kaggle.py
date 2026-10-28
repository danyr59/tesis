#!/usr/bin/env python3
"""Generador del cuaderno Jupyter maestro para orquestación experimental en Kaggle.

Crea 'scripts/orquestador_experimentos_kaggle.ipynb' estructurado de forma modular,
con selector de escenario diario, ciclo de purga total de VRAM, persistencia en
carpetas organizadas y generación de tablas comparativas.
"""

import json
from pathlib import Path

def create_notebook():
    cells = []

    def md(text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        }

    def code(text):
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        }

    # =========================================================================
    # CELDA 1: TÍTULO Y DESCRIPCIÓN
    # =========================================================================
    cells.append(md("""# 🧪 Orquestador Experimental — Tesis de Grado (Kaggle)
### Generación y Destilación de Series Temporales con LLMs Ligeros
Este cuaderno implementa el **Protocolo Experimental formal** de la tesis:
* **6 Escenarios de Destilación:** 3 Intra-familia (Qwen, LLaMA, DeepSeek) y 3 Cross-Family.
* **5 Perfiles Temporales:** $P_1$ (Estable), $P_2$ (Estacional), $P_3$ (Tendencia-Estacional), $P_4$ (Tendencia-Ruido), $P_5$ (Complejo).
* **Ciclo Automático de Purga de VRAM:** Cada modelo nace, cumple su función, se purga de memoria GPU y luego nace el siguiente.
* **Persistencia Modular:** Cada corrida guarda sus datos sintéticos, el modelo Estudiante LoRA entrenado, configuración JSON para recreación futura, checkpoint RNN, predicciones, métricas y gráficos en carpetas independientes.
* **Ejecución Diaria por Bloques:** Diseñado para correr **1 escenario por sesión (~45 min)** evitando el agotamiento de cuotas."""))

    # =========================================================================
    # CELDA 2: INSTALACIÓN DE DEPENDENCIAS Y HARDWARE
    # =========================================================================
    cells.append(md("""## 1. Instalación de Dependencias y Verificación de Hardware"""))
    cells.append(code("""# Instalación de librerías requeridas en el entorno de Kaggle
!pip install -q -U transformers datasets accelerate peft bitsandbytes trl
!pip install -q sentence-transformers chromadb evaluate

import os
import gc
import sys
import json
import time
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Verificación de aceleración GPU
print("=" * 60)
print(f"CUDA Disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPUs detectadas: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  [GPU {i}]: {props.name} ({props.total_memory / 1e9:.2f} GB VRAM)")
print("=" * 60)"""))

    # =========================================================================
    # CELDA 3: CONFIGURACIÓN DEL PROYECTO, GIT, RAG Y HUGGINGFACE
    # =========================================================================
    cells.append(md("""## 2. Configuración del Repositorio, RAG y Autenticación"""))
    cells.append(code("""# 1. Definir rutas base de trabajo
BASE_DIR = Path("/kaggle/working")
EXP_DIR = BASE_DIR / "experimentos"
TABLAS_DIR = BASE_DIR / "tablas_tesis"
PROJECT_ROOT = BASE_DIR / "proyecto_tesis"

for p in [EXP_DIR, TABLAS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# 2. Obtener código fuente del proyecto
pipeline_dir = PROJECT_ROOT / "pipeline"
if not pipeline_dir.exists() or not (pipeline_dir / "teacher_agent").exists():
    input_code_dir = Path("/kaggle/input/proyecto-tesis-codigo")
    if input_code_dir.exists():
        print("📦 Copiando paquete 'pipeline' desde dataset interno de Kaggle...")
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Si viene como archivo zip de Kaggle:
        zip_files = list(input_code_dir.rglob("*.zip"))
        if zip_files:
            for z in zip_files:
                shutil.unpack_archive(str(z), str(pipeline_dir))
        elif (input_code_dir / "pipeline").exists():
            shutil.copytree(input_code_dir / "pipeline", pipeline_dir, dirs_exist_ok=True)
        else:
            shutil.copytree(input_code_dir, pipeline_dir, dirs_exist_ok=True)
            
        # 2. Si se desempaquetó una subcarpeta interna llamada 'pipeline', aplanarla
        if (pipeline_dir / "pipeline").exists():
            for item in (pipeline_dir / "pipeline").iterdir():
                shutil.move(str(item), str(pipeline_dir))
            shutil.rmtree(pipeline_dir / "pipeline")
            
        if not (pipeline_dir / "__init__.py").exists():
            (pipeline_dir / "__init__.py").touch()
            
        print(f"✅ Paquete 'pipeline' instalado exitosamente en: {pipeline_dir}")
    else:
        # Fallback a clonación git no-bloqueante con soporte de token
        print("📥 Clonando repositorio desde GitHub...")
        gh_token = None
        try:
            from kaggle_secrets import UserSecretsClient
            gh_token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        except Exception:
            gh_token = os.getenv("GITHUB_TOKEN")
            
        if gh_token:
            print("🔑 Usando GITHUB_TOKEN para clonar repositorio privado...")
            !GIT_TERMINAL_PROMPT=0 git clone -b kaggle_experimentos https://oauth2:{gh_token}@github.com/danyr59/tesis_antes.git {PROJECT_ROOT}
        else:
            print("ℹ️ Clonando con credenciales públicas (GIT_TERMINAL_PROMPT=0)...")
            !GIT_TERMINAL_PROMPT=0 git clone -b kaggle_experimentos https://github.com/danyr59/tesis_antes.git {PROJECT_ROOT}
else:
    print(f"✅ Código del proyecto ya presente en: {pipeline_dir}")

for p in [PROJECT_ROOT, BASE_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.chdir(str(PROJECT_ROOT))

# 3. Autenticación Hugging Face (requerida para descargar modelos de la familia LLaMA)
try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("🔑 Autenticado en Hugging Face con éxito (Kaggle Secrets).")
except Exception:
    hf_env = os.getenv("HF_TOKEN")
    if hf_env:
        from huggingface_hub import login
        login(token=hf_env)
        print("🔑 Autenticado en Hugging Face (variable de entorno).")
    else:
        print("ℹ️ HF_TOKEN no detectado en secrets. Si vas a usar LLaMA, configura tu HF_TOKEN.")

# 4. Configurar Base de Conocimiento RAG (ChromaDB)
RAG_DEST = PROJECT_ROOT / "rag" / "maestro"
if not (RAG_DEST / "chroma.sqlite3").exists():
    RAG_DEST.mkdir(parents=True, exist_ok=True)
    print("🔍 Buscando archivo comprimido de la base RAG en /kaggle/input...")
    zip_encontrado = None
    for z in Path("/kaggle/input").rglob("*.zip"):
        if "memoria" in z.name.lower() or "rag" in z.name.lower():
            zip_encontrado = z
            break
            
    if zip_encontrado and zip_encontrado.exists():
        print(f"📦 Descomprimiendo base de conocimiento RAG desde {zip_encontrado}...")
        shutil.unpack_archive(str(zip_encontrado), str(RAG_DEST))
        print("✅ RAG descomprimido exitosamente.")
    elif Path("/kaggle/input/rag-maestro/chroma.sqlite3").exists():
        print("📋 Copiando base RAG desde dataset /kaggle/input/rag-maestro...")
        shutil.copytree("/kaggle/input/rag-maestro", str(RAG_DEST), dirs_exist_ok=True)
        print("✅ RAG copiado exitosamente.")
    else:
        print("⚠️ Advertencia: No se encontró la base RAG pre-indexada en /kaggle/input.")
        print("   Asegúrate de agregar el dataset 'rag-maestro' al notebook de Kaggle.")

os.environ["RAG_DB_PATH"] = str(RAG_DEST)
print(f"\\n📂 Espacio de trabajo activo: {os.getcwd()}")
print(f"📚 Ruta RAG configurada en: {os.environ['RAG_DB_PATH']}")
print(f"📁 Directorio de salida de experimentos: {EXP_DIR}")"""))

    # =========================================================================
    # CELDA 4: CATÁLOGO DE ESCENARIOS Y PERFILES
    # =========================================================================
    cells.append(md("""## 3. Catálogo Formal de Escenarios y Perfiles Matemáticos"""))
    cells.append(code("""# Matriz de los 6 Escenarios de Destilación (Capítulo IV)
ESCENARIOS = {
    "E1": {
        "nombre": "E1_qwen_qwen",
        "tipo": "Intra-Familia",
        "teacher_id": "Qwen/Qwen2.5-7B-Instruct",
        "student_id": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "E2": {
        "nombre": "E2_llama_llama",
        "tipo": "Intra-Familia",
        "teacher_id": "meta-llama/Llama-3.1-8B-Instruct",
        "student_id": "meta-llama/Llama-3.2-1B-Instruct",
    },
    "E3": {
        "nombre": "E3_deepseek_deepseek",
        "tipo": "Intra-Familia",
        "teacher_id": "deepseek-ai/DeepSeek-Coder-6.7B-instruct",
        "student_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    },
    "E4": {
        "nombre": "E4_llama_qwen",
        "tipo": "Cross-Family",
        "teacher_id": "meta-llama/Llama-3.1-8B-Instruct",
        "student_id": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "E5": {
        "nombre": "E5_deepseek_llama",
        "tipo": "Cross-Family",
        "teacher_id": "deepseek-ai/DeepSeek-Coder-6.7B-instruct",
        "student_id": "meta-llama/Llama-3.2-1B-Instruct",
    },
    "E6": {
        "nombre": "E6_deepseek_qwen",
        "tipo": "Cross-Family",
        "teacher_id": "deepseek-ai/DeepSeek-Coder-6.7B-instruct",
        "student_id": "Qwen/Qwen2.5-1.5B-Instruct",
    },
}

# Queries exactas parametrizadas para los 5 perfiles temporales
PERFILES = {
    "P1": {
        "id": "P1_ESTABLE",
        "query": "Necesito generar lecturas de frecuencia cardíaca de un paciente en reposo monitoreado en una clínica. Dominio: Medicina y salud. Variable estricta: Únicamente 'Frecuencia_Cardiaca'. Perfil matemático: P1_ESTABLE, valores constantes con bajo nivel de ruido. Escala: mantenerse estable alrededor de 70 lpm durante todo el periodo.",
        "dominio": "medicina"
    },
    "P2": {
        "id": "P2_ESTACIONAL",
        "query": "Simular el flujo diario de pasajeros en una estación de metro durante un año completo. Dominio: Transporte urbano. Variable estricta: Únicamente 'Pasajeros'. Perfil matemático: P2_ESTACIONAL, con ciclos semanales marcados (mayor afluencia en días laborales y menor en fines de semana), sin tendencia global. Escala: promedio de 15000 pasajeros diarios.",
        "dominio": "transporte"
    },
    "P3": {
        "id": "P3_TEND_ESTAC",
        "query": "Simular las ventas diarias de una plataforma de comercio electrónico en expansión durante un año. Dominio: Retail y Ventas. Variable estricta: Únicamente 'Ventas_Diarias'. Perfil matemático: P3_TENDENCIA_ESTACIONAL, con tendencia creciente sostenida y ciclos semanales marcados por mayor consumo en fines de semana. Escala: promedio inicial de 2500 USD creciendo hacia 4000 USD.",
        "dominio": "retail"
    },
    "P4": {
        "id": "P4_TEND_RUIDO",
        "query": "Simular la cotización diaria de una acción tecnológica durante un ciclo correctivo de mercado. Dominio: Finanzas y Mercados. Variable estricta: Únicamente 'Precio_Accion'. Perfil matemático: P4_TENDENCIA_RUIDO, con tendencia fluctuante descendente y alto nivel de ruido o volatilidad, sin componentes estacionales. Escala: cotizando alrededor de 120 USD.",
        "dominio": "finanzas"
    },
    "P5": {
        "id": "P5_COMPLEJO",
        "query": "Monitorear el consumo de ancho de banda en un centro de datos empresarial. Dominio: Telecomunicaciones y Servidores. Variable estricta: Únicamente 'Ancho_Banda_Gbps'. Perfil matemático: P5_COMPLEJO, con fuerte estacionalidad por horario laboral, alto ruido de tráfico y picos anómalos esporádicos por sobrecarga. Escala: promedio de 45 Gbps.",
        "dominio": "servidores"
    }
}

print(f"✅ Catálogo cargado: {len(ESCENARIOS)} escenarios de destilación y {len(PERFILES)} perfiles temporales.")"""))

    # =========================================================================
    # CELDA 5: SELECTOR DE LA SESIÓN DE HOY
    # =========================================================================
    cells.append(md("""## 4. 🎛️ Selector de la Sesión de Hoy
> **Instrucción:** Cambia `ESCENARIO_SELECCIONADO` según el día de trabajo.
> El cuaderno correrá automáticamente los 5 perfiles para este escenario."""))
    cells.append(code("""# =========================================================================
# CONFIGURACIÓN DE LA CORRIDA DE HOY:
# =========================================================================
ESCENARIO_SELECCIONADO = "E1"   # Opciones: "E1", "E2", "E3", "E4", "E5", "E6"

# Lista de perfiles a ejecutar en esta sesión:
PERFILES_A_CORRER = ["P1", "P2", "P3", "P4", "P5"]

# Si es False, salta automáticamente cualquier perfil que ya tenga métricas guardadas
FORZAR_REPETICION = False

# -------------------------------------------------------------------------
# PARÁMETROS DE GENERACIÓN SINTÉTICA (Ajustables a tu gusto):
# -------------------------------------------------------------------------
os.environ["SYNTHETIC_POINTS_PER_SERIES"] = "400"  # Puntos por serie (ej: 400, 500, 1000)
os.environ["SYNTHETIC_NUM_SERIES"] = "20"          # Cantidad de series (ej: 15, 20, 30)
os.environ["RAG_TOP_K"] = "15"                     # Semillas extraídas del RAG por consulta

print(f"🎯 Escenario activo para hoy: {ESCENARIO_SELECCIONADO} ({ESCENARIOS[ESCENARIO_SELECCIONADO]['nombre']})")
print(f"📊 Perfiles a ejecutar: {PERFILES_A_CORRER}")
print(f"📏 Longitud serie: {os.environ['SYNTHETIC_POINTS_PER_SERIES']} pts | Series: {os.environ['SYNTHETIC_NUM_SERIES']} | RAG Top-K: {os.environ['RAG_TOP_K']}")"""))

    # =========================================================================
    # CELDA 6: FUNCIÓN DE PURGA DE VRAM
    # =========================================================================
    cells.append(md("""## 5. Función de Purga Automática de Memoria GPU ("Uno se borra y el otro nace")"""))
    cells.append(code("""def purgar_memoria(mensaje=""):
    \"\"\"Libera completamente la memoria GPU y memoria RAM del sistema.\"\"\"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        vram_usada = torch.cuda.memory_allocated() / 1e9
        vram_reservada = torch.cuda.memory_reserved() / 1e9
        print(f"  [Purga VRAM] {mensaje} -> Asignada: {vram_usada:.2f} GB | Reservada: {vram_reservada:.2f} GB")

# Test inicial de purga
purgar_memoria("Estado inicial")"""))

    # =========================================================================
    # CELDA 7: MOTOR PRINCIPAL DE EJECUCIÓN MODULAR
    # =========================================================================
    cells.append(md("""## 6. Motor de Ejecución Modular (M1 → M8) con Persistencia"""))
    cells.append(code("""def ejecutar_experimento_perfil(escenario_key, perfil_key, force=False):
    esc_info = ESCENARIOS[escenario_key]
    perf_info = PERFILES[perfil_key]
    
    # 1. Definir carpeta de salida para este experimento específico
    out_dir = EXP_DIR / esc_info["nombre"] / perf_info["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metricas_file = out_dir / "metricas.json"
    
    # Si ya se ejecutó y no forzamos repetición, cargar y salir
    if metricas_file.exists() and not force:
        print(f"⏩ [{escenario_key} - {perfil_key}] Ya completado previamente. Cargando métricas...")
        with open(metricas_file, "r") as f:
            return json.load(f)
            
    print(f"\\n{'='*70}")
    print(f"🚀 INICIANDO: {esc_info['nombre']} | Perfil: {perf_info['id']}")
    print(f"   Maestro: {esc_info['teacher_id']} -> Estudiante: {esc_info['student_id']}")
    print(f"{'='*70}")
    
    t_inicio = time.time()
    
    # Configurar variables de entorno dinámicas
    os.environ["TEACHER_MODEL_ID"] = esc_info["teacher_id"]
    os.environ["TEACHER_MODEL_PATH"] = esc_info["teacher_id"]
    os.environ["STUDENT_MODEL_ID"] = esc_info["student_id"]
    os.environ["STUDENT_MODEL_PATH"] = esc_info["student_id"]
    
    # -------------------------------------------------------------------------
    # FASE 1: NACE EL MAESTRO (Razonamiento y RAG)
    # -------------------------------------------------------------------------
    print("\\n[Fase 1] Inicializando Maestro...")
    from pipeline.teacher_agent.agent import TeacherAgent
    teacher = TeacherAgent()
    structured_prompt = teacher.interpret_request(perf_info["query"])
    teacher_response = teacher.process_request(structured_prompt)
    
    # PURGA DEL MAESTRO
    print("\\n[Fase 1] Purgando Maestro de VRAM...")
    if hasattr(teacher, "embedder"):
        del teacher.embedder
    if hasattr(teacher, "llm") and teacher.llm is not None:
        if hasattr(teacher.llm, "model"): del teacher.llm.model
        if hasattr(teacher.llm, "_base_model"): del teacher.llm._base_model
        if hasattr(teacher.llm, "tokenizer"): del teacher.llm.tokenizer
        del teacher.llm
    del teacher
    purgar_memoria("Maestro destruido")
    
    # -------------------------------------------------------------------------
    # FASE 2: NACE EL ESTUDIANTE (Destilación LoRA y Síntesis)
    # -------------------------------------------------------------------------
    print("\\n[Fase 2] Inicializando Estudiante...")
    from pipeline.student_agent.agent import StudentAgent
    student = StudentAgent()
    
    current_dataset = teacher_response.get("seed_dataset", [])
    if not current_dataset:
        print("❌ Error: No se obtuvieron semillas del Maestro.")
        return None
        
    final_model_path = student.fine_tune(
        current_dataset, teacher_response.get("metadata_summary", {}), iteration=0
    )
    
    csv_temp_path = student.generate_final_series(
        current_dataset, structured_prompt, final_model_path
    )
    
    # Copiar CSV generado a la carpeta permanente del perfil
    target_csv = out_dir / "synthetic_series.csv"
    shutil.copy(csv_temp_path, target_csv)
    print(f"✅ Serie sintética guardada en: {target_csv}")
    
    # Guardar el adaptador LoRA del Estudiante (pesa solo ~20 MB)
    if final_model_path and Path(final_model_path).exists():
        target_adapter = out_dir / "student_adapter"
        shutil.copytree(final_model_path, target_adapter, dirs_exist_ok=True)
        print(f"🧠 Adaptador LoRA del Estudiante guardado en: {target_adapter}")

    # Guardar configuración y semillas para recreación futura e inmediata
    config_generacion = {
        "escenario": esc_info["nombre"],
        "perfil": perf_info["id"],
        "query": perf_info["query"],
        "dominio": perf_info["dominio"],
        "structured_prompt": structured_prompt,
        "metadata_summary": teacher_response.get("metadata_summary", {}),
        "seed_dataset": current_dataset,
    }
    with open(out_dir / "config_generacion.json", "w", encoding="utf-8") as f:
        json.dump(config_generacion, f, indent=2, ensure_ascii=False)
    print(f"📋 Configuración de generación guardada en: {out_dir / 'config_generacion.json'}")
    
    # PURGA DEL ESTUDIANTE
    print("\\n[Fase 2] Purgando Estudiante de VRAM...")
    if hasattr(student, "llm") and student.llm is not None:
        if hasattr(student.llm, "model"): del student.llm.model
        if hasattr(student.llm, "_base_model"): del student.llm._base_model
        if hasattr(student.llm, "tokenizer"): del student.llm.tokenizer
        del student.llm
    del student
    purgar_memoria("Estudiante destruido")
    
    # -------------------------------------------------------------------------
    # FASE 3: EVALUACIÓN DOWNSTREAM (M6 LSTM -> M7 Predictor -> M8 Juez & Alucinaciones)
    # -------------------------------------------------------------------------
    print("\\n[Fase 3] Evaluación Downstream (M6 RNN -> M7 Predictor -> M8 Juez)...")
    from pipeline.numeric_pipeline import run_numeric_pipeline
    
    eval_result = run_numeric_pipeline(
        csv_path=str(target_csv),
        seed_dataset=current_dataset,
        domain=perf_info["dominio"]
    )
    
    t_total = time.time() - t_inicio
    vram_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    
    # -------------------------------------------------------------------------
    # FASE 4: CONSOLIDACIÓN DE MÉTRICAS Y ARCHIVOS
    # -------------------------------------------------------------------------
    pred_dict = eval_result.get("prediction", {})
    judgment_dict = eval_result.get("judgment", {})
    
    # Copiar checkpoint de la RNN y gráfico si existen
    if "model_path" in eval_result and Path(eval_result["model_path"]).exists():
        shutil.copy(eval_result["model_path"], out_dir / "rnn_checkpoint.pt")
    if "plot_path" in eval_result and Path(eval_result["plot_path"]).exists():
        shutil.copy(eval_result["plot_path"], out_dir / "grafica_prediccion.png")
        
    resultado_consolidado = {
        "escenario": esc_info["nombre"],
        "tipo_escenario": esc_info["tipo"],
        "perfil": perf_info["id"],
        "teacher": esc_info["teacher_id"],
        "student": esc_info["student_id"],
        "downstream": {
            "r2": float(pred_dict.get("r2", 0.0)),
            "mape": float(pred_dict.get("mape", 0.0)),
            "rmse": float(pred_dict.get("rmse", 0.0)),
            "mae": float(pred_dict.get("mae", 0.0)),
        },
        "juez": {
            "aprobado": bool(judgment_dict.get("approved", False)),
            "confianza": float(judgment_dict.get("confidence", 0.0)),
            "score_alucinacion": float(judgment_dict.get("hallucination_score", 0.0)),
        },
        "alucinaciones": {
            "tasa_alucinacion": float(eval_result.get("hallucination_rate", 0.0)),
        },
        "eficiencia": {
            "tiempo_segundos": round(t_total, 2),
            "vram_pico_gb": round(vram_peak, 2)
        }
    }
    
    # Guardar métricas en JSON permanente
    with open(metricas_file, "w") as f:
        json.dump(resultado_consolidado, f, indent=2)
        
    print(f"\\n🏁 Experimento completado con éxito en {t_total:.1f}s.")
    print(f"   R²={resultado_consolidado['downstream']['r2']:.4f} | MAPE={resultado_consolidado['downstream']['mape']:.2f}% | Veredicto Juez: {resultado_consolidado['juez']['aprobado']}")
    
    return resultado_consolidado"""))

    # =========================================================================
    # CELDA 8: BUCLE PRINCIPAL DE LA SESIÓN DE HOY
    # =========================================================================
    cells.append(md("""## 7. 🚀 Ejecución de la Sesión de Hoy"""))
    cells.append(code("""resultados_sesion = []

for p_key in PERFILES_A_CORRER:
    try:
        res = ejecutar_experimento_perfil(
            ESCENARIO_SELECCIONADO, 
            p_key, 
            force=FORZAR_REPETICION
        )
        if res:
            resultados_sesion.append(res)
    except Exception as e:
        print(f"❌ Error en perfil {p_key}: {e}")
        import traceback
        traceback.print_exc()

print(f"\\n🎉 Sesión completada: {len(resultados_sesion)}/{len(PERFILES_A_CORRER)} perfiles procesados.")"""))

    # =========================================================================
    # CELDA 9: TABLA DE RESULTADOS DE LA SESIÓN
    # =========================================================================
    cells.append(md("""## 8. 📊 Visualización de Resultados y Tabla Comparativa"""))
    cells.append(code("""# Construir tabla comparativa en DataFrame
filas = []
for r in resultados_sesion:
    filas.append({
        "Escenario": r["escenario"],
        "Perfil": r["perfil"],
        "R² (LSTM)": f"{r['downstream']['r2']:.4f}",
        "MAPE (%)": f"{r['downstream']['mape']:.2f}%",
        "RMSE": f"{r['downstream']['rmse']:.2f}",
        "MAE": f"{r['downstream']['mae']:.2f}",
        "Juez": "APROBADO" if r["juez"]["aprobado"] else "RECHAZADO",
        "Tasa Aluc. (%)": f"{r['alucinaciones']['tasa_alucinacion']:.2%}",
        "VRAM Pico (GB)": f"{r['eficiencia']['vram_pico_gb']:.2f}",
        "Tiempo (s)": f"{r['eficiencia']['tiempo_segundos']:.1f}"
    })

df_resumen = pd.DataFrame(filas)
print("\\n" + "=" * 90)
print(f"RESUMEN EXPERIMENTAL — SESIÓN {ESCENARIO_SELECCIONADO}")
print("=" * 90)
display(df_resumen)

# Guardar CSV consolidado de la sesión
csv_resumen_path = TABLAS_DIR / f"resumen_{ESCENARIO_SELECCIONADO}.csv"
df_resumen.to_csv(csv_resumen_path, index=False)
print(f"\\n💾 Resumen guardado en: {csv_resumen_path}")"""))

    # =========================================================================
    # CELDA 10: EMPAQUETADO FINAL EN ZIP PARA DESCARGA
    # =========================================================================
    cells.append(md("""## 9. 📦 Empaquetado Automático en ZIP (Descarga a tu PC)"""))
    cells.append(code("""# Comprimir la carpeta del escenario actual para descarga directa
zip_name = f"{ESCENARIO_SELECCIONADO}_resultados.zip"
zip_path = BASE_DIR / zip_name

dir_a_comprimir = EXP_DIR / ESCENARIOS[ESCENARIO_SELECCIONADO]["nombre"]
if dir_a_comprimir.exists():
    shutil.make_archive(str(BASE_DIR / f"{ESCENARIO_SELECCIONADO}_resultados"), 'zip', dir_a_comprimir)
    print(f"✅ Archivo ZIP generado: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)")
    print(f"👉 Puedes descargarlo directamente desde el panel lateral derecho de Kaggle ('Output').")
else:
    print(f"⚠️ No se encontró la carpeta: {dir_a_comprimir}")"""))

    # Guardar archivo .ipynb
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    target_path = Path("scripts/orquestador_experimentos_kaggle.ipynb")
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2, ensure_ascii=False)

    print(f"✅ Cuaderno generado exitosamente en: {target_path}")

if __name__ == "__main__":
    create_notebook()
