#!/usr/bin/env python3
"""
Generador de los 3 cuadernos de Kaggle para Fine-Tuning de los Modelos Padres:
1. Qwen 2.5 7B Instruct
2. LLaMA 3.1 8B Instruct
3. DeepSeek Coder 6.7B Instruct

Genera los archivos en 'util/' y en 'scripts/'.
Optimizado para Kaggle (GPU P100 o GPU T4 x2) con:
- Detección automática de cualquier dataset .jsonl en /kaggle/input (sin importar el nombre que le dé el usuario).
- Cobertura completa: max_seq_length=768 con 120 puntos numéricos.
- LoRA enfocado en Proyecciones de Atención (q_proj, k_proj, v_proj, o_proj) para máxima velocidad.
- Dynamic Batch Padding con DataCollatorForSeq2Seq (pad_to_multiple_of=8, loss ignora padding con -100).
- Salida empaquetada en /kaggle/working/ lista para descarga o para usar en el orquestador.
"""

import json
from pathlib import Path

MODELOS = [
    {
        "id": "qwen",
        "num": "01",
        "nombre": "Qwen 2.5 7B Instruct",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "familia": "qwen",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_seq_length": 768,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "escenarios": "E1 (Intra Qwen -> Qwen)",
        "zip_name": "adapter_maestro_qwen.zip",
        "output_dir_name": "adapter_maestro_qwen",
        "gated_info": "Este modelo es público en Hugging Face y se descarga directamente sin solicitud previa."
    },
    {
        "id": "llama",
        "num": "02",
        "nombre": "LLaMA 3.1 8B Instruct",
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "familia": "llama",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_seq_length": 768,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "escenarios": "E2 (Intra LLaMA -> LLaMA) y E4 (Cross LLaMA -> Qwen)",
        "zip_name": "adapter_maestro_llama.zip",
        "output_dir_name": "adapter_maestro_llama",
        "gated_info": "⚠️ MODELO GATED: Debes haber aceptado la licencia de Meta en Hugging Face y agregar tu HF_TOKEN en Kaggle Secrets (Add-ons -> Secrets)."
    },
    {
        "id": "gemma",
        "num": "03",
        "nombre": "Gemma 2 9B Instruct",
        "repo_id": "google/gemma-2-9b-it",
        "familia": "gemma",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_seq_length": 512,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "escenarios": "E7 (Intra Gemma -> Gemma), E8 (Cross Gemma -> Qwen) y E9 (Cross Gemma -> LLaMA)",
        "zip_name": "adapter_maestro_gemma.zip",
        "output_dir_name": "adapter_maestro_gemma",
        "gated_info": "⚠️ MODELO GATED: Debes haber aceptado la licencia de Google Gemma en Hugging Face y agregar tu HF_TOKEN en Kaggle Secrets (Add-ons -> Secrets)."
    }
]

def build_kaggle_notebook(m):
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
    cells.append(md(f"""# 🎓 Fine-Tuning de Maestro en Kaggle: {m['nombre']}
### Adaptación de Dominio en Series Temporales (Tesis de Grado)

Este cuaderno está configurado para ejecutarse en **Kaggle (GPU P100 o GPU T4 x2)** para entrenar el adaptador LoRA del modelo maestro **{m['nombre']}** con la cuota gratuita de 30 horas semanales de Kaggle.

* **Modelo Base:** `{m['repo_id']}`
* **Familia:** `{m['familia'].upper()}`
* **Escenarios del Protocolo donde actúa:** **{m['escenarios']}**
* **{m['gated_info']}**
* **Salida generada:** `/kaggle/working/{m['zip_name']}` (se descarga desde el panel derecho de Kaggle).

---
### ⚙️ Configuración Previa en Kaggle:
1. En el panel lateral derecho (**Notebook options**):
   * **Accelerator:** Selecciona **GPU P100** (recomendada por ancho de banda de memoria) o **GPU T4 x2**.
   * **Internet:** Debe estar en **Internet On** (para descargar librerías y modelos de Hugging Face).
2. En la sección **Input** (panel derecho):
   * Haz clic en **Add Input** y agrega el dataset donde subiste `TS_Dataset.jsonl`.
3. Si usas LLaMA 3.1:
   * Menú superior: **Add-ons $\\to$ Secrets** $\\to$ agrega `HF_TOKEN` con tu token de Hugging Face."""))

    # =========================================================================
    # CELDA 2: INSTALACIÓN DE DEPENDENCIAS
    # =========================================================================
    cells.append(md("""## 1. Instalación de Dependencias"""))
    cells.append(code("""# Desinstalar torchao incompatible preinstalado en Kaggle y actualizar librerías
!pip uninstall -y torchao
!pip install -q -U transformers datasets accelerate peft bitsandbytes trl huggingface_hub"""))

    # =========================================================================
    # CELDA 3: GPU CHECK
    # =========================================================================
    cells.append(md("""## 2. Verificación de Hardware y GPU en Kaggle"""))
    cells.append(code("""import os
import torch

# Optimización de asignación de memoria PyTorch para evitar fragmentación en GPUs T4
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Forzar GPU 0 si hay múltiples GPUs para estabilidad con BitsAndBytes QLoRA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("=" * 65)
print(f"CUDA Disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Detectada: {props.name}")
    print(f"VRAM Total:    {props.total_memory / 1e9:.2f} GB")
else:
    print("❌ ERROR: No se detectó GPU. En el panel derecho ve a 'Settings' -> 'Accelerator' y selecciona GPU P100 o GPU T4 x2.")
print("=" * 65)"""))

    # =========================================================================
    # CELDA 4: CONFIGURACIÓN ESPECÍFICA DEL MODELO
    # =========================================================================
    cells.append(md(f"""## 3. Configuración para {m['nombre']}"""))
    cells.append(code(f"""MODELO_ID = "{m['repo_id']}"
FAMILIA = "{m['familia']}"
TARGET_MODULES = {m['target_modules']}
MAX_SEQ_LENGTH = {m['max_seq_length']}
OUTPUT_DIR = "/kaggle/working/{m['output_dir_name']}"
ZIP_OUTPUT = "/kaggle/working/{m['zip_name']}"

# Hiperparámetros de Entrenamiento (Opción 3: 1 Época Completa):
# Con 1 época el modelo ve el 100% de los 8.787 ejemplos del dataset y culmina en ~2.5 - 3.5 horas en Kaggle.
HIPERPARAMETROS = {{
    "num_train_epochs": 1,            # 1 época completa (~3h en P100 / T4)
    "per_device_train_batch_size": {m.get('batch_size', 2)}, # Batch adaptado a la VRAM de la GPU
    "gradient_accumulation_steps": {m.get('gradient_accumulation_steps', 2)}, # Batch efectivo = 4 (~1.648 pasos totales)
    "learning_rate": 2e-4,            # Tasa de convergencia LoRA recomendada
    "lora_r": 16,                     # Rango de adaptación LoRA
    "lora_alpha": 32,                 # alpha = 2 * r
    "lora_dropout": 0.05,
    "max_samples": None               # None = todo el dataset (8.787 ejemplos)
}}

print(f"🎯 Modelo a entrenar: {{MODELO_ID}}")
print(f"🎯 Módulos LoRA (Atención): {{TARGET_MODULES}}")
print(f"🎯 Longitud máxima de contexto: {{MAX_SEQ_LENGTH}} tokens")
print(f"📁 Directorio de salida: {{OUTPUT_DIR}}")
print(f"📦 Archivo comprimido final: {{ZIP_OUTPUT}}")"""))

    # =========================================================================
    # CELDA 5: AUTENTICACIÓN HUGGING FACE
    # =========================================================================
    cells.append(md("""## 4. Autenticación con Hugging Face (Esencial para LLaMA 3.1)"""))
    cells.append(code("""import os
from huggingface_hub import login

token_encontrado = False

# 1. Intentar con Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("🔑 Autenticado exitosamente con Hugging Face (vía Kaggle Secrets).")
        token_encontrado = True
except Exception:
    pass

# 2. Intentar con variable de entorno
if not token_encontrado:
    hf_env = os.getenv("HF_TOKEN")
    if hf_env:
        login(token=hf_env)
        print("🔑 Autenticado exitosamente con Hugging Face (vía variable de entorno).")
        token_encontrado = True

if not token_encontrado:
    print("ℹ️ HF_TOKEN no detectado en Kaggle Secrets. Si este modelo es público se descargará sin problema.")
    print("   Si es LLaMA 3.1, ve a 'Add-ons' -> 'Secrets' en Kaggle y agrega tu HF_TOKEN.")"""))

    # =========================================================================
    # CELDA 6: DETECCIÓN AUTOMÁTICA DEL DATASET EN KAGGLE
    # =========================================================================
    cells.append(md("""## 5. Detección Inteligente del Dataset en `/kaggle/input`"""))
    cells.append(code("""from pathlib import Path

# Detección flexible: encuentra TS_Dataset.jsonl sin importar el nombre del dataset en Kaggle
DATASET_PATH = None

posibles_rutas = [
    Path("/kaggle/input/ts-dataset/TS_Dataset.jsonl"),
    Path("/kaggle/input/dataset-series-temporales/TS_Dataset.jsonl"),
    Path("/kaggle/input/ts-dataset-jsonl/TS_Dataset.jsonl"),
    Path("/kaggle/input/time-series-generation/time_series_generation.jsonl"),
]

for p in posibles_rutas:
    if p.exists():
        DATASET_PATH = p
        break

# Búsqueda recursiva en todo /kaggle/input si no está en las rutas estándar
if DATASET_PATH is None and Path("/kaggle/input").exists():
    for f in Path("/kaggle/input").rglob("*.jsonl"):
        if "TS_Dataset" in f.name or "time_series" in f.name:
            DATASET_PATH = f
            break
    # Si aún no lo encuentra por nombre, tomar el primer .jsonl que exista
    if DATASET_PATH is None:
        for f in Path("/kaggle/input").rglob("*.jsonl"):
            DATASET_PATH = f
            break

if DATASET_PATH is None or not DATASET_PATH.exists():
    raise FileNotFoundError(
        "❌ No se encontró ningún archivo .jsonl en /kaggle/input.\\n"
        "   Pasos para resolverlo:\\n"
        "   1. En el panel derecho de Kaggle, haz clic en '+ Add Input'.\\n"
        "   2. Busca y agrega tu dataset que contiene 'TS_Dataset.jsonl'.\\n"
        "   3. Vuelve a ejecutar esta celda."
    )

print(f"✅ Dataset de entrenamiento localizado: {DATASET_PATH} ({DATASET_PATH.stat().st_size / (1024*1024):.2f} MB)")"""))

    # =========================================================================
    # CELDA 7: ESTRUCTURACIÓN DE DATOS
    # =========================================================================
    cells.append(md("""## 6. Carga y Estructuración Pedagógica de los Datos"""))
    cells.append(code("""import json
from datasets import Dataset

def cargar_datos(ruta, max_samples=None):
    items = []
    with open(ruta, "r", encoding="utf-8") as f:
        for idx, linea in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            linea = linea.strip()
            if not linea:
                continue
            try:
                items.append(json.loads(linea))
            except json.JSONDecodeError:
                continue
    return items

raw_data = cargar_datos(DATASET_PATH, max_samples=HIPERPARAMETROS["max_samples"])
print(f"📊 Registros cargados: {len(raw_data)}")

def formatear_instruccion(item):
    if "series" in item and "description" in item:
        series_vals = item["series"]
        if isinstance(series_vals, list):
            # 120 puntos numéricos con 4 decimales cubren ampliamente la dinámica temporal y entran en 768 tokens
            series_str = ", ".join([f"{float(x):.4f}" for x in series_vals[:120]])
        else:
            series_str = str(series_vals)[:400]
        meta = item.get("metadata", {})
        meta_str = f"unidades={meta.get('units','')}, frecuencia={meta.get('frequency','')}, inicio={meta.get('start','')}, fin={meta.get('end','')}"
        
        texto = (
            "### Instrucción: Describe la siguiente serie temporal y enumera sus características principales.\\n\\n"
            f"### Serie: [{series_str}]\\n"
            f"### Metadatos: {meta_str}\\n\\n"
            "### Respuesta:\\n"
            f"{item.get('description', '')}\\n\\n"
            "### Características:\\n"
            f"{item.get('characteristics', '')}"
        )
        return {"text": texto}
    elif "input" in item and "output" in item:
        meta = item.get("metadata", {})
        dominio = meta.get("original_domain", meta.get("domain", "General"))
        tarea = meta.get("task_type", "análisis")
        texto = (
            f"### Instrucción: Dominio: {dominio}. Tarea: {tarea}. {item['input']}\\n\\n"
            f"### Respuesta del Maestro Experto:\\n{item['output']}"
        )
        return {"text": texto}
    else:
        return {"text": json.dumps(item, ensure_ascii=False)}

ds_estructurado = Dataset.from_list([formatear_instruccion(x) for x in raw_data])
print("📝 Muestra del formato instructivo:")
print("-" * 60)
print(ds_estructurado[0]["text"][:450] + "\\n...")
print("-" * 60)"""))

    # =========================================================================
    # CELDA 8: TOKENIZACIÓN DINÁMICA
    # =========================================================================
    cells.append(md("""## 7. Tokenizador y Preparación de Muestras (Dynamic Padding)"""))
    cells.append(code("""from transformers import AutoTokenizer

print(f"📥 Descargando tokenizador de {MODELO_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_ID, trust_remote_code=True)

# Configuración obligatoria de pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenizar_batch(batch):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,  # Sin relleno estático para evitar desperdicio de GPU
    )
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    return tokenized

print(f"⚙️ Tokenizando ejemplos (max_length={MAX_SEQ_LENGTH}, dynamic padding listo)...")
tokenized_dataset = ds_estructurado.map(
    tokenizar_batch,
    batched=True,
    remove_columns=ds_estructurado.column_names,
)
print(f"✅ Dataset tokenizado: {len(tokenized_dataset)} muestras listas.")"""))

    # =========================================================================
    # CELDA 9: MODELO 4-BIT QLORA CON TARGETS DE ATENCIÓN
    # =========================================================================
    cells.append(md("""## 8. Carga del Modelo Base en 4-Bit (QLoRA) e Inyección LoRA (Atención)"""))
    cells.append(code("""import gc
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. Configuración BitsAndBytes 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_kwargs = {
    "quantization_config": bnb_config,
    "device_map": {"": "cuda:0"},
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
}
if "gemma" in MODELO_ID.lower():
    model_kwargs["attn_implementation"] = "eager"

print(f"🚀 Cargando modelo base {MODELO_ID} en 4-bit...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODELO_ID,
    **model_kwargs,
)

# 2. Habilitar Gradient Checkpointing para ahorro de VRAM
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
base_model.config.use_cache = False

# 3. Configurar adaptadores LoRA enfocados en Proyecciones de Atención
lora_config = LoraConfig(
    r=HIPERPARAMETROS["lora_r"],
    lora_alpha=HIPERPARAMETROS["lora_alpha"],
    target_modules=TARGET_MODULES,
    lora_dropout=HIPERPARAMETROS["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(base_model, lora_config)
print("\\n🎯 Parámetros entrenables vs congelados:")
peft_model.print_trainable_parameters()

vram_actual = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
print(f"📊 VRAM inicial ocupada por el modelo en GPU: {vram_actual:.2f} GB")"""))

    # =========================================================================
    # CELDA 10: ENTRENAMIENTO ACELERADO
    # =========================================================================
    cells.append(md("""## 9. Entrenamiento Acelerado con Hugging Face Trainer y Dynamic Collator"""))
    cells.append(code("""import os
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

# DataCollator con Dynamic Batch Padding:
# 1. Ajusta cada lote únicamente a su tamaño real.
# 2. pad_to_multiple_of=8 satura óptimamente los Tensor Cores de la GPU.
# 3. Sustituye tokens de padding por -100 en 'labels' para que CrossEntropyLoss los ignore totalmente.
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=HIPERPARAMETROS["per_device_train_batch_size"],
    gradient_accumulation_steps=HIPERPARAMETROS["gradient_accumulation_steps"],
    num_train_epochs=HIPERPARAMETROS["num_train_epochs"],
    learning_rate=HIPERPARAMETROS["learning_rate"],
    logging_steps=10,
    save_steps=200,
    save_strategy="steps",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    max_grad_norm=0.3,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Detección y reanudación automática si ya existía un checkpoint previo
resume_checkpoint = None
if os.path.exists(OUTPUT_DIR):
    ckpts = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-")
    ]
    if ckpts:
        ckpts.sort(key=lambda x: int(x.split("-")[-1]))
        resume_checkpoint = ckpts[-1]
        print(f"🔄 Se detectó un checkpoint previo. Reanudando desde: {resume_checkpoint}")

print(f"🏁 Iniciando entrenamiento QLoRA optimizado para {MODELO_ID} en Kaggle...")
gc.collect()
torch.cuda.empty_cache()

trainer.train(resume_from_checkpoint=resume_checkpoint)
print("🎉 ¡Entrenamiento completado exitosamente!")"""))

    # =========================================================================
    # CELDA 11: GUARDADO Y EMPAQUETADO EN KAGGLE
    # =========================================================================
    cells.append(md("""## 10. Guardado, Empaquetado y Enlace de Descarga en Kaggle"""))
    cells.append(code("""import shutil
from pathlib import Path
from IPython.display import FileLink

# 1. Guardar adaptadores y tokenizador en /kaggle/working/
peft_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Adaptador guardado en: {OUTPUT_DIR}")

# 2. Comprimir en archivo ZIP en /kaggle/working/
shutil.make_archive(ZIP_OUTPUT.replace('.zip', ''), 'zip', OUTPUT_DIR)
zip_size = Path(ZIP_OUTPUT).stat().st_size / (1024 * 1024)
print(f"📦 Archivo comprimido generado: {ZIP_OUTPUT} ({zip_size:.2f} MB)")

# 3. Mostrar enlace interactivo para descarga directa
print("⬇️ Haz clic en el enlace a continuación para descargar el archivo ZIP:")
FileLink(ZIP_OUTPUT)"""))

    # =========================================================================
    # CELDA 12: INFERENCIA DE PRUEBA
    # =========================================================================
    cells.append(md("""## 11. Inferencia de Verificación (Smoke Test)"""))
    cells.append(code("""print("🔬 Verificando inferencia con el adaptador entrenado...")
peft_model.eval()

serie_test = [70.1, 70.4, 69.8, 70.2, 70.5, 70.1, 69.9, 70.3, 70.0, 70.2]
serie_str = ", ".join([str(x) for x in serie_test])

prompt_test = (
    "### Instrucción: Describe la siguiente serie temporal y enumera sus características principales.\\n\\n"
    f"### Serie: [{serie_str}]\\n"
    f"### Metadatos: unidades=lpm, frecuencia=minuto, inicio=2024-01-01 00:00, fin=2024-01-01 00:10\\n\\n"
    "### Respuesta:\\n"
)

inputs = tokenizer(prompt_test, return_tensors="pt").to(peft_model.device)
with torch.no_grad():
    tokens_out = peft_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

resp = tokenizer.decode(tokens_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\\n--- RESPUESTA DEL MAESTRO CON ADAPTADOR LoRA ---")
print(resp)
print("-" * 50)"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    return notebook

if __name__ == "__main__":
    for m in MODELOS:
        nb = build_kaggle_notebook(m)
        filename = f"{m['num']}_finetuning_maestro_{m['id']}_kaggle.ipynb"
        
        # Guardar en util/ y en scripts/
        rutas = [
            Path(f"util/{filename}"),
            Path(f"scripts/{filename}"),
        ]
        for r in rutas:
            r.parent.mkdir(parents=True, exist_ok=True)
            with open(r, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=2, ensure_ascii=False)
            print(f"✅ Cuaderno Kaggle generado: {r}")

