#!/usr/bin/env python3
"""
Generador de los 3 cuadernos de Google Colab para Fine-Tuning de los Modelos Padres:
1. Qwen 2.5 7B Instruct
2. LLaMA 3.1 8B Instruct
3. DeepSeek Coder 6.7B Instruct

Genera los archivos en 'util/' y en 'scripts/'.
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
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 1024,
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
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 1024,
        "escenarios": "E2 (Intra LLaMA -> LLaMA) y E4 (Cross LLaMA -> Qwen)",
        "zip_name": "adapter_maestro_llama.zip",
        "output_dir_name": "adapter_maestro_llama",
        "gated_info": "⚠️ MODELO GATED: Debes haber aceptado la licencia de Meta en Hugging Face y contar con un HF_TOKEN válido."
    },
    {
        "id": "deepseek",
        "num": "03",
        "nombre": "DeepSeek Coder 6.7B Instruct",
        "repo_id": "deepseek-ai/DeepSeek-Coder-6.7B-instruct",
        "familia": "deepseek",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 1024,
        "escenarios": "E3 (Intra DeepSeek), E5 (DeepSeek -> LLaMA) y E6 (DeepSeek -> Qwen)",
        "zip_name": "adapter_maestro_deepseek.zip",
        "output_dir_name": "adapter_maestro_deepseek",
        "gated_info": "Este modelo es público en Hugging Face y se descarga directamente."
    }
]

def build_colab_notebook(m):
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
    cells.append(md(f"""# 🎓 Fine-Tuning en Google Colab: Maestro {m['nombre']}
### Adaptación de Dominio en Series Temporales (Tesis de Grado)

Este cuaderno está configurado para ejecutarse en **Google Colab (GPU T4 gratuita o Pro)** para entrenar el adaptador LoRA del modelo padre **{m['nombre']}** sin consumir la cuota semanal de GPU en Kaggle.

* **Modelo Base:** `{m['repo_id']}`
* **Familia:** `{m['familia'].upper()}`
* **Escenarios del Protocolo donde actúa:** **{m['escenarios']}**
* **{m['gated_info']}**
* **Archivo de Salida a Descargar:** `{m['zip_name']}` (lo subirás a Kaggle después).

---
### ⚙️ Instrucciones Previas en Google Colab:
1. En el menú superior: ve a **Entorno de ejecución $\\to$ Cambiar tipo de entorno de ejecución**.
2. Selecciona **GPU T4** (Acelerador de hardware estándar de Colab) y guarda.
3. Si usas Colab Secrets para Hugging Face, agrega la clave `HF_TOKEN` en el icono de llave 🔑 a la izquierda."""))

    # =========================================================================
    # CELDA 2: INSTALACIÓN
    # =========================================================================
    cells.append(md("""## 1. Instalación de Dependencias"""))
    cells.append(code("""# Instalación de librerías actualizadas en Google Colab
!pip install -q -U transformers datasets accelerate peft bitsandbytes trl huggingface_hub"""))

    # =========================================================================
    # CELDA 3: GPU CHECK
    # =========================================================================
    cells.append(md("""## 2. Verificación de GPU (NVIDIA T4)"""))
    cells.append(code("""import torch

print("=" * 60)
print(f"CUDA Disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Detectada: {props.name}")
    print(f"VRAM Total:    {props.total_memory / 1e9:.2f} GB")
else:
    print("❌ ERROR: No se detectó GPU. Ve a 'Entorno de ejecución' -> 'Cambiar tipo de entorno' y activa GPU T4.")
print("=" * 60)"""))

    # =========================================================================
    # CELDA 4: CONFIGURACIÓN ESPECÍFICA DEL MODELO
    # =========================================================================
    cells.append(md(f"""## 3. Configuración para {m['nombre']}"""))
    cells.append(code(f"""MODELO_ID = "{m['repo_id']}"
FAMILIA = "{m['familia']}"
TARGET_MODULES = {m['target_modules']}
MAX_SEQ_LENGTH = {m['max_seq_length']}
OUTPUT_DIR = "/content/{m['output_dir_name']}"
ZIP_OUTPUT = "/content/{m['zip_name']}"

# Hiperparámetros óptimos para QLoRA en GPU T4
HIPERPARAMETROS = {{
    "num_train_epochs": 2,            # 2 épocas completas
    "per_device_train_batch_size": 1, # 1 por GPU para máxima estabilidad de VRAM
    "gradient_accumulation_steps": 8, # Batch efectivo = 1 * 8 = 8
    "learning_rate": 2e-4,            # Tasa óptima de convergencia LoRA
    "lora_r": 16,                     # Rango de adaptación LoRA
    "lora_alpha": 32,                 # alpha = 2 * r
    "lora_dropout": 0.05,
    "max_samples": None               # None = todo el dataset, o un entero (ej. 5000) para pruebas rápidas
}}

print(f"🎯 Modelo a entrenar: {{MODELO_ID}}")
print(f"📁 Directorio de salida: {{OUTPUT_DIR}}")
print(f"📦 Archivo comprimido final: {{ZIP_OUTPUT}}")"""))

    # =========================================================================
    # CELDA 5: AUTENTICACIÓN HUGGING FACE
    # =========================================================================
    cells.append(md("""## 4. Autenticación con Hugging Face"""))
    cells.append(code("""import os
from huggingface_hub import login

# Intentar autenticar con Secret de Colab o variable de entorno
token_encontrado = False
try:
    from google.colab import userdata
    hf_token = userdata.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)
        print("🔑 Autenticado exitosamente con Hugging Face (vía Colab Secrets).")
        token_encontrado = True
except Exception:
    pass

if not token_encontrado:
    hf_env = os.getenv("HF_TOKEN")
    if hf_env:
        login(token=hf_env)
        print("🔑 Autenticado exitosamente con Hugging Face (vía variable de entorno).")
    else:
        print("ℹ️ Ingresa tu token de Hugging Face a continuación:")
        login()"""))

    # =========================================================================
    # CELDA 6: LOCALIZACIÓN O SUBIDA DEL DATASET
    # =========================================================================
    cells.append(md("""## 5. Localización o Subida del Dataset (`TS_Dataset.jsonl`)"""))
    cells.append(code("""from pathlib import Path

# Rutas donde buscar el dataset TS_Dataset.jsonl en Colab
posibles_rutas = [
    Path("/content/TS_Dataset.jsonl"),
    Path("/content/drive/MyDrive/TS_Dataset.jsonl"),
    Path("/content/time_series_generation.jsonl"),
    Path("/content/drive/MyDrive/time_series_generation.jsonl"),
]

DATASET_PATH = None
for p in posibles_rutas:
    if p.exists():
        DATASET_PATH = p
        break

# Si aún no se ha colocado en /content/, abrir selector interactivo de archivos
if DATASET_PATH is None:
    print("📤 No se detectó 'TS_Dataset.jsonl' en /content/.")
    print("   Puedes arrastrarlo al panel de archivos a la izquierda (📁), o subirlo ahora:")
    try:
        from google.colab import files
        subidos = files.upload()
        for nombre in subidos.keys():
            if nombre.endswith(".jsonl"):
                DATASET_PATH = Path(f"/content/{nombre}")
                break
    except Exception as e:
        print(f"Aviso en selector: {e}")

if DATASET_PATH is None or not DATASET_PATH.exists():
    raise FileNotFoundError(
        "❌ No se encontró 'TS_Dataset.jsonl' en /content/. Arrastra el archivo 'TS_Dataset.jsonl' al panel izquierdo de Google Colab (icono de carpeta 📁) y vuelve a ejecutar esta celda."
    )

print(f"✅ Dataset localizado y listo: {DATASET_PATH} ({DATASET_PATH.stat().st_size / (1024*1024):.2f} MB)")"""))

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
            series_str = ", ".join([f"{float(x):.4f}" for x in series_vals[:120]])
        else:
            series_str = str(series_vals)[:400]
        meta = item.get("metadata", {})
        meta_str = f"unidades: {meta.get('units','')}, frecuencia: {meta.get('frequency','')}, rango: [{meta.get('start','')} al {meta.get('end','')}]"
        
        texto = (
            "### Instrucción: Eres un Maestro experto en series temporales. "
            "Analiza el comportamiento numérico de la serie, su dinámica y sus características temporales.\\n\\n"
            f"### Metadatos: {meta_str}\\n"
            f"### Serie Numérica: [{series_str}]\\n\\n"
            "### Análisis de Características Temporales:\\n"
            f"{item.get('description', '')}\\n"
            f"Detalles: {item.get('characteristics', '')}"
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
print(ds_estructurado[0]["text"][:500] + "\\n...")
print("-" * 60)"""))

    # =========================================================================
    # CELDA 8: TOKENIZACIÓN
    # =========================================================================
    cells.append(md("""## 7. Tokenizador y Procesamiento en Lotes"""))
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
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("⚙️ Tokenizando ejemplos...")
tokenized_dataset = ds_estructurado.map(
    tokenizar_batch,
    batched=True,
    remove_columns=ds_estructurado.column_names,
)
print(f"✅ Dataset tokenizado: {len(tokenized_dataset)} muestras.")"""))

    # =========================================================================
    # CELDA 9: MODELO 4-BIT QLORA
    # =========================================================================
    cells.append(md("""## 8. Carga del Modelo Base en 4-Bit (QLoRA) e Inyección LoRA"""))
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

print(f"🚀 Cargando modelo base {MODELO_ID} en 4-bit...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODELO_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# 2. Habilitar Gradient Checkpointing para ahorro de VRAM
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
base_model.config.use_cache = False

# 3. Configurar adaptadores LoRA
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
    # CELDA 10: TRAINER
    # =========================================================================
    cells.append(md("""## 9. Entrenamiento con Hugging Face Trainer"""))
    cells.append(code("""from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=HIPERPARAMETROS["per_device_train_batch_size"],
    gradient_accumulation_steps=HIPERPARAMETROS["gradient_accumulation_steps"],
    num_train_epochs=HIPERPARAMETROS["num_train_epochs"],
    learning_rate=HIPERPARAMETROS["learning_rate"],
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    warmup_steps=20,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    report_to="none",
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print(f"🏁 Iniciando entrenamiento QLoRA para {MODELO_ID}...")
gc.collect()
torch.cuda.empty_cache()

trainer.train()
print("🎉 ¡Entrenamiento completado!")"""))

    # =========================================================================
    # CELDA 11: GUARDADO Y DESCARGA EN COLAB
    # =========================================================================
    cells.append(md("""## 10. Empaquetado y Descarga Automática del Adaptador LoRA"""))
    cells.append(code("""import shutil
from google.colab import files

# 1. Guardar adaptadores y tokenizador
peft_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Adaptador guardado localmente en: {OUTPUT_DIR}")

# 2. Comprimir en archivo ZIP
shutil.make_archive(ZIP_OUTPUT.replace('.zip', ''), 'zip', OUTPUT_DIR)
zip_size = Path(ZIP_OUTPUT).stat().st_size / (1024 * 1024)
print(f"📦 Archivo comprimido generado: {ZIP_OUTPUT} ({zip_size:.2f} MB)")

# 3. Descarga directa en el navegador
print("⬇️ Descargando archivo ZIP automáticamente a tu computadora...")
try:
    files.download(ZIP_OUTPUT)
    print("✅ Descarga iniciada. Revisa la carpeta 'Descargas' de tu PC.")
except Exception as e:
    print(f"ℹ️ Si la descarga automática no inició, descárgalo desde el panel de archivos izquierdo: {ZIP_OUTPUT}")"""))

    # =========================================================================
    # CELDA 12: INFERENCIA DE PRUEBA
    # =========================================================================
    cells.append(md("""## 11. Inferencia de Verificación (Smoke Test)"""))
    cells.append(code("""print("🔬 Verificando inferencia con el adaptador entrenado...")
peft_model.eval()

serie_test = [70.1, 70.4, 69.8, 70.2, 70.5, 70.1, 69.9, 70.3, 70.0, 70.2]
serie_str = ", ".join([str(x) for x in serie_test])

prompt_test = (
    "### Instrucción: Eres un Maestro experto en series temporales. "
    "Analiza el comportamiento numérico de la serie, su dinámica y sus características temporales.\\n\\n"
    f"### Metadatos: unidades: lpm, frecuencia: minuto, rango: [2024-01-01 00:00 al 2024-01-01 00:10]\\n"
    f"### Serie Numérica: [{serie_str}]\\n\\n"
    "### Análisis de Características Temporales:\\n"
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

    # =========================================================================
    # CELDA 13: GUÍA PARA KAGGLE
    # =========================================================================
    cells.append(md(f"""## 📋 ¿Cómo cargar este adaptador en Kaggle?

Una vez descargado `{m['zip_name']}` en tu computadora:

1. **Subirlo a Kaggle como Dataset:**
   * Entra a [kaggle.com/datasets](https://www.kaggle.com/datasets) $\\to$ **New Dataset**.
   * Nombre sugerido: `maestro-{m['familia']}-adapter` (o agrúpalos todos en `adaptadores-maestros`).
   * Sube el archivo `{m['zip_name']}` y haz clic en **Create**.

2. **Conectarlo a tu Cuaderno de Experimentos en Kaggle:**
   * Abre tu cuaderno `orquestador_experimentos_kaggle.ipynb`.
   * En el panel derecho: haz clic en **Add Input** $\\to$ busca tu dataset recién subido y agrégalo.
   * El orquestador detectará automáticamente los archivos en `/kaggle/input` y cargará el adaptador para el escenario correspondiente sin que tengas que modificar una sola línea de código."""))

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
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
        nb = build_colab_notebook(m)
        filename = f"{m['num']}_finetuning_maestro_{m['id']}_colab.ipynb"
        
        # Guardar en util/ y en scripts/
        rutas = [
            Path(f"util/{filename}"),
            Path(f"scripts/{filename}"),
        ]
        for r in rutas:
            r.parent.mkdir(parents=True, exist_ok=True)
            with open(r, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=2, ensure_ascii=False)
            print(f"✅ Cuaderno generado: {r}")

