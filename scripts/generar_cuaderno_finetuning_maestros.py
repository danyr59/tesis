#!/usr/bin/env python3
"""
Generador del Cuaderno Jupyter para Fine-Tuning de los 3 Modelos Padres (Maestros).
Crea 'util/finetuning_maestros_kaggle.ipynb' y 'scripts/finetuning_maestros_kaggle.ipynb'.
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
    # CELDA 1: TÍTULO Y PROTOCOLO FORMAL DE LA TESIS
    # =========================================================================
    cells.append(md("""# 🎓 Fine-Tuning de Modelos Padres (Maestros) para Series Temporales
### Protocolo Experimental de la Tesis de Grado (Kaggle GPU)

Este cuaderno permite realizar el **fine-tuning de adaptación de dominio (Domain Adaptation Fine-Tuning)** para los **3 Modelos Padres (Maestros)** que gobiernan los 6 escenarios experimentales ($E_1$ a $E_6$):

| # | Modelo Padre (Maestro) | Familia | Parámetros | Escenarios donde actúa |
|---|---|---|---|---|
| **1** | `Qwen/Qwen2.5-7B-Instruct` | Qwen | ~7.6B | $E_1$ (Intra Qwen $\to$ Qwen) |
| **2** | `meta-llama/Llama-3.1-8B-Instruct` | LLaMA | ~8.0B | $E_2$ (Intra LLaMA $\to$ LLaMA), $E_4$ (Cross LLaMA $\to$ Qwen) |
| **3** | `deepseek-ai/DeepSeek-Coder-6.7B-instruct` | DeepSeek | ~6.7B | $E_3$ (Intra DeepSeek), $E_5$ (DeepSeek $\to$ LLaMA), $E_6$ (DeepSeek $\to$ Qwen) |

---
### 📊 Análisis de Consumo de Memoria (RAM y VRAM) en Kaggle (Tesla T4)
* **Método de Entrenamiento:** **QLoRA (Cuantización 4-bit NF4 + Adaptadores LoRA FP16 + Paged AdamW 8-bit)**.
* **VRAM de GPU:** 
  - Pesos base cuantizados (4-bit): **~3.5 a 4.2 GB**.
  - Adaptadores LoRA y estados del optimizador: **~0.2 GB**.
  - Activaciones con Gradient Checkpointing (batch=1): **~2.0 a 2.5 GB**.
  - **VRAM Total Requerida:** **~6.0 a 7.2 GB** (¡Cabe perfectamente en los **15 GB** de una GPU Tesla T4!).
* **RAM de Sistema (CPU Host):**
  - Carga y deserialización inicial: **~8.0 a 12.0 GB** (Kaggle provee **30 GB** de System RAM)."""))

    # =========================================================================
    # CELDA 2: INSTALACIÓN DE DEPENDENCIAS
    # =========================================================================
    cells.append(md("""## 1. Instalación de Dependencias"""))
    cells.append(code("""# Desinstalar torchao preinstalado incompatible si existiera
!pip uninstall -y torchao
!pip install -q -U transformers datasets accelerate peft bitsandbytes trl huggingface_hub"""))

    # =========================================================================
    # CELDA 3: VERIFICACIÓN DE HARDWARE Y GPU
    # =========================================================================
    cells.append(md("""## 2. Verificación de Hardware y GPU"""))
    cells.append(code("""import os
import gc
import json
import shutil
import torch
from pathlib import Path

print("=" * 65)
print(f"CUDA Disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Número de GPUs detectadas: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  [GPU {i}]: {props.name} | VRAM Total: {props.total_memory / 1e9:.2f} GB")
else:
    print("⚠️ ADVERTENCIA: No se detectó GPU. Ve a Settings -> Accelerator -> GPU T4 x2")
print("=" * 65)"""))

    # =========================================================================
    # CELDA 4: CONFIGURACIÓN Y SELECTOR DE MODELO PADRE
    # =========================================================================
    cells.append(md("""## 3. Selector del Modelo Padre a Entrenar"""))
    cells.append(code("""# =============================================================================
# SELECCIONA EL MODELO PADRE A ENTRENAR CAMBIANDO LA VARIABLE 'MODELO_KEY':
# Opciones disponibles: "QWEN_7B" | "LLAMA_8B" | "DEEPSEEK_7B"
# =============================================================================
MODELO_KEY = "QWEN_7B"

CONFIG_MODELOS = {
    "QWEN_7B": {
        "nombre": "Qwen 2.5 7B Instruct",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "familia": "qwen",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 1024,
    },
    "LLAMA_8B": {
        "nombre": "LLaMA 3.1 8B Instruct",
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "familia": "llama",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 1024,
    },
    "DEEPSEEK_7B": {
        "nombre": "DeepSeek Coder 6.7B Instruct",
        "repo_id": "deepseek-ai/DeepSeek-Coder-6.7B-instruct",
        "familia": "deepseek",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 1024,
    }
}

cfg = CONFIG_MODELOS[MODELO_KEY]
print(f"🎯 MODELO SELECCIONADO: {cfg['nombre']}")
print(f"📦 Repositorio HF:      {cfg['repo_id']}")
print(f"🧬 Familia:             {cfg['familia']}")
print(f"🎯 Capas LoRA objetivo: {cfg['target_modules']}")

# Hiperparámetros de entrenamiento QLoRA
HIPERPARAMETROS = {
    "num_train_epochs": 2,            # 2 épocas completas
    "per_device_train_batch_size": 1, # 1 por GPU para prevenir OOM
    "gradient_accumulation_steps": 8, # Batch efectivo = 1 * 8 = 8
    "learning_rate": 2e-4,            # Tasa estándar óptima para LoRA en LLMs
    "lora_r": 16,                     # Rango de adaptación LoRA
    "lora_alpha": 32,                 # Escala de LoRA (alpha = 2 * r)
    "lora_dropout": 0.05,
    "max_samples": None               # None para entrenar con todo el dataset, o un entero (ej. 5000)
}
print(f"⚙️ Hiperparámetros:     {HIPERPARAMETROS}")"""))

    # =========================================================================
    # CELDA 5: AUTENTICACIÓN HUGGING FACE Y DATASET
    # =========================================================================
    cells.append(md("""## 4. Autenticación (Hugging Face) y Localización del Dataset"""))
    cells.append(code("""# 1. Autenticación con Hugging Face (Esencial para LLaMA 3.1)
try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    login(token=hf_token)
    print("🔑 Autenticado en Hugging Face con éxito (vía Kaggle Secrets).")
except Exception:
    hf_env = os.getenv("HF_TOKEN")
    if hf_env:
        from huggingface_hub import login
        login(token=hf_env)
        print("🔑 Autenticado en Hugging Face con éxito (vía variable de entorno).")
    else:
        print("ℹ️ HF_TOKEN no configurado en Secrets. Si usas LLaMA 3.1, agrégalo en Add-ons -> Secrets.")

# 2. Localizar el dataset de entrenamiento
DATASET_PATH = None
posibles_rutas = [
    Path("/kaggle/input/ts-dataset/TS_Dataset.jsonl"),
    Path("/kaggle/input/dataset-series-temporales/TS_Dataset.jsonl"),
    Path("/kaggle/working/proyecto_tesis/data/TS_Dataset.jsonl"),
    Path("./data/TS_Dataset.jsonl"),
    Path("/kaggle/working/proyecto_tesis/data/time_series_generation.jsonl"),
    Path("./data/time_series_generation.jsonl"),
]

for p in posibles_rutas:
    if p.exists():
        DATASET_PATH = p
        break

# Si no está en rutas directas, buscar recursivamente en /kaggle/input
if DATASET_PATH is None and Path("/kaggle/input").exists():
    for f in Path("/kaggle/input").rglob("*.jsonl"):
        DATASET_PATH = f
        break

if DATASET_PATH is None:
    raise FileNotFoundError("❌ No se encontró ningún archivo .jsonl de entrenamiento en /kaggle/input ni en el proyecto.")

print(f"✅ Dataset de entrenamiento localizado: {DATASET_PATH} ({DATASET_PATH.stat().st_size / (1024*1024):.2f} MB)")"""))

    # =========================================================================
    # CELDA 6: CARGA Y PROCESAMIENTO DEL DATASET
    # =========================================================================
    cells.append(md("""## 5. Carga y Estructuración de Datos de Series Temporales"""))
    cells.append(code("""from datasets import Dataset

def cargar_datos_jsonl(ruta_archivo, max_samples=None):
    registros = []
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        for idx, linea in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            linea = linea.strip()
            if not linea:
                continue
            try:
                item = json.loads(linea)
                registros.append(item)
            except json.JSONDecodeError:
                continue
    return registros

raw_data = cargar_datos_jsonl(DATASET_PATH, max_samples=HIPERPARAMETROS["max_samples"])
print(f"📊 Total de registros cargados para entrenamiento: {len(raw_data)}")

# Formatear cada ejemplo al estilo instructivo para que el Maestro aprenda a razonar
def formatear_texto_instruccion(item):
    # Formato TS_Dataset.jsonl
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
            "Analiza el comportamiento numérico de la serie, su dinámica y sus características temporales.\n\n"
            f"### Metadatos: {meta_str}\n"
            f"### Serie Numérica: [{series_str}]\n\n"
            "### Análisis de Características Temporales:\n"
            f"{item.get('description', '')}\n"
            f"Detalles: {item.get('characteristics', '')}"
        )
        return {"text": texto}
    
    # Formato alternativo TSQA (input/output)
    elif "input" in item and "output" in item:
        meta = item.get("metadata", {})
        dominio = meta.get("original_domain", meta.get("domain", "General"))
        tarea = meta.get("task_type", "análisis")
        texto = (
            f"### Instrucción: Dominio: {dominio}. Tarea: {tarea}. {item['input']}\n\n"
            f"### Respuesta del Maestro Experto:\n{item['output']}"
        )
        return {"text": texto}
    else:
        return {"text": json.dumps(item, ensure_ascii=False)}

dataset_estructurado = Dataset.from_list([formatear_texto_instruccion(x) for x in raw_data])
print("📝 Vista previa del primer ejemplo formateado:")
print("-" * 60)
print(dataset_estructurado[0]["text"][:600] + "\n...")
print("-" * 60)"""))

    # =========================================================================
    # CELDA 7: TOKENIZACIÓN
    # =========================================================================
    cells.append(md("""## 6. Tokenizador y Preparación de Tensores"""))
    cells.append(code("""from transformers import AutoTokenizer

print(f"📥 Cargando tokenizador desde: {cfg['repo_id']}...")
tokenizer = AutoTokenizer.from_pretrained(cfg["repo_id"], trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

max_length = cfg["max_seq_length"]

def tokenizar_batch(batch):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("⚙️ Tokenizando dataset en paralelo...")
tokenized_dataset = dataset_estructurado.map(
    tokenizar_batch,
    batched=True,
    remove_columns=dataset_estructurado.column_names,
)
print(f"✅ Tokenización completada: {len(tokenized_dataset)} muestras listas.")"""))

    # =========================================================================
    # CELDA 8: CARGA DEL MODELO EN QLORA (4-BIT)
    # =========================================================================
    cells.append(md("""## 7. Carga del Modelo Base en 4-Bit (QLoRA) e Inyección LoRA"""))
    cells.append(code("""from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. Configuración de Cuantización 4-bit NormalFloat (NF4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"🚀 Cargando modelo base '{cfg['repo_id']}' en cuantización 4-bit NF4...")
base_model = AutoModelForCausalLM.from_pretrained(
    cfg["repo_id"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 2. Preparar el modelo para entrenamiento en k-bit con Gradient Checkpointing
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
base_model.config.use_cache = False  # Obligatorio para Gradient Checkpointing sin advertencias

# 3. Configuración del Adaptador LoRA
lora_config = LoraConfig(
    r=HIPERPARAMETROS["lora_r"],
    lora_alpha=HIPERPARAMETROS["lora_alpha"],
    target_modules=cfg["target_modules"],
    lora_dropout=HIPERPARAMETROS["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(base_model, lora_config)
print("\\n🎯 Resumen de Parámetros Entrenables:")
peft_model.print_trainable_parameters()

vram_usada = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
print(f"📊 VRAM actual ocupada por el modelo 4-bit en GPU: {vram_usada:.2f} GB")"""))

    # =========================================================================
    # CELDA 9: ENTRENAMIENTO CON TRAINER
    # =========================================================================
    cells.append(md("""## 8. Entrenamiento con Hugging Face Trainer"""))
    cells.append(code("""from transformers import TrainingArguments, Trainer

output_dir = f"/kaggle/working/adapter_maestro_{cfg['familia']}"

training_args = TrainingArguments(
    output_dir=output_dir,
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

print(f"🏁 Iniciando fine-tuning QLoRA para {cfg['nombre']}...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer.train()
print("🎉 ¡Entrenamiento completado exitosamente!")"""))

    # =========================================================================
    # CELDA 10: GUARDADO, COMPRESIÓN Y DESCARGA
    # =========================================================================
    cells.append(md("""## 9. Guardado del Adaptador LoRA y Compresión ZIP"""))
    cells.append(code("""# 1. Guardar pesos del adaptador LoRA y tokenizador
peft_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Adaptador LoRA guardado en: {output_dir}")

# 2. Comprimir en archivo ZIP para descarga directa o uso en el pipeline
zip_filename = f"/kaggle/working/adapter_maestro_{cfg['familia']}.zip"
shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', output_dir)
print(f"📦 Archivo comprimido generado: {zip_filename} ({Path(zip_filename).stat().st_size / (1024*1024):.2f} MB)")

# 3. Enlace de descarga directa interactivo en Kaggle
from IPython.display import FileLink
print("\\n📥 Haz clic en el siguiente enlace para descargar tu adaptador:")
FileLink(f"adapter_maestro_{cfg['familia']}.zip")"""))

    # =========================================================================
    # CELDA 11: INFERENCIA DE PRUEBA (SMOKE TEST)
    # =========================================================================
    cells.append(md("""## 10. Inferencia de Verificación (Smoke Test)"""))
    cells.append(code("""print("🔬 Probando generación con el adaptador entrenado...")
peft_model.eval()

# Ejemplo sintético de prueba: serie con tendencia y estacionalidad
serie_test = [10.2, 10.8, 11.5, 12.1, 11.9, 12.8, 13.5, 14.1, 13.8, 14.9, 15.6, 16.2]
serie_str = ", ".join([str(x) for x in serie_test])

prompt_test = (
    "### Instrucción: Eres un Maestro experto en series temporales. "
    "Analiza el comportamiento numérico de la serie, su dinámica y sus características temporales.\n\n"
    f"### Metadatos: unidades: lpm, frecuencia: minuto, rango: [2024-01-01 al 2024-01-02]\n"
    f"### Serie Numérica: [{serie_str}]\n\n"
    "### Análisis de Características Temporales:\n"
)

inputs = tokenizer(prompt_test, return_tensors="pt").to(peft_model.device)
with torch.no_grad():
    output_tokens = peft_model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

respuesta = tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\\n--- RESPUESTA DEL MAESTRO CON ADAPTADOR LoRA ---")
print(respuesta)
print("-" * 50)"""))

    notebook = {
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
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook

if __name__ == "__main__":
    nb = create_notebook()
    
    # Guardar en util/ y en scripts/
    rutas_salida = [
        Path("util/finetuning_maestros_kaggle.ipynb"),
        Path("scripts/finetuning_maestros_kaggle.ipynb"),
    ]
    
    for ruta in rutas_salida:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"✅ Cuaderno generado exitosamente en: {ruta}")

