#!/usr/bin/env python3
"""
finetune_maestro.py: Script CLI para entrenamiento ligero (QLoRA 4-bit) de Modelos Maestros
de Series Temporales (Qwen2.5-7B, LLaMA-3.1-8B, Gemma-2-9B o modelos personalizados).

Permite fine-tunear localmente o en servidores de cómputo (Kaggle/Colab/Slurm/Local)
usando adaptadores PEFT LoRA sobre representaciones instructivas de series temporales.

Uso típico:
    python scripts/finetune_maestro.py --model qwen --epochs 1 --batch_size 2
    python scripts/finetune_maestro.py --model llama --dataset data/time_series_generation.jsonl
    python scripts/finetune_maestro.py --model gemma --max_samples 500
"""

import os
import sys
import gc
import json
import argparse
from pathlib import Path

# Resolver la raíz del repositorio
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── Catálogo de Modelos Docentes ──────────────────────────────────────────
MODEL_REGISTRY = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "family": "qwen",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "attn_implementation": None,
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "family": "llama",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "attn_implementation": None,
    },
    "gemma": {
        "model_id": "google/gemma-2-9b-it",
        "family": "gemma",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "attn_implementation": "eager",
    },
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning supervisado (SFT QLoRA) para Modelos Docentes en Series Temporales."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        help="Preset de modelo ('qwen', 'llama', 'gemma') o ID completo de Hugging Face.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Ruta al archivo JSONL del dataset de entrenamiento.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directorio de salida para guardar el adaptador PEFT entrenado.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Número de épocas de entrenamiento (default: 1).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size por dispositivo (default: 2).",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Pasos de acumulación de gradiente (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Tasa de aprendizaje (default: 2e-4).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=768,
        help="Longitud máxima de contexto en tokens (default: 768).",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="Rango LoRA (r, default: 16).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Escala LoRA (alpha, default: 32).",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout en capas LoRA (default: 0.05).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Límite superior de muestras a cargar (para pruebas rápidas de validación).",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Deshabilitar cuantización 4-bit (útil si se ejecuta en CPU o Apple Silicon MPS sin soporte bitsandbytes).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Token de Hugging Face para modelos con control de acceso (Llama, Gemma).",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Nombre del repositorio en Hugging Face para subir el adaptador final (opcional).",
    )
    return parser.parse_args()

def locate_dataset(provided_path: str = None) -> Path:
    if provided_path:
        p = Path(provided_path)
        if p.exists():
            return p
        # Probar relativo al repo
        p_rel = REPO_ROOT / provided_path
        if p_rel.exists():
            return p_rel
        raise FileNotFoundError(f"No se encontró el dataset en: {provided_path}")

    # Búsqueda automática en rutas habituales
    candidates = [
        REPO_ROOT / "data" / "time_series_generation.jsonl",
        REPO_ROOT / "data" / "TS_Dataset.jsonl",
        Path("/kaggle/input/ts-dataset/TS_Dataset.jsonl"),
        Path("/kaggle/input/dataset-series-temporales/TS_Dataset.jsonl"),
        Path("/kaggle/input/time-series-generation/time_series_generation.jsonl"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "No se localizó automáticamente ningún dataset JSONL. "
        "Por favor especifique la ruta mediante --dataset <ruta.jsonl>."
    )

def cargar_datos(ruta: Path, max_samples: int = None) -> list:
    items = []
    print(f"[INFO] Leyendo dataset desde: {ruta}")
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

def formatear_instruccion(item: dict) -> dict:
    """Convierte registros crudos al formato instructivo formal de la tesis."""
    if "series" in item and "description" in item:
        series_vals = item["series"]
        if isinstance(series_vals, list):
            series_str = ", ".join([f"{float(x):.4f}" for x in series_vals[:120]])
        else:
            series_str = str(series_vals)[:400]
        meta = item.get("metadata", {})
        meta_str = (
            f"unidades={meta.get('units','')}, "
            f"frecuencia={meta.get('frequency','')}, "
            f"inicio={meta.get('start','')}, "
            f"fin={meta.get('end','')}"
        )
        texto = (
            "### Instrucción: Describe la siguiente serie temporal y enumera sus características principales.\n\n"
            f"### Serie: [{series_str}]\n"
            f"### Metadatos: {meta_str}\n\n"
            "### Respuesta:\n"
            f"{item.get('description', '')}\n\n"
            "### Características:\n"
            f"{item.get('characteristics', '')}"
        )
        return {"text": texto}
    elif "input" in item and "output" in item:
        meta = item.get("metadata", {})
        dominio = meta.get("original_domain", meta.get("domain", "General"))
        tarea = meta.get("task_type", "análisis")
        texto = (
            f"### Instrucción: Dominio: {dominio}. Tarea: {tarea}. {item['input']}\n\n"
            f"### Respuesta del Maestro Experto:\n{item['output']}"
        )
        return {"text": texto}
    elif "text" in item:
        return {"text": item["text"]}
    else:
        return {"text": json.dumps(item, ensure_ascii=False)}

def main():
    args = parse_args()

    # 1. Resolver modelo y configuración
    model_key = args.model.lower()
    if model_key in MODEL_REGISTRY:
        config_entry = MODEL_REGISTRY[model_key]
        model_id = config_entry["model_id"]
        family = config_entry["family"]
        target_modules = config_entry["target_modules"]
        attn_impl = config_entry["attn_implementation"]
    else:
        model_id = args.model
        family = "custom"
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        attn_impl = "eager" if "gemma" in model_id.lower() else None

    # Autenticación opcional en Hugging Face
    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            print("[INFO] Autenticado en Hugging Face Hub.")
        except Exception as e:
            print(f"[WARN] Error en autenticacion HF: {e}")

    # Directorio de salida
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / "models" / f"adapter_maestro_{family}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINETUNING DEL MODELO MAESTRO DOCENTE")
    print("=" * 70)
    print(f"  - Modelo Base:         {model_id}")
    print(f"  - Familia:             {family}")
    print(f"  - Epocas:              {args.epochs}")
    print(f"  - Batch Size:          {args.batch_size} (Grad Accum: {args.grad_accum})")
    print(f"  - Learning Rate:       {args.lr}")
    print(f"  - Longitud Max:        {args.max_seq_length} tokens")
    print(f"  - Directorio Salida:   {output_dir}")
    print("=" * 70)

    # 2. Cargar y procesar dataset
    dataset_path = locate_dataset(args.dataset)
    raw_data = cargar_datos(dataset_path, max_samples=args.max_samples)
    print(f"[INFO] Registros procesados: {len(raw_data)}")
    if not raw_data:
        print("[ERROR] El dataset no contiene registros validos.")
        sys.exit(1)

    ds_estructurado = Dataset.from_list([formatear_instruccion(x) for x in raw_data])

    # 3. Descargar y configurar Tokenizador
    print(f"\n[INFO] Descargando tokenizador para {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=args.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Función de tokenización
    def tokenize_fn(ejemplos):
        return tokenizer(
            ejemplos["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    print("[INFO] Tokenizando secuencias de entrenamiento...")
    tokenized_dataset = ds_estructurado.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds_estructurado.column_names,
        desc="Tokenizando",
    )

    # 4. Configurar Cuantización y Cargar Modelo Base
    use_cuda = torch.cuda.is_available()
    load_in_4bit = use_cuda and not args.no_4bit

    model_kwargs = {
        "trust_remote_code": True,
        "token": args.hf_token,
        "low_cpu_mem_usage": True,
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    if load_in_4bit:
        print("[INFO] Habilitando BitsAndBytes 4-bit NF4 con doble cuantizacion...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": 0}
    elif use_cuda:
        print("[INFO] Cargando en GPU con precision float16...")
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    elif torch.backends.mps.is_available():
        print("[INFO] Dispositivo detectado: Apple Silicon MPS...")
        model_kwargs["torch_dtype"] = torch.float32
    else:
        print("[INFO] Cargando en CPU con precision float32...")
        model_kwargs["torch_dtype"] = torch.float32

    print(f"[INFO] Cargando modelo base {model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if load_in_4bit:
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.config.use_cache = False

    # 5. Configurar Adaptador LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(base_model, lora_config)
    print("\nParametros entrenables vs congelados:")
    peft_model.print_trainable_parameters()

    # 6. Preparar Trainer
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_strategy="steps",
        save_total_limit=2,
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if load_in_4bit else "adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_cuda else None,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 7. Ejecutar Entrenamiento
    print("\nIniciando ciclo de optimizacion LoRA...")
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    trainer.train()
    print("Entrenamiento completado.")

    # 8. Guardar Adaptadores y Tokenizador
    print(f"\nGuardando adaptadores en: {output_dir}")
    peft_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("[OK] Adaptadores y tokenizador guardados satisfactoriamente.")

    # 9. Subir a Hugging Face Hub (opcional)
    if args.push_to_hub:
        print(f"[INFO] Subiendo adaptadores a Hugging Face Hub: {args.push_to_hub}...")
        try:
            peft_model.push_to_hub(args.push_to_hub, token=args.hf_token)
            tokenizer.push_to_hub(args.push_to_hub, token=args.hf_token)
            print(f"[OK] Adaptador publicado en https://huggingface.co/{args.push_to_hub}")
        except Exception as e:
            print(f"[WARN] Error al subir al Hub: {e}")

if __name__ == "__main__":
    main()
