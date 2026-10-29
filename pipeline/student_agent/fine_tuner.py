"""Fine-tuning LoRA real del modelo Estudiante con dataset semilla."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset

# Parche de compatibilidad para evitar el ImportError de torchao (<0.16.0 en Kaggle)
try:
  import peft.import_utils
  peft.import_utils.is_torchao_available = lambda: False
  try:
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
  except Exception:
    pass
except Exception:
  pass

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from pipeline.config.settings import Settings
from pipeline.utils.series_utils import extract_numeric_series


def parse_hyperparams_from_text(text: str, n_samples: int, hyperparams_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  epochs = 3 if n_samples < 10 else 5 if n_samples < 30 else 8
  batch_size = 1 if n_samples < 6 else 2
  lr = 2e-4

  m = re.search(r"(\d+)\s*epochs?", text, re.I)
  if m:
    epochs = int(m.group(1))
  m = re.search(r"batch[_\s-]*size[^\d]*(\d+)", text, re.I)
  if m:
    batch_size = int(m.group(1))
  m = re.search(r"learning\s*rate[^\d]*([\d.eE+-]+)", text, re.I)
  if m:
    try:
      lr = float(m.group(1).rstrip('.'))
    except ValueError:
      lr = 2e-4

  if hyperparams_json:
    try:
      if "epochs" in hyperparams_json and hyperparams_json["epochs"] is not None:
        epochs = int(hyperparams_json["epochs"])
      if "batch_size" in hyperparams_json and hyperparams_json["batch_size"] is not None:
        batch_size = int(hyperparams_json["batch_size"])
      if "learning_rate" in hyperparams_json and hyperparams_json["learning_rate"] is not None:
        lr = float(hyperparams_json["learning_rate"])
    except (ValueError, TypeError) as e:
      print(f"[FineTuner] Error parsing JSON hyperparameters ({e}), fallback to text parsing.")

  epochs = max(1, min(epochs, 15))
  batch_size = max(1, min(batch_size, 8))
  return {"epochs": epochs, "batch_size": batch_size, "learning_rate": lr}


def _build_training_records(
  seed_dataset: List[Dict[str, Any]],
  metadata_summary: Dict[str, Any],
) -> List[Dict[str, str]]:
  records: List[Dict[str, str]] = []
  domain = metadata_summary.get("domain", "general")
  for seed in seed_dataset:
    meta = seed.get("metadata", {})
    series = extract_numeric_series(seed)
    series_preview = series[:120]
    text = (
      f"Dominio: {meta.get('domain', domain)}. "
      f"Tarea: {meta.get('task_type', 'forecast')}. "
      f"Pregunta: {str(meta.get('question', ''))[:500]}. "
      f"respuesta: {str(meta.get('response',''))}"
      f"Serie numérica: {json.dumps(series_preview)}"
    )
    records.append({"text": text})
  return records


def run_lora_fine_tune(
    base_model,
    tokenizer,
    seed_dataset: List[Dict[str, Any]],
    metadata_summary: Dict[str, Any],
    reasoning_text: str,
    continue_from_adapter: Optional[str] = None,
    hyperparams_json: Optional[Dict[str, Any]] = None,
) -> str:
  """Entrena adapter LoRA y devuelve la ruta guardada.

    Args:
        base_model: modelo base (sin adaptador)
        tokenizer: tokenizador asociado
        seed_dataset: lista de semillas
        metadata_summary: metadatos
        reasoning_text: texto con hiperparámetros
        continue_from_adapter: ruta de un adaptador previo para continuar entrenando
        hyperparams_json: parámetros extraídos en JSON (opcional)
  """
  if not seed_dataset:
    raise ValueError("No hay semillas para fine-tuning.")

  hp = parse_hyperparams_from_text(reasoning_text, len(seed_dataset), hyperparams_json)
  records = _build_training_records(seed_dataset, metadata_summary)
  if not records:
    raise ValueError("No se pudieron construir ejemplos de entrenamiento.")

  # model_path = Settings.get_student_model_path()
  out_dir = Path(Settings.STUDENT_LORA_PATH)
  out_dir.mkdir(parents=True, exist_ok=True)

  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  # Si continuamos desde un adaptador, lo cargamos sobre el modelo base
  if continue_from_adapter is not None:
    print(f"[FineTuner] Cargando adaptador previo desde {continue_from_adapter}")
    model = PeftModel.from_pretrained(base_model, continue_from_adapter)
  else:
    lora_cfg = LoraConfig(
      r=Settings.LORA_R,
      lora_alpha=Settings.LORA_ALPHA,
      lora_dropout=Settings.LORA_DROPOUT,
      bias="none",
      task_type="CAUSAL_LM",
      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    try:
      import peft.import_utils
      peft.import_utils.is_torchao_available = lambda: False
      if hasattr(peft.tuners.lora, "torchao"):
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
      pass
    model = get_peft_model(base_model, lora_cfg)
  model.print_trainable_parameters()

  def tokenize(batch):
    tokenized = tokenizer(
      batch["text"],
      truncation=True,
      max_length=512,
      padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

  ds = Dataset.from_list(records)
  tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

  train_kwargs = {
    "output_dir": str(out_dir / "checkpoints"),
    "num_train_epochs": hp["epochs"],
    "per_device_train_batch_size": 1,  # Forzar a 1 para evitar OOM
    "learning_rate": hp["learning_rate"],
    "logging_steps": 5,
    "save_strategy": "no",
    "report_to": [],
    "gradient_accumulation_steps": 4,  # Compensar el batch size 1
    "fp16": torch.cuda.is_available(),
    "gradient_checkpointing": True,  # Ahorro sustancial de VRAM
  }

  import inspect
  sig = inspect.signature(TrainingArguments.__init__)
  safe_kwargs = {k: v for k, v in train_kwargs.items() if k in sig.parameters}
  args = TrainingArguments(**safe_kwargs)

  # Habilitar checkpointing en el modelo base si es posible
  if hasattr(model, "enable_input_require_grads"):
      model.enable_input_require_grads()
  if hasattr(model, "config") and hasattr(model.config, "use_cache"):
      model.config.use_cache = False

  trainer = Trainer(model=model, args=args, train_dataset=tokenized)
  trainer.train()

  if hasattr(model, "config") and hasattr(model.config, "use_cache"):
      model.config.use_cache = True

  adapter_path = str(out_dir / "adapter")
  model.save_pretrained(adapter_path)
  tokenizer.save_pretrained(adapter_path)
  print(f"[FineTuner] Adapter LoRA guardado en {adapter_path}")
  return adapter_path
