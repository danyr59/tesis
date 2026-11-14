"""Fine-tuning LoRA real del modelo Estudiante con dataset semilla."""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
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

  # Configuración estándar controlada para el protocolo experimental de la tesis:
  # Garantiza paridad de condiciones (ceteris paribus) y reproducibilidad científica entre escenarios E1-E9.
  # El modelo razona y sugiere valores en la consola, pero el benchmark fija el estándar formal:
  # 3 épocas completas, batch size físico 1 (efectivo 4 con acumulación) y tasa de aprendizaje 2e-4.
  STD_EPOCHS = 3
  STD_BATCH_SIZE = 1
  STD_LR = 2e-4

  print(
    f"[FineTuner] Hiperparámetros propuestos por LLM: épocas={epochs}, batch={batch_size}, lr={lr} "
    f"-> Estandarizados para comparabilidad de tesis: épocas={STD_EPOCHS}, batch={STD_BATCH_SIZE} (acum=4), lr={STD_LR}"
  )
  return {"epochs": STD_EPOCHS, "batch_size": STD_BATCH_SIZE, "learning_rate": STD_LR}


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


class LogitDistillationTrainer(Trainer):
  """Trainer de Hugging Face con pérdida híbrida de Knowledge Distillation (Hinton, 2015).
  
  Combina:
  1. Cross-Entropy (Hard Loss): aprendizaje supervisado sobre tokens reales.
  2. Kullback-Leibler Divergence (Soft Loss): transferencia de probabilidades suaves del Maestro.
  """

  def __init__(
      self,
      *args,
      teacher_model=None,
      temperature: float = 2.0,
      alpha: float = 0.5,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.teacher_model = teacher_model
    self.temperature = temperature
    self.alpha = alpha
    if self.teacher_model is not None:
      self.teacher_model.eval()

  def compute_loss(
      self,
      model,
      inputs,
      return_outputs=False,
      num_items_in_batch=None,
      **kwargs,
  ):
    # 1. Forward del modelo Estudiante
    outputs = model(**inputs)
    student_logits = outputs.logits

    # 2. Hard Loss (Cross-Entropy clásica sobre los tokens de entrenamiento)
    loss_ce = outputs.loss
    if loss_ce is None:
      shift_logits = student_logits[..., :-1, :].contiguous()
      shift_labels = inputs["labels"][..., 1:].contiguous()
      loss_ce = F.cross_entropy(
          shift_logits.view(-1, shift_logits.size(-1)),
          shift_labels.view(-1),
          ignore_index=-100,
      )

    # 3. Soft Loss (Divergencia KL sobre los logits del Maestro)
    if self.teacher_model is not None:
      with torch.no_grad():
        teacher_device = next(self.teacher_model.parameters()).device
        teacher_inputs = {"input_ids": inputs["input_ids"].to(teacher_device)}
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
          teacher_inputs["attention_mask"] = inputs["attention_mask"].to(teacher_device)

        teacher_outputs = self.teacher_model(**teacher_inputs)
        teacher_logits = teacher_outputs.logits.to(student_logits.device)

      # Shift para Causal LM: predecir el siguiente token
      s_logits_shift = student_logits[..., :-1, :].contiguous()
      t_logits_shift = teacher_logits[..., :-1, :].contiguous()

      # Alinear longitud de secuencia
      min_len = min(s_logits_shift.size(1), t_logits_shift.size(1))
      s_logits_sl = s_logits_shift[:, :min_len, :]
      t_logits_sl = t_logits_shift[:, :min_len, :]

      # Verificar dimensiones de vocabulario
      v_s = s_logits_sl.size(-1)
      v_t = t_logits_sl.size(-1)
      min_v = min(v_s, v_t)
      if v_s != min_v or v_t != min_v:
        s_logits_sl = s_logits_sl[:, :, :min_v]
        t_logits_sl = t_logits_sl[:, :, :min_v]

      # CRÍTICO PARA ESTABILIDAD NUMÉRICA:
      # Calcular log_softmax y softmax en FLOAT32 para evitar underflow/overflow con vocabularios grandes (|V|=151646)
      s_log_probs = F.log_softmax(s_logits_sl.float() / self.temperature, dim=-1)
      t_probs = F.softmax(t_logits_sl.float() / self.temperature, dim=-1)

      # Divergencia KL por token individual: forma (batch, seq)
      kl_per_token = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)

      # Normalización por token activo (excluyendo tokens de padding)
      if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        mask = inputs["attention_mask"][..., 1:].contiguous()[:, :min_len].float()
        kl_per_token = kl_per_token * mask
        num_valid_tokens = mask.sum().clamp(min=1.0)
        loss_kd = (kl_per_token.sum() / num_valid_tokens) * (self.temperature ** 2)
      else:
        loss_kd = kl_per_token.mean() * (self.temperature ** 2)

      # Protección contra NaN / Inf para mantener la estabilidad del adaptador LoRA
      if torch.isnan(loss_kd) or torch.isinf(loss_kd):
        loss_kd = torch.tensor(0.0, device=student_logits.device, dtype=loss_ce.dtype)

      loss = ((1.0 - self.alpha) * loss_ce) + (self.alpha * loss_kd.to(loss_ce.dtype))
    else:
      loss = loss_ce

    return (loss, outputs) if return_outputs else loss


def run_lora_fine_tune(
    base_model,
    tokenizer,
    seed_dataset: List[Dict[str, Any]],
    metadata_summary: Dict[str, Any],
    reasoning_text: str,
    continue_from_adapter: Optional[str] = None,
    hyperparams_json: Optional[Dict[str, Any]] = None,
    teacher_model=None,
) -> str:
  """Entrena adapter LoRA (con soporte opcional de Logit Distillation) y devuelve la ruta guardada.

    Args:
        base_model: modelo base estudiante (sin adaptador)
        tokenizer: tokenizador asociado
        seed_dataset: lista de semillas
        metadata_summary: metadatos
        reasoning_text: texto con hiperparámetros
        continue_from_adapter: ruta de un adaptador previo para continuar entrenando
        hyperparams_json: parámetros extraídos en JSON (opcional)
        teacher_model: modelo maestro activo (opcional, para extraer logits directamente)
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
    pad_id = tokenizer.pad_token_id
    labels = []
    for seq in tokenized["input_ids"]:
      labels.append([(t if t != pad_id else -100) for t in seq])
    tokenized["labels"] = labels
    return tokenized

  ds = Dataset.from_list(records)
  tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

  train_kwargs = {
    "output_dir": str(out_dir / "checkpoints"),
    "num_train_epochs": hp["epochs"],
    "per_device_train_batch_size": 1,  # Forzar a 1 para evitar OOM
    "learning_rate": hp["learning_rate"],
    "max_grad_norm": 0.3,  # Recorte de gradientes para prevenir desbordamientos numéricos en FP16
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

  # Si se solicita Logit Distillation y no se pasó teacher_model desde afuera, cargarlo temporalmente
  temp_teacher_loaded = False
  if teacher_model is None and Settings.USE_LOGIT_DISTILLATION:
    teacher_path = Settings.get_teacher_model_path()
    print(f"[FineTuner] [Logit Distillation] Cargando Maestro ({teacher_path}) en 4-bit para extracción de logits...")
    try:
      from transformers import BitsAndBytesConfig
      bnb_cfg = (
          BitsAndBytesConfig(
              load_in_4bit=True,
              bnb_4bit_quant_type="nf4",
              bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
              bnb_4bit_use_double_quant=True,
          )
          if BitsAndBytesConfig
          else None
      )
      teacher_model = AutoModelForCausalLM.from_pretrained(
          teacher_path,
          quantization_config=bnb_cfg if torch.cuda.is_available() else None,
          device_map="auto" if torch.cuda.is_available() else None,
          torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
      )
      teacher_model.eval()
      temp_teacher_loaded = True
      print("✅ [FineTuner] Maestro cargado con éxito para Logit Distillation.")
    except Exception as e_load:
      print(f"[FineTuner] ⚠️ No se pudo cargar Maestro para logits: {e_load}. Fallback a SFT estándar.")
      teacher_model = None
      temp_teacher_loaded = False

  # Instanciar el Trainer correspondiente (Logit KD vs SFT tradicional)
  if Settings.USE_LOGIT_DISTILLATION and teacher_model is not None:
    print(f"[FineTuner] 🎯 Activando LogitDistillationTrainer (Temp={Settings.KD_TEMPERATURE}, Alpha={Settings.KD_ALPHA})")
    trainer = LogitDistillationTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        teacher_model=teacher_model,
        temperature=Settings.KD_TEMPERATURE,
        alpha=Settings.KD_ALPHA,
    )
  else:
    print("[FineTuner] ℹ️ Usando Trainer estándar (SFT supervisado sin logits).")
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)

  trainer.train()

  # Si cargamos un Teacher temporal, destruirlo y purgar VRAM inmediatamente
  if temp_teacher_loaded and teacher_model is not None:
    print("[FineTuner] Purgando Maestro temporal de VRAM...")
    del teacher_model
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    gc.collect()

  if hasattr(model, "config") and hasattr(model.config, "use_cache"):
      model.config.use_cache = True

  adapter_path = str(out_dir / "adapter")
  model.save_pretrained(adapter_path)
  tokenizer.save_pretrained(adapter_path)
  print(f"[FineTuner] Adapter LoRA guardado en {adapter_path}")
  return adapter_path
