"""Cliente unificado para modelos instruct (Llama 3.2, etc.)."""

from __future__ import annotations

import gc
import json
import os
import re
from typing import Any, Dict, Optional

# Prevenir fragmentación de VRAM en PyTorch (recomendado ante cargas intensivas de atención)
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from pipeline.config.settings import Settings


def fix_premature_root_closing(text: str) -> str:
  """Repara cierres prematuros del objeto JSON raíz causados por LLMs pequeños."""
  out = []
  in_str = False
  escape = False
  depth = 0
  i = 0
  while i < len(text):
    c = text[i]
    if escape:
      escape = False
      out.append(c)
      i += 1
      continue
    if c == "\\":
      escape = True
      out.append(c)
      i += 1
      continue
    if c == "\"":
      in_str = not in_str
      out.append(c)
      i += 1
      continue
    if not in_str:
      if c == "{":
        depth += 1
        out.append(c)
        i += 1
        continue
      elif c == "}":
        rest = text[i + 1 :].lstrip()
        # Si cerraría la raíz (depth == 1) pero todavía hay campos/claves más adelante
        if depth == 1 and rest.startswith((",", "\"", "\n")):
          if "}" in rest:
            i += 1
            continue
        depth -= 1
        out.append(c)
        i += 1
        continue
    out.append(c)
    i += 1
  return "".join(out)


def strip_json_comments(text: str) -> str:
  """Elimina comentarios tipo JavaScript (// y /* ... */) fuera de cadenas de texto."""
  out = []
  in_str = False
  escape = False
  i = 0
  n = len(text)
  while i < n:
    c = text[i]
    if escape:
      escape = False
      out.append(c)
      i += 1
      continue
    if c == "\\":
      escape = True
      out.append(c)
      i += 1
      continue
    if c == '"':
      in_str = not in_str
      out.append(c)
      i += 1
      continue
    if not in_str:
      if c == '/' and i + 1 < n and text[i + 1] == '/':
        i += 2
        while i < n and text[i] != '\n':
          i += 1
        continue
      elif c == '/' and i + 1 < n and text[i + 1] == '*':
        i += 2
        while i + 1 < n and not (text[i] == '*' and text[i + 1] == '/'):
          i += 1
        i += 2
        continue
    out.append(c)
    i += 1
  return "".join(out)


def extract_json_from_text(text: str) -> dict:
  if not text or not text.strip():
    return {}
  cleaned = text.strip()
  # 0.5. Si el modelo es de tipo razonamiento (ej. DeepSeek-R1), remover bloques <think>...</think>
  cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
  fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
  if fenced:
    cleaned = fenced.group(1)
  else:
    fenced_arr = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if fenced_arr:
      cleaned = fenced_arr.group(1)
    else:
      # Si viene con bloques de código pero sin cerrar las llaves correctamente
      cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
      cleaned = re.sub(r"\s*```$", "", cleaned)
      match = re.search(r"\{.*\}", cleaned, re.DOTALL)
      if match:
        cleaned = match.group(0)
      elif "{" in cleaned:
        cleaned = cleaned[cleaned.find("{") :]
      elif "[" in cleaned and "]" in cleaned:
        cleaned = cleaned[cleaned.find("[") : cleaned.rfind("]") + 1]

  # 1. Eliminar comentarios JS (// o /* */) que el LLM suele insertar
  cleaned = strip_json_comments(cleaned)

  # 1.5. Reparar llaves dobles de apertura/cierre (ej. '{{ ... }}' o '{{"key": ...}}')
  if cleaned.startswith("{{") and cleaned.endswith("}}"):
    cleaned = cleaned[1:-1].strip()
  elif cleaned.startswith("{{"):
    cleaned = cleaned[1:].strip()

  # 2. Reparar cierres con paréntesis en lugar de llaves (ej. '[[1, 2]])' -> '[[1, 2]]}')
  cleaned = re.sub(r"\)\s*$", "}", cleaned)
  cleaned = re.sub(r"\]\s*\)\s*$", "]}}", cleaned)

  # 3. Reparar cierres prematuros de la raíz (ej. '... "justification": "..." }, "suggested_search": ... }')
  cleaned = fix_premature_root_closing(cleaned)

  # 4. Reparar comillas simples o mixtas al final de strings o antes de saltos de línea/llaves
  cleaned = re.sub(r"'\s*\n\s*\"", '"', cleaned)
  cleaned = re.sub(r"'(\s*\n?\s*\})", r'"\1', cleaned)
  cleaned = cleaned.replace("'\n}", "\"\n}")

  # 5. Reparar comas faltantes entre campos (ej. "valor" \n "otro_campo" -> "valor", \n "otro_campo")
  cleaned = re.sub(r'"(\s*\n\s*)"', r'",\1"', cleaned)

  # 6. Reparar comas sobrantes antes de cerrar llaves/corchetes (ej. [1, 2, ])
  cleaned = re.sub(r",\s*([\}\]])", r"\1", cleaned)

  # 6.5. Reparar comillas dobles duplicadas antes del cierre (ej. "texto." "" })
  cleaned = re.sub(r'""\s*\}', '"}', cleaned)

  # 7. Intentar parseo estándar con strict=False (acepta saltos de línea crudos en cadenas)
  try:
    obj = json.loads(cleaned, strict=False)
    if isinstance(obj, dict):
      return obj
    elif isinstance(obj, list):
      if obj and isinstance(obj[0], (int, float)):
        return {"synthetic_series": [obj]}
      return {"synthetic_series": obj}
  except json.JSONDecodeError:
    pass

  # 8. Fallback: Intentar rescatar al menos el bloque "values" si está presente
  values_match = re.search(
    r'"values"\s*:\s*(\{[^{}]*\}|\{(?:[^{}]*|\{[^{}]*\})*\})',
    cleaned,
    re.DOTALL,
  )
  if values_match:
    try:
      v_data = json.loads(values_match.group(1), strict=False)
      conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', cleaned)
      conf = float(conf_match.group(1)) if conf_match else 0.9
      return {"values": v_data, "confidence": conf}
    except Exception:
      pass

  # 8.5. Fallback para hiperparámetros (ej. cuando el modelo responde en asignaciones Python como en P2)
  if "epochs" in text or "batch_size" in text or "learning_rate" in text:
    hp_extracted = {}
    m_ep = re.search(r"epochs?\s*[:=]\s*(\d+)", text, re.I)
    if m_ep:
      hp_extracted["epochs"] = int(m_ep.group(1))
    m_bs = re.search(r"batch[_\s-]*size\s*[:=]\s*(\d+)", text, re.I)
    if m_bs:
      hp_extracted["batch_size"] = int(m_bs.group(1))
    m_lr = re.search(r"learning[_\s-]*rate\s*[:=]\s*([\d.eE+-]+)", text, re.I)
    if m_lr:
      try:
        hp_extracted["learning_rate"] = float(m_lr.group(1).rstrip("."))
      except ValueError:
        pass
    if hp_extracted:
      return hp_extracted

  # 9. Fallback: Usar raw_decode para ignorar caracteres sobrantes al final
  try:
    decoder = json.JSONDecoder(strict=False)
    obj, _ = decoder.raw_decode(cleaned)
    if isinstance(obj, dict):
      return obj
    elif isinstance(obj, list):
      if obj and isinstance(obj[0], (int, float)):
        return {"synthetic_series": [obj]}
      return {"synthetic_series": obj}
  except Exception:
    pass

  # 9.5. Fallback: Si el modelo devolvió un arreglo crudo [[...], [...]] o [...] plano
  try:
    array_match = re.search(r"(\[\s*\[.*?\]\s*\]|\[\s*[-+\d.,eE\s]+\])", cleaned, re.DOTALL)
    if array_match:
      arr = json.loads(array_match.group(1), strict=False)
      if isinstance(arr, list):
        if arr and isinstance(arr[0], (int, float)):
          return {"synthetic_series": [arr]}
        return {"synthetic_series": arr}
  except Exception:
    pass

  # 10. Fallback para series sintéticas: rescatar números si el LLM se truncó por tokens o tuvo typos
  if "synthetic_series" in text or "[" in text or "series" in text.lower():
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    floats = [float(x) for x in nums if x != ""]
    if len(floats) >= 10:
      return {"synthetic_series": [floats[:10]]}

  print(
    f"\n[DEBUG LLMClient] JSON Parse Error in extracted block\n--- RAW LLM OUTPUT ---\n{cleaned}\n----------------------\n"
  )
  return {}


class LLMClient:
  def __init__(self, model_path: str, role_name: str = "asistente"):
    self.model_path = model_path
    self.role_name = role_name
    
    if torch.cuda.is_available():
        self.device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        self.device = "mps"
    else:
        self.device = "cpu"

    print(f"[LLMClient] Cargando {role_name} en {self.device}: {model_path}")
    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa" if self.device == "cuda" else "eager",
    }
    # En añadidura: los modelos Gemma (ej. Gemma 2) requieren atención 'eager'
    # debido a su mecanismo de logit soft-capping y sliding window attention.
    # Modelos como Qwen y LLaMA continúan usando 'sdpa' intacto.
    if "gemma" in str(model_path).lower():
        model_kwargs["attn_implementation"] = "eager"

    if self.device == "cuda":
        num_gpus = torch.cuda.device_count()
        use_4bit_env = os.environ.get("USE_4BIT", "").lower() in ("true", "1", "yes")

        gpu_capability = torch.cuda.get_device_capability(0) if num_gpus > 0 else (0, 0)
        is_large_model = any(tag in model_path.lower() for tag in ["7b", "8b", "6.7b", "9b", "14b"])
        # Tesla T4 (sm_75, capability < 8.0) no tiene soporte de hardware para bfloat16.
        # En modelos 7B+ entrenados nativamente en bf16 (Qwen 2.5, Llama 3.1), usar float16 sin cuantizar
        # causa desbordamiento de activaciones en atención (>65504).
        # En T4 o VRAM limitada, activamos 4-bit NF4 (BitsAndBytes) para estabilizar cálculos y reducir VRAM a ~5 GB.
        needs_4bit = (
            use_4bit_env
            or (is_large_model and gpu_capability[0] < 8)
            or (num_gpus == 1 and torch.cuda.get_device_properties(0).total_memory < 18 * 1e9 and is_large_model)
        )

        if needs_4bit and BitsAndBytesConfig is not None:
            try:
                print(f"[LLMClient] Activando cuantización 4-bit (BitsAndBytes NF4) para {model_path} en GPU sm_{gpu_capability[0]}{gpu_capability[1]}.")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                # CRÍTICO: torch_dtype DEBE ser float16 para que embed_tokens y lm_head
                # no se carguen en float32 (ahorra ~2.2 GB de VRAM en GPU 0).
                model_kwargs["torch_dtype"] = torch.float16
            except Exception as e:
                print(f"[LLMClient] Aviso: No se pudo configurar BitsAndBytesConfig ({e}), continuando en fp16.")
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16

        # En CUDA usamos device_map="auto" para repartir automáticamente entre GPUs disponibles (ej. 2x T4 en Kaggle)
        model_kwargs["device_map"] = "auto"
        if num_gpus > 1:
            # En configuraciones multi-GPU (ej. 2x T4 en Kaggle de ~15 GB cada una),
            # Accelerate tiende a saturar GPU 0 al 98% dejando poco margen para atención SDPA y KV cache.
            # Limitamos el peso estático del modelo en GPU 0 para garantizar ~5 GB libres para buffers dinámicos.
            max_mem = {i: "13.5GiB" for i in range(num_gpus)}
            max_mem[0] = "9.5GiB"
            model_kwargs["max_memory"] = max_mem

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )
    else:
        if self.device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
        self._base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        ).to(self.device)
   
    self._base_model.eval()
    # El modelo activo comienza siendo el base
    self.model = self._base_model
  
  def get_base_model(self):
        """Retorna el modelo base limpio (sin adaptadores LoRA)."""
        return self._base_model
  
  def load_adapter(self, adapter_path: str) -> None:
        """Carga un adaptador LoRA sobre el modelo base. Si ya había adaptador, lo reemplaza."""
        try:
            import peft.import_utils
            peft.import_utils.is_torchao_available = lambda: False
            if hasattr(peft.tuners.lora, "torchao"):
                peft.tuners.lora.torchao.is_torchao_available = lambda: False
        except Exception:
            pass
        from peft import PeftModel

        # Si el modelo activo no es el base (es decir, ya tenía adaptador), lo liberamos
        if self.model is not self._base_model:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            

        print(f"[LLMClient] Cargando adapter LoRA: {adapter_path}", flush=True)
        self.model = PeftModel.from_pretrained(self._base_model, adapter_path)
        self.model.eval()
    
  def unload_adapter(self):
        """Descarga el adaptador y vuelve al modelo base."""
        if self.model is not self._base_model:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            self.model = self._base_model
            print("[LLMClient] Adaptador descargado, modelo base restaurado.")

  def generate_text(
    self,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.0,
  ) -> str:
    system_prompt = system_prompt or (
      f"Eres un {self.role_name} experto en series temporales. "
      "Responde de forma precisa y sigue el formato solicitado."
    )
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
    ]
    if hasattr(self.tokenizer, "apply_chat_template"):
      try:
        prompt = self.tokenizer.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True
        )
      except Exception:
        # Fallback para modelos que no admiten rol 'system' directamente (ej. Gemma)
        messages_fallback = [
          {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
        ]
        try:
          prompt = self.tokenizer.apply_chat_template(
            messages_fallback, tokenize=False, add_generation_prompt=True
          )
        except Exception:
          prompt = f"{system_prompt}\n\nUsuario: {user_prompt}\n\nAsistente:"
    else:
      prompt = f"{system_prompt}\n\nUsuario: {user_prompt}\n\nAsistente:"

    target_device = getattr(self.model, "device", None) or self.device
    inputs = self.tokenizer(prompt, return_tensors="pt").to(target_device)
    max_new = max_new_tokens or Settings.LLM_MAX_NEW_TOKENS

    # En float16, temperaturas menores a 0.2 causan desbordamiento a inf/NaN en multinomial.
    # Para JSON o temperaturas bajas, usamos decodificación determinista Greedy (do_sample=False).
    use_sampling = temperature is not None and temperature >= 0.2
    gen_kwargs = {
      "max_new_tokens": max_new,
      "do_sample": use_sampling,
      "pad_token_id": self.tokenizer.pad_token_id,
      "eos_token_id": self.tokenizer.eos_token_id,
    }
    if use_sampling:
      gen_kwargs["temperature"] = float(temperature)
      gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
      output_ids = self.model.generate(
        **inputs,
        **gen_kwargs,
      )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # --- Limpieza de Memoria GPU ---
    import gc
    del inputs
    del output_ids
    del new_tokens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # -------------------------------
        
    return text

  def generate_json(
    self,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.0,
  ) -> dict:
    text = self.generate_text(
      user_prompt=user_prompt,
      system_prompt=system_prompt,
      max_new_tokens=max_new_tokens,
      temperature=temperature,
    )
    return extract_json_from_text(text)
