"""Cliente unificado para modelos instruct (Llama 3.2, etc.)."""

from __future__ import annotations

import gc
import json
import re
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.config.settings import Settings


def extract_json_from_text(text: str) -> dict:
  if not text or not text.strip():
    return {}
  cleaned = text.strip()
  fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
  if fenced:
    cleaned = fenced.group(1)
  try:
    return json.loads(cleaned)
  except json.JSONDecodeError:
    pass
  match = re.search(r"\{.*\}", cleaned, re.DOTALL)
  if match:
    try:
      return json.loads(match.group(0))
    except json.JSONDecodeError:
      return {}
  return {}


class LLMClient:
  def __init__(self, model_path: str, role_name: str = "asistente"):
    self.model_path = model_path
    self.role_name = role_name
    print(f"[LLMClient] Cargando {role_name}: {model_path}")
    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    # Cargamos el modelo base en float16 explícitamente para reducir la RAM a la mitad
    # FIX: No usamos device_map="mps" porque la librería 'accelerate' de HF tiene un bug en Mac que duplica 
    # la memoria (mantiene copias en CPU y MPS con buffers gigantes). Usamos .to("mps") para un envío directo.
    self._base_model = AutoModelForCausalLM.from_pretrained(
      model_path,
      torch_dtype=torch.float16,
      trust_remote_code=True,
      attn_implementation="eager", # FIX: Evita el OOM de 26GB en Mac M-Series (Qwen2 SDPA memory leak)
    ).to("mps")
   
    self._base_model.eval()
    # El modelo activo comienza siendo el base
    self.model = self._base_model
  
  def get_base_model(self):
        """Retorna el modelo base limpio (sin adaptadores LoRA)."""
        return self._base_model
  
  def load_adapter(self, adapter_path: str) -> None:
        """Carga un adaptador LoRA sobre el modelo base. Si ya había adaptador, lo reemplaza."""
        from peft import PeftModel

        # Si el modelo activo no es el base (es decir, ya tenía adaptador), lo liberamos
        if self.model is not self._base_model:
            del self.model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            

        print(f"[LLMClient] Cargando adapter LoRA: {adapter_path}", flush=True)
        self.model = PeftModel.from_pretrained(self._base_model, adapter_path)
        self.model.eval()
    
  def unload_adapter(self):
        """Descarga el adaptador y vuelve al modelo base."""
        if self.model is not self._base_model:
            del self.model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            self.model = self._base_model
            print("[LLMClient] Adaptador descargado, modelo base restaurado.")

  def generate_text(
    self,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.2,
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
      prompt = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
      )
    else:
      prompt = f"{system_prompt}\n\nUsuario: {user_prompt}\n\nAsistente:"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
    max_new = max_new_tokens or Settings.LLM_MAX_NEW_TOKENS
    with torch.no_grad():
      try:
        output_ids = self.model.generate(
          **inputs,
          max_new_tokens=max_new,
          do_sample=temperature > 0,
          temperature=temperature if temperature > 0 else None,
          pad_token_id=self.tokenizer.pad_token_id,
          eos_token_id=self.tokenizer.eos_token_id,
        )
      except RuntimeError as e:
        if "probability tensor" in str(e).lower() or "nan" in str(e).lower():
          print(f"[LLMClient] Aviso: Error numérico en sampling (MPS). Usando greedy decoding como fallback.")
          output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
          )
        else:
          raise
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # --- FIX: Evitar Fugas de Memoria en MPS (Apple Silicon) ---
    import gc
    del inputs
    del output_ids
    del new_tokens
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # ------------------------------------------------------------
        
    return text

  

  def generate_json(
    self,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
  ) -> dict:
    text = self.generate_text(
      user_prompt=user_prompt,
      system_prompt=system_prompt,
      max_new_tokens=max_new_tokens,
      temperature=0.1,
    )
    return extract_json_from_text(text)
