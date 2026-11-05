"""Utilidades para parseo y augmentación de series temporales."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np


def normalize_frequency(freq: Optional[str]) -> str:
  """Normaliza descripciones de frecuencia en lenguaje natural (español o inglés)
  o abreviaciones hacia alias de frecuencia compatibles con Pandas.
  
  Soporta múltiplos (ej. '5 minutos' -> '5min', '30 segundos' -> '30s', '2 horas' -> '2h').
  Si no se reconoce, devuelve 'D' (diario) por defecto.
  """
  if not freq or not isinstance(freq, str):
    return "D"

  clean = freq.strip().lower()
  clean = re.sub(r"^(por|cada|every|per|a)\s+", "", clean)
  clean = re.sub(r"\s+", " ", clean).strip()

  frequency_map = {
      # Segundos -> 's'
      "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s", "secondly": "s",
      "seg": "s", "segs": "s", "segundo": "s", "segundos": "s", "segundal": "s",
      # Minutos -> 'min'
      "min": "min", "mins": "min", "minute": "min", "minutes": "min", "minutely": "min",
      "minuto": "min", "minutos": "min", "minutal": "min", "minutario": "min", "t": "min",
      # Horas -> 'h'
      "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h", "hourly": "h",
      "hora": "h", "horas": "h", "horario": "h", "horaria": "h", "horarios": "h", "horarias": "h",
      # Días laborales / hábiles -> 'B'
      "b": "B", "bday": "B", "bdays": "B", "business day": "B", "business days": "B",
      "weekday": "B", "weekdays": "B", "laboral": "B", "laborales": "B", "habil": "B",
      "habiles": "B", "hábil": "B", "hábiles": "B", "dia laboral": "B", "dias laborales": "B",
      "dia habil": "B", "dias habiles": "B", "día hábil": "B", "días hábiles": "B",
      # Días calendario -> 'D'
      "d": "D", "day": "D", "days": "D", "daily": "D",
      "dia": "D", "dias": "D", "día": "D", "días": "D", "diario": "D", "diaria": "D",
      "diarios": "D", "diarias": "D",
      # Semanas -> 'W'
      "w": "W", "wk": "W", "wks": "W", "week": "W", "weeks": "W", "weekly": "W",
      "sem": "W", "semana": "W", "semanas": "W", "semanal": "W", "semanales": "W",
      # Meses -> 'MS' (Month Start)
      "m": "MS", "mo": "MS", "mos": "MS", "month": "MS", "months": "MS", "monthly": "MS",
      "mes": "MS", "meses": "MS", "mensual": "MS", "mensuales": "MS",
      # Trimestres -> 'QS'
      "q": "QS", "quarter": "QS", "quarters": "QS", "quarterly": "QS",
      "trimestre": "QS", "trimestres": "QS", "trimestral": "QS", "trimestrales": "QS",
      # Años -> 'YS' (Year Start)
      "y": "YS", "yr": "YS", "yrs": "YS", "year": "YS", "years": "YS", "yearly": "YS",
      "annual": "YS", "annually": "YS", "a": "YS", "año": "YS", "años": "YS",
      "anio": "YS", "anios": "YS", "anual": "YS", "anuales": "YS",
  }

  if clean in frequency_map:
    return frequency_map[clean]

  # Extraer multiplicador numérico si existe
  m_num = re.match(r"^(\d+)\s*([a-zA-Záéíóúñ\s]+)$", clean)
  if m_num:
    mult = m_num.group(1)
    u = m_num.group(2).strip()
    if u in frequency_map:
      return f"{mult}{frequency_map[u]}"

  # Verificar si ya es un código de Pandas reconocido
  valid_pandas_codes = {"s", "min", "h", "D", "B", "W", "MS", "ME", "QS", "QE", "YS", "YE"}
  for code in valid_pandas_codes:
    if freq.strip() == code:
      return code

  return "D"


def parser_search_query(value: str) -> str:
  palabras = ["series temporales", "time serie","serie", "series", "temporal", "temporales", "time", "series" ]

  for palabra in palabras:
    value = value.replace(palabra,"")

  return value  

def parse_time_series_metadata(meta_value: Any) -> Any:
  if meta_value is None:
    return {}
  if isinstance(meta_value, (dict, list)):
    return meta_value
  if isinstance(meta_value, str):
    try:
      return json.loads(meta_value)
    except json.JSONDecodeError:
      return {}
  return {}


def series_length_from_metadata(meta: Dict[str, Any]) -> int:
  ts = parse_time_series_metadata(meta.get("series"))
  if isinstance(ts, dict):
    lengths: List[int] = []
    for block_key in ("input", "output"):
      block = ts.get(block_key, [])
      if isinstance(block, list):
        for item in block:
          if isinstance(item, dict) and "serie" in item:
            serie_lenght = item["longitud_puntos"]
            if isinstance(serie_lenght, int):
              lengths.append(serie_lenght)
    return max(lengths) if lengths else 0
  if isinstance(ts, list):
    return len(ts)
  return 0


def extract_numeric_series(seed_entry: Dict[str, Any]) -> List[float]:
  """Extrae una serie numérica representativa de una semilla RAG."""
  raw = seed_entry.get("series")
  if isinstance(raw, list) and raw:
    nums = []
    for x in raw:
      try:
        nums.append(float(x))
      except (TypeError, ValueError):
        continue
    if len(nums) >= 3:
      return nums

  meta = seed_entry.get("metadata", {})
  ts = parse_time_series_metadata(seed_entry.get("series"))
  if isinstance(ts, dict):
    for block_key in ("output", "input"):
      block = ts.get(block_key, [])
      if isinstance(block, list):
        for item in block:
          if isinstance(item, dict) and "serie" in item:
            serie = item["serie"]
            nums = []
            for v in serie:
              try:
                nums.append(float(v))
              except (TypeError, ValueError):
                continue
            if len(nums) >= 3:
              return nums
  return []


def diversity_score(seeds: List[Dict[str, Any]]) -> float:
    if len(seeds) <= 1:
        return 0.0
    
    # 1. Diversidad de dominios (ignorando "unknown")
    domains = {
        s.get("metadata", {}).get("domain") 
        for s in seeds 
        if s.get("metadata", {}).get("domain")  # Filtra None/unknown
    }
    domain_div = len(domains) / len(seeds) if domains else 0.0
    
    # 2. Variabilidad de longitudes (con manejo de errores)
    lengths = [
        series_length_from_metadata(s) 
        for s in seeds
    ]
    
    
    # Normalización más robusta (usando max-min en lugar de 50 fijo)
    max_len = max(lengths) if lengths else 1
    min_len = min(lengths) if lengths else 0
    range_normalized = (max_len - min_len) / max_len if max_len > 0 else 0
    
    # Score final
    return min(1.0, 0.5 * domain_div + 0.5 * range_normalized)


def extend_series(
  values: List[float],
  target_len: int,
  rng: Optional[np.random.Generator] = None,
  profile_id: str = "P1_ESTABLE",
  target_mean: Optional[float] = None,
  series_seed: Optional[int] = None,
) -> List[float]:
  """Extiende una serie hasta target_len aplicando el comportamiento matemático del profile_id.
  
  Soporta Desacoplamiento Morfológico-Escalar: si se pasa target_mean (ej. 220.0),
  las semillas se re-escalan preservando su dinámica/forma pero ancladas a la magnitud solicitada.
  Soporta diferenciación estocástica por serie vía series_seed.
  """
  if not values:
    val = target_mean if target_mean is not None else 0.0
    return [float(val)] * target_len
  if rng is None:
    rng = np.random.default_rng(series_seed if series_seed is not None else 42)
  out = [float(x) for x in values]

  # Desacoplamiento Morfológico-Escalar:
  # Si la media de las semillas difiere sustancialmente del objetivo del usuario (>15%),
  # normalizamos la forma (Z-score) y la proyectamos sobre la escala objetivo.
  if target_mean is not None and len(out) > 0:
    vals_arr = np.array(out, dtype=float)
    current_mean = float(np.mean(vals_arr))
    current_std = float(np.std(vals_arr))
    if abs(current_mean - target_mean) > 0.15 * max(abs(target_mean), 1.0):
      s_std = current_std if current_std > 1e-4 else (abs(target_mean) * 0.02)
      target_std = min(s_std, abs(target_mean) * (0.02 if profile_id == "P1_ESTABLE" else 0.05))
      target_std = max(0.1, target_std)
      normalized_seed = (vals_arr - current_mean) / (current_std if current_std > 1e-4 else 1.0)
      rescaled_seed = target_mean + (normalized_seed * target_std)
      if target_mean > 0:
        rescaled_seed = [max(0.0, float(x)) for x in rescaled_seed]
      out = [float(x) for x in rescaled_seed]

    # Variación sutil inter-serie / inter-sujeto si se especifica una semilla de serie
    if series_seed is not None and series_seed != 42:
      subject_shift = float(rng.normal(0, min(0.15, abs(target_mean) * 0.003)))
      out = [float(x + subject_shift) for x in out]
      if target_mean > 0:
        out = [max(0.0, float(x)) for x in out]

  base_val = float(np.mean(out)) if out else (target_mean if target_mean is not None else 0.0)
  t = len(out)

  while len(out) < target_len:
    base_mean = abs(target_mean) if target_mean is not None else (np.mean(np.abs(out)) if out else 1.0)
    raw_std = np.std(out) if len(out) > 1 else 1.0

    if profile_id == "P1_ESTABLE":
      # Para P1, el ruido no debería exceder el 2% de la media
      noise_std = min(raw_std, base_mean * 0.02)
    elif profile_id in ["P2_ESTACIONAL", "P3_TENDENCIA_ESTACIONAL"]:
        # Para estacionales, ruido moderado (hasta 5%)
        noise_std = min(raw_std, base_mean * 0.05)
    else:
        # Para ruidosos o complejos, permitir más variabilidad (hasta 15%)
        noise_std = min(raw_std, base_mean * 0.15)
        
    noise_std = max(0.1, noise_std)
    noise = float(rng.normal(0, noise_std))
    
    if profile_id == "P1_ESTABLE":
        val = base_val + noise
        
    elif profile_id == "P2_ESTACIONAL":
        period = 7.0
        amplitude = max(2.0, base_mean * 0.1)
        val = base_val + amplitude * np.sin(2 * np.pi * t / period) + noise
        
    elif profile_id == "P3_TENDENCIA_ESTACIONAL":
        period = 12.0
        amplitude = max(2.0, base_mean * 0.1)
        trend = 0.05 * t
        val = base_val + trend + amplitude * np.sin(2 * np.pi * t / period) + noise
        
    elif profile_id == "P4_TENDENCIA_RUIDO":
        trend = -0.05 * t
        heavy_noise = float(rng.normal(0, noise_std * 2.0))
        val = base_val + trend + heavy_noise
        
    elif profile_id == "P5_COMPLEJO":
        if t < target_len * 0.33:
            val = base_val + noise
        elif t < target_len * 0.66:
            val = out[-1] + (base_mean * 0.05) + float(rng.normal(0, noise_std * 1.5))
        else:
            val = out[-1] + float(rng.normal(0, noise_std * 3.0))
            
    else:
        window = out[-min(12, len(out)) :]
        delta = float(np.mean(np.diff(window))) if len(window) > 1 else 0.0
        val = out[-1] + delta + noise
        
    if target_mean is not None and target_mean > 0:
        val = max(0.0, float(val))
    out.append(float(val))
    t += 1
    
  return out[:target_len]


def seeds_statistics(seeds: List[Dict[str, Any]]) -> Dict[str, float]:
  all_vals: List[float] = []
  for s in seeds:
    all_vals.extend(extract_numeric_series(s))
  if not all_vals:
    return {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
  arr = np.array(all_vals, dtype=float)
  return {
    "mean": float(np.mean(arr)),
    "std": float(max(np.std(arr), 1e-6)),
    "min": float(np.min(arr)),
    "max": float(np.max(arr)),
  }
