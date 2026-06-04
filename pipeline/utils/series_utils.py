"""Utilidades para parseo y augmentación de series temporales."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np


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
  ts = parse_time_series_metadata(meta.get("time_series"))
  if isinstance(ts, dict):
    lengths: List[int] = []
    for block_key in ("input", "output"):
      block = ts.get(block_key, [])
      if isinstance(block, list):
        for item in block:
          if isinstance(item, dict) and "serie" in item:
            serie = item["serie"]
            if isinstance(serie, list):
              lengths.append(len(serie))
    return max(lengths) if lengths else 0
  if isinstance(ts, list):
    return len(ts)
  return 0


def extract_numeric_series(seed_entry: Dict[str, Any]) -> List[float]:
  """Extrae una serie numérica representativa de una semilla RAG."""
  raw = seed_entry.get("series")
  if isinstance(raw, list) and raw and isinstance(raw[0], (int, float)):
    return [float(x) for x in raw]

  meta = seed_entry.get("metadata", {})
  ts = parse_time_series_metadata(meta.get("time_series"))
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
  domains = {
    s.get("metadata", {}).get("domain", "unknown") for s in seeds
  }
  lengths = [series_length_from_metadata(s.get("metadata", {})) for s in seeds]
  length_var = float(np.std(lengths)) if lengths else 0.0
  domain_div = len(domains) / len(seeds)
  return min(1.0, 0.5 * domain_div + 0.5 * min(1.0, length_var / 50.0))


def extend_series(
  values: List[float],
  target_len: int,
  rng: Optional[np.random.Generator] = None,
) -> List[float]:
  """Extiende una serie hasta target_len con continuidad suave + ruido leve."""
  if not values:
    return [0.0] * target_len
  rng = rng or np.random.default_rng(42)
  out = list(values)
  while len(out) < target_len:
    window = out[-min(12, len(out)) :]
    delta = float(np.mean(np.diff(window))) if len(window) > 1 else 0.0
    noise = float(rng.normal(0, max(0.01, np.std(window) * 0.05)))
    out.append(out[-1] + delta + noise)
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
