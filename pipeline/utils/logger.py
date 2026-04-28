"""
Logging unificado para el pipeline.

Cada módulo obtiene un logger con nombre `pipeline.<modulo>`.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configura el logging global del pipeline (se llama una sola vez)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("pipeline")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Obtiene un logger con prefijo `pipeline.`."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"pipeline.{name}")
