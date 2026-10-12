"""Configuration loading, path resolution and logging helpers shared by all modules."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml and apply environment-variable overrides."""
    cfg_path = Path(path or os.environ.get("CONFIG_PATH", PROJECT_ROOT / "config" / "config.yaml"))
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if os.environ.get("STREAM_DELAY_SECONDS"):
        cfg["stream"]["stream_delay_seconds"] = float(os.environ["STREAM_DELAY_SECONDS"])
    if os.environ.get("KAFKA_BOOTSTRAP_SERVERS"):
        cfg["kafka"]["bootstrap_servers"] = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    if os.environ.get("KAFKA_TOPIC"):
        cfg["kafka"]["topic"] = os.environ["KAFKA_TOPIC"]
    return cfg


def get_path(cfg: dict, key: str, create: bool = False) -> Path:
    """Absolute path for a key of cfg['paths'] (no hard-coded locations anywhere)."""
    p = PROJECT_ROOT / cfg["paths"][key]
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger(name)


# Day-of-week naming. Internally we use Python's convention: Monday=0 ... Sunday=6.
DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
