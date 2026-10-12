"""Loading of the processed dataset and small shared constants."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import get_path

LEVEL_COLORS = {"LOW": "#3b9c6d", "MODERATE": "#e0b341", "HIGH": "#e07a3a", "SEVERE": "#c2362f"}


def load_processed(cfg: dict):
    """Return (cube[days, sensors, slots] float32, days DataFrame, sensors DataFrame)."""
    p = get_path(cfg, "processed_dir")
    cube_path = p / "occupancy_cube.npy"
    if not cube_path.exists():
        raise FileNotFoundError("Processed data not found. Run: python -m preprocessing.preprocess")
    return np.load(cube_path), pd.read_csv(p / "days.csv"), pd.read_csv(p / "sensors.csv")


def n_train_days(days: pd.DataFrame) -> int:
    return int((days["split"] == "train").sum())
