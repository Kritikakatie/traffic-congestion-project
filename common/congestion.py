"""
Congestion index (CI) and level classification.

The PEMS-SF data only contains lane OCCUPANCY (fraction of time a detector is
occupied by vehicles, 0..1). Speed and flow are NOT available, so the index is
built from occupancy alone:

    CI = clip(occupancy / reference_occupancy, 0, 1)

reference_occupancy and the level cut-offs are calibrated as quantiles of the
TRAINING period only (so nothing from the test period leaks into the labels).
All numbers are configurable in config/config.yaml.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

LEVELS = ["LOW", "MODERATE", "HIGH", "SEVERE"]
LEVEL_CODE = {name: i for i, name in enumerate(LEVELS)}


@dataclass
class Calibration:
    reference_occupancy: float          # occupancy that maps to CI = 1
    cutoffs: list                        # CI cut-offs [moderate, high, severe]
    source: str = "quantiles of training occupancy"

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**json.load(fh))


def calibrate(train_occupancy: np.ndarray, cfg: dict) -> Calibration:
    """Derive reference occupancy and level cut-offs from training occupancy."""
    ccfg = cfg["congestion"]
    if ccfg.get("manual"):
        m = ccfg["manual"]
        return Calibration(float(m["reference_occupancy"]), [float(c) for c in m["cutoffs"]], "manual (config)")
    flat = np.asarray(train_occupancy, dtype=np.float32).ravel()
    q = ccfg["level_quantiles"]
    qs = np.quantile(flat, [ccfg["reference_quantile"], q["moderate"], q["high"], q["severe"]])
    ref = round(float(qs[0]), 6)
    # Occupancy has only 4 decimals, so many observations are EXACTLY equal to a percentile value. The 1e-6
    # tolerance puts such ties into the higher level deterministically, identically in float32 (numpy) and
    # float64 (Spark) arithmetic. No real observation lies inside the tolerance band.
    cutoffs = [round(float(v) / ref - 1e-6, 6) for v in qs[1:]]
    if not (cutoffs[0] < cutoffs[1] <= cutoffs[2]):
        raise ValueError(f"Level cut-offs must be increasing, got {cutoffs}")
    return Calibration(ref, cutoffs)


def congestion_index(occupancy, reference_occupancy: float):
    """Vectorised CI in [0, 1]."""
    return np.clip(np.asarray(occupancy, dtype=np.float32) / np.float32(reference_occupancy), 0.0, 1.0)


def classify(ci, cutoffs) -> np.ndarray:
    """CI -> level code 0..3 (LOW, MODERATE, HIGH, SEVERE). A value equal to a cut-off enters the higher level."""
    return np.digitize(np.asarray(ci), bins=np.asarray(cutoffs, dtype=np.float64), right=False).astype(np.int8)


def level_name(code: int) -> str:
    return LEVELS[int(code)]
