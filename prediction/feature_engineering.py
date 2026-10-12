"""
Feature engineering for congestion prediction.

Task: given what a sensor has shown up to slot t, predict its congestion LEVEL at slot t + horizon
(horizon_slots x 10 min, default 30 min).

No-leakage rules enforced here:
  * features use ONLY slots t-5 ... t of the same day (never anything after t)
  * the target is read from slot t + horizon of the same day
  * the per-sensor baseline (sensor_mean_ci) is computed on TRAIN days only
  * train/test samples come from different, chronologically separated days
"""
from __future__ import annotations

import numpy as np

from common.congestion import classify

HISTORY = 6   # slots of history needed (t-5 ... t) = 60 minutes
FEATURE_NAMES = ["ci_t", "ci_lag1", "ci_lag2", "ci_lag3", "roll_mean_3", "roll_mean_6",
                 "diff_1", "diff_3", "slot", "dow", "is_weekend", "sensor_mean_ci"]


def build_features(ci_window: np.ndarray, slot, dow, sensor_mean_ci) -> np.ndarray:
    """ci_window: (N, 6) congestion index at slots t-5..t (oldest first). slot/dow/sensor_mean_ci: scalars or (N,)."""
    w = np.asarray(ci_window, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] != HISTORY:
        raise ValueError(f"ci_window must have shape (N, {HISTORY}), got {w.shape}")
    n = len(w)
    slot, dow, smean = (np.broadcast_to(np.asarray(a, dtype=np.float32), (n,)) for a in (slot, dow, sensor_mean_ci))
    return np.column_stack([
        w[:, 5], w[:, 4], w[:, 3], w[:, 2],              # current value and lags 1..3
        w[:, 3:6].mean(axis=1), w.mean(axis=1),          # rolling means over 30 and 60 minutes
        w[:, 5] - w[:, 4], w[:, 5] - w[:, 2],            # short-term trends
        slot, dow, (dow >= 5).astype(np.float32), smean,
    ]).astype(np.float32)


def sample_dataset(ci_cube: np.ndarray, dow_by_day: np.ndarray, sensor_mean_ci: np.ndarray, cutoffs,
                   day_lo: int, day_hi: int, n_samples: int, horizon: int, rng: np.random.Generator):
    """Random (day, sensor, slot) samples from days [day_lo, day_hi). Returns X, y (level codes), meta."""
    n_sensors = ci_cube.shape[1]
    d = rng.integers(day_lo, day_hi, n_samples)
    s = rng.integers(0, n_sensors, n_samples)
    t = rng.integers(HISTORY - 1, ci_cube.shape[2] - horizon, n_samples)       # t in [5, 143-horizon]
    window = ci_cube[d[:, None], s[:, None], t[:, None] + np.arange(-(HISTORY - 1), 1)]
    X = build_features(window, t, dow_by_day[d], sensor_mean_ci[s])
    y = classify(ci_cube[d, s, t + horizon], cutoffs)
    return X, y, {"day": d, "sensor": s, "slot": t}
