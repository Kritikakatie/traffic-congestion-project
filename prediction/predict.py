"""Load the trained model and predict future congestion levels (used by the dashboard)."""
from __future__ import annotations

import joblib
import numpy as np

from common.config import get_path
from prediction.feature_engineering import build_features


def load_bundle(cfg: dict) -> dict:
    path = get_path(cfg, "models_dir") / "congestion_rf.joblib"
    if not path.exists():
        raise FileNotFoundError("Model not found. Run: python -m prediction.train_model")
    return joblib.load(path)


def predict_levels(bundle: dict, ci_window: np.ndarray, slot: int, dow: int, sensor_mean_ci: np.ndarray):
    """Returns (predicted level code per sensor, probability of HIGH-or-SEVERE per sensor)."""
    X = build_features(ci_window, slot, dow, sensor_mean_ci)
    model = bundle["model"]
    proba = model.predict_proba(X)
    pred = model.classes_[np.argmax(proba, axis=1)].astype(int)
    p_high = proba[:, [i for i, c in enumerate(model.classes_) if c >= 2]].sum(axis=1)
    return pred, p_high
