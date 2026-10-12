"""
Model selection WITHOUT touching the test days: fit on the first 80% of the TRAIN days and
validate on the last 20% of the TRAIN days. Results -> output/results/tuning_validation.json
(this is the evidence behind the settings in config.yaml, including the class-weighting decision).
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from common.config import get_path, load_config, setup_logging
from common.congestion import Calibration, classify, congestion_index
from common.data import load_processed, n_train_days
from prediction.feature_engineering import sample_dataset

log = setup_logging("tune")
CANDIDATES = {
    "RF balanced depth14 leaf50": dict(class_weight="balanced_subsample", max_depth=14, min_samples_leaf=50),
    "RF unweighted depth14 leaf50": dict(class_weight=None, max_depth=14, min_samples_leaf=50),
    "RF unweighted depth18 leaf20": dict(class_weight=None, max_depth=18, min_samples_leaf=20),
    "RF balanced depth10 leaf200": dict(class_weight="balanced_subsample", max_depth=10, min_samples_leaf=200),
}


def run() -> dict:
    cfg = load_config()
    res = get_path(cfg, "results_dir", True)
    cube, days, _ = load_processed(cfg)
    cal = Calibration.load(res / "calibration.json")
    ci = congestion_index(cube, cal.reference_occupancy)
    n_tr, dow, H = n_train_days(days), days["dow"].to_numpy(), cfg["prediction"]["horizon_slots"]
    smean = ci[:n_tr].mean(axis=(0, 2))
    rng, split = np.random.default_rng(1), int(n_tr * 0.8)
    Xf, yf, _ = sample_dataset(ci, dow, smean, cal.cutoffs, 0, split, 200000, H, rng)
    Xv, yv, _ = sample_dataset(ci, dow, smean, cal.cutoffs, split, n_tr, 100000, H, rng)

    def score(pred):
        return {"accuracy": float(accuracy_score(yv, pred)), "macro_f1": float(f1_score(yv, pred, average="macro")),
                "high_or_severe_f1": float(f1_score(yv >= 2, pred >= 2))}
    out = {"persistence_baseline": score(classify(Xv[:, 0], cal.cutoffs)), "validation_days": f"{split}-{n_tr - 1}", "candidates": {}}
    for name, kw in CANDIDATES.items():
        t0 = time.time()
        model = RandomForestClassifier(n_estimators=40, n_jobs=-1, random_state=1, **kw).fit(Xf, yf)
        out["candidates"][name] = dict(score(model.predict(Xv)), fit_seconds=round(time.time() - t0, 1))
        log.info("%s -> %s", name, out["candidates"][name])
    with open(res / "tuning_validation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    run()
