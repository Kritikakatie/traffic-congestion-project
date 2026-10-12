"""
Train and evaluate the Random Forest that predicts the congestion level `horizon` slots ahead.

* chronological split by DAY: first 70% of days train, last 30% test (no random day mixing)
* class imbalance (HIGH/SEVERE are rare): measured, reported with macro/per-class metrics, and class weighting was
  evaluated on a validation split of the train days (prediction/tune_validation.py) - config decides
* compared against two baselines evaluated on the SAME test samples:
    - persistence: "the future level equals the current level"
    - always-LOW (majority class)
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from common.config import get_path, load_config, setup_logging
from common.congestion import LEVELS, Calibration, classify, congestion_index
from common.data import load_processed, n_train_days
from prediction.feature_engineering import FEATURE_NAMES, sample_dataset

log = setup_logging("train_model")


def evaluate(y_true, y_pred) -> dict:
    labels = [0, 1, 2, 3]
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    # operational question for alerts: will the sensor be HIGH or SEVERE?
    bt, bp = y_true >= 2, y_pred >= 2
    tp, fp, fn = int((bt & bp).sum()), int((~bt & bp).sum()), int((bt & ~bp).sum())
    bprec, brec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": {"precision": float(pm), "recall": float(rm), "f1": float(fm)},
        "weighted": {"precision": float(pw), "recall": float(rw), "f1": float(fw)},
        "per_class": {LEVELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])} for i in labels},
        "high_or_severe_binary": {"precision": float(bprec), "recall": float(brec), "f1": float(2 * bprec * brec / max(bprec + brec, 1e-12))},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def run() -> dict:
    cfg = load_config()
    pc = cfg["prediction"]
    res, fig, models = get_path(cfg, "results_dir", True), get_path(cfg, "figures_dir", True), get_path(cfg, "models_dir", True)
    cube, days, _ = load_processed(cfg)
    cal = Calibration.load(res / "calibration.json")          # calibrated on train days only
    ci = congestion_index(cube, cal.reference_occupancy)
    n_tr, dow = n_train_days(days), days["dow"].to_numpy()
    sensor_mean_ci = ci[:n_tr].mean(axis=(0, 2))              # train-only baseline per sensor
    rng = np.random.default_rng(pc["random_state"])
    H = pc["horizon_slots"]

    Xtr, ytr, _ = sample_dataset(ci, dow, sensor_mean_ci, cal.cutoffs, 0, n_tr, pc["train_samples"], H, rng)
    Xte, yte, meta = sample_dataset(ci, dow, sensor_mean_ci, cal.cutoffs, n_tr, len(cube), pc["test_samples"], H, rng)
    log.info("train samples %d (days 0-%d), test samples %d (days %d-%d), horizon %d slots = %d min",
             len(Xtr), n_tr - 1, len(Xte), n_tr, len(cube) - 1, H, H * 10)
    log.info("train class counts %s | test class counts %s", np.bincount(ytr, minlength=4).tolist(), np.bincount(yte, minlength=4).tolist())

    t0 = time.time()
    model = RandomForestClassifier(n_estimators=pc["n_estimators"], max_depth=pc["max_depth"], min_samples_leaf=pc["min_samples_leaf"],
                                   class_weight=pc.get("class_weight"), n_jobs=-1, random_state=pc["random_state"])
    model.fit(Xtr, ytr)
    train_seconds = time.time() - t0
    pred_rf = model.predict(Xte)
    pred_persist = classify(Xte[:, 0], cal.cutoffs).astype(int)       # feature 0 = current CI
    pred_low = np.zeros_like(yte)

    metrics = {
        "horizon_slots": H, "horizon_minutes": H * 10, "train_days": n_tr, "test_days": len(cube) - n_tr,
        "train_samples": int(len(Xtr)), "test_samples": int(len(Xte)), "train_seconds": round(train_seconds, 1),
        "train_class_counts": dict(zip(LEVELS, np.bincount(ytr, minlength=4).tolist())),
        "test_class_counts": dict(zip(LEVELS, np.bincount(yte, minlength=4).tolist())),
        "random_forest": evaluate(yte, pred_rf), "baseline_persistence": evaluate(yte, pred_persist), "baseline_always_low": evaluate(yte, pred_low),
        "feature_importance": dict(zip(FEATURE_NAMES, model.feature_importances_.round(4).tolist())),
    }
    with open(res / "prediction_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    joblib.dump({"model": model, "horizon_slots": H, "feature_names": FEATURE_NAMES, "sensor_mean_ci": sensor_mean_ci,
                 "calibration": {"reference_occupancy": cal.reference_occupancy, "cutoffs": cal.cutoffs}}, models / "congestion_rf.joblib", compress=3)
    make_figures(metrics, fig)
    for name in ["random_forest", "baseline_persistence", "baseline_always_low"]:
        m = metrics[name]
        log.info("%-22s acc %.4f | macro P %.3f R %.3f F1 %.3f | HIGH+ F1 %.3f", name, m["accuracy"], m["macro"]["precision"],
                 m["macro"]["recall"], m["macro"]["f1"], m["high_or_severe_binary"]["f1"])
    return metrics


def make_figures(m: dict, fig) -> None:
    cm = np.array(m["random_forest"]["confusion_matrix"])
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    f, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]*100:.0f}%)", ha="center", va="center", color="white" if norm[i, j] > .5 else "black", fontsize=8)
    ax.set_xticks(range(4), LEVELS, rotation=30); ax.set_yticks(range(4), LEVELS); ax.set_xlabel("predicted level"); ax.set_ylabel("actual level")
    ax.set_title(f"Random Forest confusion matrix (+{m['horizon_minutes']} min, test days)"); f.colorbar(im, label="row-normalised"); f.tight_layout()
    f.savefig(fig / "confusion_matrix.png", dpi=150); plt.close(f)

    imp = sorted(m["feature_importance"].items(), key=lambda kv: kv[1])
    f, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh([k for k, _ in imp], [v for _, v in imp], color="#2f6db5"); ax.set_title("Random Forest feature importance"); f.tight_layout()
    f.savefig(fig / "feature_importance.png", dpi=150); plt.close(f)

    names = ["random_forest", "baseline_persistence", "baseline_always_low"]
    labels = ["Random Forest", "Persistence", "Always LOW"]
    vals = {"Accuracy": [m[n]["accuracy"] for n in names], "Macro F1": [m[n]["macro"]["f1"] for n in names],
            "HIGH+ F1": [m[n]["high_or_severe_binary"]["f1"] for n in names]}
    f, ax = plt.subplots(figsize=(7.5, 4)); w = .26
    for i, (k, v) in enumerate(vals.items()):
        ax.bar(np.arange(3) + (i - 1) * w, v, w, label=k)
    ax.set_xticks(range(3), labels); ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Model vs baselines on held-out days"); f.tight_layout()
    f.savefig(fig / "model_vs_baselines.png", dpi=150); plt.close(f)


if __name__ == "__main__":
    run()
