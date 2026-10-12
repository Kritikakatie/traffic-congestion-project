"""
Verify the output of a streaming run against the real dataset and summarise it:
completeness, value equality with the source cube, Spark-vs-NumPy level agreement, alert de-duplication, throughput.
Works for a Kafka run and for the offline file-source test.   python -m analysis.verify_stream [--label text]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.config import get_path, load_config, setup_logging
from common.congestion import Calibration, classify, congestion_index
from common.data import load_processed

log = setup_logging("verify_stream")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="unspecified source")
    args = ap.parse_args()
    cfg = load_config()
    sd, res = get_path(cfg, "stream_dir"), get_path(cfg, "results_dir", True)
    rec, ticks = pd.read_parquet(sd / "records"), pd.read_parquet(sd / "tick_summary")
    cube, days, sensors = load_processed(cfg)
    cal = Calibration.load(res / "calibration.json")
    out = {"source": args.label, "records": int(len(rec)), "distinct_sensor_ticks": int(len(rec[["sensor_id", "tick"]].drop_duplicates())),
           "ticks": int(ticks["tick"].nunique()), "days_in_stream": sorted(int(d) for d in rec["day_index"].unique())}
    sidx = {int(s): i for i, s in enumerate(sensors["sensor_id"])}
    r = rec.sort_values(["tick", "sensor_id"])
    src = cube[r["day_index"].to_numpy(), r["sensor_id"].map(sidx).to_numpy(), r["slot"].to_numpy()]
    out["max_abs_occupancy_difference_vs_source"] = float(np.abs(src - r["occupancy"].to_numpy()).max())
    exp = classify(congestion_index(src, cal.reference_occupancy), cal.cutoffs)
    out["spark_vs_numpy_level_agreement_percent"] = float(100 * (exp == r["level_code"].to_numpy()).mean())
    out["level_counts"] = {n: int((r["level_code"] == i).sum()) for i, n in enumerate(["LOW", "MODERATE", "HIGH", "SEVERE"])}
    out["raw_high_or_severe_sensor_ticks"] = int((r["level_code"] >= 2).sum())
    a = get_path(cfg, "alerts_dir") / "alerts.csv"
    if a.exists():
        al = pd.read_csv(a)
        out["alerts_raised"] = int(len(al)); out["alerts_by_type"] = al["alert_type"].value_counts().to_dict()
        out["sensors_alerted"] = int(al["sensor_id"].nunique())
        out["alert_reduction_factor"] = round(out["raw_high_or_severe_sensor_ticks"] / max(len(al), 1), 1)
    m = sd / "stream_metrics.csv"
    if m.exists():
        sm = pd.read_csv(m)
        out["micro_batches"] = int(len(sm)); out["mean_records_per_second"] = round(float(sm["records_per_second"].mean()), 1)
        out["median_records_per_second"] = round(float(sm["records_per_second"].median()), 1)
        lat = sm["avg_end_to_end_latency_s"].dropna()
        out["mean_end_to_end_latency_s"] = round(float(lat.mean()), 2) if len(lat) else None
    w = sd / "windows"
    out["window_rows"] = int(len(pd.read_parquet(w))) if w.exists() and any(w.glob("*.parquet")) else 0
    (res / "stream_verification.json").write_text(json.dumps(out, indent=2))
    log.info(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
