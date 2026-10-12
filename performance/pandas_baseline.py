"""
SERIAL baseline (single-process pandas) for the benchmark workload.
It processes the daily Parquet files one after another and merges partial sums, so memory stays bounded
even for the full 37M-record dataset. Same computation as spark_jobs.aggregation.batch_workload.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["sensor_id", "occupancy", "dow", "hour"]


def list_day_files(long_dir: Path, n_days: int) -> list[Path]:
    files = sorted(Path(long_dir).glob("day_*.parquet"))
    if n_days > len(files):
        raise ValueError(f"Requested {n_days} days but only {len(files)} files exist")
    return files[:n_days]


def _partial(df: pd.DataFrame, ref: float, cutoffs: list):
    occ = df["occupancy"].to_numpy(dtype=np.float64)
    ci = np.clip(occ / ref, 0.0, 1.0)
    code = (ci >= cutoffs[0]).astype(np.int8) + (ci >= cutoffs[1]) + (ci >= cutoffs[2])
    work = pd.DataFrame({"dow": df["dow"].to_numpy(), "hour": df["hour"].to_numpy(), "sensor_id": df["sensor_id"].to_numpy(),
                         "occ": occ, "ci": ci, "high": (code >= 2).astype(np.float64), "sev": (code >= 3).astype(np.float64)})
    prof = work.groupby(["dow", "hour"]).agg(occ=("occ", "sum"), ci=("ci", "sum"), high=("high", "sum"), n=("occ", "size"))
    sens = work.groupby("sensor_id").agg(occ=("occ", "sum"), ci=("ci", "sum"), high=("high", "sum"), sev=("sev", "sum"), n=("occ", "size"))
    return prof, sens


def run(files: list[Path], ref: float, cutoffs: list):
    """Returns (profile_df, sensor_df) with the same columns as the Spark workload."""
    prof_total = sens_total = None
    for f in files:
        prof, sens = _partial(pd.read_parquet(f, columns=COLUMNS), ref, cutoffs)
        prof_total = prof if prof_total is None else prof_total.add(prof, fill_value=0)
        sens_total = sens if sens_total is None else sens_total.add(sens, fill_value=0)
    profile = pd.DataFrame({"mean_occupancy": prof_total.occ / prof_total.n, "mean_ci": prof_total.ci / prof_total.n,
                            "pct_high_or_severe": 100 * prof_total.high / prof_total.n, "n": prof_total.n}).reset_index()
    sensors = pd.DataFrame({"mean_occupancy": sens_total.occ / sens_total.n, "mean_ci": sens_total.ci / sens_total.n,
                            "pct_high_or_severe": 100 * sens_total.high / sens_total.n, "pct_severe": 100 * sens_total.sev / sens_total.n}).reset_index()
    return profile, sensors
