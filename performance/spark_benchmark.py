"""Spark side of the benchmark: the same workload on the same Parquet files (read + compute + collect)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from spark_jobs.aggregation import batch_workload


def run(spark, files: list[Path], ref: float, cutoffs: list):
    df = spark.read.parquet(*[f.resolve().as_uri() for f in files]).select("sensor_id", "occupancy", "dow", "hour")
    profile, sensors = batch_workload(df, ref, cutoffs)
    p = pd.DataFrame([r.asDict() for r in profile.collect()])
    s = pd.DataFrame([r.asDict() for r in sensors.collect()])
    return p, s
