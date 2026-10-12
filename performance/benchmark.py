"""
Serial (pandas) vs Spark benchmark on the REAL dataset.

Dataset sizes are whole days of the real data (1 day = 138,672 records = 963 sensors x 144 slots).
Both engines read the same Parquet files and compute the same result; the result is checked for equality.
Every number is measured by this script - nothing is estimated.

  python -m performance.benchmark                    # sizes/repeats from config.yaml
  python -m performance.benchmark --sizes 1 4 15 --repeats 2
  python -m performance.benchmark --scaling          # Spark local[k] scalability on a fixed size
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import get_path, load_config, setup_logging
from common.congestion import Calibration
from common.spark_session import get_spark
from performance import pandas_baseline, spark_benchmark

log = setup_logging("benchmark")
RECORDS_PER_DAY = 963 * 144


def environment_info() -> dict:
    import pyspark
    info = {"cpu_count": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version(),
            "pandas": pd.__version__, "pyspark": pyspark.__version__}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        try:
            info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
        except (ValueError, OSError, AttributeError):
            info["ram_gb"] = None
    return info


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def max_abs_difference(a: pd.DataFrame, b: pd.DataFrame, key: list) -> float:
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        return float("inf")
    cols = [c for c in a.columns if c not in key and c != "n"]
    return float(max(np.abs(m[c + "_a"] - m[c + "_b"]).max() for c in cols))


def run_sizes(cfg, sizes, repeats) -> pd.DataFrame:
    long_dir = get_path(cfg, "processed_dir") / "traffic_long"
    cal = Calibration.load(get_path(cfg, "results_dir") / "calibration.json")
    t0 = time.perf_counter()
    spark = get_spark(cfg, "TrafficBenchmark")
    startup = time.perf_counter() - t0
    # warm-up: the first Spark job pays JVM/class-loading costs - measured and reported separately
    first_job, _ = timed(lambda: spark_benchmark.run(spark, pandas_baseline.list_day_files(long_dir, 1), cal.reference_occupancy, cal.cutoffs))
    log.info("Spark session startup %.2fs, first (warm-up) job %.2fs - both excluded from the table below", startup, first_job)

    rows = []
    for n_days in sizes:
        files = pandas_baseline.list_day_files(long_dir, n_days)
        t_pd, t_sp = [], []
        for _ in range(repeats):
            s, (pp, ps) = timed(lambda: pandas_baseline.run(files, cal.reference_occupancy, cal.cutoffs)); t_pd.append(s)
            s, (sp, ss) = timed(lambda: spark_benchmark.run(spark, files, cal.reference_occupancy, cal.cutoffs)); t_sp.append(s)
        diff = max(max_abs_difference(pp, sp, ["dow", "hour"]), max_abs_difference(ps, ss, ["sensor_id"]))
        rec = n_days * RECORDS_PER_DAY
        row = {"days": n_days, "records": rec, "pandas_seconds": statistics.median(t_pd), "spark_seconds": statistics.median(t_sp),
               "pandas_min": min(t_pd), "pandas_max": max(t_pd), "spark_min": min(t_sp), "spark_max": max(t_sp),
               "speedup": statistics.median(t_pd) / statistics.median(t_sp),
               "pandas_records_per_s": rec / statistics.median(t_pd), "spark_records_per_s": rec / statistics.median(t_sp),
               "max_abs_result_difference": diff}
        rows.append(row)
        log.info("%3d days %10d records | pandas %.2fs | spark %.2fs | speedup %.2fx | result diff %.1e", n_days, rec,
                 row["pandas_seconds"], row["spark_seconds"], row["speedup"], diff)
    spark.stop()
    df = pd.DataFrame(rows)
    res = get_path(cfg, "results_dir", True)
    df.to_csv(res / "benchmark_results.csv", index=False)
    meta = {"environment": environment_info(), "repeats": repeats, "statistic": "median of repeats",
            "spark_master": cfg["benchmark"]["spark_master"], "spark_session_startup_seconds": round(startup, 2),
            "spark_first_job_seconds": round(first_job, 2),
            "workload": "CI+level per record; weekday x hour profile; per-sensor statistics (read parquet + compute + collect)"}
    (res / "benchmark_meta.json").write_text(json.dumps(meta, indent=2))
    make_figures(df, get_path(cfg, "figures_dir", True))
    return df


def run_scaling(cfg, n_days: int) -> pd.DataFrame:
    long_dir = get_path(cfg, "processed_dir") / "traffic_long"
    cal = Calibration.load(get_path(cfg, "results_dir") / "calibration.json")
    files = pandas_baseline.list_day_files(long_dir, n_days)
    cores = [k for k in cfg["benchmark"]["scaling_cores"] if k <= (os.cpu_count() or 1)]
    if len(cores) < 2:
        log.warning("Only %s CPU core(s) available - scalability experiment needs >= 2. Run it on a multi-core machine.", os.cpu_count())
    rows = []
    for k in cores:
        spark = get_spark(cfg, f"Scaling{k}", master=f"local[{k}]")
        spark_benchmark.run(spark, files[:1], cal.reference_occupancy, cal.cutoffs)       # warm-up
        times = [timed(lambda: spark_benchmark.run(spark, files, cal.reference_occupancy, cal.cutoffs))[0] for _ in range(3)]
        spark.stop()
        rows.append({"cores": k, "days": n_days, "records": n_days * RECORDS_PER_DAY, "spark_seconds": statistics.median(times)})
        log.info("local[%d]: %.2fs", k, rows[-1]["spark_seconds"])
    df = pd.DataFrame(rows)
    if len(df):
        df["speedup_vs_1_core"] = df.spark_seconds.iloc[0] / df.spark_seconds
    df.to_csv(get_path(cfg, "results_dir", True) / "benchmark_scaling.csv", index=False)
    if len(df) >= 2:
        f, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(df.cores, df.speedup_vs_1_core, "o-", label="measured"); ax.plot(df.cores, df.cores, "k--", alpha=.4, label="ideal")
        ax.set_xlabel("Spark local cores"); ax.set_ylabel("speedup vs 1 core"); ax.legend(); ax.set_title(f"Spark scalability ({n_days} days)")
        f.tight_layout(); f.savefig(get_path(cfg, "figures_dir", True) / "benchmark_scaling.png", dpi=150); plt.close(f)
    return df


def make_figures(df: pd.DataFrame, fig) -> None:
    f, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(df.records, df.pandas_seconds, "o-", label="pandas (serial)", color="#c2362f")
    ax.plot(df.records, df.spark_seconds, "s-", label="Spark", color="#2f6db5")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("records"); ax.set_ylabel("execution time (s, median)"); ax.legend(); ax.set_title("Execution time vs dataset size")
    f.tight_layout(); f.savefig(fig / "benchmark_time.png", dpi=150); plt.close(f)
    f, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(df.records, df.speedup, "o-", color="#3b9c6d"); ax.axhline(1, color="black", ls="--"); ax.set_xscale("log")
    ax.set_xlabel("records"); ax.set_ylabel("speedup = serial time / Spark time"); ax.set_title("Speedup (below 1 = Spark slower)")
    f.tight_layout(); f.savefig(fig / "benchmark_speedup.png", dpi=150); plt.close(f)
    f, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(df.records, df.pandas_records_per_s, "o-", label="pandas (serial)", color="#c2362f"); ax.plot(df.records, df.spark_records_per_s, "s-", label="Spark", color="#2f6db5")
    ax.set_xscale("log"); ax.set_xlabel("records"); ax.set_ylabel("throughput (records / s)"); ax.legend(); ax.set_title("Processing throughput")
    f.tight_layout(); f.savefig(fig / "benchmark_throughput.png", dpi=150); plt.close(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="dataset sizes in days")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--scaling", action="store_true", help="Spark local[k] scalability experiment")
    ap.add_argument("--scaling-days", type=int, default=72)
    args = ap.parse_args()
    cfg = load_config()
    if args.scaling:
        print(run_scaling(cfg, args.scaling_days).to_string(index=False))
    else:
        print(run_sizes(cfg, args.sizes or cfg["benchmark"]["sizes_days"], args.repeats or cfg["benchmark"]["repeats"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
