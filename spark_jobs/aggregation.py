"""Spark aggregations: streaming time-window aggregation and the batch workload used in the benchmark."""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from spark_jobs.congestion_detection import add_congestion, level_name_col


def window_aggregate(enriched: DataFrame, window_duration: str, watermark: str) -> DataFrame:
    """Per sensor and per tumbling event-time window: record count, mean/max occupancy, mean/max CI, worst level.
    The watermark bounds the state: a window is finalised (and emitted in append mode) once event time has
    advanced `watermark` beyond its end."""
    agg = (enriched.withWatermark("event_time", watermark)
           .groupBy(F.window("event_time", window_duration).alias("w"), "sensor_id")
           .agg(F.count("*").alias("records"), F.avg("occupancy").alias("avg_occupancy"), F.max("occupancy").alias("max_occupancy"),
                F.avg("ci").alias("avg_ci"), F.max("ci").alias("max_ci"), F.max("level_code").alias("max_level_code")))
    return agg.select(F.col("w.start").alias("window_start"), F.col("w.end").alias("window_end"), "sensor_id", "records",
                      "avg_occupancy", "max_occupancy", "avg_ci", "max_ci", "max_level_code",
                      level_name_col("max_level_code").alias("worst_level"))


def batch_workload(df: DataFrame, reference_occupancy: float, cutoffs: list):
    """THE benchmark workload (the pandas baseline implements exactly the same computation):
       1. CI + level for every record
       2. weekday x hour profile: mean occupancy, mean CI, share of HIGH+SEVERE, record count
       3. per-sensor statistics: mean occupancy, mean CI, share of HIGH+SEVERE, share of SEVERE
    Returns (profile_df, sensor_df) - Spark DataFrames (lazy)."""
    e = add_congestion(df, reference_occupancy, cutoffs)
    high = (F.col("level_code") >= 2).cast("double")
    profile = e.groupBy("dow", "hour").agg(F.avg("occupancy").alias("mean_occupancy"), F.avg("ci").alias("mean_ci"),
                                           (100 * F.avg(high)).alias("pct_high_or_severe"), F.count("*").alias("n"))
    sensors = e.groupBy("sensor_id").agg(F.avg("occupancy").alias("mean_occupancy"), F.avg("ci").alias("mean_ci"),
                                         (100 * F.avg(high)).alias("pct_high_or_severe"),
                                         (100 * F.avg((F.col("level_code") >= 3).cast("double"))).alias("pct_severe"))
    return profile, sensors
