"""Congestion index and level as Spark column expressions (same formula as common/congestion.py)."""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

LEVEL_NAMES = ["LOW", "MODERATE", "HIGH", "SEVERE"]


def add_congestion(df: DataFrame, reference_occupancy: float, cutoffs: list, occ_col: str = "occupancy") -> DataFrame:
    """Adds `ci` = clip(occupancy / reference, 0, 1) and `level_code` 0..3 (cut-offs are CI values)."""
    ci = F.least(F.lit(1.0), F.greatest(F.lit(0.0), F.col(occ_col) / F.lit(float(reference_occupancy))))
    df = df.withColumn("ci", ci)
    return df.withColumn("level_code",
                         F.when(F.col("ci") >= cutoffs[2], 3).when(F.col("ci") >= cutoffs[1], 2)
                          .when(F.col("ci") >= cutoffs[0], 1).otherwise(0))


def level_name_col(code_col: str = "level_code"):
    """level_code -> LOW/MODERATE/HIGH/SEVERE column."""
    c = F.col(code_col)
    return F.when(c == 3, "SEVERE").when(c == 2, "HIGH").when(c == 1, "MODERATE").otherwise("LOW")
