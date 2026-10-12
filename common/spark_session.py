"""SparkSession factory (local mode). The Kafka connector jar is requested via spark.jars.packages."""
from __future__ import annotations

import os
import sys


def get_spark(cfg: dict, app_name: str, master: str | None = None, with_kafka: bool = False):
    import pyspark
    from pyspark.sql import SparkSession

    # make sure the Spark Python workers use the same interpreter (important inside a Windows venv)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    sc = cfg["spark"]
    builder = (SparkSession.builder.appName(app_name).master(master or cfg["benchmark"]["spark_master"])
               .config("spark.driver.memory", sc["driver_memory"])
               .config("spark.sql.shuffle.partitions", str(sc["shuffle_partitions"]))
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.ui.showConsoleProgress", "false")
               .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true"))
    if with_kafka:
        # Scala 2.12 build matches the pip-installed PySpark 3.5.x; version must equal the PySpark version
        builder = builder.config("spark.jars.packages", f"org.apache.spark:spark-sql-kafka-0-10_2.12:{pyspark.__version__}")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
