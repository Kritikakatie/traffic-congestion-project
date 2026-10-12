"""
Spark Structured Streaming job:  Kafka topic `traffic_data`  ->  parse JSON  ->  congestion index/level
  -> (query 1) per-micro-batch processing: record sink, per-tick network summary, ALERTS, throughput metrics
  -> (query 2) event-time WINDOWED aggregation per sensor (watermark, append mode, parquet sink, checkpoint)

Run (Kafka must be running and the topic must exist):
    python -m spark_jobs.spark_streaming --reset
Offline test of the same transformation code without a broker (test sink only):
    python -m spark_jobs.spark_streaming --source files --input-dir <spool dir> --reset --stop-after-idle 20
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType

from common.alerts import AlertManager
from common.config import get_path, load_config, setup_logging
from common.congestion import LEVEL_CODE, Calibration
from common.spark_session import get_spark
from spark_jobs.aggregation import window_aggregate
from spark_jobs.congestion_detection import add_congestion

log = setup_logging("spark_streaming")

MESSAGE_SCHEMA = StructType([
    StructField("sensor_id", LongType()), StructField("day_index", IntegerType()), StructField("slot", IntegerType()),
    StructField("dow", IntegerType()), StructField("occupancy", DoubleType()), StructField("event_time", StringType()),
    StructField("tick", LongType()), StructField("replay_ts", DoubleType()),
])


def read_stream(spark, cfg: dict, source: str, input_dir: str | None) -> DataFrame:
    """Streaming source: Kafka (production path) or a directory of JSON files (offline test only)."""
    if source == "kafka":
        reader = (spark.readStream.format("kafka")
                  .option("kafka.bootstrap.servers", cfg["kafka"]["bootstrap_servers"])
                  .option("subscribe", cfg["kafka"]["topic"])
                  .option("startingOffsets", cfg["spark"]["starting_offsets"])
                  .option("failOnDataLoss", "false"))
        if cfg["spark"].get("max_offsets_per_trigger"):
            reader = reader.option("maxOffsetsPerTrigger", int(cfg["spark"]["max_offsets_per_trigger"]))
        raw = reader.load()
        parsed = raw.select(F.from_json(F.col("value").cast("string"), MESSAGE_SCHEMA).alias("d"), F.col("timestamp").alias("kafka_ts"))
        df = parsed.select("d.*", "kafka_ts")
    else:
        df = spark.readStream.schema(MESSAGE_SCHEMA).option("maxFilesPerTrigger", 6).json(input_dir).withColumn("kafka_ts", F.current_timestamp())
    # drop malformed JSON / records with missing key fields, then type the event time
    df = df.filter(F.col("sensor_id").isNotNull() & F.col("occupancy").isNotNull() & F.col("event_time").isNotNull())
    return df.withColumn("event_time", F.to_timestamp("event_time"))


class BatchProcessor:
    """foreachBatch handler: runs once per micro-batch on a static DataFrame."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.stream_dir = get_path(cfg, "stream_dir", True)
        self.alerts_dir = get_path(cfg, "alerts_dir", True)
        self.results_dir = get_path(cfg, "results_dir", True)
        ac = cfg["alerts"]
        self.manager = AlertManager(LEVEL_CODE[ac["min_level"]], ac["clear_after_ticks"], ac["cooldown_ticks"], cfg["dataset"]["slot_minutes"])
        self.alert_csv = self.alerts_dir / "alerts.csv"
        self.metrics_csv = self.stream_dir / "stream_metrics.csv"
        self.total_rows = 0

    def __call__(self, batch: DataFrame, batch_id: int) -> None:
        t0 = time.time()
        batch.persist()
        try:
            n = batch.count()
            if n == 0:
                return
            cols = ["sensor_id", "day_index", "slot", "dow", "occupancy", "ci", "level_code", "tick", "event_time"]
            batch.select(*cols).coalesce(1).write.mode("append").parquet(str(self.stream_dir / "records"))

            # per-tick network summary (computed by Spark)
            summary = batch.groupBy("tick", "day_index", "slot").agg(
                F.countDistinct("sensor_id").alias("active_sensors"), F.avg("occupancy").alias("mean_occupancy"),
                F.avg("ci").alias("mean_ci"), F.max("ci").alias("max_ci"),
                *[F.sum(F.when(F.col("level_code") == k, 1).otherwise(0)).alias(f"n_level_{k}") for k in range(4)])
            summary.coalesce(1).write.mode("append").parquet(str(self.stream_dir / "tick_summary"))

            # alerts (de-duplicated by AlertManager, which needs every sensor of every tick in order)
            rows = [r.asDict() for r in batch.select("sensor_id", "tick", "day_index", "slot", "occupancy", "ci", "level_code").collect()]
            alerts = self.manager.process(rows)
            self._write_alerts(alerts)

            lat = batch.agg(F.avg(F.unix_millis(F.current_timestamp()) / 1000.0 - F.col("replay_ts"))).first()[0]
            dt_s = time.time() - t0
            self.total_rows += n
            self._write_metrics(batch_id, n, dt_s, lat)
            log.info("batch %d | %d records | %d ticks | %d new alerts (suppressed so far %d) | %.2fs | %.0f rec/s",
                     batch_id, n, len({r["tick"] for r in rows}), len(alerts), self.manager.suppressed, dt_s, n / max(dt_s, 1e-9))
        finally:
            batch.unpersist()

    def _write_alerts(self, alerts: list[dict]) -> None:
        if not alerts:
            return
        new_file = not self.alert_csv.exists()
        with open(self.alert_csv, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(alerts[0].keys()))
            if new_file:
                w.writeheader()
            w.writerows(alerts)
        for a in alerts[:3]:
            log.warning(a["message"])

    def _write_metrics(self, batch_id: int, n: int, secs: float, latency) -> None:
        new_file = not self.metrics_csv.exists()
        with open(self.metrics_csv, "a", newline="") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["wall_time", "batch_id", "records", "batch_seconds", "records_per_second", "avg_end_to_end_latency_s"])
            w.writerow([time.strftime("%H:%M:%S"), batch_id, n, round(secs, 3), round(n / max(secs, 1e-9), 1), "" if latency is None else round(latency, 3)])


def reset_outputs(cfg: dict) -> None:
    for d in (get_path(cfg, "stream_dir"), get_path(cfg, "alerts_dir")):
        if d.exists():
            for item in d.iterdir():
                shutil.rmtree(item, ignore_errors=True) if item.is_dir() else item.unlink(missing_ok=True)
    log.info("Cleared previous streaming outputs and checkpoints")


def main() -> int:
    ap = argparse.ArgumentParser(description="Kafka -> Spark Structured Streaming congestion detection")
    ap.add_argument("--source", choices=["kafka", "files"], default="kafka")
    ap.add_argument("--input-dir", default=None, help="only for --source files (offline test)")
    ap.add_argument("--reset", action="store_true", help="delete old checkpoints/outputs before starting")
    ap.add_argument("--timeout", type=int, default=0, help="stop after N seconds (0 = run until Ctrl+C)")
    ap.add_argument("--stop-after-idle", type=int, default=0, help="stop after N seconds without new input (0 = disabled)")
    args = ap.parse_args()

    cfg = load_config()
    if args.source == "files" and not args.input_dir:
        ap.error("--input-dir is required with --source files")
    res = get_path(cfg, "results_dir")
    cal_path = res / "calibration.json"
    if not cal_path.exists():
        log.error("calibration.json missing - run: python -m analysis.historical")
        return 2
    cal = Calibration.load(cal_path)
    if args.reset:
        reset_outputs(cfg)

    spark = get_spark(cfg, "TrafficCongestionStreaming", master="local[*]", with_kafka=(args.source == "kafka"))
    sc = cfg["spark"]
    stream_dir = get_path(cfg, "stream_dir", True)
    trigger = f"{sc['trigger_seconds']} seconds"

    enriched = add_congestion(read_stream(spark, cfg, args.source, args.input_dir), cal.reference_occupancy, cal.cutoffs)

    q1 = (enriched.writeStream.foreachBatch(BatchProcessor(cfg)).outputMode("update").trigger(processingTime=trigger)
          .option("checkpointLocation", str(stream_dir / "checkpoints" / "records")).queryName("records_and_alerts").start())
    q2 = (window_aggregate(enriched, sc["window_duration"], sc["watermark"]).writeStream.outputMode("append").format("parquet")
          .option("path", str(stream_dir / "windows")).option("checkpointLocation", str(stream_dir / "checkpoints" / "windows"))
          .trigger(processingTime=trigger).queryName("window_aggregates").start())
    log.info("Streaming started (%s source). Trigger %s, window %s, watermark %s. Ctrl+C to stop.", args.source, trigger, sc["window_duration"], sc["watermark"])

    started, last_active = time.time(), time.time()
    try:
        while q1.isActive and q2.isActive:
            time.sleep(2)
            prog = q1.lastProgress
            if prog and prog.get("numInputRows", 0) > 0:
                last_active = time.time()
            if args.timeout and time.time() - started > args.timeout:
                break
            if args.stop_after_idle and time.time() - last_active > args.stop_after_idle and q1.lastProgress is not None and time.time() - started > 15:
                break
        for q in (q1, q2):
            if q.exception():
                log.error("Query %s failed: %s", q.name, q.exception())
                return 1
    except KeyboardInterrupt:
        log.warning("Stopping on user request")
    finally:
        for q in (q1, q2):
            try:
                q.stop()
            except Exception:
                pass
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
