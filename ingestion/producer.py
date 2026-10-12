"""
Kafka producer: REPLAYS real PEMS-SF observations to the topic `traffic_data` at a controlled pace.

The dataset is historical, so the "live" stream is a replay of real recorded observations (nothing is
generated). One TICK = one 10-minute slot = one message per sensor (963 messages). After each tick the
producer sleeps STREAM_DELAY_SECONDS, so with 0.5 s one simulated day (144 ticks) takes ~72 seconds.

Usage:
  python -m ingestion.producer --recreate-topic                # normal demo run (default: first weekday of the test period)
  python -m ingestion.producer --day 200 --num-days 2 --delay 0.2
  python -m ingestion.producer --max-speed --num-days 5        # raw Kafka throughput measurement
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time

import numpy as np

from common.config import get_path, load_config, setup_logging
from common.data import load_processed
from ingestion.kafka_utils import FileSpoolSink, build_tick_messages, ensure_topic, make_producer

log = setup_logging("producer")


def pick_start_day(days, requested) -> int:
    if str(requested) != "auto":
        return int(requested)
    test = days[(days["split"] == "test") & (days["dow"] < 5)]
    return int(test["day_index"].iloc[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay real PEMS-SF observations into Kafka")
    ap.add_argument("--day", default=None, help="first day_index to replay (default: config stream.start_day)")
    ap.add_argument("--num-days", type=int, default=None)
    ap.add_argument("--delay", type=float, default=None, help="seconds between ticks (default: config / STREAM_DELAY_SECONDS)")
    ap.add_argument("--max-speed", action="store_true", help="no delay: measure maximum throughput")
    ap.add_argument("--recreate-topic", action="store_true", help="delete and recreate the topic first (clean demo)")
    ap.add_argument("--topic-only", action="store_true", help="only create/recreate the topic, then exit")
    ap.add_argument("--sink", choices=["kafka", "file"], default="kafka", help="'file' = offline test sink (no broker)")
    ap.add_argument("--spool-dir", default=None)
    args = ap.parse_args()

    cfg = load_config()
    kc, sc = cfg["kafka"], cfg["stream"]
    delay = 0.0 if args.max_speed else (args.delay if args.delay is not None else sc["stream_delay_seconds"])
    cube, days, sensors = load_processed(cfg)
    stations = sensors["sensor_id"].to_numpy()
    dow = days["dow"].to_numpy()
    start_day = pick_start_day(days, args.day if args.day is not None else sc["start_day"])
    num_days = args.num_days or sc["num_days"]
    end_day = min(start_day + num_days, len(days))
    dates = [dt.date.fromisoformat(x) for x in days["approx_date"]]
    slots = cube.shape[2]

    sent = errors = 0
    producer = None
    try:
        if args.sink == "kafka":
            ensure_topic(kc["bootstrap_servers"], kc["topic"], kc["partitions"], kc["replication_factor"], args.recreate_topic)
            if args.topic_only:
                return 0
            producer = make_producer(kc["bootstrap_servers"])
        else:
            producer = FileSpoolSink(args.spool_dir or (get_path(cfg, "stream_dir", True) / "spool"))
    except ConnectionError as exc:
        log.error("%s", exc)
        return 2

    def on_error(exc):
        nonlocal errors
        errors += 1
        log.error("send failed: %s", exc)

    log.info("Replaying REAL observations: days %d-%d (%s ...), %d sensors/tick, delay %.3fs/tick, topic %s",
             start_day, end_day - 1, days.loc[start_day, "dow_name"], len(stations), delay, kc["topic"])
    t_start = time.time()
    t_send = 0.0
    try:
        for d in range(start_day, end_day):
            for slot in range(slots):
                t0 = time.time()
                for m in build_tick_messages(cube, stations, d, slot, int(dow[d]), dates[d], cfg["dataset"]["slot_minutes"]):
                    if args.sink == "kafka":
                        producer.send(kc["topic"], key=m["sensor_id"], value=m).add_errback(on_error)
                    else:
                        producer.send(kc["topic"], m["sensor_id"], m)
                    sent += 1
                producer.flush()
                t_send += time.time() - t0
                if slot % 12 == 0:
                    el = time.time() - t_start
                    log.info("day %d %02d:%02d | sent %d msgs | %.0f msg/s overall | errors %d", d, slot // 6, (slot % 6) * 10, sent, sent / max(el, 1e-9), errors)
                if delay:
                    time.sleep(delay)
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
    except Exception as exc:                                   # broker lost mid-stream, serialization problems, ...
        log.error("Producer stopped: %s: %s", type(exc).__name__, exc)
        errors += 1
    finally:
        if producer is not None:
            try:
                producer.close()
            except Exception:
                pass

    total = time.time() - t_start
    stats = {"sink": args.sink, "messages_sent": sent, "errors": errors, "days_replayed": f"{start_day}-{end_day - 1}",
             "delay_seconds_per_tick": delay, "wall_seconds": round(total, 2), "messages_per_second_wall": round(sent / max(total, 1e-9), 1),
             "messages_per_second_send_only": round(sent / max(t_send, 1e-9), 1), "max_speed_mode": bool(args.max_speed)}
    log.info("DONE %s", json.dumps(stats))
    if args.sink == "kafka":
        out = get_path(cfg, "results_dir", True) / ("producer_stats_maxspeed.json" if args.max_speed else "producer_stats.json")
        out.write_text(json.dumps(stats, indent=2))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
