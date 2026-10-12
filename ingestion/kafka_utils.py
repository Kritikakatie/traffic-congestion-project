"""Kafka helpers: topic management, producer creation, message construction, and an offline file sink."""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np

from common.config import setup_logging

log = setup_logging("kafka_utils")


def _kafka():
    try:
        import kafka  # noqa: F401  (package: kafka-python)
        from kafka import KafkaProducer
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka import errors
        return KafkaProducer, KafkaAdminClient, NewTopic, errors
    except ImportError as exc:      # pragma: no cover
        raise SystemExit("kafka-python is not installed. Run: pip install -r requirements.txt") from exc


def ensure_topic(bootstrap: str, topic: str, partitions: int = 3, replication: int = 1, recreate: bool = False,
                 timeout_s: int = 20) -> None:
    """Create the topic if missing (optionally delete + recreate it so a demo starts from an empty topic)."""
    _, KafkaAdminClient, NewTopic, errors = _kafka()
    try:
        admin = KafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=timeout_s * 1000)
    except errors.NoBrokersAvailable as exc:
        raise ConnectionError(f"Cannot reach Kafka at {bootstrap}. Is the broker running? (see README, step 'Start Kafka')") from exc
    try:
        existing = set(admin.list_topics())
        if topic in existing and recreate:
            admin.delete_topics([topic])
            log.info("Deleted topic %s, waiting for deletion ...", topic)
            for _ in range(30):
                time.sleep(1)
                if topic not in set(admin.list_topics()):
                    break
            existing.discard(topic)
        if topic not in existing:
            try:
                admin.create_topics([NewTopic(name=topic, num_partitions=partitions, replication_factor=replication)])
                log.info("Created topic %s (%d partitions)", topic, partitions)
            except errors.TopicAlreadyExistsError:
                pass
        else:
            log.info("Topic %s already exists", topic)
    finally:
        admin.close()


def make_producer(bootstrap: str):
    """JSON producer. Key = sensor id, so all observations of one sensor stay ordered in one partition."""
    KafkaProducer, _, _, errors = _kafka()
    try:
        return KafkaProducer(
            bootstrap_servers=bootstrap, acks=1, linger_ms=20, batch_size=64 * 1024, retries=3,
            key_serializer=lambda k: str(k).encode("utf-8"),
            value_serializer=lambda v: json.dumps(v, separators=(",", ":")).encode("utf-8"),
        )
    except errors.NoBrokersAvailable as exc:
        raise ConnectionError(f"Cannot reach Kafka at {bootstrap}. Start the broker first (see README).") from exc


def build_tick_messages(cube: np.ndarray, stations: np.ndarray, day_index: int, slot: int, dow: int,
                        day_date: dt.date, slot_minutes: int = 10) -> list[dict]:
    """All sensor observations of one 10-minute tick. Occupancy / sensor id / position are the REAL dataset values.
    event_time = APPROXIMATE calendar date of the day (see preprocessing) + slot time of day."""
    event_time = (dt.datetime.combine(day_date, dt.time()) + dt.timedelta(minutes=int(slot) * slot_minutes)).strftime("%Y-%m-%d %H:%M:%S")
    tick = int(day_index) * cube.shape[2] + int(slot)
    now = time.time()
    occ = cube[day_index, :, slot]
    return [{"sensor_id": int(stations[i]), "day_index": int(day_index), "slot": int(slot), "dow": int(dow),
             "occupancy": round(float(occ[i]), 4), "event_time": event_time, "tick": tick, "replay_ts": now}
            for i in range(len(stations))]


class FileSpoolSink:
    """OFFLINE TEST SINK ONLY: writes each tick as a JSON-lines file so the Spark transformation code can be
    tested on a machine without a Kafka broker. The real pipeline always uses Kafka."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._buf: list[str] = []
        self._n = 0

    def send(self, topic, key, value):
        self._buf.append(json.dumps(value, separators=(",", ":")))

    def flush(self):
        if not self._buf:
            return
        tmp = self.dir / f".tick_{self._n:07d}.tmp"
        tmp.write_text("\n".join(self._buf) + "\n")
        os.replace(tmp, self.dir / f"tick_{self._n:07d}.json")
        self._n += 1
        self._buf = []

    def close(self):
        self.flush()
