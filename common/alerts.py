"""
Alert manager with de-duplication.

A congestion EPISODE for a sensor starts when its level reaches `min_level`.
Only ONE alert is raised per episode (plus one ESCALATION alert if the level
rises further, e.g. HIGH -> SEVERE). An episode ends after `clear_after`
consecutive ticks below `min_level`. A new alert for the same sensor is
suppressed for `cooldown` ticks after the previous alert (anti-flapping).

The manager is plain Python (no Spark dependency), so it is unit-testable and is
called from the Spark foreachBatch sink.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from common.congestion import LEVELS

SLOTS_PER_DAY = 144


@dataclass
class _Episode:
    alerted_level: int
    below_count: int = 0


@dataclass
class AlertManager:
    min_level_code: int = 2
    clear_after: int = 3
    cooldown: int = 6
    slot_minutes: int = 10
    active: dict = field(default_factory=dict)
    last_alert_tick: dict = field(default_factory=dict)
    suppressed: int = 0

    def process(self, rows: list[dict]) -> list[dict]:
        """rows: dicts with sensor_id, tick, day_index, slot, occupancy, ci, level_code.
        Must contain every sensor of each tick so that episode clearing works. Returns new alerts."""
        alerts: list[dict] = []
        for r in sorted(rows, key=lambda x: (x["tick"], x["sensor_id"])):
            sid, code, tick = r["sensor_id"], int(r["level_code"]), r["tick"]
            ep = self.active.get(sid)
            if code >= self.min_level_code:
                if ep is None:
                    last = self.last_alert_tick.get(sid)
                    if last is not None and tick - last < self.cooldown:
                        self.suppressed += 1
                        self.active[sid] = _Episode(alerted_level=code)   # same episode continues silently
                        continue
                    self.active[sid] = _Episode(alerted_level=code)
                    self.last_alert_tick[sid] = tick
                    alerts.append(self._make(r, "NEW"))
                else:
                    ep.below_count = 0
                    if code > ep.alerted_level:
                        ep.alerted_level = code
                        self.last_alert_tick[sid] = tick
                        alerts.append(self._make(r, "ESCALATION"))
            elif ep is not None:
                ep.below_count += 1
                if ep.below_count >= self.clear_after:
                    del self.active[sid]
        return alerts

    def _make(self, r: dict, kind: str) -> dict:
        minutes = int(r["slot"]) * self.slot_minutes
        level = LEVELS[int(r["level_code"])]
        a = {
            "alert_type": kind, "sensor_id": int(r["sensor_id"]), "day_index": int(r["day_index"]),
            "slot": int(r["slot"]), "time": f"{minutes // 60:02d}:{minutes % 60:02d}",
            "occupancy": round(float(r["occupancy"]), 4), "congestion_index": round(float(r["ci"]), 3),
            "level": level,
        }
        a["message"] = (f"CONGESTION ALERT [{kind}] Sensor: {a['sensor_id']} | Day#{a['day_index']} {a['time']} | "
                        f"Occupancy: {a['occupancy']:.3f} | Congestion Index: {a['congestion_index']:.2f} | Level: {level}")
        return a
