import numpy as np
import pytest

from common.alerts import AlertManager
from common.config import load_config
from common.congestion import LEVELS, Calibration, calibrate, classify, congestion_index


def test_congestion_index_is_clipped_and_monotonic():
    ci = congestion_index(np.array([0.0, 0.1, 0.25, 0.9]), 0.25)
    assert ci.min() == 0 and ci.max() == 1 and np.all(np.diff(ci) >= 0)


def test_classification_boundaries():
    cut = [0.4, 0.5, 1.0]
    assert classify(np.array([0.0, 0.39, 0.4, 0.49, 0.5, 0.99, 1.0]), cut).tolist() == [0, 0, 1, 1, 2, 2, 3]
    assert [LEVELS[i] for i in range(4)] == ["LOW", "MODERATE", "HIGH", "SEVERE"]


def test_calibration_from_quantiles_and_manual_override():
    cfg = load_config()
    occ = np.random.default_rng(0).gamma(2.0, 0.03, size=100_000).astype(np.float32)
    cal = calibrate(occ, cfg)
    assert cal.cutoffs[0] < cal.cutoffs[1] <= cal.cutoffs[2] <= 1.0
    share = np.bincount(classify(congestion_index(occ, cal.reference_occupancy), cal.cutoffs), minlength=4) / occ.size
    assert share[0] == pytest.approx(0.90, abs=0.01) and share[3] == pytest.approx(0.01, abs=0.005)   # 90/95/99 percentile design
    cfg["congestion"]["manual"] = {"reference_occupancy": 0.3, "cutoffs": [0.3, 0.5, 0.9]}
    assert calibrate(occ, cfg).cutoffs == [0.3, 0.5, 0.9]


def _rows(levels, sensor=1):
    return [dict(sensor_id=sensor, tick=t, day_index=0, slot=t, occupancy=0.2, ci=0.9, level_code=c) for t, c in enumerate(levels)]


def test_alert_one_per_episode_and_escalation():
    m = AlertManager(min_level_code=2, clear_after=3, cooldown=0)
    alerts = m.process(_rows([0, 2, 2, 2, 3, 3, 2, 0, 0, 0, 2]))
    assert [(a["alert_type"], a["slot"]) for a in alerts] == [("NEW", 1), ("ESCALATION", 4), ("NEW", 10)]


def test_alert_cooldown_suppresses_flapping():
    m = AlertManager(min_level_code=2, clear_after=1, cooldown=6)
    alerts = m.process(_rows([2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 2]))
    assert len(alerts) == 2 and m.suppressed == 2      # first at tick 0, next only after the cooldown (tick 11)


def test_alert_message_format():
    a = AlertManager().process(_rows([2]))[0]
    assert "CONGESTION ALERT" in a["message"] and a["level"] == "HIGH" and a["time"] == "00:00"


@pytest.mark.slow
def test_spark_expression_matches_numpy():
    pytest.importorskip("pyspark")
    from common.spark_session import get_spark
    from spark_jobs.congestion_detection import add_congestion
    cfg = load_config()
    cal = Calibration(0.25, [0.369999, 0.469999, 0.999999])   # same 1e-6 tie margin as calibrate()
    occ = np.round(np.random.default_rng(3).gamma(2, 0.05, 2000), 4)
    occ = np.concatenate([occ, [0.0925, 0.1175, 0.25]])                  # values exactly on the cut-offs (ties)
    spark = get_spark(cfg, "test", master="local[1]")
    try:
        df = spark.createDataFrame([(float(v),) for v in occ], ["occupancy"])
        got = np.array([r.level_code for r in add_congestion(df, cal.reference_occupancy, cal.cutoffs).collect()])
    finally:
        spark.stop()
    expected = classify(congestion_index(occ.astype(np.float32), cal.reference_occupancy), cal.cutoffs)
    assert (got == expected).all()
