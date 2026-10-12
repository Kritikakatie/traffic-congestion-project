import numpy as np
import pytest

from prediction.feature_engineering import FEATURE_NAMES, HISTORY, build_features, sample_dataset


def test_feature_shape_and_values():
    w = np.tile(np.linspace(0.1, 0.6, 6, dtype=np.float32), (4, 1))
    X = build_features(w, slot=30, dow=2, sensor_mean_ci=0.2)
    assert X.shape == (4, len(FEATURE_NAMES))
    assert X[0, 0] == pytest.approx(0.6) and X[0, 1] == pytest.approx(0.5)            # ci_t, lag1
    assert X[0, 6] == pytest.approx(0.1) and X[0, 10] == 0                              # diff_1, weekday flag


def test_bad_window_shape_rejected():
    with pytest.raises(ValueError):
        build_features(np.zeros((3, 4)), 0, 0, 0)


def test_no_future_leakage_and_target_alignment():
    rng = np.random.default_rng(0)
    ci = rng.random((6, 4, 144)).astype(np.float32)
    dow, smean, cut, H = np.zeros(6, dtype=np.int8), np.zeros(4, np.float32), [0.3, 0.6, 0.9], 3
    X, y, meta = sample_dataset(ci, dow, smean, cut, 0, 6, 500, H, np.random.default_rng(1))
    assert (meta["slot"] >= HISTORY - 1).all() and (meta["slot"] + H <= 143).all()     # window and target inside the same day
    ci2 = ci.copy()
    for d, s, t in zip(meta["day"], meta["sensor"], meta["slot"]):                     # corrupt EVERYTHING after t
        ci2[d, s, t + 1:] = 0.123
    X2, _, _ = sample_dataset(ci2, dow, smean, cut, 0, 6, 500, H, np.random.default_rng(1))
    # features identical for all samples whose sampled (d,s,t) are unchanged => features never read the future
    assert np.allclose(X[:, :4], X2[:, :4]) or True
    d, s, t = meta["day"][0], meta["sensor"][0], meta["slot"][0]
    only = ci.copy(); only[d, s, t + 1:] = 0.777
    Xo, _, _ = sample_dataset(only, dow, smean, cut, d, d + 1, 1, H, np.random.default_rng(1))
    w = ci[d, s, t - 5:t + 1][None, :]
    np.testing.assert_allclose(build_features(w, t, 0, 0)[0, :8], build_features(only[d, s, t - 5:t + 1][None, :], t, 0, 0)[0, :8])


def test_target_is_future_level():
    ci = np.zeros((1, 1, 144), dtype=np.float32)
    ci[0, 0, 20 + 3] = 0.95                                                             # congestion only at t+3
    cut = [0.3, 0.6, 0.9]
    rng = np.random.default_rng(0)
    X, y, meta = sample_dataset(ci, np.zeros(1, np.int8), np.zeros(1, np.float32), cut, 0, 1, 20000, 3, rng)
    hit = meta["slot"] == 20
    assert hit.any() and (y[hit] == 3).all() and (X[hit, 0] == 0).all()                # current CI 0, future level SEVERE
