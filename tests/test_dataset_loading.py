"""Checks on the REAL processed dataset (skipped if preprocessing has not been run)."""
import numpy as np
import pytest

from common.config import get_path, load_config

cfg = load_config()
CUBE = get_path(cfg, "processed_dir") / "occupancy_cube.npy"
pytestmark = pytest.mark.skipif(not CUBE.exists(), reason="run python -m preprocessing.preprocess first")


def test_real_cube_shape_and_range():
    cube = np.load(CUBE, mmap_mode="r")
    assert cube.shape[1:] == (963, 144) and cube.shape[0] == 267
    assert np.isfinite(cube).all() and cube.min() >= 0 and cube.max() <= 1


def test_days_table_consistent_with_cube():
    import pandas as pd
    days = pd.read_csv(get_path(cfg, "processed_dir") / "days.csv")
    assert len(days) == 267 and days["day_index"].is_monotonic_increasing
    assert set(days["split"]) == {"train", "test"} and (days["split"].iloc[:187] == "train").all()   # chronological split
    steps = np.diff(days["dow"].to_numpy()) % 7
    assert np.isin(steps, [1, 2]).all()                                                  # weekday chain intact


def test_no_duplicate_days_remain():
    cube = np.load(CUBE, mmap_mode="r")
    assert len({np.asarray(d).tobytes() for d in cube}) == len(cube)
