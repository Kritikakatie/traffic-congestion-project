"""
Preprocessing of the UCI PEMS-SF files (raw files are never modified).

Raw format: each line of PEMS_train / PEMS_test is one DAY written as a MATLAB-style
matrix "[v v v ...; v v v ...; ...]" with 963 rows (sensors) x 144 columns (10-min slots).
Values are lane occupancy rates in [0, 1].

Steps
  1. parse files + labels + randperm + stations_list
  2. restore chronological order (inverse of randperm) and VERIFY it with the weekday chain
  3. remove exact duplicate days (the supplied files contain only 267 distinct days)
  4. infer the weekday encoding from the data (lowest-occupancy label = Sunday)
  5. validate / repair values (NaN, out of range -> interpolation)
  6. derive time features and write cube, metadata and long-format Parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import DOW_NAMES, get_path, load_config, setup_logging

log = setup_logging("preprocess")
RAW_FILES = ["PEMS_train", "PEMS_test", "PEMS_trainlabels", "PEMS_testlabels", "randperm", "stations_list"]


# ----------------------------------------------------------------------------- parsing
def parse_vector(path: Path) -> np.ndarray:
    """Read a '[1 2 3 ...]' file into an int array."""
    text = Path(path).read_text().strip().strip("[]")
    return np.array(text.split(), dtype=np.int64)


def parse_matrix_line(line: str) -> np.ndarray:
    """One raw line -> (n_sensors, n_slots) float32 matrix."""
    rows = line.strip().strip("[]").split(";")
    return np.array([r.split() for r in rows], dtype=np.float32)


def load_days(path: Path, n_sensors: int = 963, n_slots: int = 144) -> np.ndarray:
    """Parse a PEMS_train/PEMS_test file -> (n_days, n_sensors, n_slots)."""
    days = []
    with open(path, "r") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            m = parse_matrix_line(line)
            if m.shape != (n_sensors, n_slots):
                raise ValueError(f"{path.name} line {i}: expected {(n_sensors, n_slots)}, got {m.shape}")
            days.append(m)
    return np.stack(days)


def extract_zip_if_needed(raw_dir: Path, zip_path: Path | None) -> None:
    if all((raw_dir / f).exists() for f in RAW_FILES):
        return
    if zip_path is None or not Path(zip_path).exists():
        raise FileNotFoundError(f"Raw files missing in {raw_dir}. Put the 6 PEMS-SF files there or pass --zip pems_sf.zip")
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    log.info("Extracted %s -> %s", zip_path, raw_dir)


# ----------------------------------------------------------------------------- chronology
def restore_chronology(randperm: np.ndarray) -> np.ndarray:
    """Return `order` such that cube_chronological = cube_stored[order].
    UCI shuffled the chronological days with randperm: stored[i] = chrono[randperm[i]-1]."""
    n = len(randperm)
    order = np.empty(n, dtype=np.int64)
    order[randperm - 1] = np.arange(n)
    return order


def weekday_chain_score(labels_chrono: np.ndarray) -> float:
    """Share of consecutive-day steps that advance the weekday by 1 or 2 (2 = a skipped holiday).
    A random order scores ~2/7; the true chronological order scores close to 1."""
    steps = (np.diff(labels_chrono) % 7)
    return float(np.mean((steps == 1) | (steps == 2)))


def find_duplicates(cube: np.ndarray) -> np.ndarray:
    """Boolean mask over days: True = exact copy of an EARLIER day."""
    seen, dup = {}, np.zeros(len(cube), dtype=bool)
    for i, day in enumerate(cube):
        h = hashlib.sha1(day.tobytes()).hexdigest()
        if h in seen:
            dup[i] = True
        else:
            seen[h] = i
    return dup


def infer_weekday_encoding(cube: np.ndarray, labels: np.ndarray) -> dict:
    """The label encoding in the files is checked against the data, not trusted blindly.
    Weekend days have clearly lower occupancy, so the label with the lowest mean occupancy is Sunday.
    Returns the python weekday (Mon=0..Sun=6) for each label value."""
    means = {int(l): float(cube[labels == l].mean()) for l in np.unique(labels)}
    sunday_label = min(means, key=means.get)
    ordered = sorted(means, key=means.get)
    lo2 = ordered[:2]
    adjacent = (abs(lo2[0] - lo2[1]) in (1, 6))
    if not adjacent:
        raise ValueError(f"Two lowest-occupancy labels {lo2} are not adjacent weekdays; cannot infer encoding")
    mapping = {l: (l - sunday_label - 1) % 7 for l in means}
    return {"sunday_label": sunday_label, "label_to_pyweekday": mapping, "mean_occupancy_by_label": means}


def approximate_dates(dow: np.ndarray, start: dt.date) -> list[dt.date]:
    """Approximate calendar dates from the weekday chain (gap of 2+ days = removed holiday).
    ONLY approximate: the raw files carry no dates."""
    first = start
    while first.weekday() != int(dow[0]):
        first += dt.timedelta(days=1)
    dates = [first]
    for i in range(1, len(dow)):
        step = (int(dow[i]) - int(dow[i - 1])) % 7 or 7
        dates.append(dates[-1] + dt.timedelta(days=step))
    return dates


# ----------------------------------------------------------------------------- cleaning
def clean_values(cube: np.ndarray) -> tuple[np.ndarray, dict]:
    """Mark invalid values (NaN, <0, >1) and repair them by linear interpolation along time,
    falling back to the sensor median. Genuine zeros are KEPT (night-time occupancy can be ~0)."""
    cube = cube.copy()
    invalid = ~np.isfinite(cube) | (cube < 0) | (cube > 1)
    n_invalid = int(invalid.sum())
    if n_invalid:
        cube[invalid] = np.nan
        medians = np.nanmedian(cube, axis=(0, 2))
        for d, s in zip(*np.where(invalid.any(axis=2))):
            series = pd.Series(cube[d, s]).interpolate(limit_direction="both")
            cube[d, s] = series.fillna(medians[s]).to_numpy(dtype=np.float32)
    return cube, {"invalid_values_repaired": n_invalid, "zero_values_kept": int((cube == 0).sum())}


# ----------------------------------------------------------------------------- pipeline
def build_dataset(raw_dir: Path, n_sensors: int = 963, n_slots: int = 144) -> dict:
    """Run steps 1-5 and return arrays + a quality report (no files written)."""
    train, test = load_days(raw_dir / "PEMS_train", n_sensors, n_slots), load_days(raw_dir / "PEMS_test", n_sensors, n_slots)
    labels = np.concatenate([parse_vector(raw_dir / "PEMS_trainlabels"), parse_vector(raw_dir / "PEMS_testlabels")])
    randperm = parse_vector(raw_dir / "randperm")
    stations = parse_vector(raw_dir / "stations_list")
    stored = np.concatenate([train, test])
    if not (len(stored) == len(labels) == len(randperm)):
        raise ValueError("train+test days, labels and randperm lengths differ")
    if len(stations) != n_sensors:
        raise ValueError("stations_list length != number of sensors")

    order = restore_chronology(randperm)
    cube, lab = stored[order], labels[order]
    chain_after = weekday_chain_score(lab)
    chain_before = weekday_chain_score(labels)
    if chain_after < 0.9:
        raise ValueError(f"Chronology check failed (weekday chain score {chain_after:.2f})")

    dup = find_duplicates(cube)
    keep = ~dup
    cube, lab, stored_rows = cube[keep], lab[keep], order[keep]
    enc = infer_weekday_encoding(cube, lab)
    dow = np.array([enc["label_to_pyweekday"][int(l)] for l in lab], dtype=np.int8)
    cube, clean_report = clean_values(cube)

    report = {
        "stored_days": int(len(stored)), "distinct_days": int(keep.sum()), "duplicate_days_removed": int(dup.sum()),
        "n_sensors": n_sensors, "slots_per_day": n_slots,
        "weekday_chain_score_stored_order": round(chain_before, 4),
        "weekday_chain_score_restored_order": round(chain_after, 4),
        "weekday_encoding": {"sunday_label": enc["sunday_label"],
                             "mean_occupancy_by_label": {str(k): round(v, 5) for k, v in enc["mean_occupancy_by_label"].items()}},
        "removed_day_gaps_in_chain": int(np.sum((np.diff(lab) % 7) == 2)),
        **clean_report,
        "occupancy_min": float(cube.min()), "occupancy_max": float(cube.max()), "occupancy_mean": float(cube.mean()),
    }
    return {"cube": cube, "dow": dow, "label": lab, "stations": stations, "stored_row": stored_rows, "report": report}


def write_outputs(ds: dict, cfg: dict) -> None:
    out = get_path(cfg, "processed_dir", create=True)
    cube, dow, stations = ds["cube"], ds["dow"], ds["stations"]
    n_days, n_sensors, n_slots = cube.shape
    start = dt.datetime.strptime(cfg["dataset"]["nominal_start"], "%Y-%m-%d %H:%M:%S").date()
    dates = approximate_dates(dow, start)
    train_days = int(round(n_days * cfg["dataset"]["train_fraction"]))

    np.save(out / "occupancy_cube.npy", cube)
    days = pd.DataFrame({
        "day_index": np.arange(n_days), "original_label": ds["label"], "dow": dow,
        "dow_name": [DOW_NAMES[d] for d in dow], "is_weekend": dow >= 5,
        "approx_date": [d.isoformat() for d in dates], "stored_row": ds["stored_row"],
        "split": np.where(np.arange(n_days) < train_days, "train", "test"),
    })
    days.to_csv(out / "days.csv", index=False)
    pd.DataFrame({"sensor_index": np.arange(n_sensors), "sensor_id": stations}).to_csv(out / "sensors.csv", index=False)

    # Long format, one Parquet file per day (what Spark / the pandas baseline read in the benchmark)
    long_dir = out / "traffic_long"
    long_dir.mkdir(exist_ok=True)
    for old in long_dir.glob("*.parquet"):
        old.unlink()
    slot = np.tile(np.arange(n_slots, dtype=np.int16), n_sensors)
    sensor = np.repeat(stations.astype(np.int32), n_slots)
    for d in range(n_days):
        pd.DataFrame({
            "day_index": np.int16(d), "slot": slot, "sensor_id": sensor, "occupancy": cube[d].reshape(-1),
            "dow": np.int8(dow[d]), "hour": (slot // 6).astype(np.int8), "minute": ((slot % 6) * 10).astype(np.int8),
            "is_weekend": bool(dow[d] >= 5),
        }).to_parquet(long_dir / f"day_{d:03d}.parquet", index=False)

    report = dict(ds["report"], train_days=train_days, test_days=n_days - train_days,
                  total_records=int(cube.size), approx_first_date=dates[0].isoformat(), approx_last_date=dates[-1].isoformat())
    res = get_path(cfg, "results_dir", create=True)
    with open(res / "preprocessing_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    log.info("Wrote cube %s, %d daily parquet files, report", cube.shape, n_days)


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess PEMS-SF raw files")
    ap.add_argument("--zip", type=Path, default=None, help="pems_sf.zip (extracted to data/raw if raw files are missing)")
    args = ap.parse_args()
    cfg = load_config()
    raw = get_path(cfg, "raw_dir", create=True)
    extract_zip_if_needed(raw, args.zip)
    ds = build_dataset(raw, cfg["dataset"].get("n_sensors", 963), cfg["dataset"]["slots_per_day"])
    for k, v in ds["report"].items():
        log.info("%s: %s", k, v)
    write_outputs(ds, cfg)


if __name__ == "__main__":
    main()
