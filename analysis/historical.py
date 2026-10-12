"""
Historical analysis on the full real dataset:
  * calibration of the congestion index (TRAIN days only)
  * weekday x hour profile and data-driven PEAK periods
  * congested-location ranking
  * congestion-level distribution, daily summary, figures
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import DOW_NAMES, get_path, load_config, setup_logging
from common.congestion import LEVELS, calibrate, classify, congestion_index
from common.data import LEVEL_COLORS, load_processed, n_train_days

log = setup_logging("historical")


def hourly_network_stats(values: np.ndarray) -> np.ndarray:
    """(days, sensors, 144) -> (days, 24) network mean per hour."""
    d, s, _ = values.shape
    return values.reshape(d, s, 24, 6).mean(axis=(1, 3))


def peak_mask_from_days(ci: np.ndarray, dow: np.ndarray, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    """Peak cells = (weekday, hour) whose mean network CI is above the given percentile of all 168 cells.
    Returns (mask[7,24], mean_ci[7,24])."""
    hourly = hourly_network_stats(ci)
    grid = np.full((7, 24), np.nan)
    for d in range(7):
        if (dow == d).any():
            grid[d] = hourly[dow == d].mean(axis=0)
    thr = np.nanpercentile(grid, percentile)
    return grid >= thr, grid


def peak_periods_table(mask: np.ndarray, prof: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive peak hours into periods per weekday."""
    rows = []
    for d in range(7):
        h = 0
        while h < 24:
            if mask[d, h]:
                start = h
                while h < 24 and mask[d, h]:
                    h += 1
                sel = prof[(prof.dow == d) & (prof.hour >= start) & (prof.hour < h)]
                rows.append({"dow": d, "day": DOW_NAMES[d], "start": f"{start:02d}:00", "end": f"{h:02d}:00",
                             "hours": h - start, "mean_occupancy": sel.mean_occupancy.mean(),
                             "mean_ci": sel.mean_ci.mean(), "pct_high_or_severe": sel.pct_high_or_severe.mean()})
            else:
                h += 1
    return pd.DataFrame(rows)


def run() -> dict:
    cfg = load_config()
    res, fig = get_path(cfg, "results_dir", True), get_path(cfg, "figures_dir", True)
    cube, days, sensors = load_processed(cfg)
    n_tr = n_train_days(days)
    dow = days["dow"].to_numpy()

    # 1. calibration on TRAIN days only
    cal = calibrate(cube[:n_tr], cfg)
    cal.save(res / "calibration.json")
    log.info("Calibration: reference occupancy %.4f, CI cut-offs %s", cal.reference_occupancy, np.round(cal.cutoffs, 3))
    ci = congestion_index(cube, cal.reference_occupancy)
    lvl = classify(ci, cal.cutoffs)

    # 2. weekday x hour profile over all days
    occ_h, ci_h = hourly_network_stats(cube), hourly_network_stats(ci)
    high_h = hourly_network_stats((lvl >= 2).astype(np.float32))
    n_sensors = cube.shape[1]
    rows = []
    for d in range(7):
        sel = dow == d
        for h in range(24):
            rows.append({"dow": d, "day": DOW_NAMES[d], "hour": h, "mean_occupancy": occ_h[sel, h].mean(),
                         "mean_ci": ci_h[sel, h].mean(), "pct_high_or_severe": 100 * high_h[sel, h].mean(),
                         "n_observations": int(sel.sum()) * n_sensors * 6})
    prof = pd.DataFrame(rows)
    mask, _ = peak_mask_from_days(ci, dow, cfg["peaks"]["percentile"])
    prof["is_peak"] = [bool(mask[r.dow, r.hour]) for r in prof.itertuples()]
    prof.to_csv(res / "hourly_profile.csv", index=False)
    peaks = peak_periods_table(mask, prof)
    peaks.to_csv(res / "peak_periods.csv", index=False)
    log.info("Peak periods:\n%s", peaks[["day", "start", "end", "mean_occupancy"]].to_string(index=False))

    # 3. congested-location ranking
    slot_peak = mask[dow][:, np.arange(144) // 6]                   # (days, 144) bool
    peak_mean_ci = (ci * slot_peak[:, None, :]).sum(axis=(0, 2)) / slot_peak.sum()
    rank = pd.DataFrame({
        "sensor_id": sensors["sensor_id"], "mean_occupancy": cube.mean(axis=(0, 2)), "mean_ci": ci.mean(axis=(0, 2)),
        "peak_period_mean_ci": peak_mean_ci,
        "p95_ci": np.quantile(ci.reshape(len(cube), n_sensors, -1).transpose(1, 0, 2).reshape(n_sensors, -1), 0.95, axis=1),
        "pct_high_or_severe": 100 * (lvl >= 2).mean(axis=(0, 2)), "pct_severe": 100 * (lvl >= 3).mean(axis=(0, 2)),
    }).sort_values(["pct_high_or_severe", "mean_ci"], ascending=False).reset_index(drop=True)
    rank.insert(0, "rank", rank.index + 1)
    rank.to_csv(res / "sensor_ranking.csv", index=False)

    # 4. level distribution + daily summary
    dist = pd.DataFrame({"level": LEVELS,
                         "all_days": [(lvl == i).mean() * 100 for i in range(4)],
                         "train_days": [(lvl[:n_tr] == i).mean() * 100 for i in range(4)],
                         "test_days": [(lvl[n_tr:] == i).mean() * 100 for i in range(4)]})
    dist.to_csv(res / "level_distribution.csv", index=False)
    daily = days[["day_index", "dow_name", "is_weekend", "approx_date", "split"]].copy()
    daily["mean_occupancy"] = cube.mean(axis=(1, 2))
    daily["mean_ci"] = ci.mean(axis=(1, 2))
    daily["pct_high_or_severe"] = 100 * (lvl >= 2).mean(axis=(1, 2))
    daily.to_csv(res / "daily_summary.csv", index=False)

    summary = {
        "records": int(cube.size), "days": int(len(cube)), "sensors": int(n_sensors),
        "calibration": {"reference_occupancy": cal.reference_occupancy, "cutoffs_ci": cal.cutoffs},
        "level_share_percent_all_days": dict(zip(LEVELS, np.round(dist.all_days.to_numpy(), 3).tolist())),
        "n_peak_cells": int(mask.sum()), "peak_percentile": cfg["peaks"]["percentile"],
        "top5_sensors": rank.head(5)[["sensor_id", "pct_high_or_severe", "mean_occupancy"]].round(4).to_dict("records"),
        "sensors_with_over_5pct_high_or_severe": int((rank.pct_high_or_severe > 5).sum()),
    }
    with open(res / "historical_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    make_figures(prof, mask, rank, dist, daily, cube, dow, fig, cfg)
    return summary


def make_figures(prof, mask, rank, dist, daily, cube, dow, fig, cfg) -> None:
    # (a) heatmap weekday x hour with peak cells outlined
    grid = prof.pivot(index="dow", columns="hour", values="mean_occupancy").to_numpy()
    f, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7), DOW_NAMES); ax.set_xticks(range(24)); ax.set_xlabel("Hour of day"); ax.set_title("Mean network occupancy by weekday and hour (black outline = peak cells)")
    for d in range(7):
        for h in range(24):
            if mask[d, h]:
                ax.add_patch(plt.Rectangle((h - .5, d - .5), 1, 1, fill=False, ec="black", lw=1.4))
    f.colorbar(im, ax=ax, label="mean occupancy"); f.tight_layout(); f.savefig(fig / "heatmap_weekday_hour.png", dpi=150); plt.close(f)

    # (b) average daily profile weekday vs weekend
    wk = ~np.isin(dow, [5, 6])
    x = np.arange(144) / 6
    f, ax = plt.subplots(figsize=(9, 3.8))
    for name, sel, c in [("Weekdays (Mon-Fri)", wk, "#c2362f"), ("Weekend (Sat-Sun)", ~wk, "#2f6db5")]:
        daily_net = cube[sel].mean(axis=1)
        m, s = daily_net.mean(axis=0), daily_net.std(axis=0)
        ax.plot(x, m, label=name, color=c); ax.fill_between(x, m - s, m + s, color=c, alpha=.15)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("network mean occupancy"); ax.set_xticks(range(0, 25, 3)); ax.legend(); ax.set_title("Average daily occupancy profile (band = ±1 std over days)")
    f.tight_layout(); f.savefig(fig / "daily_profile_weekday_weekend.png", dpi=150); plt.close(f)

    # (c) top congested sensors
    top = rank.head(15).iloc[::-1]
    f, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.sensor_id.astype(str), top.pct_high_or_severe, color="#e07a3a"); ax.set_xlabel("% of observations at HIGH or SEVERE"); ax.set_ylabel("sensor id")
    ax.set_title("15 most congested sensors"); f.tight_layout(); f.savefig(fig / "top_congested_sensors.png", dpi=150); plt.close(f)

    # (d) level distribution
    f, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(dist.level, dist.all_days, color=[LEVEL_COLORS[l] for l in dist.level]); ax.set_yscale("log"); ax.set_ylabel("% of all sensor-slot observations (log)")
    for i, v in enumerate(dist.all_days):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    ax.set_title("Congestion level distribution"); f.tight_layout(); f.savefig(fig / "level_distribution.png", dpi=150); plt.close(f)

    # (e) daily network occupancy
    f, ax = plt.subplots(figsize=(10, 3.4))
    colors = np.where(daily.is_weekend, "#2f6db5", "#c2362f")
    ax.bar(daily.day_index, daily.mean_occupancy, color=colors, width=1.0)
    ax.axvline(int(daily.split.eq("train").sum()) - .5, color="black", ls="--"); ax.text(int(daily.split.eq("train").sum()) + 1, daily.mean_occupancy.max(), "test period →", va="top")
    ax.set_xlabel("day index (chronological)"); ax.set_ylabel("mean occupancy"); ax.set_title("Daily network mean occupancy (red = weekday, blue = weekend)")
    f.tight_layout(); f.savefig(fig / "daily_mean_occupancy.png", dpi=150); plt.close(f)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
