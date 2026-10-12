"""
t-tests on the real data.

H0: mean occupancy in PEAK periods = mean occupancy in OFF-PEAK periods
H1: they differ (two-sided), alpha from config (0.05).

Statistical design (explained in the report):
 * Unit of analysis = one DAY. Raw 10-minute values of neighbouring slots/sensors are strongly
   autocorrelated, so using them as independent samples would inflate the t-statistic. Each weekday
   gives ONE peak mean and ONE off-peak mean (network average) -> n = number of weekdays.
 * The two means come from the same day, so the PRIMARY test is the PAIRED t-test. The Welch
   two-sample t-test is reported as well.
 * Peaks are selected because occupancy is high there, so testing them on the data used to select
   them is circular. PRIMARY analysis: peaks defined on TRAIN days, tested on held-out TEST days.
   The in-sample version is reported only for comparison.
 * Extra test: weekday vs weekend daily mean occupancy (Welch).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.historical import peak_mask_from_days
from common.config import DOW_NAMES, get_path, load_config, setup_logging
from common.congestion import Calibration, congestion_index
from common.data import load_processed, n_train_days

log = setup_logging("t_test")


def day_peak_offpeak_means(cube: np.ndarray, dow: np.ndarray, mask: np.ndarray):
    """Per day network-mean occupancy over peak slots and over off-peak slots. Days without peak slots are dropped."""
    slot_peak = mask[dow][:, np.arange(cube.shape[2]) // 6]          # (days, 144)
    net = cube.mean(axis=1, dtype=np.float64)                                           # (days, 144) network mean per slot
    keep = slot_peak.any(axis=1)
    peak = np.array([net[i, slot_peak[i]].mean() for i in np.where(keep)[0]])
    off = np.array([net[i, ~slot_peak[i]].mean() for i in np.where(keep)[0]])
    return peak, off, np.where(keep)[0]


def paired_and_welch(peak: np.ndarray, off: np.ndarray, alpha: float) -> dict:
    diff = peak - off
    n = len(diff)
    t_rel, p_rel = stats.ttest_rel(peak, off)
    t_w, p_w = stats.ttest_ind(peak, off, equal_var=False)
    se = diff.std(ddof=1) / np.sqrt(n)
    ci = stats.t.interval(1 - alpha, n - 1, loc=diff.mean(), scale=se)
    return {
        "n_days": int(n), "peak_mean_occupancy": float(peak.mean()), "offpeak_mean_occupancy": float(off.mean()),
        "mean_difference": float(diff.mean()), "difference_ci95": [float(ci[0]), float(ci[1])],
        "paired_t_statistic": float(t_rel), "paired_p_value": float(p_rel), "paired_df": int(n - 1),
        "cohens_d_paired": float(diff.mean() / diff.std(ddof=1)),
        "welch_t_statistic": float(t_w), "welch_p_value": float(p_w),
        "alpha": alpha, "reject_H0_paired": bool(p_rel < alpha), "reject_H0_welch": bool(p_w < alpha),
    }


def run() -> dict:
    cfg = load_config()
    alpha = cfg["statistics"]["alpha"]
    res, fig = get_path(cfg, "results_dir", True), get_path(cfg, "figures_dir", True)
    cube, days, _ = load_processed(cfg)
    cal = Calibration.load(res / "calibration.json")
    ci = congestion_index(cube, cal.reference_occupancy)
    dow, n_tr = days["dow"].to_numpy(), n_train_days(days)

    # PRIMARY: peaks from TRAIN days, test on held-out days
    mask_tr, _ = peak_mask_from_days(ci[:n_tr], dow[:n_tr], cfg["peaks"]["percentile"])
    peak, off, idx = day_peak_offpeak_means(cube[n_tr:], dow[n_tr:], mask_tr)
    primary = paired_and_welch(peak, off, alpha)
    # COMPARISON: in-sample (peaks and test on all days)
    mask_all, _ = peak_mask_from_days(ci, dow, cfg["peaks"]["percentile"])
    p_all, o_all, _ = day_peak_offpeak_means(cube, dow, mask_all)
    insample = paired_and_welch(p_all, o_all, alpha)
    # EXTRA: weekday vs weekend daily mean occupancy
    daily = cube.mean(axis=(1, 2), dtype=np.float64)
    wk, we = daily[dow < 5], daily[dow >= 5]
    t_ww, p_ww = stats.ttest_ind(wk, we, equal_var=False)
    weekend = {"weekday_mean": float(wk.mean()), "weekend_mean": float(we.mean()), "n_weekdays": int(len(wk)), "n_weekend_days": int(len(we)),
               "welch_t_statistic": float(t_ww), "welch_p_value": float(p_ww), "reject_H0": bool(p_ww < alpha)}

    out = {"H0": "mean occupancy in peak periods == mean occupancy in off-peak periods", "H1": "means differ (two-sided)",
           "alpha": alpha, "primary_heldout_test_days": primary, "comparison_in_sample_all_days": insample,
           "weekday_vs_weekend": weekend,
           "peak_cells_from_train_days": [{"day": DOW_NAMES[d], "hour": int(h)} for d, h in zip(*np.where(mask_tr))]}
    with open(res / "t_test_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    log.info("Primary (held-out, paired): peak=%.4f off=%.4f t=%.2f p=%.3g n=%d", primary["peak_mean_occupancy"],
             primary["offpeak_mean_occupancy"], primary["paired_t_statistic"], primary["paired_p_value"], primary["n_days"])

    f, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].boxplot([peak, off], tick_labels=["peak", "off-peak"], widths=.5)
    for a, b in zip(peak, off):
        ax[0].plot([1, 2], [a, b], color="gray", alpha=.25, lw=.8)
    ax[0].set_ylabel("network mean occupancy (one line per held-out weekday)"); ax[0].set_title("Peak vs off-peak (paired days)")
    ax[1].hist(peak - off, bins=15, color="#e07a3a"); ax[1].axvline(0, color="black", ls="--"); ax[1].set_xlabel("peak − off-peak occupancy"); ax[1].set_title("Daily differences")
    f.tight_layout(); f.savefig(fig / "t_test_peak_vs_offpeak.png", dpi=150); plt.close(f)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2)[:2500])
