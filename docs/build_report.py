"""
Builds REPORT.md and PRESENTATION_CONTENT.md from the templates in docs/ using the REAL result files in
output/results. Every number in the report comes from those files; a missing result becomes
"NOT YET EXECUTED" plus the command that produces it.

    python docs\build_report.py            (re-run after repeating the benchmark on your own computer)
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_config  # noqa: E402

cfg = load_config()
RES = ROOT / cfg["paths"]["results_dir"]


def jload(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def cload(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def md(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def build_values() -> dict:
    V: dict[str, str] = {"today": date.today().isoformat()}
    pre, cal, hist = jload("preprocessing_report.json"), jload("calibration.json"), jload("historical_summary.json")
    ref = cal["reference_occupancy"]
    V.update(n_stored=pre["stored_days"], n_distinct=pre["distinct_days"], n_dups=pre["duplicate_days_removed"], n_records=f"{pre['total_records']:,}",
             n_train=pre["train_days"], n_test=pre["test_days"], first_date=pre["approx_first_date"], last_date=pre["approx_last_date"],
             chain_before=f"{pre['weekday_chain_score_stored_order']:.2f}", chain_after=f"{pre['weekday_chain_score_restored_order']:.2f}",
             zeros=f"{pre['zero_values_kept']:,}", gaps=pre["removed_day_gaps_in_chain"])
    c = cal["cutoffs"]
    V.update(ref=f"{ref:.4f}", cut_mod=f"{c[0]:.3f}", cut_high=f"{c[1]:.3f}", cut_sev=f"{c[2]:.3f}",
             occ_mod=f"{c[0] * ref:.4f}", occ_high=f"{c[1] * ref:.4f}", occ_sev=f"{ref:.4f}")
    sh = hist["level_share_percent_all_days"]
    V.update(sh_low=f"{sh['LOW']:.2f}", sh_mod=f"{sh['MODERATE']:.2f}", sh_high=f"{sh['HIGH']:.2f}", sh_sev=f"{sh['SEVERE']:.2f}")
    peaks = cload("peak_periods.csv")
    V["peak_table"] = md(peaks[["day", "start", "end", "mean_occupancy", "mean_ci", "pct_high_or_severe"]].rename(
        columns={"day": "Day", "start": "From", "end": "To", "mean_occupancy": "Mean occupancy", "mean_ci": "Mean CI", "pct_high_or_severe": "% HIGH+SEVERE"}))
    V["n_peak_cells"] = hist["n_peak_cells"]
    V["peak_pct"] = hist["peak_percentile"]
    rank = cload("sensor_ranking.csv")
    V["top10_table"] = md(rank.head(10)[["rank", "sensor_id", "mean_occupancy", "pct_high_or_severe", "pct_severe", "peak_period_mean_ci"]].rename(
        columns={"rank": "Rank", "sensor_id": "Sensor", "mean_occupancy": "Mean occ.", "pct_high_or_severe": "% HIGH+", "pct_severe": "% SEVERE", "peak_period_mean_ci": "Peak-period CI"}))
    V["top3_ids"] = ", ".join(str(int(s)) for s in rank.head(3).sensor_id)
    V["top1_pct"] = f"{rank.iloc[0].pct_high_or_severe:.1f}"
    V["n_sens_5"] = hist["sensors_with_over_5pct_high_or_severe"]

    t = jload("t_test_results.json")
    p, ins, wk = t["primary_heldout_test_days"], t["comparison_in_sample_all_days"], t["weekday_vs_weekend"]
    V.update(alpha=t["alpha"], t_n=p["n_days"], t_df=p["paired_df"], peak_mean=f"{p['peak_mean_occupancy']:.4f}", off_mean=f"{p['offpeak_mean_occupancy']:.4f}",
             t_diff=f"{p['mean_difference']:.4f}", t_lo=f"{p['difference_ci95'][0]:.4f}", t_hi=f"{p['difference_ci95'][1]:.4f}", t_stat=f"{p['paired_t_statistic']:.2f}",
             t_p=fmt_p(p["paired_p_value"]), t_d=f"{p['cohens_d_paired']:.1f}", w_stat=f"{p['welch_t_statistic']:.2f}", w_p=fmt_p(p["welch_p_value"]),
             ins_n=ins["n_days"], ins_t=f"{ins['paired_t_statistic']:.2f}", ins_p=fmt_p(ins["paired_p_value"]),
             wk_mean=f"{wk['weekday_mean']:.4f}", we_mean=f"{wk['weekend_mean']:.4f}", wk_t=f"{wk['welch_t_statistic']:.2f}", wk_p=fmt_p(wk["welch_p_value"]),
             t_verdict="REJECT H0" if p["reject_H0_paired"] else "FAIL TO REJECT H0")

    m = jload("prediction_metrics.json")
    rf, pe, lo = m["random_forest"], m["baseline_persistence"], m["baseline_always_low"]
    V.update(horizon=m["horizon_minutes"], p_train_samples=f"{m['train_samples']:,}", p_test_samples=f"{m['test_samples']:,}")
    rows = []
    for name, r in [("Random Forest", rf), ("Persistence baseline", pe), ("Always LOW baseline", lo)]:
        rows.append({"Model": name, "Accuracy": r["accuracy"], "Macro precision": r["macro"]["precision"], "Macro recall": r["macro"]["recall"],
                     "Macro F1": r["macro"]["f1"], "HIGH+ precision": r["high_or_severe_binary"]["precision"],
                     "HIGH+ recall": r["high_or_severe_binary"]["recall"], "HIGH+ F1": r["high_or_severe_binary"]["f1"]})
    V["pred_table"] = md(pd.DataFrame(rows))
    pc = pd.DataFrame(rf["per_class"]).T.reset_index().rename(columns={"index": "Level", "precision": "Precision", "recall": "Recall", "f1": "F1", "support": "Support"})
    pc["Support"] = pc["Support"].astype(int)
    V["perclass_table"] = md(pc)
    cm = pd.DataFrame(rf["confusion_matrix"], index=["actual " + l for l in ["LOW", "MODERATE", "HIGH", "SEVERE"]], columns=["pred LOW", "pred MODERATE", "pred HIGH", "pred SEVERE"])
    V["cm_table"] = md(cm.reset_index().rename(columns={"index": ""}))
    V["train_counts"] = ", ".join(f"{k} {v:,}" for k, v in m["train_class_counts"].items())
    V["test_counts"] = ", ".join(f"{k} {v:,}" for k, v in m["test_class_counts"].items())
    better = [n for n, a, b in [("accuracy", rf["accuracy"], pe["accuracy"]), ("macro F1", rf["macro"]["f1"], pe["macro"]["f1"]),
                                 ("HIGH+ F1", rf["high_or_severe_binary"]["f1"], pe["high_or_severe_binary"]["f1"])] if a > b]
    V.update(rf_acc=f"{rf['accuracy']:.3f}", rf_mf1=f"{rf['macro']['f1']:.3f}", pe_acc=f"{pe['accuracy']:.3f}", pe_mf1=f"{pe['macro']['f1']:.3f}")
    V["rf_vs_persist"] = (f"The Random Forest is above the persistence baseline on {', '.join(better) if better else 'none of the headline metrics'} "
                          f"(accuracy {rf['accuracy']:.3f} vs {pe['accuracy']:.3f}; macro F1 {rf['macro']['f1']:.3f} vs {pe['macro']['f1']:.3f}; "
                          f"HIGH+ F1 {rf['high_or_severe_binary']['f1']:.3f} vs {pe['high_or_severe_binary']['f1']:.3f}). The margin is modest because the congestion state is strongly persistent over 30 minutes; "
                          "the always-LOW baseline shows that accuracy alone is misleading for this imbalanced problem.")
    imp = sorted(m["feature_importance"].items(), key=lambda kv: -kv[1])[:4]
    V["top_features"] = ", ".join(f"{k} ({v:.2f})" for k, v in imp)
    tune = jload("tuning_validation.json")
    tt = pd.DataFrame(tune["candidates"]).T.reset_index().rename(columns={"index": "Candidate (validation days)", "accuracy": "Accuracy", "macro_f1": "Macro F1", "high_or_severe_f1": "HIGH+ F1", "fit_seconds": "Fit s"})
    pers = tune["persistence_baseline"]
    tt.loc[len(tt)] = ["Persistence baseline", pers["accuracy"], pers["macro_f1"], pers["high_or_severe_f1"], 0.0]
    V["tune_table"] = md(tt[["Candidate (validation days)", "Accuracy", "Macro F1", "HIGH+ F1"]])

    bm, meta = cload("benchmark_results.csv"), jload("benchmark_meta.json")
    V["bench_table"] = md(bm[["days", "records", "pandas_seconds", "spark_seconds", "speedup", "pandas_records_per_s", "spark_records_per_s"]].rename(
        columns={"days": "Days", "records": "Records", "pandas_seconds": "Serial pandas (s)", "spark_seconds": "Spark (s)", "speedup": "Speedup",
                 "pandas_records_per_s": "pandas rec/s", "spark_records_per_s": "Spark rec/s"}).astype({"Records": int}), "{:,.2f}")
    env = meta["environment"]
    V["bench_env"] = f"{env['cpu_count']} CPU core(s), {env.get('ram_gb')} GB RAM, Python {env['python']}, PySpark {env['pyspark']}, pandas {env['pandas']}, Spark master {meta['spark_master']}"
    V.update(spark_startup=meta["spark_session_startup_seconds"], spark_first=meta["spark_first_job_seconds"], repeats=meta["repeats"])
    wins = bm[bm["speedup"] >= 1]
    last_loss = bm[bm["speedup"] < 1]
    if len(wins):
        V["crossover"] = (f"Spark was slower than the serial baseline for datasets up to {int(last_loss['records'].max()):,} records and faster from {int(wins['records'].iloc[0]):,} records "
                          f"({wins['speedup'].iloc[0]:.2f}x). The crossover therefore lies between {int(last_loss['records'].max()):,} and {int(wins['records'].iloc[0]):,} records on this machine.")
    else:
        V["crossover"] = f"Spark did not overtake the serial baseline for any tested size (largest {int(bm['records'].max()):,} records) on this machine."
    V["max_speedup"] = f"{bm['speedup'].max():.2f}"
    sc = cload("benchmark_scaling.csv")
    if sc is not None and len(sc) >= 2:
        V["scaling_text"] = "Spark local[k] scalability (" + ", ".join(f"{int(r.cores)} cores: {r.spark_seconds:.2f} s, speedup {r.speedup_vs_1_core:.2f}x" for r in sc.itertuples()) + ")."
    else:
        V["scaling_text"] = ("Scalability over cores: NOT YET EXECUTED - the development machine had a single core. "
                             "Run `python -m performance.benchmark --scaling` on a multi-core computer.")

    sv = jload("stream_verification.json")
    V.update(sv_records=f"{sv['records']:,}", sv_ticks=sv["ticks"], sv_agree=f"{sv['spark_vs_numpy_level_agreement_percent']:.2f}", sv_raw=f"{sv['raw_high_or_severe_sensor_ticks']:,}",
             sv_alerts=f"{sv['alerts_raised']:,}", sv_new=sv["alerts_by_type"].get("NEW", 0), sv_esc=sv["alerts_by_type"].get("ESCALATION", 0), sv_sensors=sv["sensors_alerted"],
             sv_red=sv["alert_reduction_factor"], sv_batches=sv["micro_batches"], sv_rps=f"{sv['mean_records_per_second']:,.0f}", sv_windows=f"{sv['window_rows']:,}",
             sv_source=sv["source"], sv_diff=f"{sv['max_abs_occupancy_difference_vs_source']:.1e}",
             sv_levels=", ".join(f"{k} {v:,}" for k, v in sv["level_counts"].items()))
    ps = jload("producer_stats_maxspeed.json")
    V["kafka_text"] = (f"Kafka producer maximum throughput: {ps['messages_per_second_send_only']:,.0f} messages/s ({ps['messages_sent']:,} messages)." if ps else
                       "Kafka throughput: NOT YET EXECUTED - requires a Kafka broker. Run `run_project.bat bench` (`python -m ingestion.producer --max-speed --num-days 5`) and fill in "
                       "`output\\results\\producer_stats_maxspeed.json`, then re-run `python docs\\build_report.py`.")
    return {k: str(v) for k, v in V.items()}


def render(template: Path, out: Path, values: dict) -> None:
    text = template.read_text(encoding="utf-8")
    missing = set(re.findall(r"\{\{(\w+)\}\}", text)) - set(values)
    if missing:
        raise KeyError(f"Template placeholders without values: {sorted(missing)}")
    out.write_text(re.sub(r"\{\{(\w+)\}\}", lambda m: values[m.group(1)], text), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    vals = build_values()
    render(ROOT / "docs" / "REPORT.template.md", ROOT / "REPORT.md", vals)
    render(ROOT / "docs" / "PRESENTATION.template.md", ROOT / "PRESENTATION_CONTENT.md", vals)
