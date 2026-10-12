"""
Streamlit dashboard - Real-Time Traffic Congestion Detection and Prediction.

Live sections read the output of the Spark Structured Streaming job (output/stream, output/alerts).
Static sections read the result files produced by the offline pipeline (output/results).
Nothing here is generated or simulated: if a file is missing, the section says which command creates it.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DOW_NAMES, get_path, load_config  # noqa: E402
from common.congestion import LEVELS  # noqa: E402
from common.data import LEVEL_COLORS  # noqa: E402

cfg = load_config()
RES, STREAM, ALERTS = get_path(cfg, "results_dir"), get_path(cfg, "stream_dir"), get_path(cfg, "alerts_dir")
FIGS = get_path(cfg, "figures_dir")
H = cfg["prediction"]["horizon_slots"]
TOP_N = cfg["dashboard"]["top_n"]
LEVEL_ORDER = LEVELS

st.set_page_config(page_title="Traffic Congestion Detection & Prediction", layout="wide", page_icon="🚦")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1400px}
h1{font-size:1.9rem!important;margin-bottom:.1rem}
[data-testid="stMetricValue"]{font-size:1.6rem}
.note{color:#52656f;font-size:.9rem}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Controls")
    auto = st.toggle("Auto-refresh live sections", value=True)
    refresh = st.slider("Refresh every (s)", 1, 10, int(cfg["dashboard"]["refresh_seconds"]))
    st.divider()
    st.caption("Data: UCI PEMS-SF (Caltrans PeMS), 963 freeway detectors, lane occupancy every 10 min. "
               "The live feed is a **replay of real recorded observations** through Kafka, not generated data.")
    st.caption("Only occupancy exists in this dataset: there is no speed or flow, so none is shown.")
REFRESH = refresh if auto else None


# ------------------------------------------------------------------ data access helpers
def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _csv(path: str, mtime: float):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def static_csv(name: str):
    p = RES / name
    return _csv(str(p), _mtime(p)) if p.exists() else None


def static_json(name: str):
    p = RES / name
    return _json(str(p), _mtime(p)) if p.exists() else None


def missing(cmd: str) -> None:
    st.info(f"Not generated yet. Run: `{cmd}`")


def read_live(sub: str):
    """Read a Spark parquet sink directory; Spark may be writing, so any error simply means 'try next refresh'."""
    d = STREAM / sub
    if not d.exists():
        return None
    try:
        df = pd.read_parquet(d)
        return df if len(df) else None
    except Exception:
        return None


def read_alerts():
    p = ALERTS / "alerts.csv"
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _bundle(mtime: float):
    from prediction.predict import load_bundle
    return load_bundle(cfg)


def bundle():
    p = get_path(cfg, "models_dir") / "congestion_rf.joblib"
    return _bundle(_mtime(p)) if p.exists() else None


@st.cache_data(show_spinner=False)
def sensor_index():
    s = pd.read_csv(get_path(cfg, "processed_dir") / "sensors.csv")
    return {int(v): i for i, v in enumerate(s["sensor_id"])}


def fmt_time(slot: int) -> str:
    return f"{slot * 10 // 60:02d}:{slot * 10 % 60:02d}"


def level_bar(codes: pd.Series, title: str = ""):
    counts = codes.map(dict(enumerate(LEVELS))).value_counts().reindex(LEVELS, fill_value=0)
    fig = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_map=LEVEL_COLORS,
                 labels={"x": "level", "y": "sensors"}, title=title)
    fig.update_layout(showlegend=False, height=320, margin=dict(t=40, b=10))
    return fig


# ------------------------------------------------------------------ header + live KPIs
st.title("Real-Time Traffic Congestion Detection and Prediction")
st.markdown('<span class="note">Kafka → Spark Structured Streaming → congestion index → alerts → ML prediction. '
            'Real PEMS-SF observations replayed as a stream.</span>', unsafe_allow_html=True)


@st.fragment(run_every=REFRESH)
def live_kpis():
    rec, al = read_live("records"), read_alerts()
    c = st.columns(5)
    if rec is None:
        for col, name in zip(c, ["Active Sensors", "Average Occupancy", "Congested Locations", "Current Alerts", "Stream Time"]):
            col.metric(name, "–")
        st.warning("Waiting for the stream. Start Kafka, the Spark job and the producer (README → Demo). "
                   "Historical sections below are already available.")
        return
    t = rec["tick"].max()
    cur = rec[rec["tick"] == t]
    day, slot = int(cur["day_index"].iloc[0]), int(cur["slot"].iloc[0])
    n_recent = 0 if al is None else int((al["day_index"] * 144 + al["slot"] >= t - 5).sum())
    c[0].metric("Active Sensors", f"{cur['sensor_id'].nunique()}")
    c[1].metric("Average Occupancy", f"{cur['occupancy'].mean():.3f}")
    c[2].metric("Congested Locations", f"{int((cur['level_code'] >= 2).sum())}", help="Sensors at HIGH or SEVERE right now")
    c[3].metric("Current Alerts", f"{n_recent}", help="Alerts raised in the last 60 minutes of stream time. Total so far: "
                f"{0 if al is None else len(al)}")
    c[4].metric("Stream Time", f"{fmt_time(slot)}", f"day {day} · {DOW_NAMES[int(cur['dow'].iloc[0])]}", delta_color="off")


live_kpis()

tabs = st.tabs(["Live Traffic Status", "Congested Locations", "Peak Traffic Periods", "Congestion Trend",
                "Congestion Alerts", "Prediction", "Statistical Analysis", "Performance Analysis"])

# ------------------------------------------------------------------ 1. live status
with tabs[0]:
    @st.fragment(run_every=REFRESH)
    def live_status():
        rec, ticks = read_live("records"), read_live("tick_summary")
        if rec is None or ticks is None:
            st.info("No streaming data yet.")
            return
        t = rec["tick"].max()
        cur = rec[rec["tick"] == t]
        day = int(cur["day_index"].iloc[0])
        left, right = st.columns([3, 2])
        tk = ticks[ticks["day_index"] == day].sort_values("tick")
        fig = px.line(tk, x=tk["slot"].map(fmt_time), y="mean_occupancy", labels={"x": "time of day", "mean_occupancy": "network mean occupancy"},
                      title="Network average occupancy (Spark tick summary)")
        fig.add_vline(x=fmt_time(int(cur["slot"].iloc[0])), line_dash="dot", line_color="#c2362f")
        fig.update_layout(height=340, margin=dict(t=40, b=10))
        left.plotly_chart(fig, width="stretch")
        right.plotly_chart(level_bar(cur["level_code"], "Sensors per congestion level (now)"), width="stretch")
        top3 = cur.sort_values("ci", ascending=False)["sensor_id"].head(3).tolist()
        chosen = st.multiselect("Sensors to plot (default: 3 most congested now)", sorted(cur["sensor_id"].unique().tolist()),
                                default=top3, key="live_sensors")
        if chosen:
            sub = rec[(rec["day_index"] == day) & rec["sensor_id"].isin(chosen)].sort_values("tick")
            fig2 = px.line(sub, x=sub["slot"].map(fmt_time), y="occupancy", color=sub["sensor_id"].astype(str),
                           labels={"x": "time of day", "occupancy": "occupancy", "color": "sensor"}, title="Occupancy of selected sensors")
            fig2.update_layout(height=320, margin=dict(t=40, b=10))
            st.plotly_chart(fig2, width="stretch")
    live_status()

# ------------------------------------------------------------------ 2. congested locations
with tabs[1]:
    @st.fragment(run_every=REFRESH)
    def live_top():
        rec = read_live("records")
        st.subheader("Most congested sensors right now")
        if rec is None:
            st.info("No streaming data yet.")
            return
        cur = rec[rec["tick"] == rec["tick"].max()].sort_values("ci", ascending=False).head(TOP_N)
        fig = px.bar(cur.iloc[::-1], x="ci", y=cur.iloc[::-1]["sensor_id"].astype(str), orientation="h", color=cur.iloc[::-1]["level_code"].map(dict(enumerate(LEVELS))),
                     color_discrete_map=LEVEL_COLORS, labels={"ci": "congestion index", "y": "sensor", "color": "level"})
        fig.update_layout(height=420, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    live_top()
    st.subheader("Historically most congested sensors (all 267 days)")
    rank = static_csv("sensor_ranking.csv")
    if rank is None:
        missing("python -m analysis.historical")
    else:
        a, b = st.columns([2, 3])
        a.dataframe(rank.head(TOP_N).round(4), hide_index=True, width="stretch")
        top = rank.head(TOP_N).iloc[::-1]
        fig = px.bar(top, x="pct_high_or_severe", y=top["sensor_id"].astype(str), orientation="h",
                     labels={"pct_high_or_severe": "% of observations at HIGH or SEVERE", "y": "sensor"}, color_discrete_sequence=["#e07a3a"])
        fig.update_layout(height=430, margin=dict(t=10, b=10))
        b.plotly_chart(fig, width="stretch")
    st.caption("No map: the PEMS-SF files contain sensor IDs only, no coordinates, and none were invented.")

# ------------------------------------------------------------------ 3. peak periods
with tabs[2]:
    prof, peaks = static_csv("hourly_profile.csv"), static_csv("peak_periods.csv")
    if prof is None:
        missing("python -m analysis.historical")
    else:
        grid = prof.pivot(index="dow", columns="hour", values="mean_occupancy").sort_index()
        pk = prof.pivot(index="dow", columns="hour", values="is_peak").sort_index()
        fig = go.Figure(go.Heatmap(z=grid.values, x=list(range(24)), y=DOW_NAMES, colorscale="YlOrRd", colorbar=dict(title="mean occupancy")))
        ys, xs = np.where(pk.values.astype(bool))
        fig.add_trace(go.Scatter(x=xs, y=[DOW_NAMES[i] for i in ys], mode="markers", marker=dict(symbol="square-open", size=22, color="black", line=dict(width=2)), name="peak cell"))
        fig.update_layout(height=380, xaxis_title="hour of day", title=f"Mean network occupancy by weekday and hour (peak = top {100 - cfg['peaks']['percentile']}% of cells by congestion index)", margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(peaks.drop(columns=["dow"]).round(4), hide_index=True, width="stretch")
        st.caption("Peak periods are derived from the data (percentile rule), not assumed.")
        if (FIGS / "daily_profile_weekday_weekend.png").exists():
            st.image(str(FIGS / "daily_profile_weekday_weekend.png"))

# ------------------------------------------------------------------ 4. trend
with tabs[3]:
    @st.fragment(run_every=REFRESH)
    def live_trend():
        ticks = read_live("tick_summary")
        if ticks is None:
            st.info("No streaming data yet.")
            return
        day = int(ticks.sort_values("tick")["day_index"].iloc[-1])
        tk = ticks[ticks["day_index"] == day].sort_values("tick")
        x = tk["slot"].map(fmt_time)
        fig = px.line(tk, x=x, y="mean_ci", labels={"x": "time of day", "mean_ci": "network mean congestion index"}, title="Congestion trend (day being replayed)")
        fig.update_layout(height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        long = tk.melt(id_vars=["slot"], value_vars=[f"n_level_{k}" for k in range(4)], var_name="level", value_name="sensors")
        long["level"] = long["level"].map({f"n_level_{k}": LEVELS[k] for k in range(4)})
        fig2 = px.area(long, x=long["slot"].map(fmt_time), y="sensors", color="level", color_discrete_map=LEVEL_COLORS, category_orders={"level": LEVEL_ORDER},
                       labels={"x": "time of day"}, title="Sensors per congestion level over time")
        fig2.update_layout(height=330, margin=dict(t=40, b=10))
        st.plotly_chart(fig2, width="stretch")
    live_trend()
    dist = static_csv("level_distribution.csv")
    if dist is not None:
        st.subheader("Congestion level distribution (historical, % of all observations)")
        st.dataframe(dist.round(3), hide_index=True, width="stretch")
    daily = static_csv("daily_summary.csv")
    if daily is not None:
        fig = px.bar(daily, x="day_index", y="mean_occupancy", color="is_weekend", color_discrete_map={False: "#c2362f", True: "#2f6db5"},
                     labels={"day_index": "day index", "mean_occupancy": "mean occupancy", "is_weekend": "weekend"}, title="Daily network mean occupancy, all days")
        fig.update_layout(height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ 5. alerts
with tabs[4]:
    @st.fragment(run_every=REFRESH)
    def live_alerts():
        al, ticks = read_alerts(), read_live("tick_summary")
        if al is None:
            st.info("No alerts yet (or stream not started).")
            return
        c = st.columns(4)
        c[0].metric("Alerts raised", len(al))
        c[1].metric("Sensors alerted", al["sensor_id"].nunique())
        c[2].metric("SEVERE", int((al["level"] == "SEVERE").sum()))
        if ticks is not None:
            raw = int(ticks[["n_level_2", "n_level_3"]].sum().sum())
            c[3].metric("Raw HIGH+ sensor-ticks", raw, help="Number of alerts that would be raised WITHOUT de-duplication/cooldown")
        for _, r in al.tail(3).iloc[::-1].iterrows():
            st.error(r["message"])
        st.dataframe(al.drop(columns=["message"]).iloc[::-1].head(100), hide_index=True, width="stretch")
        st.caption("One alert per congestion episode per sensor (+ one ESCALATION if the level rises); cooldown suppresses flapping.")
    live_alerts()

# ------------------------------------------------------------------ 6. prediction
with tabs[5]:
    m = static_json("prediction_metrics.json")
    b = bundle()
    st.subheader(f"Predicted congestion level {H * 10} minutes ahead")

    @st.fragment(run_every=REFRESH)
    def live_prediction():
        rec = read_live("records")
        if rec is None or b is None:
            st.info("Needs streaming data and a trained model (`python -m prediction.train_model`).")
            return
        from prediction.predict import predict_levels
        idx = sensor_index()
        t = rec["tick"].max()
        day, slot = int(rec.loc[rec["tick"] == t, "day_index"].iloc[0]), int(rec.loc[rec["tick"] == t, "slot"].iloc[0])
        dow = int(rec.loc[rec["tick"] == t, "dow"].iloc[0])
        today = rec[rec["day_index"] == day]
        sm = b["sensor_mean_ci"]

        def predict_at(s: int):
            w = today[(today["slot"] >= s - 5) & (today["slot"] <= s)]
            piv = w.pivot_table(index="sensor_id", columns="slot", values="ci")
            if piv.shape[1] < 6:
                return None
            piv = piv.dropna()
            order = [idx[int(sid)] for sid in piv.index]
            pred, p_high = predict_levels(b, piv.to_numpy(dtype=np.float32), s, dow, sm[order])
            return pd.DataFrame({"sensor_id": piv.index, "pred_code": pred, "p_high": p_high})

        if slot < 5:
            st.info("Collecting the first 60 minutes of the day before predicting (the model uses the last 6 ticks).")
            return
        now = predict_at(slot)
        c1, c2 = st.columns([2, 3])
        c1.plotly_chart(level_bar(now["pred_code"], f"Predicted levels at {fmt_time(min(slot + H, 143))}"), width="stretch")
        cur = today[today["slot"] == slot].set_index("sensor_id")["level_code"]
        top = now.sort_values("p_high", ascending=False).head(TOP_N).copy()
        top["current_level"] = top["sensor_id"].map(cur).map(dict(enumerate(LEVELS)))
        top["predicted_level"] = top["pred_code"].map(dict(enumerate(LEVELS)))
        top["P(HIGH or SEVERE)"] = top["p_high"].round(3)
        c2.markdown("**Sensors most likely to be congested soon**")
        c2.dataframe(top[["sensor_id", "current_level", "predicted_level", "P(HIGH or SEVERE)"]], hide_index=True, width="stretch")
        # live check: predictions made H ticks ago versus what actually happened now
        if slot - H >= 5:
            past = predict_at(slot - H)
            if past is not None:
                merged = past.merge(cur.rename("actual").reset_index(), on="sensor_id")
                acc = float((merged["pred_code"] == merged["actual"]).mean())
                st.success(f"Live check: predictions made {H * 10} minutes ago were correct for {acc * 100:.1f}% of {len(merged)} sensors "
                           f"(HIGH+ actual now: {int((merged['actual'] >= 2).sum())}, predicted: {int((merged['pred_code'] >= 2).sum())}).")
    live_prediction()

    st.subheader("Model evaluation on held-out (chronologically later) days")
    if m is None:
        missing("python -m prediction.train_model")
    else:
        rows = []
        for name, key in [("Random Forest", "random_forest"), ("Persistence baseline", "baseline_persistence"), ("Always LOW", "baseline_always_low")]:
            r = m[key]
            rows.append({"model": name, "accuracy": r["accuracy"], "macro precision": r["macro"]["precision"], "macro recall": r["macro"]["recall"],
                         "macro F1": r["macro"]["f1"], "HIGH+ F1": r["high_or_severe_binary"]["f1"]})
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
        a, bcol = st.columns(2)
        cm = np.array(m["random_forest"]["confusion_matrix"])
        fig = px.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), x=LEVELS, y=LEVELS, text_auto=".2f", color_continuous_scale="Blues",
                        labels=dict(x="predicted", y="actual", color="row share"), title="Confusion matrix (row-normalised)")
        fig.update_layout(height=380, margin=dict(t=50, b=10))
        a.plotly_chart(fig, width="stretch")
        imp = pd.Series(m["feature_importance"]).sort_values()
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "importance", "y": ""}, title="Feature importance")
        fig.update_layout(height=380, margin=dict(t=50, b=10))
        bcol.plotly_chart(fig, width="stretch")
        st.caption(f"Train days {m['train_days']}, test days {m['test_days']}; horizon {m['horizon_minutes']} min. "
                   "Class counts (test): " + ", ".join(f"{k} {v}" for k, v in m["test_class_counts"].items()))
        tune = static_json("tuning_validation.json")
        if tune:
            with st.expander("Model selection on a validation split of the training days (class weighting decision)"):
                st.dataframe(pd.DataFrame(tune["candidates"]).T.round(4), width="stretch")

# ------------------------------------------------------------------ 7. statistics
with tabs[6]:
    t = static_json("t_test_results.json")
    if t is None:
        missing("python -m stats.t_test")
    else:
        p = t["primary_heldout_test_days"]
        st.markdown(f"**H0:** {t['H0']}  \n**H1:** {t['H1']}  \n**α = {t['alpha']}**")
        c = st.columns(4)
        c[0].metric("Peak mean occupancy", f"{p['peak_mean_occupancy']:.4f}")
        c[1].metric("Off-peak mean occupancy", f"{p['offpeak_mean_occupancy']:.4f}")
        c[2].metric("t-statistic (paired)", f"{p['paired_t_statistic']:.2f}", f"df = {p['paired_df']}", delta_color="off")
        c[3].metric("p-value", f"{p['paired_p_value']:.2e}")
        verdict = "rejected" if p["reject_H0_paired"] else "not rejected"
        st.success(f"H0 is **{verdict}** at α = {t['alpha']}: over {p['n_days']} held-out weekdays, occupancy in peak periods differs from off-peak by "
                   f"{p['mean_difference']:.4f} (95% CI {p['difference_ci95'][0]:.4f} to {p['difference_ci95'][1]:.4f}); Cohen's d = {p['cohens_d_paired']:.1f}.")
        st.dataframe(pd.DataFrame({"primary: held-out days (peaks from train days)": p, "comparison: in-sample, all days": t["comparison_in_sample_all_days"]})
                     .drop(index=["alpha"], errors="ignore").astype(str), width="stretch")
        w = t["weekday_vs_weekend"]
        st.markdown(f"**Weekday vs weekend** (Welch t-test on daily mean occupancy): weekday {w['weekday_mean']:.4f} vs weekend {w['weekend_mean']:.4f}, "
                    f"t = {w['welch_t_statistic']:.2f}, p = {w['welch_p_value']:.2e}.")
        st.caption("Unit of analysis = one day (neighbouring 10-minute values are autocorrelated). Peaks are defined on training days and tested on later days to avoid circular reasoning.")
        if (FIGS / "t_test_peak_vs_offpeak.png").exists():
            st.image(str(FIGS / "t_test_peak_vs_offpeak.png"))

# ------------------------------------------------------------------ 8. performance
with tabs[7]:
    bm, meta = static_csv("benchmark_results.csv"), static_json("benchmark_meta.json")
    if bm is None:
        missing("python -m performance.benchmark")
    else:
        env = meta["environment"]
        st.caption(f"Measured on: {env['cpu_count']} CPU core(s), {env.get('ram_gb')} GB RAM, Python {env['python']}, PySpark {env['pyspark']}, pandas {env['pandas']}. "
                   f"Median of {meta['repeats']} runs. Spark session startup ({meta['spark_session_startup_seconds']} s) and first-job warm-up ({meta['spark_first_job_seconds']} s) are excluded.")
        show = bm[["days", "records", "pandas_seconds", "spark_seconds", "speedup", "pandas_records_per_s", "spark_records_per_s", "max_abs_result_difference"]]
        st.dataframe(show.round(3), hide_index=True, width="stretch")
        a, bcol = st.columns(2)
        fig = go.Figure([go.Scatter(x=bm["records"], y=bm["pandas_seconds"], name="pandas (serial)", mode="lines+markers"),
                         go.Scatter(x=bm["records"], y=bm["spark_seconds"], name="Spark", mode="lines+markers")])
        fig.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="records", yaxis_title="seconds", title="Execution time", height=340, margin=dict(t=50, b=10))
        a.plotly_chart(fig, width="stretch")
        fig = px.line(bm, x="records", y="speedup", markers=True, log_x=True, title="Speedup = serial / Spark (below 1: Spark slower)")
        fig.add_hline(y=1, line_dash="dash")
        fig.update_layout(height=340, margin=dict(t=50, b=10))
        bcol.plotly_chart(fig, width="stretch")
        wins = bm[bm["speedup"] >= 1]
        if len(wins):
            st.success(f"Spark first overtakes the serial baseline at {int(wins['records'].iloc[0]):,} records on this machine.")
        else:
            st.warning("On this machine Spark did NOT overtake the serial baseline for any tested size (JVM startup/scheduling overhead; "
                       f"the serial job is a vectorised pandas pass over only {int(bm['records'].max()):,} records at most). "
                       "Run the benchmark on a multi-core machine to see scaling.")
    sc = static_csv("benchmark_scaling.csv")
    if sc is not None and len(sc):
        st.subheader("Spark scalability (local[k])")
        st.dataframe(sc.round(3), hide_index=True, width="stretch")
    st.subheader("Kafka and Spark streaming throughput")
    ps, pm = static_json("producer_stats_maxspeed.json"), STREAM / "stream_metrics.csv"
    cols = st.columns(2)
    if ps:
        cols[0].metric("Kafka producer max throughput", f"{ps['messages_per_second_send_only']:,.0f} msg/s")
    else:
        cols[0].info("Producer throughput not measured yet: `python -m ingestion.producer --max-speed --num-days 5`")
    if pm.exists():
        sm = pd.read_csv(pm)
        cols[1].metric("Spark streaming (mean over micro-batches)", f"{sm['records_per_second'].mean():,.0f} rec/s")
        lat = sm["avg_end_to_end_latency_s"].dropna()
        if len(lat):
            st.caption(f"Mean producer→Spark latency: {lat.mean():.2f} s over {len(sm)} micro-batches.")
    else:
        cols[1].info("Run the streaming job to collect micro-batch metrics.")
