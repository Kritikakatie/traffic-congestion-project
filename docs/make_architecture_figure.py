"""Draws the architecture diagram used in the report -> output/results/figures/architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common.config import get_path, load_config


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#16252d", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#16252d"))


def main():
    cfg = load_config()
    f, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 5.6); ax.axis("off")
    data, kafka, spark, out, ml = "#dbe9f4", "#fde3c8", "#e1f0dc", "#f4dfe3", "#ece3f4"
    box(ax, .1, 4.3, 2.1, .9, "Real PEMS-SF dataset\n(UCI / Caltrans PeMS)\n963 sensors, occupancy", data)
    box(ax, .1, 3.0, 2.1, .9, "Preprocessing\nchronology, de-duplication,\nvalidation", data)
    box(ax, 2.8, 3.0, 2.0, .9, "Kafka producer\nreplays real records\n(963 msgs / 10-min tick)", kafka)
    box(ax, 2.8, 4.3, 2.0, .9, "Kafka topic\ntraffic_data\n3 partitions\nkey = sensor id", kafka)
    box(ax, 5.4, 3.65, 2.3, 1.55, "Spark Structured Streaming\nparse JSON, congestion index,\n30-min event-time windows,\nwatermark, checkpoint", spark)
    box(ax, 8.3, 4.5, 2.5, .7, "Alerts\n(de-duplicated)", out)
    box(ax, 8.3, 3.7, 2.5, .7, "Tick summary +\nwindow aggregates", out)
    box(ax, 8.3, 2.9, 2.5, .7, "Streamlit dashboard", out)
    box(ax, 5.4, 1.9, 2.3, .9, "Historical analysis\npeak periods, ranking,\nt-test (paired)", ml)
    box(ax, 5.4, .6, 2.3, .9, "Random Forest\nprediction (+30 min)", ml)
    box(ax, 2.8, 1.25, 2.0, 1.0, "Benchmark\npandas (serial) vs Spark\ntime and speedup", data)
    arrow(ax, 1.15, 4.3, 1.15, 3.9); arrow(ax, 2.2, 3.45, 2.8, 3.45); arrow(ax, 3.8, 3.9, 3.8, 4.3)
    arrow(ax, 4.8, 4.75, 5.4, 4.5); arrow(ax, 7.7, 4.6, 8.3, 4.8); arrow(ax, 7.7, 4.3, 8.3, 4.05); arrow(ax, 9.55, 3.7, 9.55, 3.6)
    arrow(ax, 6.55, 1.9, 6.55, 1.5); arrow(ax, 7.7, 1.2, 8.9, 2.9); arrow(ax, 7.7, 2.4, 8.5, 2.9)
    arrow(ax, 1.15, 3.0, 2.8, 1.9)
    ax.text(5.5, 5.45, "Real-Time Traffic Congestion Detection and Prediction - architecture", ha="center", fontsize=11, weight="bold")
    out_path = get_path(cfg, "figures_dir", True) / "architecture.png"
    f.savefig(out_path, dpi=160, bbox_inches="tight"); print("wrote", out_path)


if __name__ == "__main__":
    main()
