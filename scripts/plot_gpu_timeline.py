"""Plot actual hardware telemetry and pipeline progress over the sampled interval."""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager


def main():
    font = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    chinese = font.exists()
    if chinese:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font)
        ).get_name()

    def label(zh, en):
        return zh if chinese else en

    with Path("logs/local/gpu_samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError("At least two telemetry samples are required")
    zone = ZoneInfo("Asia/Shanghai")
    ts = []
    series = {k: [] for k in ("gpu_percent", "used_mib", "layout", "ocr", "reviews")}
    previous = None
    for row in rows:
        moment = datetime.fromtimestamp(float(row["epoch"]), zone)
        if previous is not None and (moment - previous).total_seconds() > 15:
            ts.append(previous + timedelta(seconds=1))
            for values in series.values():
                values.append(float("nan"))
        ts.append(moment)
        for key, values in series.items():
            values.append(float(row[key]))
        previous = moment
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 6.6),
        sharex=True,
        layout="constrained",
        gridspec_kw=dict(height_ratios=[1.25, 1, 1]),
    )
    fig.suptitle(
        label(
            "本机 RTX 4070 实际负载与流水线进度", "Local RTX 4070 production telemetry"
        ),
        fontsize=15,
        fontweight="bold",
    )
    colors = ["#135f9c", "#a55a00", "#007d69"]
    for key, target, name, color in zip(
        ("layout", "ocr", "reviews"),
        (13247, 1000, 1000),
        (label("版面", "Layout"), "OCR", label("视觉审核", "Visual review")),
        colors,
    ):
        axes[0].plot(
            ts, [100 * v / target for v in series[key]], color=color, label=name, lw=1.8
        )
    axes[0].set(ylabel=label("目标完成率 (%)", "Completion (%)"), ylim=(0, 106))
    axes[0].legend(ncol=3, loc="lower right", frameon=False)
    axes[1].plot(ts, series["gpu_percent"], color=colors[0], lw=1.1)
    axes[1].set(ylabel=label("GPU 利用率 (%)", "GPU utilization (%)"), ylim=(0, 105))
    axes[2].plot(ts, [v / 1024 for v in series["used_mib"]], color=colors[2], lw=1.2)
    axes[2].set(
        ylabel=label("已用显存 (GiB)", "VRAM used (GiB)"),
        ylim=(0, 12),
        xlabel=label("北京时间", "Beijing time"),
    )
    complete = next((t for t, v in zip(ts, series["layout"]) if v >= 13247), None)
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        if complete is not None:
            ax.axvline(complete, color="#505968", ls="--", lw=0.9)
    if complete is not None:
        axes[1].annotate(
            label("版面完成", "Layout complete"),
            xy=(complete, 5),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=zone))
    fig.savefig("reports/performance/local_gpu_timeline.png", dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    main()
