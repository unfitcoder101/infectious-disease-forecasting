"""
Figures for the history-window sensitivity study.

Design notes:
  - Windows (1,2,4,8,12) are plotted at EVEN spacing, not on a numeric axis.
    They are five experimental conditions; even spacing keeps 1/2/4 readable
    instead of crowding them against the left edge.
  - Two model series get fixed categorical hues (blue/orange, validated for
    colour-vision deficiency). The persistence baseline is neutral grey and
    dashed, because it is a reference line, not a third competing series.
  - Every chart carries a legend, since two series are present.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")           # no GUI needed; write straight to file
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

SERIES_COLORS = {"XGBoost": "#2a78d6", "LightGBM": "#eb6834"}
BASELINE_COLOR = "#8a8984"
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
RANDOM_SEED_FOR_JITTER = 42     # only affects horizontal scatter jitter, not any metric

METRICS = [
    ("MAE",            "MAE (cases)",              "Mean absolute error vs history window", False),
    ("RMSE",           "RMSE (cases)",             "Root mean squared error vs history window", False),
    ("R2",             "R² (variance explained)",  "R² vs history window", False),
    ("training_time",  "Training time (seconds)",  "Training cost vs history window", True),
    ("inference_time", "Inference time (seconds)", "Inference cost vs history window", True),
]


def _style(ax, title, ylabel, windows, title_fontsize=13):
    ax.set_title(title, fontsize=title_fontsize, color=TEXT_PRIMARY, pad=12, loc="left", fontweight="bold")
    ax.set_xlabel("History window (weeks of past data given to the model)",
                  fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def plot_metric(results, metric, ylabel, title, is_cost, outfile):
    model_rows = results[~results["model"].str.startswith("Persistence")]
    windows = sorted(model_rows["history_window"].unique())
    x = range(len(windows))

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for name, color in SERIES_COLORS.items():
        sub = model_rows[model_rows["model"] == name].sort_values("history_window")
        ax.plot(x, sub[metric], color=color, linewidth=2, marker="o",
                markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                label=name, zorder=3)

    # Accuracy charts get the persistence baseline as a reference line.
    if not is_cost and metric in ("MAE", "RMSE", "R2"):
        base = results[results["model"].str.startswith("Persistence")][metric].iloc[0]
        ax.axhline(base, color=BASELINE_COLOR, linewidth=1.6, linestyle="--", zorder=2,
                   label=f"Persistence baseline ({base:.2f})")

    if is_cost:
        ax.set_ylim(bottom=0)

    _style(ax, title, ylabel, windows)
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_forecast(series, preds_by_window, outfile, model_name):
    """Actual vs predicted on the test period, for the shortest and longest window."""
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    dates, actual = series
    ax.plot(dates, actual, color=TEXT_PRIMARY, linewidth=2, label="Actual cases", zorder=3)
    for (w, preds), color in zip(preds_by_window.items(), ["#2a78d6", "#eb6834"]):
        ax.plot(dates, preds, color=color, linewidth=2, alpha=0.9,
                label=f"{model_name}, {w}-week history", zorder=2)

    ax.set_title(f"Forecasts on the held-out test period — {model_name}",
                 fontsize=13, color=TEXT_PRIMARY, pad=12, loc="left", fontweight="bold")
    ax.set_ylabel("Weekly dengue cases", fontsize=10, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_robustness(results, outfile):
    """
    Small multiples: the same MAE-vs-window curve on two disjoint future
    periods. Both panels share one y-scale, so the reader can see both the
    SHAPE of each curve and the fact that the two periods sit at different
    error levels. (Two panels rather than two y-axes on one chart -- a
    dual-axis chart would let either story be drawn at will.)
    """
    model_rows = results[~results["model"].str.startswith("Persistence")]
    windows = sorted(model_rows["history_window"].unique())
    x = range(len(windows))
    base = results[results["model"].str.startswith("Persistence")].iloc[0]

    panels = [("val_MAE", "Validation period\n(2002-12 → 2005-08)", base["val_MAE"]),
              ("MAE",     "Test period\n(2005-08 → 2008-04)",       base["MAE"])]

    lo = min(model_rows[["MAE", "val_MAE"]].min().min(), base["val_MAE"], base["MAE"])
    hi = max(model_rows[["MAE", "val_MAE"]].max().max(), base["MAE"])
    pad = (hi - lo) * 0.08

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")

    for ax, (col, title, baseval) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for name, color in SERIES_COLORS.items():
            sub = model_rows[model_rows["model"] == name].sort_values("history_window")
            ax.plot(x, sub[col], color=color, linewidth=2, marker="o", markersize=8,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=name, zorder=3)
        ax.axhline(baseval, color=BASELINE_COLOR, linewidth=1.6, linestyle="--",
                   zorder=2, label="Persistence baseline")
        ax.set_ylim(lo - pad, hi + pad)
        _style(ax, title, "MAE (cases)" if col == "val_MAE" else "", windows)

    axes[0].set_xlabel("History window (weeks)", fontsize=10, color=TEXT_SECONDARY)
    axes[1].set_xlabel("History window (weeks)", fontsize=10, color=TEXT_SECONDARY)
    leg = axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    for tx in leg.get_texts():
        tx.set_color(TEXT_SECONDARY)
    fig.suptitle("Robustness: the same comparison on two disjoint future periods",
                 fontsize=13, color=TEXT_PRIMARY, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outfile, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cv_summary(cv_summary, outfile, tolerance_pct=5):
    """
    MAE vs history window, but now each point is a MEAN across CV folds with
    an ERROR BAR of +/-1 standard deviation, instead of a single number.

    This is the figure that actually answers "did CV change the story":
    on the single-split chart, the lines are clean and the 8-12 week jump
    looks dramatic. Here, if the error bars of adjacent windows overlap,
    that jump is not distinguishable from fold-to-fold noise.
    """
    windows = sorted(cv_summary["history_window"].unique())
    x = list(range(len(windows)))

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for name, color in SERIES_COLORS.items():
        sub = cv_summary[cv_summary["model"] == name].sort_values("history_window")
        ax.errorbar(x, sub["MAE_mean"], yerr=sub["MAE_std"], color=color, linewidth=2,
                    marker="o", markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                    capsize=4, elinewidth=1.4, ecolor=color, alpha=0.95,
                    label=name, zorder=3)

    base = cv_summary[cv_summary["model"].str.startswith("Persistence")]
    if len(base):
        base_mean = base["MAE_mean"].mean()
        ax.axhline(base_mean, color=BASELINE_COLOR, linewidth=1.6, linestyle="--", zorder=2,
                   label=f"Persistence baseline (mean {base_mean:.2f})")

    n_folds = int(cv_summary["n_folds"].iloc[0]) if len(cv_summary) else "?"
    _style(ax, f"CV mean MAE vs history window ({n_folds} rolling-origin folds, error bars = ±1 std)",
           "MAE (cases)", windows, title_fontsize=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cv_fold_detail(cv_results, outfile):
    """
    Every individual fold's MAE as a faint point, model mean as a bold line.
    Shows the RAW spread the error-bar chart only summarizes -- useful for
    spotting whether variance comes from a couple of outlier folds (e.g. a
    fold that happens to contain an outbreak peak) or is spread evenly.
    """
    model_rows = cv_results[~cv_results["model"].str.startswith("Persistence")]
    windows = sorted(model_rows["history_window"].unique())
    x_pos = {w: i for i, w in enumerate(windows)}

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    rng = np.random.default_rng(RANDOM_SEED_FOR_JITTER)
    for name, color in SERIES_COLORS.items():
        sub = model_rows[model_rows["model"] == name]
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        xs = [x_pos[w] for w in sub["history_window"]] + jitter
        ax.scatter(xs, sub["MAE"], color=color, alpha=0.35, s=28, zorder=2, linewidths=0)

        means = sub.groupby("history_window")["MAE"].mean().reindex(windows)
        ax.plot(range(len(windows)), means, color=color, linewidth=2.4, marker="o",
                markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.3,
                label=f"{name} (fold mean)", zorder=3)

    _style(ax, "Every CV fold's MAE (faint dots) with fold-mean overlaid", "MAE (cases)", windows)
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_all(results, fig_dir=FIG_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric, ylabel, title, is_cost in METRICS:
        out = fig_dir / f"{metric.lower()}_vs_history_window.png"
        plot_metric(results, metric, ylabel, title, is_cost, out)
        written.append(out)
    return written
