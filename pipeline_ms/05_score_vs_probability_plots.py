"""
Session-level scatter plots of clinical severity score vs. predicted P(severe),
with a LOESS trend line, for every results_ms/* prediction file.

Every transcript/session is plotted as its own point (no participant-level
averaging); participants with multiple sessions contribute one point per session.

DAIC uses PHQ-8; PDCH and all EPI evaluations (01, 02, 04_*) use HAM-D.
03_daic_pdch_epi_2class is excluded: it is a separately trained combined model,
so its per-source predictions are not the same as the individual DAIC/PDCH models.
"""
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.1,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

RESULTS_ROOT = "/zi/home/luke.bregulla/Desktop/DSS/results_ms"

# folder -> (prediction_file, scale)
FOLDER_CONFIG = {
    "01_zero_shot_epi": ("subject_predictions.csv", "HAM-D"),
    "02_epi": ("subject_predictions.csv", "HAM-D"),
    "03_daic": ("subject_predictions.csv", "PHQ-8"),
    "03_pdch": ("subject_predictions.csv", "HAM-D"),
    "04_daic": ("subject_predictions.csv", "HAM-D"),
    "04_pdch": ("subject_predictions.csv", "HAM-D"),
    "04_daic_pdch": ("subject_predictions.csv", "HAM-D"),
}


def compute_axis_limits(values: np.ndarray, pad_frac: float = 0.08, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0

    lo = float(np.min(finite))
    hi = float(np.max(finite))

    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.1, 0.05)
        lo -= pad
        hi += pad
    else:
        span = hi - lo
        pad = span * pad_frac
        lo -= pad
        hi += pad

    if lower_bound is not None:
        lo = min(lo, float(lower_bound))
    if upper_bound is not None:
        hi = max(hi, float(upper_bound))

    return lo, hi


def _render_scatter(
    x: np.ndarray,
    y: np.ndarray,
    scale_name: str,
    folder_name: str,
    plot_path: str,
    stat_name: str,
    stat_symbol: str,
    stat_value: float,
    stat_p: float,
    trend: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    ax.scatter(
        x,
        y,
        s=42,
        alpha=0.75,
        color="#2c7fb8",
        edgecolor="white",
        linewidth=0.5,
        label="Session",
        zorder=2,
    )

    if trend == "lowess":
        order = np.argsort(x)
        smoothed = lowess(y[order], x[order], frac=0.66, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="#d62728", linewidth=2.4, label="LOESS trend", zorder=3)
    else:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(line_x, slope * line_x + intercept, color="#d62728", linewidth=2.4, label="Linear fit", zorder=3)

    x_min, x_max = compute_axis_limits(x)
    y_min, y_max = compute_axis_limits(y)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    value_text = "nan" if not np.isfinite(stat_value) else f"{stat_value:.3f}"
    p_text = "nan" if not np.isfinite(stat_p) else f"{stat_p:.4f}"

    ax.set_xlabel(f"{scale_name} score", labelpad=8)
    ax.set_ylabel("Predicted P(severe)", labelpad=8)
    ax.set_title(f"{folder_name}: {scale_name} vs. P(severe)\n{stat_name} correlation", pad=16)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linewidth=0.8, alpha=0.2)

    ax.annotate(
        f"{stat_name} {stat_symbol} = {value_text}\np = {p_text}\nn = {len(x)}",
        xy=(x_min + 0.02 * (x_max - x_min), y_max - 0.06 * (y_max - y_min)),
        xycoords="data",
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cfcfcf"},
        zorder=5,
    )

    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_score_vs_probability(df: pd.DataFrame, scale_name: str, folder_name: str, output_dir: str, suffix: str = "") -> None:
    x = df["continuous_score"].to_numpy(dtype=float)
    y = df["p_severe"].to_numpy(dtype=float)

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if x.size == 0 or y.size == 0:
        raise ValueError(f"[{folder_name}] no valid score/probability pairs available for plotting")

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        rho, p_value = np.nan, np.nan
        pearson_r, pearson_p = np.nan, np.nan
    else:
        rho, p_value = spearmanr(x, y)
        pearson_r, pearson_p = pearsonr(x, y)

    os.makedirs(output_dir, exist_ok=True)
    scale_tag = scale_name.lower().replace("-", "").replace(" ", "_")

    spearman_path = os.path.join(output_dir, f"{scale_tag}_vs_p_severe{suffix}_spearman.png")
    pearson_path = os.path.join(output_dir, f"{scale_tag}_vs_p_severe{suffix}_pearson.png")
    _render_scatter(x, y, scale_name, folder_name, spearman_path, "Spearman", "ρ", rho, p_value, trend="lowess")
    _render_scatter(x, y, scale_name, folder_name, pearson_path, "Pearson", "r", pearson_r, pearson_p, trend="linear")

    csv_path = os.path.join(output_dir, f"{scale_tag}_vs_p_severe{suffix}_session_level.csv")
    out = df[["id", "participant", "continuous_score", "p_severe"]].rename(columns={"continuous_score": f"{scale_tag}_score"}).copy()
    out["spearman_r"] = rho
    out["spearman_p_value"] = p_value
    out["pearson_r"] = pearson_r
    out["pearson_p_value"] = pearson_p
    out.to_csv(csv_path, index=False)

    if np.isfinite(rho) and np.isfinite(p_value):
        print(f"[{folder_name}] {scale_name}: n={len(x)} sessions, Spearman r={rho:.3f} (p={p_value:.4f}), Pearson r={pearson_r:.3f} (p={pearson_p:.4f})")
    else:
        print(f"[{folder_name}] {scale_name}: n={len(x)} sessions, Spearman r=nan, Pearson r=nan (insufficient variation)")
    print(f"  Saved: {spearman_path}")
    print(f"  Saved: {pearson_path}")
    print(f"  Saved: {csv_path}")


def process_folder(folder_name: str, file_name: str, scale: str) -> None:
    folder_path = os.path.join(RESULTS_ROOT, folder_name)
    csv_path = os.path.join(folder_path, file_name)
    if not os.path.isfile(csv_path):
        print(f"[{folder_name}] Skipped: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    output_dir = os.path.join(folder_path, "score_vs_probability")
    plot_score_vs_probability(df, scale, folder_name, output_dir)


if __name__ == "__main__":
    for folder_name, (file_name, scale) in FOLDER_CONFIG.items():
        process_folder(folder_name, file_name, scale)
