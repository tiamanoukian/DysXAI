# # `analyze_true_duration.py`
#
# Converted from Python for notebook workflow.
#
# **Module docstring**
#
# Investigate whether the Time channel leaks sequence length via padding.
#
# Computes true writing duration from raw timestamps (last - first) per recording,
# aggregates total duration per subject, and plots distributions, group tests, and
# age confounding. Requires: pandas, numpy, matplotlib, seaborn, scipy, openpyxl.
#
# Run from project root:
#     python analyze_true_duration.py
#
# In Jupyter: ``%matplotlib inline`` then ``from analyze_true_duration import ...`` and call
# ``plot_histogram_kde(df, show=True)`` (and pass ``out_path=...`` if you also want files saved).

"""
Investigate whether the Time channel leaks sequence length via padding.

Computes true writing duration from raw timestamps (last - first) per recording,
aggregates total duration per subject, and plots distributions, group tests, and
age confounding. Requires: pandas, numpy, matplotlib, seaborn, scipy, openpyxl.

Run from project root:
    python analyze_true_duration.py

In Jupyter: ``%matplotlib inline`` then ``from analyze_true_duration import ...`` and call
``plot_histogram_kde(df, show=True)`` (and pass ``out_path=...`` if you also want files saved).
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate
from scipy.stats import gaussian_kde, mannwhitneyu, ttest_ind

# Project imports
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dysxai_init import Config, discover_files_and_map_subjects, load_metadata, load_raw_timeseries

# Raw channel order: x, y, t, pressure, azimuth, altitude, pen_status (see 00_initialization)
TIME_COL_IDX = 2


def resolve_raw_filepath(data_root: str, file_name) -> str:
    """Match HandwritingDataset / notebook file resolution."""
    file_name = str(file_name)
    candidates = [
        os.path.join(data_root, f"{file_name}.svc"),
        os.path.join(data_root, f"{file_name}.txt"),
        os.path.join(data_root, file_name),
    ]
    for cp in candidates:
        if os.path.exists(cp):
            return cp
    raise FileNotFoundError(f"Could not find raw file for sample '{file_name}'")


def true_duration_one_recording(filepath: str, t_idx: int = TIME_COL_IDX) -> float:
    """
    Total writing time for one raw recording: last_timestamp - first_timestamp.
    Uses pre-pad/pre-truncate data from load_raw_timeseries (same as training pipeline).
    """
    ts = load_raw_timeseries(filepath)
    if ts.shape[0] < 2:
        return 0.0
    if t_idx >= ts.shape[1]:
        raise ValueError(f"Time column index {t_idx} out of bounds for shape {ts.shape}")
    t = ts[:, t_idx].astype(np.float64)
    return float(t[-1] - t[0])


def load_subject_age_label(meta_xlsx: str) -> pd.DataFrame:
    """Full demographics from Excel (ID, diag, age, ...)."""
    df = pd.read_excel(meta_xlsx)
    out = pd.DataFrame(
        {
            "Subject_ID": df["ID"].astype(int),
            "Label": np.where(
                df["diag"].astype(str).str.upper() == "DYSGR",
                "Dysgraphic",
                "Control",
            ),
            "Age": pd.to_numeric(df["age"], errors="coerce"),
        }
    )
    return out


def build_duration_table(
    data_root: str,
    meta_xlsx: str,
    t_idx: int = TIME_COL_IDX,
) -> pd.DataFrame:
    """
    One row per subject: total True_Duration = sum over all recordings for that subject.
    """
    # discover_files_and_map_subjects expects columns subject_id, label (same as dysxai_init.run_init)
    subject_meta = load_metadata(meta_xlsx)[["subject_id", "label"]].copy()
    base = discover_files_and_map_subjects(data_root, subject_meta, verbose=False)

    rows = []
    for _, row in base.iterrows():
        fp = resolve_raw_filepath(data_root, row["file_name"])
        dur = true_duration_one_recording(fp, t_idx=t_idx)
        rows.append({"subject_id": row["subject_id"], "file_duration": dur})

    per_file = pd.DataFrame(rows)
    agg = per_file.groupby("subject_id", as_index=False)["file_duration"].sum()
    agg = agg.rename(columns={"file_duration": "True_Duration", "subject_id": "Subject_ID"})

    demo = load_subject_age_label(meta_xlsx)
    out = agg.merge(demo, on="Subject_ID", how="left")
    return out


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Hedges' g / Cohen's d for independent samples (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled


def kde_overlap_coefficient(x: np.ndarray, y: np.ndarray, n_grid: int = 512) -> float:
    """
    Overlap coefficient OVL = integral min(f(x), g(x)) dx for KDE estimates of two samples.
    Returns value in [0, 1]; higher means more distributional overlap.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    kde_x = gaussian_kde(x)
    kde_y = gaussian_kde(y)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    fx = kde_x(grid)
    fy = kde_y(grid)
    m = np.minimum(fx, fy)
    return float(integrate.simpson(m, x=grid))


def plot_histogram_kde(df: pd.DataFrame, out_path: str | None = None, show: bool = False) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    palette = {"Control": "#2171b5", "Dysgraphic": "#cb181d"}

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    plot_df = df.copy()
    plot_df["Label"] = pd.Categorical(plot_df["Label"], categories=["Control", "Dysgraphic"], ordered=True)

    sns.histplot(
        data=plot_df,
        x="True_Duration",
        hue="Label",
        stat="density",
        bins=25,
        alpha=0.4,
        palette=palette,
        ax=ax,
        element="step",
        common_norm=False,
    )
    for label in ["Control", "Dysgraphic"]:
        sub = plot_df[plot_df["Label"] == label]["True_Duration"]
        sns.kdeplot(sub, color=palette[label], linewidth=2.2, ax=ax, warn_singular=False)

    ax.set_title("True total writing duration per subject (raw timestamps, pre-padding)")
    ax.set_xlabel("True duration (time units as in raw files, typically ms)")
    ax.set_ylabel("Density")
    leg_handles = [Patch(facecolor=palette[k], edgecolor="white", alpha=0.4, label=k) for k in ["Control", "Dysgraphic"]]
    ax.legend(handles=leg_handles, title="Group", frameon=True)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved figure: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_group_stats(df: pd.DataFrame) -> None:
    ctrl = df[df["Label"] == "Control"]["True_Duration"].values
    dys = df[df["Label"] == "Dysgraphic"]["True_Duration"].values

    print("\n--- Descriptive statistics ---")
    print(df.groupby("Label")["True_Duration"].describe())

    print("\n--- Parametric: Welch's t-test (unequal variances) ---")
    tt = ttest_ind(ctrl, dys, equal_var=False, nan_policy="omit")
    print(f"  t-statistic = {tt.statistic:.4f}, p-value = {tt.pvalue:.6g}")

    print("\n--- Nonparametric: Mann-Whitney U ---")
    mw = mannwhitneyu(ctrl, dys, alternative="two-sided")
    print(f"  U = {mw.statistic:.4f}, p-value = {mw.pvalue:.6g}")

    d = cohens_d(dys, ctrl)
    print("\n--- Effect size (Cohen's d; Dysgraphic vs Control) ---")
    print(f"  Cohen's d = {d:.4f}  (positive => Dysgraphic longer on average)")

    ovl = kde_overlap_coefficient(ctrl, dys)
    print("\n--- Distributional overlap (KDE overlap coefficient OVL) ---")
    print(f"  OVL = integral min(f,g) dx ~ {ovl:.4f}  (1 = identical, 0 = no overlap)")


def plot_age_scatter(df: pd.DataFrame, out_path: str | None = None, show: bool = False) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    palette = {"Control": "#2171b5", "Dysgraphic": "#cb181d"}

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for label in ["Control", "Dysgraphic"]:
        g = df[df["Label"] == label].dropna(subset=["Age", "True_Duration"])
        ax.scatter(
            g["Age"],
            g["True_Duration"],
            label=label,
            color=palette[label],
            alpha=0.75,
            s=55,
            edgecolor="white",
            linewidth=0.5,
        )
        if len(g) >= 2:
            z = np.polyfit(g["Age"].values, g["True_Duration"].values, 1)
            xp = np.linspace(g["Age"].min(), g["Age"].max(), 100)
            ax.plot(xp, np.poly1d(z)(xp), color=palette[label], linewidth=2.2, linestyle="--")

    ax.set_title("Age vs true total duration (linear fit per group)")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("True duration (raw time units)")
    ax.legend(title="Group")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved figure: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    # Full-session tree (not task clips) — duration analysis needs raw timestamps across the whole hw file.
    df = build_duration_table(Config.RAW_DATASCI_ROOT, Config.META_XLSX)
    print("Subject-level duration table (head):")
    print(df.head(10).to_string(index=False))
    print(f"\nN subjects: {len(df)}")

    csv_path = os.path.join(_PROJECT_ROOT, "true_duration_per_subject.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    print_group_stats(df)

    fig_dir = _PROJECT_ROOT
    plot_histogram_kde(df, out_path=os.path.join(fig_dir, "true_duration_hist_kde.png"))
    plot_age_scatter(df, out_path=os.path.join(fig_dir, "true_duration_age_scatter.png"))

if __name__ == "__main__":
    main()
