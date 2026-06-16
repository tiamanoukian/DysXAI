# # `calculate_age_compensation.py`
#
# Converted from Python for notebook workflow.
#
# **Module docstring**
#
# Task 7 (hračkárstvo): fit Control Age -> mean Velocity_X (slope + intercept), then for each
# subject compute residual Velocity_Deviation = Raw - (slope * Age + intercept).
#
# Reuses analyze_velocity_age.build_subject_velocity_age_df (Task 7 .svc only + metadata).
#
# Requires: numpy, pandas, matplotlib, seaborn, scipy, openpyxl.
#
# Run from project root:
#     python calculate_age_compensation.py
#     python calculate_age_compensation.py --output figures/Age_Compensation_Proof.png

"""
Task 7 (hračkárstvo): fit Control Age -> mean Velocity_X (slope + intercept), then for each
subject compute residual Velocity_Deviation = Raw - (slope * Age + intercept).

Reuses analyze_velocity_age.build_subject_velocity_age_df (Task 7 .svc only + metadata).

Requires: numpy, pandas, matplotlib, seaborn, scipy, openpyxl.

Run from project root:
    python calculate_age_compensation.py
    python calculate_age_compensation.py --output figures/Age_Compensation_Proof.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

FIGURE_SAVE_DPI = 300

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analyze_velocity_age import _task7_dir, build_subject_velocity_age_df

PALETTE = {"Control": "#1f77b4", "Dysgraphic": "#d62728"}

RAW_Y = "Raw_Mean_Velocity_X"
EXPECTED_Y = "Expected_Mean_Velocity_X"
DEV_Y = "Velocity_Deviation"


def build_compensation_table(
    task_dir: str | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """Return analysis DataFrame and Control linregress (slope, intercept) for Age -> raw Vx."""
    base = build_subject_velocity_age_df(task_dir=task_dir)
    df = pd.DataFrame(
        {
            "Subject_ID": base["subject_id"],
            "Age": base["age"],
            "Diagnosis": base["Group"],
            RAW_Y: base["mean_velocity_x"],
        }
    )

    ctrl = df[df["Diagnosis"] == "Control"]
    if len(ctrl) < 2:
        raise RuntimeError(
            f"Need at least 2 Control subjects with age for linear regression; got {len(ctrl)}."
        )

    result = linregress(
        ctrl["Age"].to_numpy(dtype=float),
        ctrl[RAW_Y].to_numpy(dtype=float),
    )
    slope = float(result.slope)
    intercept = float(result.intercept)

    df[EXPECTED_Y] = slope * df["Age"] + intercept
    df[DEV_Y] = df[RAW_Y] - df[EXPECTED_Y]
    return df, slope, intercept


def plot_age_compensation_proof(
    df: pd.DataFrame,
    *,
    out_path: str,
    show: bool = False,
) -> None:
    """1×2 panels: raw Velocity_X vs age; deviation from Control-expected vs age."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    panels: list[tuple[plt.Axes, str, str, str, bool]] = [
        (axes[0], RAW_Y, "Before compensation", "Mean Velocity_X (raw)", False),
        (
            axes[1],
            DEV_Y,
            "Deviation from Control-expected Velocity_X",
            "Velocity deviation (raw − expected)",
            True,
        ),
    ]

    for ax, ycol, title, ylabel, show_zero_line in panels:
        assert ycol in (RAW_Y, DEV_Y)
        for name, color in PALETTE.items():
            sub = df[df["Diagnosis"] == name]
            sns.regplot(
                data=sub,
                x="Age",
                y=ycol,
                ax=ax,
                color=color,
                label=name,
                scatter_kws={"alpha": 0.75, "s": 45},
                line_kws={"linewidth": 2},
            )
        if show_zero_line:
            ax.axhline(0, color="black", linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax.legend(title="Diagnosis", loc="best")

    fig.suptitle(
        "Task 7 (hračkárstvo): mean Velocity_X vs age and residual vs Control baseline",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_SAVE_DPI, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control-based age compensation for Task 7 mean Velocity_X."
    )
    parser.add_argument(
        "--task-dir",
        default=None,
        help=f"Override Task 7 folder (default: {_task7_dir()})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(_PROJECT_ROOT, "Age_Compensation_Proof.png"),
        help="Path to save the figure (PNG).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window after saving.",
    )
    args = parser.parse_args()

    task_dir = args.task_dir or _task7_dir()
    print(f"Task 7 clips directory: {os.path.abspath(task_dir)}")

    df, slope, intercept = build_compensation_table(task_dir=task_dir)
    print(f"Subjects (Task 7 + metadata + age): {len(df)}")
    print(
        f"Control baseline (Age -> {RAW_Y}): slope={slope:.6g}, intercept={intercept:.6g}"
    )

    ctrl = df[df["Diagnosis"] == "Control"]
    slope_dev = linregress(
        ctrl["Age"].to_numpy(dtype=float),
        ctrl[DEV_Y].to_numpy(dtype=float),
    ).slope
    print(
        f"Control slope (Age -> {DEV_Y}, expect ~0): {float(slope_dev):.6g}"
    )

    plot_age_compensation_proof(df, out_path=args.output, show=args.show)
    print(f"Saved figure: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
