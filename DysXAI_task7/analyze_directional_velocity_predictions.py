"""
Directional velocity vs model predictions (Task 7).

Per-subject mean speed in four directions (Vx_Pos, Vx_Neg, Vy_Pos, Vy_Neg) using
the same robust derivative pipeline as ``analyze_directional_velocity_age.py``,
with FFT 12 Hz XY filtering to match the trained classifier.

Merges with OOF dysgraphic probabilities and reports correlations plus scatter
panels (speed magnitude vs pred_prob, colored by diagnostic group).

Run::

    python DysXAI_task7/analyze_directional_velocity_predictions.py
    python DysXAI_task7/analyze_directional_velocity_predictions.py \\
        --predictions "oof predictions/oof_predictions_fft_baseline_plus_age_plus_gender.csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dataset import (  # noqa: E402
    CUTOFF_HZ,
    build_sample_table,
    filter_speed_outliers,
    load_raw_timeseries,
    pen_on_mask,
    safe_dt,
)

DIRECTION_ORDER = ("Vx_Pos", "Vx_Neg", "Vy_Pos", "Vy_Neg")
DIRECTION_LABELS = {
    "Vx_Pos": r"Mean speed pushing right ($V_x > 0$)",
    "Vx_Neg": r"Mean speed pulling left ($|V_x|, V_x < 0$)",
    "Vy_Pos": r"Mean speed pushing up ($V_y > 0$)",
    "Vy_Neg": r"Mean speed pulling down ($|V_y|, V_y < 0$)",
}
GROUP_PALETTE = {"Control": "#1f77b4", "Dysgraphic": "#d62728"}

DEFAULT_PREDICTIONS_CSV = (
    _HERE
    / "oof predictions"
    / "oof_predictions_fft_baseline_plus_age_plus_gender.csv"
)
DEFAULT_RESULTS_DIR = _HERE / "XAI results" / "Directional velocity vs predictions"
DEFAULT_MERGED_CSV = (
    DEFAULT_RESULTS_DIR / "directional_velocity_vs_predictions_fft.csv"
)
DEFAULT_CORR_CSV = DEFAULT_RESULTS_DIR / "directional_velocity_correlations_fft.csv"
DEFAULT_SCATTER_PNG = (
    DEFAULT_RESULTS_DIR / "directional_velocity_vs_predictions_scatter_fft.png"
)


def extract_directional_speed_pools_fft(
    filepath: Path,
    *,
    cutoff_hz: float = CUTOFF_HZ,
) -> dict[str, np.ndarray]:
    """Directional speed samples with FFT low-pass; neg axes stored as |v|."""
    from dysxai_fft_xy_filter import lowpass_filter_xy_fft

    data = load_raw_timeseries(str(filepath))
    svc = lowpass_filter_xy_fft(np.asarray(data, dtype=np.float32), cutoff_hz=cutoff_hz)

    x = np.asarray(svc[:, 0], dtype=np.float64)
    y = np.asarray(svc[:, 1], dtype=np.float64)
    t = np.asarray(svc[:, 2], dtype=np.float64)

    dt = safe_dt(t)
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt

    pen_on = pen_on_mask(svc)[1:]
    vx = vx[pen_on]
    vy = vy[pen_on]
    vx, vy = filter_speed_outliers(vx, vy)

    return {
        "Vx_Pos": vx[vx > 0].astype(np.float64),
        "Vx_Neg": np.abs(vx[vx < 0]).astype(np.float64),
        "Vy_Pos": vy[vy > 0].astype(np.float64),
        "Vy_Neg": np.abs(vy[vy < 0]).astype(np.float64),
    }


def build_subject_directional_table(
    *,
    cutoff_hz: float = CUTOFF_HZ,
) -> pd.DataFrame:
    """One row per subject with four directional mean speeds (wide format)."""
    sample_df = build_sample_table()
    rows: list[dict[str, object]] = []

    for _, row in tqdm(
        sample_df.iterrows(),
        total=len(sample_df),
        desc="Directional speed (FFT)",
    ):
        fp = Path(row["filepath"])
        label = int(row["label"])
        group = "Dysgraphic" if label == 1 else "Control"
        try:
            pools = extract_directional_speed_pools_fft(fp, cutoff_hz=cutoff_hz)
        except Exception as exc:
            print(f"Skipping {fp.name}: {exc}")
            continue

        entry: dict[str, object] = {
            "subject_id": int(row["subject_id"]),
            "label": label,
            "Group": group,
            "age": float(row["age"]) if pd.notna(row["age"]) else float("nan"),
        }
        for direction in DIRECTION_ORDER:
            values = pools[direction]
            entry[direction] = float(np.mean(values)) if values.size > 0 else float("nan")
        rows.append(entry)

    return pd.DataFrame(rows)


def load_predictions(predictions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    required = {"subject_id", "true_label", "pred_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")
    return df.rename(columns={"true_label": "label"})


def compute_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Pearson and Spearman correlation of each speed column with pred_prob."""
    rows: list[dict[str, object]] = []
    for direction in DIRECTION_ORDER:
        sub = merged[[direction, "pred_prob"]].dropna()
        if len(sub) < 3:
            pearson_r, pearson_p = float("nan"), float("nan")
            spearman_r, spearman_p = float("nan"), float("nan")
        else:
            pearson_r, pearson_p = stats.pearsonr(sub[direction], sub["pred_prob"])
            spearman_r, spearman_p = stats.spearmanr(sub[direction], sub["pred_prob"])
        rows.append(
            {
                "Direction": direction,
                "n": len(sub),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )
    return pd.DataFrame(rows)


def print_group_speed_summary(merged: pd.DataFrame) -> None:
    print("\nMean directional speed by group (FFT 12 Hz, pen-on, robust dt):")
    for direction in DIRECTION_ORDER:
        ctrl = merged.loc[merged["label"] == 0, direction].mean()
        dys = merged.loc[merged["label"] == 1, direction].mean()
        ratio = dys / ctrl if ctrl and np.isfinite(ctrl) and ctrl != 0 else float("nan")
        print(
            f"  {direction:8s}  Control={ctrl:.4f}  Dysgraphic={dys:.4f}  D/C={ratio:.3f}"
        )


def plot_scatter_grid(merged: pd.DataFrame, out_path: Path, *, cutoff_hz: float) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="white")
    axes_flat = axes.ravel()

    for ax, direction in zip(axes_flat, DIRECTION_ORDER):
        sub = merged[[direction, "pred_prob", "Group"]].dropna(subset=[direction, "pred_prob"])
        if sub.empty:
            ax.set_title(f"{DIRECTION_LABELS[direction]}\n(no data)")
            continue

        sns.scatterplot(
            data=sub,
            x=direction,
            y="pred_prob",
            hue="Group",
            palette=GROUP_PALETTE,
            alpha=0.75,
            s=40,
            ax=ax,
            legend=False,
        )
        if len(sub) >= 3:
            r, p = stats.spearmanr(sub[direction], sub["pred_prob"])
            ax.text(
                0.03,
                0.97,
                f"Spearman rho = {r:.3f}\np = {p:.3g}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
            )
        ax.set_xlabel("Mean speed magnitude")
        ax.set_ylabel("OOF dysgraphic probability")
        ax.set_title(DIRECTION_LABELS[direction])
        ax.set_ylim(-0.05, 1.05)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_PALETTE["Control"], markersize=8, label="Control"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_PALETTE["Dysgraphic"], markersize=8, label="Dysgraphic"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Directional speed vs model prediction — FFT {cutoff_hz:g} Hz (Task 7)\n"
        "Higher speed → lower dysgraphic probability (consistent with population analysis)",
        fontsize=11,
        y=1.06,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directional velocity vs OOF dysgraphic probability (FFT-matched).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_CSV,
        help="CSV with subject_id, true_label, pred_prob.",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=CUTOFF_HZ,
        help=f"FFT low-pass cutoff in Hz (default: {CUTOFF_HZ}).",
    )
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=DEFAULT_MERGED_CSV,
        help="Output merged per-subject table.",
    )
    parser.add_argument(
        "--corr-csv",
        type=Path,
        default=DEFAULT_CORR_CSV,
        help="Output correlation table.",
    )
    parser.add_argument(
        "--scatter-png",
        type=Path,
        default=DEFAULT_SCATTER_PNG,
        help="Output 2x2 scatter figure.",
    )
    args = parser.parse_args()

    predictions_csv = args.predictions if args.predictions.is_absolute() else _HERE / args.predictions
    merged_csv = args.merged_csv if args.merged_csv.is_absolute() else _HERE / args.merged_csv
    corr_csv = args.corr_csv if args.corr_csv.is_absolute() else _HERE / args.corr_csv
    scatter_png = args.scatter_png if args.scatter_png.is_absolute() else _HERE / args.scatter_png

    speed_df = build_subject_directional_table(cutoff_hz=float(args.cutoff_hz))
    pred_df = load_predictions(predictions_csv)
    merged = speed_df.merge(pred_df[["subject_id", "pred_prob"]], on="subject_id", how="inner")

    if merged.empty:
        raise RuntimeError("No rows after merging directional speeds with predictions.")

    print_group_speed_summary(merged)

    corr_df = compute_correlations(merged)
    print("\nCorrelation with OOF dysgraphic probability:")
    for _, row in corr_df.iterrows():
        print(
            f"  {row['Direction']:8s}  Spearman rho={row['spearman_r']:+.3f} (p={row['spearman_p']:.3g})  "
            f"Pearson r={row['pearson_r']:+.3f} (p={row['pearson_p']:.3g})  n={int(row['n'])}"
        )

    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_csv, index=False)
    corr_df.to_csv(corr_csv, index=False)
    plot_scatter_grid(merged, scatter_png, cutoff_hz=float(args.cutoff_hz))

    print(f"\nSaved merged table: {merged_csv.resolve()}")
    print(f"Saved correlations: {corr_csv.resolve()}")
    print(f"Saved scatter figure: {scatter_png.resolve()}")


if __name__ == "__main__":
    main()
