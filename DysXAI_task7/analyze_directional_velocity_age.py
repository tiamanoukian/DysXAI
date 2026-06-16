"""
Population-level directional velocity vs age (Task 7).

One mean speed per subject per direction (four rows per subject):
  Vx_Pos, Vx_Neg, Vy_Pos, Vy_Neg

Produces two 2x2 seaborn lineplot figures (Control vs Dysgraphic, ±1 SE):
  - Unfiltered (raw) XY derivatives
  - 12 Hz Butterworth low-pass on XY before derivatives

Run::

    python DysXAI_task7/analyze_directional_velocity_age.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from dataset import (
    build_sample_table,
    butterworth_lowpass_xy,
    filter_speed_outliers,
    load_raw_timeseries,
    pen_on_mask,
    safe_dt,
)

DIRECTION_ORDER = ("Vx_Pos", "Vx_Neg", "Vy_Pos", "Vy_Neg")
GROUP_PALETTE = {"Control": "#1f77b4", "Dysgraphic": "#d62728"}

FILTER_SPECS: tuple[tuple[bool, str, str, str], ...] = (
    (
        False,
        "RAW",
        "population_directional_velocity_age_RAW.png",
        "population_directional_velocity_age_RAW.csv",
    ),
    (
        True,
        "Butterworth 12 Hz",
        "population_directional_velocity_age_BUTTERWORTH_12Hz.png",
        "population_directional_velocity_age_BUTTERWORTH_12Hz.csv",
    ),
)


def extract_directional_speed_pools(filepath: Path, *, butterworth: bool) -> dict[str, np.ndarray]:
    """Directional speed samples; neg axes stored as |v| for magnitude plots."""
    data = load_raw_timeseries(str(filepath))
    svc = butterworth_lowpass_xy(data) if butterworth else np.asarray(data, dtype=np.float64)

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


def build_directional_dataframe(*, butterworth: bool) -> pd.DataFrame:
    """Exactly four rows per subject (one mean per direction)."""
    sample_df = build_sample_table()
    rows: list[dict[str, object]] = []
    tag = "butterworth" if butterworth else "raw"

    for _, row in tqdm(
        sample_df.iterrows(),
        total=len(sample_df),
        desc=f"Task 7 clips ({tag})",
    ):
        age = row["age"]
        if pd.isna(age):
            continue
        age_val = float(age)
        group = "Dysgraphic" if int(row["label"]) == 1 else "Control"
        subject_id = int(row["subject_id"])
        fp = Path(row["filepath"])

        try:
            pools = extract_directional_speed_pools(fp, butterworth=butterworth)
        except Exception as exc:
            print(f"Skipping {fp.name}: {exc}")
            continue

        for direction in DIRECTION_ORDER:
            values = pools[direction]
            mean_speed = float(np.mean(values)) if values.size > 0 else float("nan")
            rows.append(
                {
                    "subject_id": subject_id,
                    "Age": age_val,
                    "Group": group,
                    "Direction_Type": direction,
                    "Speed_Magnitude": mean_speed,
                    "XY_Filter": tag,
                }
            )

    return pd.DataFrame(rows)


def plot_population_directional_curves(
    df: pd.DataFrame,
    out_path: Path,
    *,
    filter_label: str,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")

    panel_specs = [
        ("Vx_Pos", r"Mean speed pushing right ($V_x > 0$)", axes[0, 0]),
        ("Vx_Neg", r"Mean speed pulling left ($V_x < 0$)", axes[0, 1]),
        ("Vy_Pos", r"Mean speed pushing up ($V_y > 0$)", axes[1, 0]),
        ("Vy_Neg", r"Mean speed pulling down ($V_y < 0$)", axes[1, 1]),
    ]

    for direction, title, ax in panel_specs:
        sub = df[(df["Direction_Type"] == direction) & df["Speed_Magnitude"].notna()]
        if sub.empty:
            ax.set_title(f"{title}\n(no data)")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Mean speed magnitude")
            continue

        sns.lineplot(
            data=sub,
            x="Age",
            y="Speed_Magnitude",
            hue="Group",
            palette=GROUP_PALETTE,
            errorbar="se",
            markers=True,
            dashes=False,
            ax=ax,
            legend=False,
        )
        ax.set_title(title)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Mean speed magnitude")
        ax.set_ylim(bottom=0)

    handles = [
        plt.Line2D([0], [0], color=GROUP_PALETTE["Control"], lw=2, label="Control"),
        plt.Line2D([0], [0], color=GROUP_PALETTE["Dysgraphic"], lw=2, label="Dysgraphic"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.02))

    n_subj = df["subject_id"].nunique()
    fig.suptitle(
        f"Directional velocity across age — Task 7 (hračkárstvo), {filter_label}\n"
        f"Subject-level means, n = {n_subj} children; shaded bands = ±1 SE across subjects",
        fontsize=12,
        y=1.05,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    for butterworth, filter_label, png_name, csv_name in FILTER_SPECS:
        print(f"\n=== {filter_label} ===")
        df = build_directional_dataframe(butterworth=butterworth)
        if df.empty:
            raise RuntimeError(f"No rows for {filter_label}; check Task 7 data.")

        n_subj = df["subject_id"].nunique()
        print(f"Subjects: {n_subj} | Rows: {len(df)} (expected ~{4 * n_subj})")

        speeds = df["Speed_Magnitude"].dropna().to_numpy()
        if speeds.size:
            print(
                f"Mean speed: min={speeds.min():.4f}, median={np.median(speeds):.4f}, "
                f"max={speeds.max():.4f}"
            )

        out_png = out_dir / png_name
        out_csv = out_dir / csv_name
        df.to_csv(out_csv, index=False)
        plot_population_directional_curves(df, out_png, filter_label=filter_label)
        print(f"Saved CSV:  {out_csv.resolve()}")
        print(f"Saved figure: {out_png.resolve()}")

    # Backward-compatible alias for the Butterworth figure
    bw_png = out_dir / "population_directional_velocity_age.png"
    bw_csv = out_dir / "population_directional_velocity_age.csv"
    src_png = out_dir / "population_directional_velocity_age_BUTTERWORTH_12Hz.png"
    src_csv = out_dir / "population_directional_velocity_age_BUTTERWORTH_12Hz.csv"
    if src_png.is_file():
        import shutil

        shutil.copy2(src_png, bw_png)
        shutil.copy2(src_csv, bw_csv)
        print(f"\nAlso copied Butterworth outputs to {bw_png.name} / {bw_csv.name}")


if __name__ == "__main__":
    main()
