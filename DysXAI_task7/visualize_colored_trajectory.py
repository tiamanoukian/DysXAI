"""
Task 7: side-by-side Control vs Dysgraphic handwriting colored by signed Vx / Vy.

2×2 figure per filter variant:
  - Top row: horizontal velocity (Vx) — Control | Dysgraphic
  - Bottom row: vertical velocity (Vy) — Control | Dysgraphic

By default saves both unfiltered (raw) and 12 Hz Butterworth versions.

Velocity color limits are **shared** between control and dysgraphic within each row
(Vx row and Vy row), so the same color means the same signed speed in both panels.
Use ``--per-subject-scale`` for independent limits (more vivid dys trace, not comparable).

Run::

    python DysXAI_task7/visualize_colored_trajectory.py
    python DysXAI_task7/visualize_colored_trajectory.py --raw-only
    python DysXAI_task7/visualize_colored_trajectory.py --per-subject-scale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from dataset import (  # noqa: E402
    build_sample_table,
    butterworth_lowpass_xy,
    load_raw_timeseries,
    pen_on_mask,
    velocities_xy,
)

OUT_DIR = _PKG_DIR / "trajectory_velocity_outputs"
DEFAULT_CONTROL_ID = 50
DEFAULT_DYSGRAPHIC_ID = 22

# Saturated blue→red (ColorBrewer-style); light gray at zero — visible, not white/black.
VIVID_BLUE_RED_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "vivid_blue_red",
    ["#2166ac", "#67a9cf", "#d1d5db", "#ef8a62", "#b2182b"],
    N=256,
)
CMAP_CHOICES = ("vivid_blue_red", "coolwarm", "RdBu_r", "dark_blue_red")


def resolve_colormap(name: str) -> mcolors.Colormap:
    if name == "vivid_blue_red":
        return VIVID_BLUE_RED_CMAP
    if name == "dark_blue_red":
        return mcolors.LinearSegmentedColormap.from_list(
            "dark_blue_red", ["#0d47a1", "#455a64", "#b71c1c"], N=256
        )
    return plt.get_cmap(name)


def resolve_subject(subject_id: int, label: int | None) -> tuple[Path, int]:
    df = build_sample_table()
    row = df[df["subject_id"] == subject_id]
    if row.empty:
        raise ValueError(f"Subject {subject_id} not found in Task 7 table.")
    if label is not None and int(row.iloc[0]["label"]) != label:
        raise ValueError(f"Subject {subject_id} is not label={label}.")
    return Path(row.iloc[0]["filepath"]), int(subject_id)


def load_trajectory(
    filepath: Path,
    *,
    butterworth: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pen-on X/Y and signed Vx/Vy; NaN at pen lifts."""
    raw = load_raw_timeseries(str(filepath))
    svc = butterworth_lowpass_xy(raw) if butterworth else np.asarray(raw, dtype=np.float32)
    on = pen_on_mask(svc)

    x = svc[:, 0].astype(np.float64).copy()
    y = svc[:, 1].astype(np.float64).copy()
    vx, vy = velocities_xy(x, y, svc[:, 2].astype(np.float64))

    x[~on] = np.nan
    y[~on] = np.nan
    vx[~on] = np.nan
    vy[~on] = np.nan
    return x, y, vx, vy


def subject_vmax(v: np.ndarray, *, percentile: float, fixed: float | None) -> float:
    if fixed is not None:
        return max(float(fixed), 1e-6)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return 1.0
    return max(float(np.nanpercentile(np.abs(finite), percentile)), 1e-6)


def row_vmax(
    control_v: np.ndarray,
    dys_v: np.ndarray,
    *,
    percentile: float,
    fixed: float | None,
    sync_scale: bool,
) -> tuple[float, float]:
    """Return (vmax_control, vmax_dys). Identical when sync_scale or fixed override."""
    if fixed is not None:
        lim = max(float(fixed), 1e-6)
        return lim, lim
    c_lim = subject_vmax(control_v, percentile=percentile, fixed=None)
    d_lim = subject_vmax(dys_v, percentile=percentile, fixed=None)
    if sync_scale:
        shared = max(c_lim, d_lim)
        return shared, shared
    return c_lim, d_lim


def _near_center_fraction(v: np.ndarray, vmax_val: float, frac: float = 0.1) -> float:
    finite = v[np.isfinite(v)]
    if finite.size == 0 or vmax_val <= 0:
        return 0.0
    return float(np.mean(np.abs(finite) < frac * vmax_val))


def plot_velocity_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    velocity: np.ndarray,
    *,
    title: str,
    vmax_val: float,
    cmap: mcolors.Colormap,
) -> None:
    norm = mcolors.TwoSlopeNorm(vmin=-vmax_val, vcenter=0.0, vmax=vmax_val)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(velocity)
    sc = ax.scatter(
        x[mask],
        y[mask],
        c=velocity[mask],
        cmap=cmap,
        norm=norm,
        s=5,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (tablet)")
    ax.set_ylabel("Y (tablet)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Velocity")


def save_comparison_figure(
    ctrl: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dys: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    control_id: int,
    dys_id: int,
    filter_label: str,
    out_path: Path,
    cmap: mcolors.Colormap,
    percentile: float,
    vmax_vx: float | None,
    vmax_vy: float | None,
    sync_scale: bool,
) -> None:
    cx, cy, cvx, cvy = ctrl
    dx, dy, dvx, dvy = dys

    vmax_x_c, vmax_x_d = row_vmax(
        cvx, dvx, percentile=percentile, fixed=vmax_vx, sync_scale=sync_scale
    )
    vmax_y_c, vmax_y_d = row_vmax(
        cvy, dvy, percentile=percentile, fixed=vmax_vy, sync_scale=sync_scale
    )
    scale_note = (
        "shared row scales (same |v| limits control vs dys)"
        if sync_scale
        else "per-subject row scales"
    )
    print(
        f"  [{filter_label}] Vx vmax - control: {vmax_x_c:.4g}, dys: {vmax_x_d:.4g} | "
        f"Vy - control: {vmax_y_c:.4g}, dys: {vmax_y_d:.4g} ({scale_note})"
    )
    if sync_scale:
        print(
            f"    dys near-gray (|v|<10% vmax): "
            f"Vx {_near_center_fraction(dvx, vmax_x_d):.0%}, Vy {_near_center_fraction(dvy, vmax_y_d):.0%}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor="white")

    plot_velocity_scatter(
        axes[0, 0], cx, cy, cvx,
        title="Control — Horizontal Velocity (Red=Right, Blue=Left)",
        vmax_val=vmax_x_c,
        cmap=cmap,
    )
    plot_velocity_scatter(
        axes[0, 1], dx, dy, dvx,
        title="Dysgraphic — Horizontal Velocity (Red=Right, Blue=Left)",
        vmax_val=vmax_x_d,
        cmap=cmap,
    )
    plot_velocity_scatter(
        axes[1, 0], cx, cy, cvy,
        title="Control — Vertical Velocity (Red=Up, Blue=Down)",
        vmax_val=vmax_y_c,
        cmap=cmap,
    )
    plot_velocity_scatter(
        axes[1, 1], dx, dy, dvy,
        title="Dysgraphic — Vertical Velocity (Red=Up, Blue=Down)",
        vmax_val=vmax_y_d,
        cmap=cmap,
    )

    fig.suptitle(
        f"Task 7 (hračkárstvo) — Control u{control_id:05d} vs Dysgraphic u{dys_id:05d}\n"
        f"{filter_label} | {scale_note} | blue = negative, gray = 0, red = positive",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved: {out_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="2×2 Control vs Dysgraphic trajectory colored by signed Vx/Vy.",
    )
    p.add_argument("--control-id", type=int, default=DEFAULT_CONTROL_ID)
    p.add_argument("--dys-id", type=int, default=DEFAULT_DYSGRAPHIC_ID)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--percentile",
        type=float,
        default=98.0,
        help="Percentile of |v| used to set vmax (per subject if --per-subject-scale; else max of both).",
    )
    p.add_argument(
        "--per-subject-scale",
        action="store_true",
        help="Independent vmax per panel (more vivid for slower writers; colors not comparable across columns).",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="vivid_blue_red",
        choices=CMAP_CHOICES,
        help="Colormap (default: saturated blue / light gray / saturated red).",
    )
    p.add_argument("--vmax-vx", type=float, default=None, help="Override shared Vx row limit.")
    p.add_argument("--vmax-vy", type=float, default=None, help="Override shared Vy row limit.")
    p.add_argument("--raw-only", action="store_true", help="Only save unfiltered figure.")
    p.add_argument("--butterworth-only", action="store_true", help="Only save Butterworth figure.")
    args = p.parse_args()

    if args.raw_only and args.butterworth_only:
        raise SystemExit("Choose at most one of --raw-only and --butterworth-only.")

    sns.set_theme(style="white")
    cmap = resolve_colormap(args.cmap)

    c_path, c_id = resolve_subject(args.control_id, 0)
    d_path, d_id = resolve_subject(args.dys_id, 1)
    print(f"Control u{c_id:05d} | Dysgraphic u{d_id:05d}")

    variants: list[tuple[bool, str, str]] = []
    if not args.butterworth_only:
        variants.append((False, "Unfiltered XY (raw derivatives)", "RAW"))
    if not args.raw_only:
        variants.append((True, "12 Hz Butterworth on XY", "BUTTERWORTH"))

    for butterworth, filter_label, suffix in variants:
        ctrl = load_trajectory(c_path, butterworth=butterworth)
        dys = load_trajectory(d_path, butterworth=butterworth)
        out = args.out_dir / f"control_u{c_id:05d}_vs_dys_u{d_id:05d}_velocity_compare_{suffix}.png"
        save_comparison_figure(
            ctrl,
            dys,
            control_id=c_id,
            dys_id=d_id,
            filter_label=filter_label,
            out_path=out,
            cmap=cmap,
            percentile=args.percentile,
            vmax_vx=args.vmax_vx,
            vmax_vy=args.vmax_vy,
            sync_scale=not args.per_subject_scale,
        )

    print(f"\nSaved under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
