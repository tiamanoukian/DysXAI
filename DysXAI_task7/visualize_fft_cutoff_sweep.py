"""
FFT low-pass cutoff sweep on handwriting (Task 7) — spatial plots colored by jerk.

Produces **two figures** (same style as ``visualize_filter_overlay.py``):

1. **Handwriting** — Control | Dysgraphic columns; rows: Raw XY, then FFT at each cutoff Hz.
2. **Residuals** — same layout; rows: ``Raw kinematic − FFT kinematic`` per cutoff.

All panels in figure (1) share one diverging color scale; residual figure uses the same
bar limits as figure (1) with a **display gain** on Δ so faint residuals are visible.

Display gain (auto, unless ``--residual-amplify`` sets a higher floor)::

    typical = nanpercentile(|Δ|, 90)   # after clipping |Δ| at the 99.5th percentile
    gain = clip(signal_lim × target_fraction / typical, 1.0, 5000)
    color_on_residual_row = gain × Δ
    colorbar_limits = ±signal_lim   (same as jerk rows; gain is display-only)

Examples::

    python DysXAI_task7/visualize_fft_cutoff_sweep.py
    python DysXAI_task7/visualize_fft_cutoff_sweep.py --control-id 50 --dys-id 6
    python DysXAI_task7/visualize_fft_cutoff_sweep.py --cutoffs 8,10,12,15 --metric jerk_x
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visualize_filter_overlay import (  # noqa: E402
    COLOR_METRICS,
    SIGNED_CMAP,
    _annotate_residual_amp_axis,
    _format_amplify,
    kinematic_scalar_from_svc,
    load_raw_timeseries,
    pen_on_mask,
    pick_control_and_dysgraphic,
    plot_xy_colored_by_scalar,
    robust_abs_percentile,
    shared_symmetric_lim_from_scalars,
    trim_svc_edges,
    xy_with_stroke_breaks,
)

DEFAULT_CUTOFFS_HZ = (8.0, 10.0, 12.0, 15.0)
DEFAULT_CONTROL_ID = 50
DEFAULT_DYS_ID = 6
DEFAULT_METRIC = "jerk_x"
DEFAULT_OUT_DIR = _PKG_DIR / "filter_overlay_outputs"
RESIDUAL_TARGET_FRACTION = 1.0
RESIDUAL_GAIN_PERCENTILE = 90.0
SIGNAL_PERCENTILE = 95.0
RESIDUAL_PERCENTILE = 99.0


def fft_lowpass_xy_hz(ts: np.ndarray, cutoff_hz: float) -> np.ndarray:
    from dysxai_fft_xy_filter import lowpass_filter_xy_fft

    return lowpass_filter_xy_fft(np.asarray(ts, dtype=np.float32), cutoff_hz=float(cutoff_hz))


@dataclass(frozen=True)
class SweepPanel:
    """One spatial row: geometry from ``geom_svc``, stroke colored by ``scalar``."""

    geom_svc: np.ndarray
    scalar: np.ndarray
    label: str
    kind: str = "signal"  # signal | residual


def signal_panels(raw_svc: np.ndarray, metric: str, cutoffs_hz: tuple[float, ...]) -> list[SweepPanel]:
    rows: list[SweepPanel] = [
        SweepPanel(
            raw_svc,
            kinematic_scalar_from_svc(raw_svc, metric),
            "Raw XY",
            kind="signal",
        )
    ]
    for hz in cutoffs_hz:
        fft_svc = fft_lowpass_xy_hz(raw_svc, hz)
        rows.append(
            SweepPanel(
                fft_svc,
                kinematic_scalar_from_svc(fft_svc, metric),
                f"FFT XY ({_fmt_hz(hz)})",
                kind="signal",
            )
        )
    return rows


def residual_panels(raw_svc: np.ndarray, metric: str, cutoffs_hz: tuple[float, ...]) -> list[SweepPanel]:
    s_raw = kinematic_scalar_from_svc(raw_svc, metric)
    rows: list[SweepPanel] = []
    for hz in cutoffs_hz:
        fft_svc = fft_lowpass_xy_hz(raw_svc, hz)
        s_fft = kinematic_scalar_from_svc(fft_svc, metric)
        rows.append(
            SweepPanel(
                raw_svc,
                s_raw - s_fft,
                f"Noise residual (Raw - FFT {_fmt_hz(hz)})\n[display amplified]",
                kind="residual",
            )
        )
    return rows


def _fmt_hz(hz: float) -> str:
    return f"{hz:g} Hz"


def parse_cutoffs(text: str) -> tuple[float, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("At least one cutoff frequency is required.")
    return tuple(float(p) for p in parts)


def collect_signal_limit(
    *raw_svcs: np.ndarray,
    metric: str,
    cutoffs_hz: tuple[float, ...],
) -> float:
    scalars: list[np.ndarray] = []
    for raw_svc in raw_svcs:
        for panel in signal_panels(raw_svc, metric, cutoffs_hz):
            scalars.append(panel.scalar)
    return shared_symmetric_lim_from_scalars(*scalars, percentile=SIGNAL_PERCENTILE)


def collect_residual_gain(
    *raw_svcs: np.ndarray,
    metric: str,
    cutoffs_hz: tuple[float, ...],
    signal_lim: float,
    residual_amplify: float,
    auto_amplify: bool,
    target_fraction: float,
) -> tuple[float, float]:
    """Return (display_gain, true_residual_lim)."""
    residual_scalars: list[np.ndarray] = []
    for raw_svc in raw_svcs:
        for panel in residual_panels(raw_svc, metric, cutoffs_hz):
            residual_scalars.append(panel.scalar)

    true_lim = shared_symmetric_lim_from_scalars(
        *residual_scalars, percentile=RESIDUAL_PERCENTILE
    )
    typical = robust_abs_percentile(
        *residual_scalars,
        percentile=RESIDUAL_GAIN_PERCENTILE,
        outlier_percentile=99.5,
    )
    gain = max(float(residual_amplify), 1.0)
    if auto_amplify and typical > 0:
        target = signal_lim * target_fraction
        auto = float(np.clip(target / typical, 1.0, 5000.0))
        gain = max(gain, auto)
    return gain, true_lim


def draw_column(
    axes: list,
    raw_svc: np.ndarray,
    panels: list[SweepPanel],
    *,
    group_label: str,
    color_metric: str,
    signal_lim: float,
    residual_gain: float,
    true_residual_lim: float,
) -> list[ScalarMappable]:
    _, cbar_label, color_desc = COLOR_METRICS[color_metric]
    sms: list[ScalarMappable] = []
    amp_s = _format_amplify(residual_gain)
    for ax, panel in zip(axes, panels):
        x_path, y_path = xy_with_stroke_breaks(panel.geom_svc)
        if panel.kind == "residual":
            scalar = panel.scalar * residual_gain
            lim = signal_lim
            row_cbar = f"Δ{cbar_label} (×{amp_s} display; jerk limits)"
            _annotate_residual_amp_axis(
                ax,
                amplify=residual_gain,
                signal_lim=signal_lim,
                true_lim=true_residual_lim,
            )
        else:
            scalar = panel.scalar
            lim = signal_lim
            row_cbar = cbar_label
        sm = plot_xy_colored_by_scalar(
            ax,
            x_path,
            y_path,
            scalar,
            title=f"{group_label} — {panel.label}\n{color_desc}",
            cbar_label=row_cbar,
            symmetric_lim=lim,
            cmap=SIGNED_CMAP,
        )
        sms.append(sm)
    return sms


def save_two_column_figure(
    *,
    control_svc: np.ndarray,
    dys_svc: np.ndarray,
    control_id: int,
    dys_id: int,
    panels_left: list[SweepPanel],
    panels_right: list[SweepPanel],
    out_path: Path,
    color_metric: str,
    signal_lim: float,
    residual_gain: float,
    true_residual_lim: float,
    suptitle: str,
) -> None:
    nrows = len(panels_left)
    if len(panels_right) != nrows:
        raise ValueError("Control and dysgraphic panel counts must match.")
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3.15 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    draw_kw = dict(
        color_metric=color_metric,
        signal_lim=signal_lim,
        residual_gain=residual_gain,
        true_residual_lim=true_residual_lim,
    )
    sms_c = draw_column(
        list(axes[:, 0]),
        control_svc,
        panels_left,
        group_label=f"Control (u{control_id:05d})",
        **draw_kw,
    )
    sms_d = draw_column(
        list(axes[:, 1]),
        dys_svc,
        panels_right,
        group_label=f"Dysgraphic (u{dys_id:05d})",
        **draw_kw,
    )
    for row_ax, sm_c, sm_d in zip(axes, sms_c, sms_d):
        fig.colorbar(sm_c, ax=row_ax[0], fraction=0.046, pad=0.03)
        fig.colorbar(sm_d, ax=row_ax[1], fraction=0.046, pad=0.03)

    fig.suptitle(suptitle, fontsize=9.5, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_sweep(
    *,
    out_dir: Path,
    cutoffs_hz: tuple[float, ...],
    color_metric: str,
    control_id: int | None,
    dys_id: int | None,
    edge_trim: int,
    residual_amplify: float,
    auto_amplify: bool,
    target_fraction: float,
) -> None:
    if color_metric not in COLOR_METRICS:
        raise ValueError(f"Unknown metric {color_metric!r}; choose from {list(COLOR_METRICS)}")

    (c_path, c_id), (d_path, d_id) = pick_control_and_dysgraphic(control_id, dys_id)
    control_svc = load_raw_timeseries(str(c_path))
    dys_svc = load_raw_timeseries(str(d_path))
    if edge_trim > 0:
        control_svc = trim_svc_edges(control_svc, edge_trim)
        dys_svc = trim_svc_edges(dys_svc, edge_trim)

    signal_lim = collect_signal_limit(control_svc, dys_svc, metric=color_metric, cutoffs_hz=cutoffs_hz)
    residual_gain, true_residual_lim = collect_residual_gain(
        control_svc,
        dys_svc,
        metric=color_metric,
        cutoffs_hz=cutoffs_hz,
        signal_lim=signal_lim,
        residual_amplify=residual_amplify,
        auto_amplify=auto_amplify,
        target_fraction=target_fraction,
    )

    hz_tag = "_".join(f"{int(h) if h == int(h) else h}" for h in cutoffs_hz)
    base = f"control_u{c_id:05d}_vs_dys_u{d_id:05d}_{color_metric}_fft_cutoffs_{hz_tag}"

    panels_c_sig = signal_panels(control_svc, color_metric, cutoffs_hz)
    panels_d_sig = signal_panels(dys_svc, color_metric, cutoffs_hz)
    _, _, color_desc = COLOR_METRICS[color_metric]
    cutoffs_txt = ", ".join(_fmt_hz(h) for h in cutoffs_hz)

    handwriting_path = out_dir / f"{base}_handwriting.png"
    save_two_column_figure(
        control_svc=control_svc,
        dys_svc=dys_svc,
        control_id=c_id,
        dys_id=d_id,
        panels_left=panels_c_sig,
        panels_right=panels_d_sig,
        out_path=handwriting_path,
        color_metric=color_metric,
        signal_lim=signal_lim,
        residual_gain=1.0,
        true_residual_lim=true_residual_lim,
        suptitle=(
            f"Control vs dysgraphic — {color_desc} on handwriting (FFT cutoff sweep: {cutoffs_txt}). "
            f"Shared color scale on all rows (±{signal_lim:.2e})."
        ),
    )

    panels_c_res = residual_panels(control_svc, color_metric, cutoffs_hz)
    panels_d_res = residual_panels(dys_svc, color_metric, cutoffs_hz)
    residual_path = out_dir / f"{base}_residuals.png"
    amp_lbl = _format_amplify(residual_gain)
    save_two_column_figure(
        control_svc=control_svc,
        dys_svc=dys_svc,
        control_id=c_id,
        dys_id=d_id,
        panels_left=panels_c_res,
        panels_right=panels_d_res,
        out_path=residual_path,
        color_metric=color_metric,
        signal_lim=signal_lim,
        residual_gain=residual_gain,
        true_residual_lim=true_residual_lim,
        suptitle=(
            f"Control vs dysgraphic — FFT residual {color_desc} (Raw − FFT per cutoff). "
            f"Shared jerk limits ±{signal_lim:.2e}; display gain ×{amp_lbl} on Δ only. "
            f"Typical true |Δ| scale ≈ ±{true_residual_lim:.2e}."
        ),
    )

    print(f"Control u{c_id:05d} | {c_path.name}")
    print(f"Dysgraphic u{d_id:05d} | {d_path.name}")
    print(f"Cutoffs (Hz): {list(cutoffs_hz)}")
    print(f"Shared signal color limit: +/-{signal_lim:.3e}")
    print(f"True residual color scale (physical |delta|): +/-{true_residual_lim:.3e}")
    print(
        "Residual display gain: "
        f"gain = max(manual, clip(signal_lim * {target_fraction} / P90(|delta|), 1, 5000)) "
        f"=> x{residual_gain:.2f}"
    )
    print(f"Wrote: {handwriting_path.resolve()}")
    print(f"Wrote: {residual_path.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="FFT cutoff sweep: handwriting + residual figures (control vs dysgraphic).",
    )
    p.add_argument(
        "--cutoffs",
        type=str,
        default=",".join(str(int(h)) if h == int(h) else str(h) for h in DEFAULT_CUTOFFS_HZ),
        help="Comma-separated FFT cutoff frequencies in Hz (default: 8,10,12,15).",
    )
    p.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        choices=tuple(COLOR_METRICS),
        help="Kinematic scalar for stroke color (default: jerk_x).",
    )
    p.add_argument("--control-id", type=int, default=DEFAULT_CONTROL_ID)
    p.add_argument("--dys-id", type=int, default=DEFAULT_DYS_ID)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--edge-trim", type=int, default=30, help="Trim samples at clip ends.")
    p.add_argument(
        "--residual-amplify",
        type=float,
        default=1.0,
        help="Minimum display gain on residuals (auto gain can increase it).",
    )
    p.add_argument(
        "--no-auto-amplify",
        action="store_true",
        help="Disable automatic residual gain; use --residual-amplify only.",
    )
    p.add_argument(
        "--residual-target-fraction",
        type=float,
        default=RESIDUAL_TARGET_FRACTION,
        help="Auto gain target: signal_lim × this / P90(|Δ|).",
    )
    args = p.parse_args()

    cutoffs = parse_cutoffs(args.cutoffs)
    out_dir = args.out_dir if args.out_dir.is_absolute() else _PKG_DIR / args.out_dir

    run_sweep(
        out_dir=out_dir,
        cutoffs_hz=cutoffs,
        color_metric=args.metric,
        control_id=args.control_id,
        dys_id=args.dys_id,
        edge_trim=args.edge_trim,
        residual_amplify=args.residual_amplify,
        auto_amplify=not args.no_auto_amplify,
        target_fraction=args.residual_target_fraction,
    )


if __name__ == "__main__":
    main()
