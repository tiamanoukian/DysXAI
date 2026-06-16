"""
Raw vs filtered handwriting dynamics for Task 7 (professor overlay figures).

Style: trajectory colored by **signed** Vx/Vy/jerk (RdBu_r); per-panel symmetric limits.

Run from repo root::

    # Single-panel Jerk_X proof (quick)
    python DysXAI_task7/visualize_filter_overlay.py --plot jerk

    # Full 8-panel: X, Y, Vx, Vy, Ax, Ay, Jerk_X, Jerk_Y (main deliverable)
    python DysXAI_task7/visualize_filter_overlay.py --plot dynamics

    # Control vs dysgraphic: handwriting colored by jerk + residual row (shared color scale)
    python DysXAI_task7/visualize_filter_overlay.py --plot jerk

    # FFT + amplified residuals + true |Δ| rows (physical scale); see in-plot notes
    python DysXAI_task7/visualize_filter_overlay.py --plot jerk --compare-fft --find-noisiest

    # Shorter figure (no true-scale rows)
    python DysXAI_task7/visualize_filter_overlay.py --plot jerk --compare-fft --no-true-residual-row

    # Extra manual gain if residuals still faint
    python DysXAI_task7/visualize_filter_overlay.py --plot jerk --compare-fft --residual-amplify 5

    # Handwriting trace (X vs Y) — paper style + gray/blue overlay (professor example)
    python DysXAI_task7/visualize_filter_overlay.py --plot trajectory

    # Control vs dysgraphic, colored by Vx / Jerk_X / Jerk_Y (side-by-side)
    python DysXAI_task7/visualize_filter_overlay.py --compare-groups \\
        --color-metrics vx,jerk_x,jerk_y --compare-fft

Outputs under ``DysXAI_task7/filter_overlay_outputs/`` by default.
"""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dysxai_init import Config, load_metadata  # noqa: E402

from dataset import (  # noqa: E402
    TASK7_DATA_DIR,
    build_sample_table,
    butterworth_lowpass_xy,
    compute_derivatives,
    load_raw_timeseries,
    pen_on_mask,
    subject_id_from_svc_path,
)

# After compute_derivatives on 7-channel .svc: vx..jy at 7..12
_DERIV_BASE = 7
CHANNEL_SPECS: tuple[tuple[str, int], ...] = (
    ("X", 0),
    ("Y", 1),
    ("Vx", _DERIV_BASE + 0),
    ("Vy", _DERIV_BASE + 1),
    ("Ax", _DERIV_BASE + 2),
    ("Ay", _DERIV_BASE + 3),
    ("Jerk_X", _DERIV_BASE + 4),
    ("Jerk_Y", _DERIV_BASE + 5),
)

# Kinematic scalars for trajectory color (signed; index in kinematics_tensor output).
COLOR_METRICS: dict[str, tuple[int, str, str]] = {
    "vx": (_DERIV_BASE + 0, "Vx", "horizontal velocity Vx"),
    "vy": (_DERIV_BASE + 1, "Vy", "vertical velocity Vy"),
    "jerk_x": (_DERIV_BASE + 4, "Jerk_X", "jerk along X"),
    "jerk_y": (_DERIV_BASE + 5, "Jerk_Y", "jerk along Y"),
}
SIGNED_CMAP = "RdBu_r"  # diverging: negative vs positive along axis

RAW_STYLE = {"color": "#b0b0b0", "alpha": 0.45, "linewidth": 0.75, "zorder": 1}
BW_STYLE = {"color": "#2196f3", "alpha": 0.95, "linewidth": 1.35, "zorder": 3}
FFT_STYLE = {"color": "#ff9800", "alpha": 0.9, "linewidth": 1.15, "zorder": 2, "linestyle": "-"}
def fft_lowpass_xy(ts: np.ndarray) -> np.ndarray:
    from dysxai_fft_xy_filter import lowpass_filter_xy_fft

    return lowpass_filter_xy_fft(np.asarray(ts, dtype=np.float32))


def kinematics_tensor(ts: np.ndarray) -> np.ndarray:
    """7-channel clip -> 13-channel with vx..jy."""
    return compute_derivatives(np.asarray(ts, dtype=np.float32))


def mask_pen_off(arr: np.ndarray, on: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    out[~on] = np.nan
    return out


def trim_edges(t: np.ndarray, *arrays: np.ndarray, n: int = 30) -> tuple:
    if t.size <= 2 * n:
        return (t, *arrays)
    sl = slice(n, -n)
    return (t[sl], *(a[sl] for a in arrays))


def trim_svc_edges(raw_svc: np.ndarray, n: int = 30) -> np.ndarray:
    """Drop first/last ``n`` samples (reduces derivative edge spikes in spatial plots)."""
    if raw_svc.shape[0] <= 2 * n:
        return raw_svc
    return raw_svc[n:-n]


def y_limits_symmetric(*arrays: np.ndarray, percentile: float = 99.0) -> float:
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size])
    if vals.size == 0:
        return 1.0
    cap = float(np.percentile(np.abs(vals), percentile))
    return max(cap, float(np.max(np.abs(vals)) * 0.2), 1e-6)


def jerk_noise_score(raw_svc: np.ndarray, *, axis: str = "X", edge_trim: int = 30) -> float:
    """Variance of pen-on Jerk residual (raw XY derivatives minus Butterworth XY derivatives)."""
    if edge_trim > 0:
        raw_svc = trim_svc_edges(raw_svc, edge_trim)
    key = f"Jerk_{axis}"
    idx = dict(CHANNEL_SPECS)[key]
    on = pen_on_mask(raw_svc)
    kin_raw = kinematics_tensor(raw_svc)
    kin_bw = kinematics_tensor(butterworth_lowpass_xy(raw_svc))
    j_raw = mask_pen_off(kin_raw[:, idx], on)
    j_bw = mask_pen_off(kin_bw[:, idx], on)
    return float(np.nanvar(j_raw - j_bw))


def scan_noisiest_subjects(
    top_k: int = 3,
    *,
    axis: str = "X",
) -> list[tuple[int, float, int, str]]:
    """
    Rank subjects by ``nanvar(jerk_raw - jerk_butterworth)`` on pen-on samples.

    Returns ``[(subject_id, score, label, filepath), ...]`` descending by score.
    """
    df = build_sample_table()
    scored: list[tuple[int, float, int, str]] = []
    for _, row in df.iterrows():
        fp = str(row["filepath"])
        try:
            raw_svc = load_raw_timeseries(fp)
            if raw_svc.shape[1] < 7 or not np.any(pen_on_mask(raw_svc)):
                continue
            score = jerk_noise_score(raw_svc, axis=axis)
            if not np.isfinite(score):
                continue
            scored.append((int(row["subject_id"]), score, int(row["label"]), fp))
        except (OSError, ValueError, RuntimeError):
            continue
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def print_noisiest_ranking(ranked: list[tuple[int, float, int, str]], *, axis: str = "X") -> None:
    label_names = {0: "control", 1: "dysgraphic"}
    print(f"\nTop {len(ranked)} noisiest subjects (nanvar of Jerk_{axis} residual: raw - Butterworth):")
    for rank, (sid, score, lab, fp) in enumerate(ranked, start=1):
        group = label_names.get(lab, f"label={lab}")
        print(f"  {rank}. subject_id={sid:05d}  score={score:.6e}  ({group})  {Path(fp).name}")


def xy_with_stroke_breaks(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pen-on X/Y with NaN at pen-up and large within-stroke time gaps."""
    on = pen_on_mask(ts)
    x = ts[:, 0].astype(np.float64).copy()
    y = ts[:, 1].astype(np.float64).copy()
    t = ts[:, 2].astype(np.float64)
    x[~on] = np.nan
    y[~on] = np.nan
    if on.sum() >= 2:
        dt = np.diff(t)
        pen_down = on[:-1] & on[1:]
        good = pen_down & np.isfinite(dt) & (dt > 0)
        if np.any(good):
            gap = float(np.percentile(dt[good], 95) * 3.0)
            for i in np.where(pen_down & (dt > gap))[0]:
                x[i + 1] = np.nan
                y[i + 1] = np.nan
    return x, y


def split_xy_at_nan(x: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split into continuous strokes (no NaN interior)."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    xs: list[float] = []
    ys: list[float] = []
    for xi, yi in zip(x, y):
        if np.isfinite(xi) and np.isfinite(yi):
            xs.append(float(xi))
            ys.append(float(yi))
        elif xs:
            segments.append((np.asarray(xs), np.asarray(ys)))
            xs, ys = [], []
    if xs:
        segments.append((np.asarray(xs), np.asarray(ys)))
    return segments


def split_xyz_at_nan(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split (x, y, z) into continuous strokes."""
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for xi, yi, zi in zip(x, y, z):
        if np.isfinite(xi) and np.isfinite(yi):
            xs.append(float(xi))
            ys.append(float(yi))
            zs.append(float(zi) if np.isfinite(zi) else 0.0)
        elif xs:
            segments.append((np.asarray(xs), np.asarray(ys), np.asarray(zs)))
            xs, ys, zs = [], [], []
    if xs:
        segments.append((np.asarray(xs), np.asarray(ys), np.asarray(zs)))
    return segments


def plot_xy_strokes(ax, x: np.ndarray, y: np.ndarray, *, label: str | None = None, **line_kw) -> None:
    first = True
    for xs, ys in split_xy_at_nan(x, y):
        if xs.size < 2:
            continue
        ax.plot(
            xs,
            ys,
            label=label if first and label else None,
            **line_kw,
        )
        first = False


def kinematic_scalar_from_svc(svc: np.ndarray, metric: str) -> np.ndarray:
    """Pen-on kinematic scalar from this clip's XY (raw, Butterworth, or FFT)."""
    if metric not in COLOR_METRICS:
        raise KeyError(f"Unknown metric {metric!r}; choose from {list(COLOR_METRICS)}")
    idx, _, _ = COLOR_METRICS[metric]
    kin = kinematics_tensor(svc)
    return mask_pen_off(kin[:, idx], pen_on_mask(svc))


def raw_kinematic_scalar(raw_svc: np.ndarray, metric: str) -> np.ndarray:
    """Alias: scalar from unfiltered XY."""
    return kinematic_scalar_from_svc(raw_svc, metric)


def signed_twoslope_norm(
    scalar: np.ndarray,
    percentile: float = 95.0,
    *,
    symmetric_lim: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> TwoSlopeNorm:
    """
    Diverging norm with neutral color pinned at zero.

    If ``symmetric_lim`` is set, use ``[-L, 0, +L]`` with ``L = symmetric_lim``.
    Otherwise ``L = nanpercentile(|value|, percentile)`` on ``scalar``.
    """
    if symmetric_lim is not None:
        lim = max(float(symmetric_lim), 1e-6)
    elif vmin is not None and vmax is not None:
        lim = max(abs(float(vmin)), abs(float(vmax)), 1e-6)
    else:
        finite = scalar[np.isfinite(scalar)]
        if finite.size == 0:
            lim = 1.0
        else:
            lim = float(np.nanpercentile(np.abs(finite), percentile))
            lim = max(lim, 1e-6)
    return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)


def plot_xy_colored_by_scalar(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    scalar: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str = SIGNED_CMAP,
    percentile: float = 95.0,
    symmetric_lim: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> ScalarMappable:
    """
    Spatial trace colored by **signed** scalar along the path (Vx, Vy, jerk, …).

    Each panel uses ``TwoSlopeNorm(vcenter=0)`` with ``L = nanpercentile(|scalar|)``
    or a fixed ``symmetric_lim`` when comparing subjects side-by-side.
    """
    norm = signed_twoslope_norm(
        scalar, percentile, symmetric_lim=symmetric_lim, vmin=vmin, vmax=vmax
    )
    sm = ScalarMappable(norm=norm, cmap=cmap)

    for sx, sy, sv in split_xyz_at_nan(x, y, scalar):
        if sx.size < 2:
            continue
        pts = np.column_stack([sx, sy])
        seg_lines = np.stack([pts[:-1], pts[1:]], axis=1)
        c = sv[:-1]  # signed — direction of motion / acceleration change matters
        lc = LineCollection(seg_lines, cmap=cmap, norm=norm, linewidths=1.4, alpha=0.95)
        lc.set_array(c)
        ax.add_collection(lc)

    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (tablet)")
    ax.set_ylabel("Y (tablet)")
    ax.grid(True, alpha=0.2)
    return sm


@dataclass(frozen=True)
class SpatialPanel:
    """One spatial handwriting row."""

    svc: np.ndarray
    scalar: np.ndarray
    label: str
    kind: str = "signal"  # signal | residual_amp | residual_true

    @property
    def is_residual(self) -> bool:
        return self.kind.startswith("residual")


def spatial_panel_specs(
    raw_svc: np.ndarray,
    color_metric: str,
    *,
    compare_fft: bool,
    include_residual: bool,
    include_true_residual: bool = False,
) -> list[SpatialPanel]:
    """
    Rows for spatial plots. Residual rows use **raw** stroke geometry and separate color scaling.

    Residuals: ``raw_kinematic - filtered_kinematic`` for Butterworth and (if enabled) FFT.
    """
    bw_svc = butterworth_lowpass_xy(raw_svc)
    s_raw = kinematic_scalar_from_svc(raw_svc, color_metric)
    s_bw = kinematic_scalar_from_svc(bw_svc, color_metric)
    panels: list[SpatialPanel] = [
        SpatialPanel(raw_svc, s_raw, "Raw XY"),
        SpatialPanel(bw_svc, s_bw, "Butterworth XY (12 Hz)"),
    ]
    fft_svc: np.ndarray | None = None
    if compare_fft:
        fft_svc = fft_lowpass_xy(raw_svc)
        panels.append(
            SpatialPanel(
                fft_svc,
                kinematic_scalar_from_svc(fft_svc, color_metric),
                "FFT XY (12 Hz)",
            )
        )
    res_bw = s_raw - s_bw
    res_fft = (
        s_raw - kinematic_scalar_from_svc(fft_svc, color_metric)
        if compare_fft and fft_svc is not None
        else None
    )
    if include_residual:
        panels.append(
            SpatialPanel(
                raw_svc,
                res_bw,
                "Noise residual (Raw - Butterworth)\n[display amplified]",
                kind="residual_amp",
            )
        )
        if res_fft is not None:
            panels.append(
                SpatialPanel(
                    raw_svc,
                    res_fft,
                    "Noise residual (Raw - FFT)\n[display amplified]",
                    kind="residual_amp",
                )
            )
    if include_true_residual:
        panels.append(
            SpatialPanel(
                raw_svc,
                res_bw,
                "True |Δ| (Raw - Butterworth)\n[physical scale, gain x1]",
                kind="residual_true",
            )
        )
        if res_fft is not None:
            panels.append(
                SpatialPanel(
                    raw_svc,
                    res_fft,
                    "True |Δ| (Raw - FFT)\n[physical scale, gain x1]",
                    kind="residual_true",
                )
            )
    return panels


def shared_symmetric_lim_from_scalars(
    *scalars: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """One diverging color scale ``[-L, +L]`` shared across panels."""
    vals = np.concatenate([s[np.isfinite(s)] for s in scalars if s.size])
    if vals.size == 0:
        return 1.0
    lim = float(np.nanpercentile(np.abs(vals), percentile))
    return max(lim, 1e-6)


def robust_abs_percentile(
    *scalars: np.ndarray,
    percentile: float = 90.0,
    outlier_percentile: float = 99.5,
) -> float:
    """Typical |value| for gain estimation; clips extreme stroke-edge spikes first."""
    vals = np.concatenate([s[np.isfinite(s)] for s in scalars if s.size])
    if vals.size == 0:
        return 1.0
    absv = np.abs(vals.astype(np.float64))
    if outlier_percentile < 100.0 and absv.size > 8:
        cap = float(np.nanpercentile(absv, outlier_percentile))
        if cap > 0:
            absv = absv[absv <= cap]
    if absv.size == 0:
        return 1.0
    return max(float(np.nanpercentile(absv, percentile)), 1e-12)


def collect_signal_and_residual_limits(
    *raw_svcs: np.ndarray,
    color_metric: str,
    compare_fft: bool,
    include_residual: bool,
    signal_percentile: float = 95.0,
    residual_percentile: float = 99.0,
    residual_amplify: float = 1.0,
    residual_auto_amplify: bool = True,
    residual_target_fraction: float = 1.0,
    residual_gain_percentile: float = 90.0,
) -> tuple[float, float, float, float]:
    """
    Return ``(signal_lim, residual_amp_lim, effective_amplify, true_residual_lim)``.

    Amplified residual rows use ``residual_amp_lim`` (= ``signal_lim``) with display gain.
    True residual rows use ``true_residual_lim`` on unmultiplied |Δ|.
    """
    signal_scalars: list[np.ndarray] = []
    residual_scalars: list[np.ndarray] = []
    for raw_svc in raw_svcs:
        for panel in spatial_panel_specs(
            raw_svc,
            color_metric,
            compare_fft=compare_fft,
            include_residual=include_residual,
            include_true_residual=False,
        ):
            if panel.kind.startswith("residual"):
                residual_scalars.append(panel.scalar)
            elif panel.kind == "signal":
                signal_scalars.append(panel.scalar)

    signal_lim = shared_symmetric_lim_from_scalars(*signal_scalars, percentile=signal_percentile)
    if not residual_scalars:
        return signal_lim, signal_lim, 1.0, signal_lim

    true_residual_lim = shared_symmetric_lim_from_scalars(
        *residual_scalars, percentile=residual_percentile
    )
    typical_residual = robust_abs_percentile(
        *residual_scalars,
        percentile=residual_gain_percentile,
        outlier_percentile=99.5,
    )
    amplify = max(float(residual_amplify), 1.0)
    if residual_auto_amplify and typical_residual > 0:
        target = signal_lim * residual_target_fraction
        auto = float(np.clip(target / typical_residual, 1.0, 5000.0))
        amplify = max(amplify, auto)

    return signal_lim, signal_lim, amplify, true_residual_lim


def collect_scalars_for_shared_lim(
    *raw_svcs: np.ndarray,
    color_metric: str,
    compare_fft: bool,
    include_residual: bool,
) -> float:
    """Legacy helper: single limit from signal rows only."""
    lim, _, _, _ = collect_signal_and_residual_limits(
        *raw_svcs,
        color_metric=color_metric,
        compare_fft=compare_fft,
        include_residual=include_residual,
        residual_auto_amplify=False,
    )
    return lim


def _format_amplify(amplify: float) -> str:
    return f"{amplify:.0f}" if amplify >= 10 else f"{amplify:.1f}"


def _annotate_residual_amp_axis(ax, *, amplify: float, signal_lim: float, true_lim: float) -> None:
    """Explain that colors are boosted for display, not larger physics than jerk."""
    ax.text(
        0.02,
        0.98,
        (
            f"Color = Δ × {_format_amplify(amplify)} (display only)\n"
            f"True |Δ| ≈ color / {_format_amplify(amplify)}\n"
            f"Bar limits = jerk (±{signal_lim:.1e})\n"
            f"Typical true |Δ| scale ≈ ±{true_lim:.1e}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8e1", edgecolor="#c9a227", alpha=0.92),
        zorder=10,
    )


def _draw_spatial_column(
    axes: list,
    raw_svc: np.ndarray,
    *,
    group_label: str,
    color_metric: str,
    compare_fft: bool,
    include_residual: bool,
    include_true_residual: bool,
    signal_lim: float,
    residual_amp_lim: float,
    true_residual_lim: float,
    residual_amplify: float,
) -> list[ScalarMappable]:
    """Fill one column: jerk rows + amplified residuals + optional true-scale residuals."""
    _, cbar_label, color_desc = COLOR_METRICS[color_metric]
    panels = spatial_panel_specs(
        raw_svc,
        color_metric,
        compare_fft=compare_fft,
        include_residual=include_residual,
        include_true_residual=include_true_residual,
    )
    scalar_mappables: list[ScalarMappable] = []
    amp_s = _format_amplify(residual_amplify)
    for ax, panel in zip(axes, panels):
        x_path, y_path = xy_with_stroke_breaks(panel.svc)
        if panel.kind == "residual_amp":
            scalar = panel.scalar * residual_amplify
            lim = residual_amp_lim
            row_cbar = f"Δ{cbar_label} (×{amp_s} display; jerk limits)"
            _annotate_residual_amp_axis(
                ax,
                amplify=residual_amplify,
                signal_lim=signal_lim,
                true_lim=true_residual_lim,
            )
        elif panel.kind == "residual_true":
            scalar = panel.scalar
            lim = true_residual_lim
            row_cbar = f"Δ{cbar_label} (true |Δ| scale)"
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
        )
        scalar_mappables.append(sm)
    return scalar_mappables


def _draw_paper_style_column(
    axes: list,
    raw_svc: np.ndarray,
    *,
    subject_id: int,
    group_label: str,
    color_metric: str,
    compare_fft: bool,
    percentile: float = 95.0,
) -> list[ScalarMappable]:
    """
    Fill one column: raw / Butterworth / FFT (per-panel color limits).

    Prefer ``_draw_spatial_column`` when a shared scale or residual row is needed.
    """
    _ = subject_id
    lim = collect_scalars_for_shared_lim(
        raw_svc,
        color_metric=color_metric,
        compare_fft=compare_fft,
        include_residual=False,
    )
    return _draw_spatial_column(
        axes,
        raw_svc,
        group_label=group_label,
        color_metric=color_metric,
        compare_fft=compare_fft,
        include_residual=False,
        include_true_residual=False,
        signal_lim=lim,
        residual_amp_lim=lim,
        true_residual_lim=lim,
        residual_amplify=1.0,
    )


def save_trajectory_paper_style(
    raw_svc: np.ndarray,
    *,
    out_path: Path,
    subject_id: int,
    compare_fft: bool,
    color_metric: str = "vy",
) -> None:
    """Paper-style figure: raw path colored by kinematic scalar; filtered XY below."""
    nrows = 3 if compare_fft else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.2 * nrows))
    if nrows == 1:
        axes = [axes]
    col_axes = axes if isinstance(axes, np.ndarray) else [axes]
    if not isinstance(col_axes, list):
        col_axes = list(col_axes)

    sms = _draw_paper_style_column(
        col_axes,
        raw_svc,
        subject_id=subject_id,
        group_label=f"Subject {subject_id}",
        color_metric=color_metric,
        compare_fft=compare_fft,
    )
    _, cbar_label, color_desc = COLOR_METRICS[color_metric]
    for ax, sm in zip(col_axes, sms):
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label(
            f"{cbar_label} (arb. units)"
        )
    fig.suptitle(
        f"Spatial handwriting: raw vs filtered — subject {subject_id} ({color_desc})",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pick_control_and_dysgraphic(
    control_id: int | None,
    dys_id: int | None,
) -> tuple[tuple[Path, int], tuple[Path, int]]:
    """Return ((control_path, id), (dys_path, id))."""
    df = build_sample_table().sort_values("subject_id")
    ctrl_df = df[df["label"] == 0]
    dys_df = df[df["label"] == 1]
    if ctrl_df.empty or dys_df.empty:
        raise RuntimeError("Need at least one control and one dysgraphic Task 7 sample.")

    if control_id is not None:
        row = ctrl_df[ctrl_df["subject_id"] == control_id]
        if row.empty:
            raise ValueError(f"Control subject_id={control_id} not found.")
        c_path, c_id = Path(row.iloc[0]["filepath"]), int(control_id)
    else:
        c_path = Path(ctrl_df.iloc[0]["filepath"])
        c_id = int(ctrl_df.iloc[0]["subject_id"])

    if dys_id is not None:
        row = dys_df[dys_df["subject_id"] == dys_id]
        if row.empty:
            raise ValueError(f"Dysgraphic subject_id={dys_id} not found.")
        d_path, d_id = Path(row.iloc[0]["filepath"]), int(dys_id)
    else:
        d_path = Path(dys_df.iloc[0]["filepath"])
        d_id = int(dys_df.iloc[0]["subject_id"])

    return (c_path, c_id), (d_path, d_id)


def save_control_dysgraphic_paper_pair(
    control_svc: np.ndarray,
    dys_svc: np.ndarray,
    *,
    control_id: int,
    dys_id: int,
    color_metric: str,
    out_path: Path,
    compare_fft: bool,
    include_residual: bool = False,
    include_true_residual: bool = True,
    residual_amplify: float = 1.0,
    residual_auto_amplify: bool = True,
    signal_percentile: float = 95.0,
    residual_percentile: float = 99.0,
) -> None:
    """Side-by-side spatial handwriting: control (left) vs dysgraphic (right)."""
    nrows = len(
        spatial_panel_specs(
            control_svc,
            color_metric,
            compare_fft=compare_fft,
            include_residual=include_residual,
            include_true_residual=include_true_residual and include_residual,
        )
    )
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3.2 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    signal_lim, residual_amp_lim, effective_amp, true_residual_lim = (
        collect_signal_and_residual_limits(
            control_svc,
            dys_svc,
            color_metric=color_metric,
            compare_fft=compare_fft,
            include_residual=include_residual,
            signal_percentile=signal_percentile,
            residual_percentile=residual_percentile,
            residual_amplify=residual_amplify,
            residual_auto_amplify=residual_auto_amplify,
        )
    )

    draw_kw = dict(
        color_metric=color_metric,
        compare_fft=compare_fft,
        include_residual=include_residual,
        include_true_residual=include_true_residual and include_residual,
        signal_lim=signal_lim,
        residual_amp_lim=residual_amp_lim,
        true_residual_lim=true_residual_lim,
        residual_amplify=effective_amp,
    )
    sms_c = _draw_spatial_column(
        list(axes[:, 0]),
        control_svc,
        group_label=f"Control (u{control_id:05d})",
        **draw_kw,
    )
    sms_d = _draw_spatial_column(
        list(axes[:, 1]),
        dys_svc,
        group_label=f"Dysgraphic (u{dys_id:05d})",
        **draw_kw,
    )
    for row_ax, sm_c, sm_d in zip(axes, sms_c, sms_d):
        fig.colorbar(sm_c, ax=row_ax[0], fraction=0.046, pad=0.03)
        fig.colorbar(sm_d, ax=row_ax[1], fraction=0.046, pad=0.03)

    _, _cbar_label, color_desc = COLOR_METRICS[color_metric]
    if include_residual:
        res_rows = "Raw-BW and Raw-FFT" if compare_fft else "Raw-BW"
        amp_lbl = _format_amplify(effective_amp)
        true_note = (
            " Bottom rows: true |Δ| at physical scale (gain x1)."
            if include_true_residual
            else ""
        )
        residual_note = (
            f" Jerk rows +/-{signal_lim:.2e}. Amplified residual rows ({res_rows}):"
            f" color = Δx{amp_lbl} (same bar limits as jerk; stronger color does not mean"
            f" larger physics).{true_note}"
        )
    else:
        residual_note = ""
    fig.suptitle(
        f"Control vs dysgraphic — {color_desc} on handwriting.{residual_note}",
        fontsize=9.5,
        y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_compare_groups(
    *,
    out_dir: Path,
    color_metrics: list[str],
    compare_fft: bool,
    control_id: int | None,
    dys_id: int | None,
    include_residual: bool = False,
    include_true_residual: bool = True,
    edge_trim: int = 30,
    residual_amplify: float = 1.0,
    residual_auto_amplify: bool = True,
    signal_percentile: float = 95.0,
    residual_percentile: float = 99.0,
) -> None:
    (c_path, c_id), (d_path, d_id) = pick_control_and_dysgraphic(control_id, dys_id)
    control_svc = load_raw_timeseries(str(c_path))
    dys_svc = load_raw_timeseries(str(d_path))
    if edge_trim > 0:
        control_svc = trim_svc_edges(control_svc, edge_trim)
        dys_svc = trim_svc_edges(dys_svc, edge_trim)

    fft_tag = "_raw_bw_fft" if compare_fft else "_raw_vs_butterworth"
    res_tag = "_with_residual"
    if include_residual:
        res_tag += "_true_scale" if include_true_residual else ""
    else:
        res_tag = ""
    for metric in color_metrics:
        if metric not in COLOR_METRICS:
            raise ValueError(f"Unknown color metric {metric!r}")
        out_path = out_dir / (
            f"control_u{c_id:05d}_vs_dys_u{d_id:05d}_{metric}{fft_tag}{res_tag}_spatial.png"
        )
        save_control_dysgraphic_paper_pair(
            control_svc,
            dys_svc,
            control_id=c_id,
            dys_id=d_id,
            color_metric=metric,
            out_path=out_path,
            compare_fft=compare_fft,
            include_residual=include_residual,
            include_true_residual=include_true_residual,
            residual_amplify=residual_amplify,
            residual_auto_amplify=residual_auto_amplify,
            signal_percentile=signal_percentile,
            residual_percentile=residual_percentile,
        )
        if include_residual:
            _sl, _, _amp, _true = collect_signal_and_residual_limits(
                control_svc,
                dys_svc,
                color_metric=metric,
                compare_fft=compare_fft,
                include_residual=True,
                signal_percentile=signal_percentile,
                residual_percentile=residual_percentile,
                residual_amplify=residual_amplify,
                residual_auto_amplify=residual_auto_amplify,
            )
            print(
                f"  {metric}: {out_path.name}  |  jerk_lim={_sl:.3e}  "
                f"true_|delta|_lim={_true:.3e}  display_gain=x{_format_amplify(_amp)}"
            )
        else:
            print(f"  {metric}: {out_path.name}")

    print(f"Control u{c_id:05d} | {c_path.name} | pen-on {int(pen_on_mask(control_svc).sum())}")
    print(f"Dysgraphic u{d_id:05d} | {d_path.name} | pen-on {int(pen_on_mask(dys_svc).sum())}")


def save_trajectory_overlay(
    raw_svc: np.ndarray,
    *,
    out_path: Path,
    subject_id: int,
    compare_fft: bool,
) -> None:
    """Single panel: faint raw XY under bold Butterworth (and optional FFT)."""
    x_raw, y_raw = xy_with_stroke_breaks(raw_svc)
    x_bw, y_bw = xy_with_stroke_breaks(butterworth_lowpass_xy(raw_svc))

    fig, ax = plt.subplots(figsize=(11, 8))
    plot_xy_strokes(ax, x_raw, y_raw, color="#c8c8c8", linewidth=1.0, alpha=0.55, label="Raw XY")
    plot_xy_strokes(
        ax,
        x_bw,
        y_bw,
        color="#1565c0",
        linewidth=2.0,
        alpha=0.92,
        label="Butterworth XY",
    )
    if compare_fft:
        x_fft, y_fft = xy_with_stroke_breaks(fft_lowpass_xy(raw_svc))
        plot_xy_strokes(
            ax,
            x_fft,
            y_fft,
            color="#ef6c00",
            linewidth=1.5,
            alpha=0.85,
            label="FFT XY",
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(
        f"Handwriting overlay — raw vs filtered (subject {subject_id})",
        fontsize=11,
    )
    ax.set_xlabel("X (tablet)")
    ax.set_ylabel("Y (tablet)")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def resolve_svc_path(subject_id: int | None, label: int | None) -> tuple[Path, int]:
    meta = load_metadata(Config.META_XLSX)
    candidates: list[tuple[Path, int]] = []
    for fp in sorted(glob.glob(str(TASK7_DATA_DIR / "*.svc"))):
        sid_str = subject_id_from_svc_path(fp)
        if sid_str is None:
            continue
        sid = int(sid_str)
        if subject_id is not None and sid != subject_id:
            continue
        row = meta.loc[meta["subject_id"] == sid]
        if row.empty:
            continue
        lab = int(row["label"].iloc[0])
        if label is not None and lab != label:
            continue
        candidates.append((Path(fp), sid))
    if not candidates:
        raise FileNotFoundError(
            f"No Task 7 .svc matched subject_id={subject_id!r} label={label!r} under {TASK7_DATA_DIR}"
        )
    return candidates[0]


def plot_series(ax, t: np.ndarray, raw: np.ndarray, bw: np.ndarray, fft: np.ndarray | None, title: str) -> None:
    ax.plot(t, raw, label="Raw", **RAW_STYLE)
    ax.plot(t, bw, label="12 Hz Butterworth", **BW_STYLE)
    if fft is not None:
        ax.plot(t, fft, label="12 Hz FFT", **FFT_STYLE)
    cap = y_limits_symmetric(raw, bw, fft if fft is not None else bw)
    ax.set_ylim(-cap, cap)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.22)
    ax.set_xlabel("Time", fontsize=8)


def build_signals(raw_svc: np.ndarray, *, compare_fft: bool) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    on = pen_on_mask(raw_svc)
    t = raw_svc[:, 2].astype(np.float64)
    t_rel = t - t[0]

    kin_raw = kinematics_tensor(raw_svc)
    kin_bw = kinematics_tensor(butterworth_lowpass_xy(raw_svc))
    kin_fft = kinematics_tensor(fft_lowpass_xy(raw_svc)) if compare_fft else None

    def extract(kin: np.ndarray) -> dict[str, np.ndarray]:
        return {name: mask_pen_off(kin[:, idx], on) for name, idx in CHANNEL_SPECS}

    return t_rel, extract(kin_raw), extract(kin_bw), extract(kin_fft) if kin_fft is not None else None


def save_dynamics_figure(
    t: np.ndarray,
    raw: dict[str, np.ndarray],
    bw: dict[str, np.ndarray],
    fft: dict[str, np.ndarray] | None,
    *,
    out_path: Path,
    subject_id: int,
    svc_name: str,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    axes_flat = axes.ravel()
    for ax, (name, _) in zip(axes_flat, CHANNEL_SPECS):
        plot_series(ax, t, raw[name], bw[name], fft[name] if fft else None, name)
    axes_flat[0].legend(loc="upper right", fontsize=7, framealpha=0.92)
    fig.suptitle(
        f"Raw vs filtered dynamics — subject {subject_id} ({svc_name})\n"
        "Gray = raw derivatives; blue = Butterworth XY then derivatives"
        + ("; orange = FFT XY then derivatives" if fft else ""),
        fontsize=11,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_jerk_spatial_compare(
    *,
    out_dir: Path,
    compare_fft: bool,
    control_id: int | None,
    dys_id: int | None,
    edge_trim: int = 30,
    include_true_residual: bool = True,
    residual_amplify: float = 1.0,
    residual_auto_amplify: bool = True,
    signal_percentile: float = 95.0,
    residual_percentile: float = 99.0,
) -> None:
    """Control vs dysgraphic handwriting colored by jerk + amplified residual rows."""
    print(
        "Generating spatial jerk figures (amplified residuals + optional true |delta| scale rows)."
    )
    run_compare_groups(
        out_dir=out_dir,
        color_metrics=["jerk_x", "jerk_y"],
        compare_fft=compare_fft,
        control_id=control_id,
        dys_id=dys_id,
        include_residual=True,
        include_true_residual=include_true_residual,
        edge_trim=edge_trim,
        residual_amplify=residual_amplify,
        residual_auto_amplify=residual_auto_amplify,
        signal_percentile=signal_percentile,
        residual_percentile=residual_percentile,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Raw vs Butterworth/FFT overlay plots for Task 7.")
    p.add_argument(
        "--plot",
        choices=("trajectory", "jerk", "dynamics", "all"),
        default="trajectory",
        help=(
            "trajectory = single-subject spatial overlays; "
            "jerk = control vs dysgraphic spatial jerk + residual row; "
            "dynamics = 8-panel time series; all = everything."
        ),
    )
    p.add_argument("--compare-fft", action="store_true", help="Overlay FFT filtered trace (orange).")
    p.add_argument("--subject-id", type=int, default=None, help="Subject ID (default: first dysgraphic).")
    p.add_argument("--label", type=int, default=None, help="Filter metadata label (0=control, 1=dysgraphic).")
    p.add_argument("--out-dir", type=Path, default=_PKG_DIR / "filter_overlay_outputs")
    p.add_argument("--edge-trim", type=int, default=30, help="Trim samples at clip edges.")
    p.add_argument("--light-theme", action="store_true", help="White background (default; use for reports).")
    p.add_argument(
        "--compare-groups",
        action="store_true",
        help="One control + one dysgraphic, side-by-side paper-style figures.",
    )
    p.add_argument(
        "--color-metrics",
        default="vx,jerk_x,jerk_y",
        help="Comma-separated raw-trace colors: vx, vy, jerk_x, jerk_y (with --compare-groups).",
    )
    p.add_argument("--control-id", type=int, default=None, help="Control subject ID for --compare-groups.")
    p.add_argument("--dys-id", type=int, default=None, help="Dysgraphic subject ID for --compare-groups.")
    p.add_argument(
        "--find-noisiest",
        action="store_true",
        help=(
            "Scan all Task 7 samples for max Jerk_X residual variance (raw − Butterworth), "
            "print top 3, and plot the noisiest subject."
        ),
    )
    p.add_argument(
        "--residual-amplify",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Extra multiply on residuals before coloring (on top of auto gain). "
            "Default auto gain targets the full shared jerk color range."
        ),
    )
    p.add_argument(
        "--no-residual-auto-amplify",
        action="store_true",
        help="Disable automatic residual gain so only --residual-amplify applies.",
    )
    p.add_argument(
        "--residual-percentile",
        type=float,
        default=99.0,
        help="Percentile for residual color limits (lower = more saturated noise). Default 99.",
    )
    p.add_argument(
        "--signal-percentile",
        type=float,
        default=95.0,
        help="Percentile for raw/filtered jerk color limits. Default 95.",
    )
    p.add_argument(
        "--no-true-residual-row",
        action="store_true",
        help="Omit bottom rows that show true |Δ| at physical scale (gain x1).",
    )
    args = p.parse_args()
    residual_kw = dict(
        residual_amplify=args.residual_amplify,
        residual_auto_amplify=not args.no_residual_auto_amplify,
        signal_percentile=args.signal_percentile,
        residual_percentile=args.residual_percentile,
        include_true_residual=not args.no_true_residual_row,
    )

    if args.find_noisiest:
        ranked = scan_noisiest_subjects(top_k=3, axis="X")
        if not ranked:
            p.error("--find-noisiest: no valid samples in build_sample_table().")
        print_noisiest_ranking(ranked, axis="X")
        args.dys_id = ranked[0][0]
        if args.plot in ("trajectory", "jerk"):
            args.plot = "jerk"
        print(f"\nUsing noisiest dysgraphic subject_id={args.dys_id:05d} (--plot {args.plot}).\n")

    if args.plot == "jerk":
        run_jerk_spatial_compare(
            out_dir=args.out_dir,
            compare_fft=args.compare_fft,
            control_id=args.control_id,
            dys_id=args.dys_id,
            edge_trim=args.edge_trim,
            **residual_kw,
        )
        print(f"Saved figures under {args.out_dir.resolve()}")
        return

    if args.compare_groups:
        metrics = [m.strip().lower() for m in args.color_metrics.split(",") if m.strip()]
        if not metrics:
            p.error("--color-metrics must list at least one of: vx, vy, jerk_x, jerk_y")
        include_residual = any(m.startswith("jerk") for m in metrics)
        print(f"Generating control vs dysgraphic figures for: {', '.join(metrics)}")
        run_compare_groups(
            out_dir=args.out_dir,
            color_metrics=metrics,
            compare_fft=args.compare_fft,
            control_id=args.control_id,
            dys_id=args.dys_id,
            include_residual=include_residual,
            include_true_residual=(
                residual_kw["include_true_residual"] if include_residual else False
            ),
            edge_trim=args.edge_trim,
            residual_amplify=residual_kw["residual_amplify"],
            residual_auto_amplify=residual_kw["residual_auto_amplify"],
            signal_percentile=residual_kw["signal_percentile"],
            residual_percentile=residual_kw["residual_percentile"],
        )
        print(f"Saved figures under {args.out_dir.resolve()}")
        return

    sid_default = args.subject_id
    label_default = args.label if args.label is not None else (None if sid_default else 1)
    svc_path, subject_id = resolve_svc_path(sid_default, label_default)

    raw_svc = load_raw_timeseries(str(svc_path))
    if raw_svc.shape[1] < 7:
        raise ValueError(f"Expected 7 channels in {svc_path}, got {raw_svc.shape[1]}")
    if not np.any(pen_on_mask(raw_svc)):
        raise RuntimeError(f"No pen-on samples in {svc_path.name}")

    need_timeseries = args.plot in ("dynamics", "all")
    sig_raw = sig_bw = sig_fft = None
    t = np.array([])
    if need_timeseries:
        t, sig_raw, sig_bw, sig_fft = build_signals(raw_svc, compare_fft=args.compare_fft)
        if args.edge_trim > 0:
            names = [n for n, _ in CHANNEL_SPECS]
            arrays = [sig_raw[n] for n in names] + [sig_bw[n] for n in names]
            if sig_fft:
                arrays += [sig_fft[n] for n in names]
            t, *masked = trim_edges(t, *arrays, n=args.edge_trim)
            n_ch = len(names)
            for i, name in enumerate(names):
                sig_raw[name] = masked[i]
                sig_bw[name] = masked[n_ch + i]
            if sig_fft:
                for i, name in enumerate(names):
                    sig_fft[name] = masked[2 * n_ch + i]

    out_dir = args.out_dir
    tag = f"subject_{subject_id:05d}"
    if args.compare_fft:
        tag += "_raw_bw_fft"
    else:
        tag += "_raw_vs_butterworth"

    if args.plot in ("trajectory", "all"):
        save_trajectory_paper_style(
            raw_svc,
            out_path=out_dir / f"{tag}_trajectory_paper_style.png",
            subject_id=subject_id,
            compare_fft=args.compare_fft,
            color_metric="vy",
        )
        save_trajectory_overlay(
            raw_svc,
            out_path=out_dir / f"{tag}_trajectory_overlay.png",
            subject_id=subject_id,
            compare_fft=args.compare_fft,
        )

    if args.plot == "all":
        run_jerk_spatial_compare(
            out_dir=out_dir,
            compare_fft=args.compare_fft,
            control_id=args.control_id,
            dys_id=args.dys_id,
            edge_trim=args.edge_trim,
            **residual_kw,
        )
    if args.plot in ("dynamics", "all"):
        save_dynamics_figure(
            t,
            sig_raw,
            sig_bw,
            sig_fft,
            out_path=out_dir / f"{tag}_dynamics_8panel.png",
            subject_id=subject_id,
            svc_name=svc_path.name,
        )

    print(f"Subject {subject_id} | {svc_path.name} | pen-on: {int(pen_on_mask(raw_svc).sum())}")
    print(f"Saved figures under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
