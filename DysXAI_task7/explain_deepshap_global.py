"""
Global DeepSHAP explainability for Task 7 (Baseline + Age + Gender, FFT late-fusion).

Uses the holdout model from ``train_final_evaluation.py`` (no retraining):
``checkpoints/holdout_fft_age_gender.pt``. That file is written when you run the
final evaluation with age + gender and FFT; scaling/age bounds come from the
checkpoint metadata (80% tuning split).

OOF ablation checkpoints (``oof_fft_*_age_gender_foldNN.pt``) are per-fold models
for outer-test evaluation only — this script needs one model for all 120 clips, so
it does not use those by default.

Outputs (under ``XAI results/DeepSHAP global/``):

- Demographics-only and kinematics beeswarms (split x-axis scales)
- Full kinematics beeswarm (directional Vx/Vy/acc/jerk Pos/Neg + X/Y/pen channels)
- Kinematics with directional velocity rows (Vx+/Vx-, Vy+/Vy-) and speed-colored beeswarms
- Mean |SHAP| bar chart for global importance ranking
- Per-sample SHAP / feature CSVs

Examples::

    python DysXAI_task7/explain_deepshap_global.py
    python DysXAI_task7/explain_deepshap_global.py --checkpoint checkpoints/holdout_fft_age_gender.pt
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from checkpoint_io import (  # noqa: E402
    XAI_XY_FILTER,
    checkpoint_basename,
    checkpoint_search_roots,
    find_checkpoint_file,
    list_matching_checkpoints,
    load_task7_checkpoint,
)
from analyze_directional_velocity_predictions import (  # noqa: E402
    DIRECTION_ORDER as DIRECTIONAL_VELOCITY_NAMES,
    build_subject_directional_table,
)
from dataset import Task5TrajectoryDataset, build_sample_table, collate_task7  # noqa: E402
from train_ab_test import RANDOM_STATE, set_seed  # noqa: E402

FEATURE_NAMES = [
    "X",
    "Y",
    "Pressure",
    "Azimuth",
    "Tilt",
    "PenStatus",
    "Vx",
    "Vy",
    "Ax",
    "Ay",
    "Jx",
    "Jy",
    "Age",
    "Gender",
]
KINEMATIC_FEATURE_NAMES = FEATURE_NAMES[:12]
DEMOGRAPHIC_FEATURE_NAMES = ["Age", "Gender"]
N_KINEMATIC = len(KINEMATIC_FEATURE_NAMES)

DEFAULT_CONFIGURATION = "Baseline + Age + Gender"
DEFAULT_XY_FILTER = "fft"
DEFAULT_RESULTS_DIR = _HERE / "XAI results" / "DeepSHAP global"
DEFAULT_OUTPUT_DEMOGRAPHICS = (
    DEFAULT_RESULTS_DIR / "task7_deepshap_global_demographics_age_gender.png"
)
DEFAULT_OUTPUT_KINEMATICS_SIGNED = (
    DEFAULT_RESULTS_DIR / "task7_deepshap_global_kinematics_signed_vx_vy_age_gender.png"
)
DEFAULT_OUTPUT_KINEMATICS_SIGNED_BY_LABEL = (
    DEFAULT_RESULTS_DIR / "task7_deepshap_global_kinematics_signed_vx_vy_by_label.png"
)
DEFAULT_OUTPUT_KINEMATICS_DIRECTIONAL = (
    DEFAULT_RESULTS_DIR / "task7_deepshap_global_kinematics_directional_velocity_age_gender.png"
)
DEFAULT_OUTPUT_KINEMATICS_DIRECTIONAL_BY_LABEL = (
    DEFAULT_RESULTS_DIR
    / "task7_deepshap_global_kinematics_directional_velocity_by_label.png"
)
DEFAULT_OUTPUT_DIRECTIONAL_ONLY = (
    DEFAULT_RESULTS_DIR / "task7_deepshap_global_directional_velocity_only_age_gender.png"
)
DEFAULT_OUTPUT_DIRECTIONAL_ONLY_BY_LABEL = (
    DEFAULT_RESULTS_DIR / "task7_deepshap_global_directional_velocity_only_by_label.png"
)
DEFAULT_OUTPUT_MEAN_ABS_SHAP = (
    DEFAULT_RESULTS_DIR / "task7_deepshap_global_mean_abs_shap_age_gender.png"
)
VX_CHANNEL_INDEX = FEATURE_NAMES.index("Vx")
VY_CHANNEL_INDEX = FEATURE_NAMES.index("Vy")
AX_CHANNEL_INDEX = FEATURE_NAMES.index("Ax")
AY_CHANNEL_INDEX = FEATURE_NAMES.index("Ay")
JX_CHANNEL_INDEX = FEATURE_NAMES.index("Jx")
JY_CHANNEL_INDEX = FEATURE_NAMES.index("Jy")
KINEMATIC_BASE_NAMES = ["X", "Y", "Pressure", "Azimuth", "Tilt", "PenStatus"]
N_KINEMATIC_BASE = len(KINEMATIC_BASE_NAMES)
DIRECTIONAL_DERIV_CHANNELS: tuple[tuple[str, int], ...] = (
    ("Vx", VX_CHANNEL_INDEX),
    ("Vy", VY_CHANNEL_INDEX),
    ("Ax", AX_CHANNEL_INDEX),
    ("Ay", AY_CHANNEL_INDEX),
    ("Jx", JX_CHANNEL_INDEX),
    ("Jy", JY_CHANNEL_INDEX),
)
KINEMATIC_WITH_DIRECTIONAL_NAMES = [
    "X",
    "Y",
    "Pressure",
    "Azimuth",
    "Tilt",
    "PenStatus",
    *list(DIRECTIONAL_VELOCITY_NAMES),
    "Ax",
    "Ay",
    "Jx",
    "Jy",
]
KINEMATIC_FULL_FEATURE_NAMES = [
    *KINEMATIC_BASE_NAMES,
    *list(DIRECTIONAL_VELOCITY_NAMES),
    "Ax_Pos",
    "Ax_Neg",
    "Ay_Pos",
    "Ay_Neg",
    "Jx_Pos",
    "Jx_Neg",
    "Jy_Pos",
    "Jy_Neg",
]
DEFAULT_OUTPUT_KINEMATICS_FULL = (
    DEFAULT_RESULTS_DIR / "task7_deepshap_global_kinematics_full_age_gender.png"
)
DEFAULT_OUTPUT_KINEMATICS_FULL_BY_LABEL = (
    DEFAULT_RESULTS_DIR / "task7_deepshap_global_kinematics_full_by_label.png"
)
MIN_DIRECTIONAL_TIMESTEPS = 5
N_BACKGROUND = 100

# Match population velocity plots (analyze_directional_velocity_age.py).
LABEL_NAMES = {0: "Control", 1: "Dysgraphic"}
LABEL_COLORS = {0: "#1f77b4", 1: "#d62728"}

DEMOGRAPHICS_SUMMARY_TITLE = "Global DeepSHAP — demographics (mean-pooled SHAP)"
DEMOGRAPHICS_SUMMARY_CAPTION = (
    "One dot per subject. Positive SHAP → higher dysgraphic logit."
)
KINEMATICS_SIGNED_TITLE = "Global DeepSHAP — signed Vx/Vy (reference; mean-pooled z-score)"
KINEMATICS_SIGNED_CAPTION = (
    "Reference panel using raw model channels. Color = signed training z-score, so red is "
    "not speed magnitude. Dysgraphic subjects have slightly higher signed mean Vx/Vy in model "
    "space (pen-up / dt artifacts), which can invert the clinical speed story — use the "
    "directional-velocity panels for interpretation."
)
KINEMATICS_SIGNED_BY_LABEL_TITLE = "Global DeepSHAP — signed Vx/Vy by diagnostic group"
KINEMATICS_SIGNED_BY_LABEL_CAPTION = (
    "Same signed mean-pooled SHAP as the reference panel. Blue = Control, red = Dysgraphic."
)
DIRECTIONAL_VELOCITY_TITLE = "Global DeepSHAP — directional velocity (speed magnitude)"
DIRECTIONAL_VELOCITY_CAPTION = (
    "Vx/Vy SHAP mean-pooled over timesteps where signed model Vx/Vy is >0 (Pos) or <0 (Neg). "
    "Color = clinical mean speed magnitude (FFT 12 Hz, pen-on, robust dt) — red = faster. "
    "Positive SHAP → higher dysgraphic logit. Expect faster (red) → left/negative SHAP."
)
DIRECTIONAL_VELOCITY_BY_LABEL_CAPTION = (
    "Same directional SHAP as the speed-colored panel. Blue = Control, red = Dysgraphic."
)
KINEMATICS_DIRECTIONAL_TITLE = (
    "Global DeepSHAP — kinematics (directional Vx/Vy + other channels)"
)
KINEMATICS_FULL_TITLE = "Global DeepSHAP — full kinematics (directional velocity, acc, jerk)"
KINEMATICS_FULL_CAPTION = (
    "Pos/Neg rows pool SHAP over timesteps where the model channel is >0 or <0. "
    "Vx/Vy color = clinical mean speed (FFT 12 Hz, pen-on); acc/jerk color = mean |z-score| "
    "on those timesteps; X–PenStatus = mean-pooled model z-score. Colors are cohort z-scored "
    "per row for one shared scale. Positive SHAP → higher dysgraphic logit. "
    "Age and Gender are shown in a separate demographics panel."
)
KINEMATICS_FULL_BY_LABEL_CAPTION = (
    "Same full kinematic SHAP as the feature-value panel. Blue = Control, red = Dysgraphic."
)
MEAN_ABS_SHAP_TITLE = "Global DeepSHAP — mean |SHAP| per feature (ranking)"
MEAN_ABS_SHAP_CAPTION = (
    "Mean absolute SHAP across subjects (mean-pooled over timesteps for kinematics). "
    "Ranking metric for time-series features; signed beeswarms can cancel across time."
)


def holdout_age_gender_basename(xy_filter: str = DEFAULT_XY_FILTER) -> str:
    """Filename written by ``train_final_evaluation.py`` for FFT + age + gender."""
    return checkpoint_basename(
        split="holdout",
        xy_filter=xy_filter,
        use_age=True,
        use_gender=True,
    )


def resolve_checkpoint_path(
    pkg_dir: Path,
    user_path: Path | None,
    *,
    checkpoint_dir: Path | None,
) -> Path:
    """
    Resolve checkpoint path: explicit file, nested search under ``--checkpoint-dir``,
    then ``checkpoints/`` and package root.
    """
    search_roots = checkpoint_search_roots(pkg_dir, checkpoint_dir)
    target_name = holdout_age_gender_basename()

    if user_path is not None:
        explicit = user_path if user_path.is_absolute() else pkg_dir / user_path
        if explicit.is_file():
            return explicit.resolve()
        if explicit.suffix != ".pt":
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        found = find_checkpoint_file(explicit.name, search_roots)
        if found is not None:
            return found
        raise FileNotFoundError(
            f"Checkpoint not found: {explicit}\nSearched under:\n"
            + "\n".join(f"  - {r.resolve()}" for r in search_roots if r.exists())
        )

    found = find_checkpoint_file(target_name, search_roots)
    if found is not None:
        return found

    candidates = list_matching_checkpoints(
        search_roots,
        "holdout_fft_age_gender.pt",
        limit=5,
    )
    candidates += list_matching_checkpoints(
        search_roots,
        "holdout_fft_*_age_gender.pt",
        limit=10,
    )
    msg_lines = [
        f"Could not find {target_name!r} under any search root.",
        "Expected from train_final_evaluation.py (Baseline + Age + Gender, FFT holdout).",
        "Search roots:",
    ]
    for root in search_roots:
        status = "exists" if root.exists() else "missing"
        msg_lines.append(f"  - {root.resolve()} ({status})")
    if candidates:
        msg_lines.append("Closest matches found:")
        for p in candidates[:15]:
            try:
                rel = p.relative_to(pkg_dir.resolve())
            except ValueError:
                rel = p
            msg_lines.append(f"  - {rel}")
        msg_lines.append(
            f"Try: python explain_deepshap_global.py --checkpoint {candidates[0]}"
        )
    raise FileNotFoundError("\n".join(msg_lines))


class SHAPWrapper(nn.Module):
    """Expose (kinematics, age, gender) as positional args for shap.DeepExplainer."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        kinematics: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(kinematics, lengths=None, age=age, gender=gender)


def _normalize_age(age: torch.Tensor, age_min: float, age_max: float) -> torch.Tensor:
    denom = (age_max - age_min) + 1e-8
    return (age.float() - age_min) / denom


def _unpack_three_input_shap(shap_values: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (shap_kinematics, shap_age, shap_gender) from DeepExplainer output."""
    if isinstance(shap_values, list) and len(shap_values) == 3:
        if all(isinstance(v, np.ndarray) for v in shap_values):
            kin, age, gender = shap_values
            return np.asarray(kin), np.asarray(age), np.asarray(gender)
        if isinstance(shap_values[0], list) and len(shap_values[0]) == 3:
            kin, age, gender = shap_values[0]
            return np.asarray(kin), np.asarray(age), np.asarray(gender)

    if isinstance(shap_values, np.ndarray) and shap_values.ndim >= 2:
        raise TypeError(
            "Got a single stacked SHAP array; expected one array per input "
            "(kinematics, age, gender)."
        )

    raise TypeError(
        "Unexpected shap_values structure; expected a list of three arrays "
        "(kinematics, age, gender) or a per-class wrapper around that list."
    )


def _kinematic_shap_to_2d(shap_kin: np.ndarray, *, n_channels: int = 12) -> np.ndarray:
    """
    Collapse kinematic SHAP (N, C, T[, 1]) to (N, C) for summary_plot.

    Mean-pools over time so pooling matches ``_kinematic_features_to_2d``.
    SHAP 0.52 stacks single-output attributions with an extra trailing axis.
    """
    arr = np.squeeze(np.asarray(shap_kin, dtype=np.float64))
    if arr.ndim == 2:
        if arr.shape[1] == n_channels:
            return arr
        if arr.shape[0] == n_channels:
            return arr.T
        raise ValueError(f"Expected {n_channels} kinematic channels, got shape {arr.shape}")
    if arr.ndim >= 3:
        n = arr.shape[0]
        if arr.shape[1] != n_channels:
            raise ValueError(f"Expected {n_channels} kinematic channels, got shape {arr.shape}")
        return arr.reshape(n, n_channels, -1).mean(axis=-1)
    raise ValueError(f"Cannot reduce kinematic SHAP with shape {arr.shape}")


def _kinematic_features_to_2d(x_kin: np.ndarray, *, n_channels: int = 12) -> np.ndarray:
    """Mean-pool kinematic inputs (N, C, T) -> (N, C)."""
    arr = np.squeeze(np.asarray(x_kin, dtype=np.float64))
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[1] == n_channels:
        return arr.mean(axis=-1)
    raise ValueError(f"Cannot reduce kinematic features with shape {arr.shape}")


def _kinematic_array_to_3d(arr: np.ndarray, *, n_channels: int = 12) -> np.ndarray:
    """Normalize kinematic SHAP or features to (N, C, T)."""
    data = np.squeeze(np.asarray(arr, dtype=np.float64))
    if data.ndim == 3 and data.shape[1] == n_channels:
        return data
    if data.ndim == 3 and data.shape[2] == n_channels:
        return np.transpose(data, (0, 2, 1))
    raise ValueError(f"Expected (N, C, T) kinematic array, got shape {data.shape}")


def _clinical_speeds_for_subjects(subject_ids: np.ndarray) -> np.ndarray:
    """(N, 4) mean speed magnitudes in DIRECTIONAL_VELOCITY_NAMES order."""
    speed_df = build_subject_directional_table()
    speed_df = speed_df.set_index("subject_id")
    ordered = speed_df.loc[np.asarray(subject_ids, dtype=np.int64), list(DIRECTIONAL_VELOCITY_NAMES)]
    return ordered.to_numpy(dtype=np.float64)


def _signed_timestep_mask(feat_ct: np.ndarray, *, positive: bool) -> np.ndarray:
    return feat_ct > 0.0 if positive else feat_ct < 0.0


def _mean_shap_on_signed_mask(
    shap_ct: np.ndarray,
    feat_ct: np.ndarray,
    *,
    positive: bool,
    min_points: int,
) -> float:
    """Mean channel SHAP on timesteps with signed z-scored value >0 or <0."""
    mask = _signed_timestep_mask(feat_ct, positive=positive)
    if int(mask.sum()) < min_points:
        return float("nan")
    return float(np.mean(shap_ct[mask]))


def _mean_feature_on_signed_mask(
    feat_ct: np.ndarray,
    *,
    positive: bool,
    min_points: int,
    magnitude: bool,
) -> float:
    """Mean feature value (optionally |value|) on signed timesteps."""
    mask = _signed_timestep_mask(feat_ct, positive=positive)
    if int(mask.sum()) < min_points:
        return float("nan")
    vals = feat_ct[mask]
    if magnitude:
        vals = np.abs(vals)
    return float(np.mean(vals))


def build_directional_derivative_splits(
    shap_kin: np.ndarray,
    x_kin: np.ndarray,
    clinical_speeds: np.ndarray,
    *,
    min_points: int = MIN_DIRECTIONAL_TIMESTEPS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Directional Pos/Neg SHAP and feature values for Vx, Vy, Ax, Ay, Jx, Jy.

    Vx/Vy colors use clinical speed magnitudes; acc/jerk use mean |z-score| on signed timesteps.
    """
    shap_3d = _kinematic_array_to_3d(shap_kin)
    feat_3d = _kinematic_array_to_3d(x_kin)
    n_samples = shap_3d.shape[0]
    if clinical_speeds.shape != (n_samples, len(DIRECTIONAL_VELOCITY_NAMES)):
        raise ValueError(
            f"Expected clinical speeds ({n_samples}, {len(DIRECTIONAL_VELOCITY_NAMES)}), "
            f"got {clinical_speeds.shape}"
        )

    speed_by_name = {
        name: clinical_speeds[:, idx] for idx, name in enumerate(DIRECTIONAL_VELOCITY_NAMES)
    }
    row_names: list[str] = []
    shap_cols: list[np.ndarray] = []
    feat_cols: list[np.ndarray] = []

    for prefix, ch_idx in DIRECTIONAL_DERIV_CHANNELS:
        for positive, suffix in ((True, "Pos"), (False, "Neg")):
            row_name = f"{prefix}_{suffix}"
            row_names.append(row_name)
            shap_col = np.full(n_samples, np.nan, dtype=np.float64)
            feat_col = np.full(n_samples, np.nan, dtype=np.float64)
            for i in range(n_samples):
                shap_col[i] = _mean_shap_on_signed_mask(
                    shap_3d[i, ch_idx],
                    feat_3d[i, ch_idx],
                    positive=positive,
                    min_points=min_points,
                )
                if prefix in ("Vx", "Vy"):
                    feat_col[i] = float(speed_by_name[row_name][i])
                else:
                    feat_col[i] = _mean_feature_on_signed_mask(
                        feat_3d[i, ch_idx],
                        positive=positive,
                        min_points=min_points,
                        magnitude=True,
                    )
            shap_cols.append(shap_col)
            feat_cols.append(feat_col)

    return np.column_stack(shap_cols), np.column_stack(feat_cols), row_names


def build_directional_velocity_shap_panel(
    shap_kin: np.ndarray,
    x_kin: np.ndarray,
    clinical_speeds: np.ndarray,
    *,
    min_points: int = MIN_DIRECTIONAL_TIMESTEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity-only slice (Vx_Pos, Vx_Neg, Vy_Pos, Vy_Neg) for standalone velocity panels."""
    shap_all, feat_all, names = build_directional_derivative_splits(
        shap_kin, x_kin, clinical_speeds, min_points=min_points
    )
    vel_idx = [names.index(name) for name in DIRECTIONAL_VELOCITY_NAMES]
    return shap_all[:, vel_idx], feat_all[:, vel_idx]


def _replace_vx_vy_with_directional(
    shap_kin_2d: np.ndarray,
    feat_kin_2d: np.ndarray,
    shap_dir: np.ndarray,
    feat_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert four directional velocity rows where Vx/Vy sat in the 12-channel kinematic block."""
    drop = [VX_CHANNEL_INDEX, VY_CHANNEL_INDEX]
    shap_base = np.delete(shap_kin_2d, drop, axis=1)
    feat_base = np.delete(feat_kin_2d, drop, axis=1)
    insert_at = VX_CHANNEL_INDEX
    shap_out = np.hstack([shap_base[:, :insert_at], shap_dir, shap_base[:, insert_at:]])
    feat_out = np.hstack([feat_base[:, :insert_at], feat_dir, feat_base[:, insert_at:]])
    return shap_out, feat_out


def build_directional_full_kinematics_panel(
    shap_kin_2d: np.ndarray,
    feat_kin_2d: np.ndarray,
    shap_splits: np.ndarray,
    feat_splits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Base channels (X–PenStatus) plus directional Vx/Vy/Ax/Ay/Jx/Jy (Pos/Neg each).

    No signed mean-pooled Vx/Vy/Ax/Ay/Jx/Jy rows — only directional splits.
    """
    shap_out = np.hstack([shap_kin_2d[:, :N_KINEMATIC_BASE], shap_splits])
    feat_out = np.hstack([feat_kin_2d[:, :N_KINEMATIC_BASE], feat_splits])
    if shap_out.shape[1] != len(KINEMATIC_FULL_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(KINEMATIC_FULL_FEATURE_NAMES)} kinematic columns, got {shap_out.shape[1]}"
        )
    return shap_out, feat_out


def _zscore_columns_for_plot(features: np.ndarray) -> np.ndarray:
    """Per-column cohort z-score so one beeswarm color scale works across mixed units."""
    feat = np.asarray(features, dtype=np.float64)
    mu = np.nanmean(feat, axis=0)
    sd = np.nanstd(feat, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((feat - mu) / sd).astype(np.float64)


def _nan_shap_for_plot(shap_values: np.ndarray) -> np.ndarray:
    """Replace NaN directional SHAP (too few timesteps) with 0 for beeswarm plotting."""
    out = np.asarray(shap_values, dtype=np.float64).copy()
    n_nan = int(np.isnan(out).sum())
    if n_nan:
        print(f"Directional SHAP: {n_nan} cells NaN (too few timesteps) → 0 for plotting.")
    out[np.isnan(out)] = 0.0
    return out


def _to_column(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _default_csv_paths() -> tuple[Path, Path]:
    """Default per-sample SHAP / feature CSV paths under ``DEFAULT_RESULTS_DIR``."""
    shap_path = DEFAULT_RESULTS_DIR / "task7_deepshap_global_values_age_gender.csv"
    feat_path = DEFAULT_RESULTS_DIR / "task7_deepshap_global_features_age_gender.csv"
    return shap_path, feat_path


def _slice_feature_columns(
    shap_combined: np.ndarray,
    features_combined: np.ndarray,
    feature_names: list[str],
    subset: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    idx = [feature_names.index(name) for name in subset]
    return (
        shap_combined[:, idx],
        features_combined[:, idx],
        list(subset),
    )


def _add_figure_caption(fig: plt.Figure, caption: str, *, top: float = 0.92) -> None:
    fig.text(
        0.5,
        top,
        caption,
        ha="center",
        va="top",
        fontsize=8,
        wrap=True,
        transform=fig.transFigure,
    )


def _wrap_caption(caption: str, *, width: int = 108) -> str:
    return "\n".join(textwrap.wrap(caption, width=width))


def _save_caption_text(path: Path, *, title: str, caption: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{title}\n\n{_wrap_caption(caption)}\n", encoding="utf-8")
    return path.resolve()


def _finalize_beeswarm_figure(
    fig: plt.Figure,
    *,
    title: str,
    caption: str,
    caption_position: str = "top",
    caption_file: Path | None = None,
    plot_top: float = 0.93,
    plot_bottom: float = 0.14,
) -> None:
    """Move title/caption away from the beeswarm so labels do not overlap."""
    for ax in fig.axes:
        ax_title = ax.get_title()
        if ax_title:
            ax.set_title("")
    fig.suptitle(title, fontsize=11, y=0.985)

    if caption_position == "bottom":
        fig.subplots_adjust(top=plot_top, bottom=plot_bottom)
        fig.text(
            0.5,
            0.02,
            _wrap_caption(caption),
            ha="center",
            va="bottom",
            fontsize=7.5,
            transform=fig.transFigure,
        )
    elif caption_position == "top":
        fig.subplots_adjust(top=0.82, bottom=0.08)
        _add_figure_caption(fig, caption, top=0.80)

    if caption_file is not None:
        _save_caption_text(caption_file, title=title, caption=caption)


def save_mean_abs_shap_barplot(
    shap_combined: np.ndarray,
    *,
    feature_names: list[str],
    out_path: Path,
    demographic_names: list[str] | None = None,
    title: str = MEAN_ABS_SHAP_TITLE,
    caption: str = MEAN_ABS_SHAP_CAPTION,
) -> Path:
    """Horizontal bar chart of mean |SHAP| per feature (global importance ranking)."""
    demographic_names = demographic_names or list(DEMOGRAPHIC_FEATURE_NAMES)
    mean_abs = np.abs(shap_combined).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ordered_names = [feature_names[i] for i in order]
    ordered_vals = mean_abs[order]
    colors = [
        "#d62728" if name in demographic_names else "steelblue"
        for name in ordered_names
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(ordered_names))
    ax.barh(y_pos, ordered_vals, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title, pad=20)
    ax.grid(axis="x", alpha=0.3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="steelblue", alpha=0.85, label="Kinematics"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728", alpha=0.85, label="Demographics"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    fig.subplots_adjust(top=0.88)
    _add_figure_caption(fig, caption, top=0.84)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def save_summary_plot_feature_values(
    shap_combined: np.ndarray,
    features_combined: np.ndarray,
    *,
    feature_names: list[str],
    out_path: Path,
    title: str = KINEMATICS_DIRECTIONAL_TITLE,
    caption: str = DIRECTIONAL_VELOCITY_CAPTION,
    color_bar_label: str = "Feature value",
    figsize: tuple[float, float] = (10, 8),
    caption_position: str = "top",
    caption_file: Path | None = None,
    plot_bottom: float = 0.14,
    seed: int = RANDOM_STATE,
) -> Path:
    """Standard SHAP beeswarm colored by feature value (z-scored or speed magnitude)."""
    plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_combined,
        features_combined,
        feature_names=feature_names,
        title="",
        color_bar_label=color_bar_label,
        show=False,
        rng=np.random.default_rng(seed),
    )
    fig = plt.gcf()
    _finalize_beeswarm_figure(
        fig,
        title=title,
        caption=caption,
        caption_position=caption_position,
        caption_file=caption_file,
        plot_bottom=plot_bottom,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def save_summary_plot_by_label(
    shap_combined: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: list[str],
    out_path: Path,
    title: str = DIRECTIONAL_VELOCITY_TITLE,
    caption: str = DIRECTIONAL_VELOCITY_BY_LABEL_CAPTION,
    figsize: tuple[float, float] = (10, 8),
    caption_position: str = "top",
    caption_file: Path | None = None,
    plot_bottom: float = 0.14,
    seed: int = RANDOM_STATE,
) -> Path:
    """Beeswarm layout matching SHAP summary_plot, colored by true diagnostic group."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if shap_combined.shape[0] != labels.shape[0]:
        raise ValueError(
            f"SHAP rows ({shap_combined.shape[0]}) must match labels ({labels.shape[0]})"
        )

    rng = np.random.default_rng(seed)
    mean_abs = np.abs(shap_combined).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ordered_names = [feature_names[i] for i in order]
    n_features = len(ordered_names)

    fig, ax = plt.subplots(figsize=figsize)
    y_centers = np.arange(n_features)

    for rank, feat_idx in enumerate(order):
        y_base = float(n_features - 1 - rank)
        shap_row = shap_combined[:, feat_idx]
        jitter = rng.uniform(-0.35, 0.35, size=shap_row.shape[0])
        for label_val in (0, 1):
            mask = labels == label_val
            if not np.any(mask):
                continue
            ax.scatter(
                shap_row[mask],
                y_base + jitter[mask],
                c=LABEL_COLORS[label_val],
                s=12,
                alpha=0.75,
                linewidths=0,
                zorder=2,
            )

    ax.axvline(0.0, color="#999999", linewidth=0.8, zorder=1)
    ax.set_yticks(y_centers)
    ax.set_yticklabels(list(reversed(ordered_names)))
    ax.set_xlabel("SHAP value (impact on model output)")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[0], markersize=8, label="Control"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[1], markersize=8, label="Dysgraphic"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9)

    _finalize_beeswarm_figure(
        fig,
        title=title,
        caption=caption,
        caption_position=caption_position,
        caption_file=caption_file,
        plot_bottom=plot_bottom,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def _sample_metadata_table(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    *,
    n_rows: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": np.asarray(subject_ids, dtype=np.int64).reshape(-1)[:n_rows],
            "label": np.asarray(labels, dtype=np.int64).reshape(-1)[:n_rows],
        }
    )


def _feature_value_table(
    values: np.ndarray,
    feature_names: list[str],
    subject_ids: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    n_rows = values.shape[0]
    meta = _sample_metadata_table(subject_ids, labels, n_rows=n_rows)
    feat = pd.DataFrame(values, columns=feature_names)
    return pd.concat([meta, feat], axis=1)


def save_shap_tables(
    shap_combined: np.ndarray,
    features_combined: np.ndarray,
    *,
    feature_names: list[str],
    subject_ids: np.ndarray,
    labels: np.ndarray,
    shap_csv: Path,
    features_csv: Path,
) -> tuple[Path, Path]:
    """Write per-sample SHAP values and beeswarm feature values to CSV."""
    shap_df = _feature_value_table(
        shap_combined, feature_names, subject_ids, labels
    )
    feat_df = _feature_value_table(
        features_combined, feature_names, subject_ids, labels
    )
    shap_csv.parent.mkdir(parents=True, exist_ok=True)
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(shap_csv, index=False)
    feat_df.to_csv(features_csv, index=False)
    return shap_csv.resolve(), features_csv.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global DeepSHAP beeswarm for Task 7 (FFT + age + gender late-fusion).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Folder to search for .pt files (includes subfolders). "
            "Default: DysXAI_task7/checkpoints/ then package root."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Override checkpoint .pt path. Default: checkpoints/holdout_fft_age_gender.pt "
            "(from train_final_evaluation.py)."
        ),
    )
    parser.add_argument(
        "--output-demographics",
        type=Path,
        default=DEFAULT_OUTPUT_DEMOGRAPHICS,
        help="Demographics-only beeswarm PNG (Age + Gender).",
    )
    parser.add_argument(
        "--output-kinematics-signed",
        type=Path,
        default=DEFAULT_OUTPUT_KINEMATICS_SIGNED,
        help="Reference beeswarm with signed mean Vx/Vy (12 channels).",
    )
    parser.add_argument(
        "--output-kinematics-signed-by-label",
        type=Path,
        default=DEFAULT_OUTPUT_KINEMATICS_SIGNED_BY_LABEL,
        help="Signed Vx/Vy beeswarm colored by diagnostic group.",
    )
    parser.add_argument(
        "--output-kinematics-directional",
        type=Path,
        default=DEFAULT_OUTPUT_KINEMATICS_DIRECTIONAL,
        help="Kinematics beeswarm with Vx+/Vx-/Vy+/Vy- speed-colored rows.",
    )
    parser.add_argument(
        "--output-kinematics-directional-by-label",
        type=Path,
        default=DEFAULT_OUTPUT_KINEMATICS_DIRECTIONAL_BY_LABEL,
        help="Directional kinematics beeswarm colored by diagnostic group.",
    )
    parser.add_argument(
        "--output-directional-only",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTIONAL_ONLY,
        help="Four-row directional velocity beeswarm (speed-colored).",
    )
    parser.add_argument(
        "--output-directional-only-by-label",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTIONAL_ONLY_BY_LABEL,
        help="Four-row directional velocity beeswarm by diagnostic group.",
    )
    parser.add_argument(
        "--output-kinematics-full",
        type=Path,
        default=DEFAULT_OUTPUT_KINEMATICS_FULL,
        help="Full kinematics beeswarm: signed Vx/Vy + directional + X/Y/acc/jerk.",
    )
    parser.add_argument(
        "--output-kinematics-full-by-label",
        type=Path,
        default=DEFAULT_OUTPUT_KINEMATICS_FULL_BY_LABEL,
        help="Full kinematics beeswarm colored by diagnostic group.",
    )
    parser.add_argument(
        "--output-mean-abs-shap",
        type=Path,
        default=DEFAULT_OUTPUT_MEAN_ABS_SHAP,
        help="Mean |SHAP| bar chart for global feature ranking.",
    )
    parser.add_argument(
        "--shap-csv",
        type=Path,
        default=None,
        help=(
            "Per-sample SHAP values CSV (default: "
            "XAI results/DeepSHAP global/task7_deepshap_global_values_age_gender.csv)."
        ),
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=None,
        help=(
            "Per-sample feature values CSV for the beeswarm (default: "
            "XAI results/DeepSHAP global/task7_deepshap_global_features_age_gender.csv)."
        ),
    )
    parser.add_argument(
        "--n-background",
        type=int,
        default=N_BACKGROUND,
        help="Number of random background samples (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="RNG seed for background subsampling.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = None
    if args.checkpoint_dir is not None:
        ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else _HERE / args.checkpoint_dir
    user_ckpt = None
    if args.checkpoint is not None:
        user_ckpt = args.checkpoint if args.checkpoint.is_absolute() else _HERE / args.checkpoint
    ckpt_path = resolve_checkpoint_path(_HERE, user_ckpt, checkpoint_dir=ckpt_dir)
    model, meta = load_task7_checkpoint(ckpt_path, device)
    use_age = bool(meta["use_age"])
    use_gender = bool(meta["use_gender"])
    if not (use_age and use_gender):
        raise ValueError(
            f"Checkpoint must use late-fusion age and gender; got use_age={use_age}, "
            f"use_gender={use_gender} in {ckpt_path}"
        )

    xy_filter = str(meta.get("xy_filter", XAI_XY_FILTER))
    channel_mean = np.asarray(meta["channel_mean"], dtype=np.float32)
    channel_std = np.asarray(meta["channel_std"], dtype=np.float32)
    age_min = float(meta["age_min"])
    age_max = float(meta["age_max"])

    sample_df = build_sample_table()
    n_samples = len(sample_df)
    all_idx = np.arange(n_samples, dtype=np.int64)

    full_ds = Task5TrajectoryDataset(
        sample_df,
        all_idx,
        use_age_channel=True,
        channel_mean=channel_mean,
        channel_std=channel_std,
        xy_filter=xy_filter,
    )
    full_loader = DataLoader(
        full_ds,
        batch_size=n_samples,
        shuffle=False,
        collate_fn=collate_task7,
        num_workers=0,
    )
    batch = next(iter(full_loader))

    test_kinematics = batch["x"].to(device)
    test_age = _normalize_age(batch["age"].to(device), age_min, age_max)
    test_gender = batch["gender"].to(device).float()

    rng = np.random.default_rng(args.seed)
    n_bg = min(int(args.n_background), n_samples)
    bg_indices = rng.choice(n_samples, size=n_bg, replace=False)

    bg_kinematics = test_kinematics[bg_indices]
    bg_age = test_age[bg_indices]
    bg_gender = test_gender[bg_indices]

    wrapper_model = SHAPWrapper(model).to(device)
    wrapper_model.eval()

    print(f"Device: {device}")
    print(f"Configuration: {DEFAULT_CONFIGURATION}")
    try:
        ckpt_display = ckpt_path.relative_to(_HERE.resolve())
    except ValueError:
        ckpt_display = ckpt_path
    print(f"Checkpoint: {ckpt_display}")
    print(f"xy_filter: {xy_filter} | samples: {n_samples} | background: {n_bg}")
    print(f"Kinematics shape: {tuple(test_kinematics.shape)}")

    # SHAP PyTorch backend requires a list (not tuple) for multi-input models;
    # a tuple is wrapped as one argument and forward() only receives kinematics.
    background_data = [bg_kinematics, bg_age, bg_gender]
    test_data = [test_kinematics, test_age, test_gender]

    print("Building DeepExplainer (this may take several minutes)...")
    explainer = shap.DeepExplainer(wrapper_model, background_data)
    shap_values = explainer.shap_values(test_data)

    shap_kinematics, shap_age, shap_gender = _unpack_three_input_shap(shap_values)
    print(
        "SHAP shapes (raw): "
        f"kin={np.asarray(shap_kinematics).shape}, "
        f"age={np.asarray(shap_age).shape}, "
        f"gender={np.asarray(shap_gender).shape}"
    )

    shap_kinematics_2d = _kinematic_shap_to_2d(shap_kinematics)
    test_kinematics_2d = _kinematic_features_to_2d(test_kinematics.detach().cpu().numpy())

    shap_age_2d = _to_column(np.squeeze(np.asarray(shap_age)))
    shap_gender_2d = _to_column(np.squeeze(np.asarray(shap_gender)))
    test_age_2d = _to_column(test_age.detach().cpu().numpy())
    test_gender_2d = _to_column(test_gender.detach().cpu().numpy())

    shap_combined = np.hstack([shap_kinematics_2d, shap_age_2d, shap_gender_2d])
    features_combined = np.hstack([test_kinematics_2d, test_age_2d, test_gender_2d])

    if shap_combined.shape != (n_samples, len(FEATURE_NAMES)):
        raise ValueError(
            f"Expected SHAP matrix ({n_samples}, {len(FEATURE_NAMES)}), got {shap_combined.shape}"
        )

    def _resolve_out(path: Path) -> Path:
        return path if path.is_absolute() else _HERE / path

    out_demographics = _resolve_out(args.output_demographics)
    out_kin_signed = _resolve_out(args.output_kinematics_signed)
    out_kin_signed_by_label = _resolve_out(args.output_kinematics_signed_by_label)
    out_kin_directional = _resolve_out(args.output_kinematics_directional)
    out_kin_directional_by_label = _resolve_out(args.output_kinematics_directional_by_label)
    out_dir_only = _resolve_out(args.output_directional_only)
    out_dir_only_by_label = _resolve_out(args.output_directional_only_by_label)
    out_kin_full = _resolve_out(args.output_kinematics_full)
    out_kin_full_by_label = _resolve_out(args.output_kinematics_full_by_label)
    out_mean_abs_shap = _resolve_out(args.output_mean_abs_shap)

    default_shap_csv, default_features_csv = _default_csv_paths()
    if args.shap_csv is not None:
        shap_csv = args.shap_csv if args.shap_csv.is_absolute() else _HERE / args.shap_csv
    else:
        shap_csv = default_shap_csv
    if args.features_csv is not None:
        features_csv = (
            args.features_csv if args.features_csv.is_absolute() else _HERE / args.features_csv
        )
    else:
        features_csv = default_features_csv

    subject_ids = batch["subject_id"].cpu().numpy()
    labels = batch["y"].cpu().numpy().astype(np.int64)

    shap_csv_path, features_csv_path = save_shap_tables(
        shap_combined,
        features_combined,
        feature_names=FEATURE_NAMES,
        subject_ids=subject_ids,
        labels=labels,
        shap_csv=shap_csv,
        features_csv=features_csv,
    )

    shap_demo, feat_demo, demo_names = _slice_feature_columns(
        shap_combined, features_combined, FEATURE_NAMES, DEMOGRAPHIC_FEATURE_NAMES
    )
    shap_kin, feat_kin, kin_names = _slice_feature_columns(
        shap_combined, features_combined, FEATURE_NAMES, KINEMATIC_FEATURE_NAMES
    )

    print("Computing clinical directional speeds (FFT) for velocity-colored panels...")
    clinical_speeds = _clinical_speeds_for_subjects(subject_ids)
    x_kin_np = test_kinematics.detach().cpu().numpy()
    shap_splits, feat_splits, _split_names = build_directional_derivative_splits(
        shap_kinematics,
        x_kin_np,
        clinical_speeds,
    )
    shap_splits_plot = _nan_shap_for_plot(shap_splits)
    shap_dir, feat_dir = build_directional_velocity_shap_panel(
        shap_kinematics,
        x_kin_np,
        clinical_speeds,
    )
    shap_dir_plot = shap_splits_plot[:, : len(DIRECTIONAL_VELOCITY_NAMES)]
    shap_kin_dir, feat_kin_dir = _replace_vx_vy_with_directional(
        shap_kin_2d=shap_kinematics_2d,
        feat_kin_2d=test_kinematics_2d,
        shap_dir=shap_dir_plot,
        feat_dir=feat_dir,
    )
    shap_kin_full, feat_kin_full = build_directional_full_kinematics_panel(
        shap_kinematics_2d,
        test_kinematics_2d,
        shap_splits_plot,
        feat_splits,
    )
    feat_kin_full_plot = _zscore_columns_for_plot(feat_kin_full)

    full_shap_csv = DEFAULT_RESULTS_DIR / "task7_deepshap_global_kinematics_full_values_age_gender.csv"
    full_feat_csv = DEFAULT_RESULTS_DIR / "task7_deepshap_global_kinematics_full_features_age_gender.csv"
    save_shap_tables(
        shap_kin_full,
        feat_kin_full,
        feature_names=KINEMATIC_FULL_FEATURE_NAMES,
        subject_ids=subject_ids,
        labels=labels,
        shap_csv=full_shap_csv,
        features_csv=full_feat_csv,
    )

    dir_shap_csv = DEFAULT_RESULTS_DIR / "task7_deepshap_global_directional_velocity_values_age_gender.csv"
    dir_feat_csv = DEFAULT_RESULTS_DIR / "task7_deepshap_global_directional_velocity_features_age_gender.csv"
    save_shap_tables(
        shap_dir_plot,
        feat_dir,
        feature_names=list(DIRECTIONAL_VELOCITY_NAMES),
        subject_ids=subject_ids,
        labels=labels,
        shap_csv=dir_shap_csv,
        features_csv=dir_feat_csv,
    )

    demographics_path = save_summary_plot_feature_values(
        shap_demo,
        feat_demo,
        feature_names=demo_names,
        out_path=out_demographics,
        title=DEMOGRAPHICS_SUMMARY_TITLE,
        caption=DEMOGRAPHICS_SUMMARY_CAPTION,
        seed=int(args.seed),
    )
    kin_signed_path = save_summary_plot_feature_values(
        shap_kin,
        feat_kin,
        feature_names=kin_names,
        out_path=out_kin_signed,
        title=KINEMATICS_SIGNED_TITLE,
        caption=KINEMATICS_SIGNED_CAPTION,
        seed=int(args.seed),
    )
    kin_signed_by_label_path = save_summary_plot_by_label(
        shap_kin,
        labels,
        feature_names=kin_names,
        out_path=out_kin_signed_by_label,
        title=KINEMATICS_SIGNED_BY_LABEL_TITLE,
        caption=KINEMATICS_SIGNED_BY_LABEL_CAPTION,
        seed=int(args.seed),
    )
    dir_only_path = save_summary_plot_feature_values(
        shap_dir_plot,
        feat_dir,
        feature_names=list(DIRECTIONAL_VELOCITY_NAMES),
        out_path=out_dir_only,
        title=DIRECTIONAL_VELOCITY_TITLE,
        caption=DIRECTIONAL_VELOCITY_CAPTION,
        color_bar_label="Mean speed magnitude",
        seed=int(args.seed),
    )
    dir_only_by_label_path = save_summary_plot_by_label(
        shap_dir_plot,
        labels,
        feature_names=list(DIRECTIONAL_VELOCITY_NAMES),
        out_path=out_dir_only_by_label,
        title=DIRECTIONAL_VELOCITY_TITLE,
        caption=DIRECTIONAL_VELOCITY_BY_LABEL_CAPTION,
        seed=int(args.seed),
    )
    kin_directional_path = save_summary_plot_feature_values(
        shap_kin_dir,
        feat_kin_dir,
        feature_names=KINEMATIC_WITH_DIRECTIONAL_NAMES,
        out_path=out_kin_directional,
        title=KINEMATICS_DIRECTIONAL_TITLE,
        caption=DIRECTIONAL_VELOCITY_CAPTION,
        color_bar_label="Mean speed magnitude (directional rows only)",
        seed=int(args.seed),
    )
    kin_directional_by_label_path = save_summary_plot_by_label(
        shap_kin_dir,
        labels,
        feature_names=KINEMATIC_WITH_DIRECTIONAL_NAMES,
        out_path=out_kin_directional_by_label,
        title=KINEMATICS_DIRECTIONAL_TITLE,
        caption=DIRECTIONAL_VELOCITY_BY_LABEL_CAPTION,
        seed=int(args.seed),
    )
    full_caption_txt = out_kin_full.with_suffix(".txt")
    kin_full_path = save_summary_plot_feature_values(
        shap_kin_full,
        feat_kin_full_plot,
        feature_names=KINEMATIC_FULL_FEATURE_NAMES,
        out_path=out_kin_full,
        title=KINEMATICS_FULL_TITLE,
        caption=KINEMATICS_FULL_CAPTION,
        color_bar_label="Cohort z-score (per feature)",
        figsize=(10, 14),
        caption_position="bottom",
        caption_file=full_caption_txt,
        plot_bottom=0.17,
        seed=int(args.seed),
    )
    kin_full_by_label_path = save_summary_plot_by_label(
        shap_kin_full,
        labels,
        feature_names=KINEMATIC_FULL_FEATURE_NAMES,
        out_path=out_kin_full_by_label,
        title=KINEMATICS_FULL_TITLE,
        caption=KINEMATICS_FULL_BY_LABEL_CAPTION,
        figsize=(10, 14),
        caption_position="bottom",
        caption_file=out_kin_full_by_label.with_suffix(".txt"),
        plot_bottom=0.17,
        seed=int(args.seed),
    )
    mean_abs_path = save_mean_abs_shap_barplot(
        shap_combined,
        feature_names=FEATURE_NAMES,
        out_path=out_mean_abs_shap,
    )

    print(f"Saved demographics beeswarm: {demographics_path}")
    print(f"Saved signed Vx/Vy reference beeswarm: {kin_signed_path}")
    print(f"Saved signed Vx/Vy by-label beeswarm: {kin_signed_by_label_path}")
    print(f"Saved directional velocity only (speed-colored): {dir_only_path}")
    print(f"Saved directional velocity only by-label: {dir_only_by_label_path}")
    print(f"Saved kinematics + directional velocity: {kin_directional_path}")
    print(f"Saved kinematics + directional by-label: {kin_directional_by_label_path}")
    print(f"Saved full kinematics (kinematics only): {kin_full_path}")
    print(f"Saved full kinematics caption: {full_caption_txt.resolve()}")
    print(f"Saved full kinematics by-label: {kin_full_by_label_path}")
    print(f"Saved mean |SHAP| bar chart: {mean_abs_path}")
    print(f"Saved per-sample SHAP values: {shap_csv_path}")
    print(f"Saved per-sample feature values: {features_csv_path}")
    print(f"Saved directional SHAP CSV: {dir_shap_csv.resolve()}")
    print(f"Saved directional speed CSV: {dir_feat_csv.resolve()}")
    print(f"Saved full kinematics SHAP CSV: {full_shap_csv.resolve()}")
    print(f"Saved full kinematics feature CSV: {full_feat_csv.resolve()}")


if __name__ == "__main__":
    main()
