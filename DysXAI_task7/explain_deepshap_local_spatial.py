"""
Local spatial DeepSHAP for Task 7 (Baseline + Age + Gender, FFT late-fusion).

Selects eight extreme OOF cases (two per TP / TN / FP / FN), runs
shap.DeepExplainer, crops each clip to one complete ``hračkárstvo`` occurrence by default,
optionally the first letter ``h``, or an auto-detected loop segment, and saves one figure
per kinematic feature: a 4x2 grid (columns = case type, rows = two confident
examples). Pen-down ink is drawn with stroke-wise colored lines plus scatter;
near-zero SHAP maps to light grey (not white). Also saves a Jana-style Vx/Vy
figure on the same segments. Color scales are shared per feature across panels.

Examples::

    python DysXAI_task7/explain_deepshap_local_spatial.py
    python DysXAI_task7/explain_deepshap_local_spatial.py --segment-modes full_word
    python DysXAI_task7/explain_deepshap_local_spatial.py --segment-modes letter_h loop
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import pandas as pd
import shap
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from checkpoint_io import XAI_XY_FILTER, load_task7_checkpoint  # noqa: E402
from dataset import (  # noqa: E402
    Task5TrajectoryDataset,
    build_sample_table,
    collate_task7,
    load_processed_tensor,
    pad_truncate,
)
from explain_deepshap_global import (  # noqa: E402
    FEATURE_NAMES,
    SHAPWrapper,
    _normalize_age,
    _unpack_three_input_shap,
    resolve_checkpoint_path,
)
from train_ab_test import RANDOM_STATE, set_seed  # noqa: E402

DEFAULT_PREDICTIONS_CSV = (
    _HERE / "oof predictions" / "oof_predictions_fft_baseline_plus_age_plus_gender.csv"
)
DEFAULT_RESULTS_DIR = _HERE / "XAI results" / "DeepSHAP local"
N_BACKGROUND = 100
PROB_HI = 0.90
PROB_LO = 0.10
N_CASES_PER_CLASS = 2
CASE_COLUMN_KEYS = ("extreme_tp", "extreme_tn", "extreme_fp", "extreme_fn")
KINEMATIC_FEATURES = FEATURE_NAMES[:12]
N_KINEMATIC = len(KINEMATIC_FEATURES)
PRESSURE_IDX = 2
PEN_STATUS_IDX = 5
MIN_LOOP_POINTS = 30
MAX_LOOP_POINTS = 250
LOOP_PAD = 4
MAX_H_SEARCH_POINTS = 150
MIN_H_POINTS = 10
H_BASELINE_FRAC = 0.20
H_SLICE_PAD = 2
VX_IDX = 6
VY_IDX = 7
DEFAULT_SEGMENT_MODES = ("full_word", "letter_h")
SHAP_CONTRAST_PERCENTILE = 88.0
SHAP_ZERO_GREY = "#d1d5db"
SHAP_COLOR_GAMMA = 0.65
LINE_WIDTH_LOOP = 2.0
MIN_CLIP_LENGTH = 500
AXIS_PAD_FRAC = 0.08
WORD_GROUP_GAP = 120
WORD_GROUP_SPACE = 1200.0
WORD_GROUP_X_RESET = 800.0
MIN_WORD_POINTS = 280
MAX_WORD_POINTS = 1400
MIN_WORD_XSPAN = 2800.0
MAX_WORD_XSPAN = 12500.0
OVERSIZED_WORD_SPACE = 700.0
TIGHT_WORD_SPACE = 500.0
MIN_MAJOR_STROKE_POINTS = 100
MIN_MAJOR_STROKE_XSPAN = 1800.0


@dataclass(frozen=True)
class ExtremeCase:
    key: str
    title: str
    subject_id: int
    true_label: int
    pred_prob: float
    example_rank: int = 1


@dataclass(frozen=True)
class LoopSlice:
    start: int
    end: int

    @property
    def length(self) -> int:
        return int(self.end - self.start)


def shap_colormap() -> mcolors.Colormap:
    """Diverging blue-red with light grey at SHAP=0 (visible, not white)."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "shap_blue_red_grey",
        ["#2166ac", "#67a9cf", SHAP_ZERO_GREY, "#ef8a62", "#b2182b"],
        N=256,
    )
    cmap.set_bad(SHAP_ZERO_GREY)
    return cmap


def build_subject_sequence_lengths(sample_df: pd.DataFrame, *, xy_filter: str) -> dict[int, int]:
    """Model sequence length per subject after pad/truncate (for clip-quality filtering)."""
    lengths: dict[int, int] = {}
    for _, row in sample_df.iterrows():
        processed = load_processed_tensor(
            Path(row["filepath"]),
            age=row["age"],
            append_age_channel=False,
            xy_filter=xy_filter,
        )
        _, length = pad_truncate(processed)
        lengths[int(row["subject_id"])] = int(length)
    return lengths


def select_extreme_cases(
    predictions_df: pd.DataFrame,
    *,
    prob_hi: float = PROB_HI,
    prob_lo: float = PROB_LO,
    n_per_class: int = N_CASES_PER_CLASS,
    eligible_subject_ids: set[int] | None = None,
) -> list[ExtremeCase]:
    """Pick ``n_per_class`` subjects per TP / TN / FN / FP (row-major 4x2 grid)."""
    required = {"subject_id", "true_label", "pred_prob"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")

    df = predictions_df.copy()
    df["subject_id"] = df["subject_id"].astype(int)
    df["true_label"] = df["true_label"].astype(int)
    df["pred_prob"] = df["pred_prob"].astype(float)
    if eligible_subject_ids is not None:
        df = df[df["subject_id"].isin(eligible_subject_ids)].copy()

    specs = [
        (
            "extreme_tp",
            "Extreme True Positive",
            lambda d: d[(d["true_label"] == 1) & (d["pred_prob"] > prob_hi)].sort_values(
                "pred_prob", ascending=False
            ),
        ),
        (
            "extreme_tn",
            "Extreme True Negative",
            lambda d: d[(d["true_label"] == 0) & (d["pred_prob"] < prob_lo)].sort_values(
                "pred_prob", ascending=True
            ),
        ),
        (
            "extreme_fn",
            "Extreme False Negative",
            lambda d: d[(d["true_label"] == 1) & (d["pred_prob"] < prob_lo)].sort_values(
                "pred_prob", ascending=True
            ),
        ),
        (
            "extreme_fp",
            "Extreme False Positive",
            lambda d: d[(d["true_label"] == 0) & (d["pred_prob"] > prob_hi)].sort_values(
                "pred_prob", ascending=False
            ),
        ),
    ]

    if n_per_class < 1:
        raise ValueError(f"n_per_class must be >= 1, got {n_per_class}")

    relaxed_specs = [
        (
            "extreme_tp",
            lambda d: d[d["true_label"] == 1].sort_values("pred_prob", ascending=False),
        ),
        (
            "extreme_tn",
            lambda d: d[d["true_label"] == 0].sort_values("pred_prob", ascending=True),
        ),
        (
            "extreme_fn",
            lambda d: d[(d["true_label"] == 1) & (d["pred_prob"] < 0.5)].sort_values(
                "pred_prob", ascending=True
            ),
        ),
        (
            "extreme_fp",
            lambda d: d[(d["true_label"] == 0) & (d["pred_prob"] > 0.5)].sort_values(
                "pred_prob", ascending=False
            ),
        ),
    ]
    relaxed_by_key = {key: selector(df) for key, selector in relaxed_specs}

    picks_by_key: dict[str, list[ExtremeCase]] = {}
    for key, title, selector in specs:
        hits = selector(df)
        relaxed = relaxed_by_key[key]
        picked_rows: list[pd.Series] = []
        used_ids: set[int] = set()
        for pool in (hits, relaxed):
            for _, row in pool.iterrows():
                if len(picked_rows) >= n_per_class:
                    break
                sid = int(row["subject_id"])
                if sid in used_ids:
                    continue
                picked_rows.append(row)
                used_ids.add(sid)
            if len(picked_rows) >= n_per_class:
                break
        if len(picked_rows) < n_per_class:
            raise ValueError(
                f"Need {n_per_class} subjects for {title}, found {len(picked_rows)}."
            )
        if len(hits) < n_per_class:
            print(
                f"Warning: only {len(hits)} strict match(es) for {title}; "
                f"filling to {n_per_class} from next-most-extreme in class."
            )
        class_cases = [
            ExtremeCase(
                key=key,
                title=title,
                subject_id=int(row["subject_id"]),
                true_label=int(row["true_label"]),
                pred_prob=float(row["pred_prob"]),
                example_rank=rank + 1,
            )
            for rank, row in enumerate(picked_rows)
        ]
        picks_by_key[key] = class_cases

    cases: list[ExtremeCase] = []
    for rank in range(n_per_class):
        for key in CASE_COLUMN_KEYS:
            cases.append(picks_by_key[key][rank])
    return cases


def _subject_id_to_row_index(sample_df: pd.DataFrame, subject_id: int) -> int:
    hits = sample_df.index[sample_df["subject_id"] == subject_id].tolist()
    if not hits:
        raise ValueError(f"subject_id={subject_id} not found in Task 7 sample table.")
    if len(hits) > 1:
        raise ValueError(f"subject_id={subject_id} maps to multiple clips: {hits}")
    return int(hits[0])


def denormalize_xy(
    kinematics_np: np.ndarray,
    sub_idx: int,
    length: int,
    *,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover physical tablet coordinates for spatial plotting."""
    x_z = kinematics_np[sub_idx, 0, :length]
    y_z = kinematics_np[sub_idx, 1, :length]
    x = x_z * channel_std[0] + channel_mean[0]
    y = y_z * channel_std[1] + channel_mean[1]
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def pen_down_mask_from_processed(processed_t: np.ndarray, length: int) -> np.ndarray:
    """Pen-down mask from unstandardized 12-channel kinematics (T, C)."""
    pressure = processed_t[:length, PRESSURE_IDX]
    pen_status = processed_t[:length, PEN_STATUS_IDX]
    return ~((pen_status <= 0.0) & (pressure <= 0.0))


def load_pen_down_mask(
    filepath: Path,
    *,
    age: object,
    xy_filter: str,
    length: int,
) -> np.ndarray:
    processed = load_processed_tensor(
        filepath,
        age=age,
        append_age_channel=False,
        xy_filter=xy_filter,
    )
    padded, _ = pad_truncate(processed)
    return pen_down_mask_from_processed(padded, length)


def _pen_down_strokes(mask: np.ndarray, *, min_stroke_len: int) -> list[tuple[int, int]]:
    strokes: list[tuple[int, int]] = []
    start: int | None = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_stroke_len:
                strokes.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_stroke_len:
        strokes.append((start, len(mask)))
    return strokes


def find_first_letter_h_slice(
    kinematics_np: np.ndarray,
    sub_idx: int,
    length: int,
    *,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    pen_mask: np.ndarray,
    max_search_points: int = MAX_H_SEARCH_POINTS,
    min_h_points: int = MIN_H_POINTS,
    baseline_frac: float = H_BASELINE_FRAC,
    pad: int = H_SLICE_PAD,
) -> LoopSlice:
    """
    Crop to the first letter ``h`` at the start of ``hračkárstvo``.

    Task 7 is often one long pen-down stroke, so we only search the opening
    window: ascend to the first loop peak, then end when Y returns near the
    starting baseline (before the stroke continues into ``r``.
    """
    x, y = denormalize_xy(
        kinematics_np,
        sub_idx,
        length,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
    mask = np.asarray(pen_mask[:length], dtype=bool)
    strokes = _pen_down_strokes(mask, min_stroke_len=min_h_points)
    if not strokes:
        end = min(length, max_search_points)
        return LoopSlice(0, end)

    stroke_start, stroke_end = strokes[0]
    search_end = min(stroke_end, stroke_start + max_search_points, length)
    if search_end - stroke_start < min_h_points:
        return LoopSlice(stroke_start, min(length, stroke_start + min_h_points))

    y_seg = y[stroke_start:search_end]
    y0 = float(y[stroke_start])
    peak_rel = int(np.argmax(y_seg))
    peak = float(y_seg[peak_rel])
    ascend = peak - y0

    end_rel = peak_rel
    if ascend > 1.0:
        baseline = y0 + baseline_frac * ascend
        for k in range(peak_rel + 1, len(y_seg)):
            if y_seg[k] <= baseline:
                end_rel = k
                break
        else:
            end_rel = min(len(y_seg) - 1, peak_rel + max(20, len(y_seg) // 6))
    else:
        end_rel = min(len(y_seg) - 1, max(peak_rel, min_h_points - 1))

    end_rel = max(end_rel, min_h_points - 1)
    start = max(0, stroke_start - pad)
    end = min(length, stroke_start + end_rel + 1 + pad)
    if end - start < min_h_points:
        end = min(length, start + min_h_points)
    return LoopSlice(start, end)


def resolve_trajectory_slice(
    segment_mode: str,
    kinematics_np: np.ndarray,
    sub_idx: int,
    length: int,
    *,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    pen_mask: np.ndarray,
    min_loop_points: int,
    max_loop_points: int,
) -> LoopSlice:
    mode = segment_mode.strip().lower()
    if mode == "letter_h":
        return find_first_letter_h_slice(
            kinematics_np,
            sub_idx,
            length,
            channel_mean=channel_mean,
            channel_std=channel_std,
            pen_mask=pen_mask,
        )
    if mode == "loop":
        return find_best_loop_slice(
            kinematics_np,
            sub_idx,
            length,
            channel_mean=channel_mean,
            channel_std=channel_std,
            pen_mask=pen_mask,
            min_loop_points=min_loop_points,
            max_loop_points=max_loop_points,
        )
    if mode == "full_word":
        return find_single_word_slice(
            kinematics_np,
            sub_idx,
            length,
            channel_mean=channel_mean,
            channel_std=channel_std,
            pen_mask=pen_mask,
        )
    raise ValueError(
        f"Unknown segment mode {segment_mode!r}; expected 'full_word', 'letter_h', or 'loop'."
    )


def _stroke_spatial_meta(x: np.ndarray, strokes: list[tuple[int, int]]) -> list[dict[str, float | int]]:
    meta: list[dict[str, float | int]] = []
    for start, end in strokes:
        xs = x[start:end]
        meta.append(
            {
                "start": start,
                "end": end,
                "n": end - start,
                "xmin": float(np.min(xs)),
                "xmax": float(np.max(xs)),
                "xstart": float(xs[0]),
                "xend": float(xs[-1]),
            }
        )
    return meta


def _split_stroke_groups(
    meta: list[dict[str, float | int]],
    *,
    word_space: float = WORD_GROUP_SPACE,
) -> list[list[int]]:
    """Group pen-down strokes into word-sized clusters (pen lifts / spacing / line resets)."""
    if not meta:
        return []
    groups: list[list[int]] = [[0]]
    for idx in range(1, len(meta)):
        prev, nxt = meta[idx - 1], meta[idx]
        gap = int(nxt["start"]) - int(prev["end"])
        dx = float(nxt["xstart"]) - float(prev["xend"])
        x_reset = float(nxt["xstart"]) < float(prev["xend"]) - WORD_GROUP_X_RESET
        split = gap >= WORD_GROUP_GAP or dx > word_space or x_reset
        if split:
            groups.append([idx])
        else:
            groups[-1].append(idx)
    return groups


def _stroke_group_is_oversized(
    meta: list[dict[str, float | int]],
    group: list[int],
) -> bool:
    _start, _end, points, xspan = _word_group_metrics(meta, group)
    return points > MAX_WORD_POINTS or xspan > MAX_WORD_XSPAN


def _subsplit_stroke_group(
    meta: list[dict[str, float | int]],
    group: list[int],
    *,
    word_space: float,
) -> list[list[int]]:
    if len(group) <= 1:
        return [group]
    sub_meta = [meta[idx] for idx in group]
    sub_groups = _split_stroke_groups(sub_meta, word_space=word_space)
    return [[group[local_idx] for local_idx in sub_group] for sub_group in sub_groups]


def _refine_word_groups(meta: list[dict[str, float | int]], groups: list[list[int]]) -> list[list[int]]:
    """Split clusters that accidentally contain multiple word copies."""
    refined: list[list[int]] = []
    for group in groups:
        if not _stroke_group_is_oversized(meta, group):
            refined.append(group)
            continue
        parts = _subsplit_stroke_group(meta, group, word_space=OVERSIZED_WORD_SPACE)
        for part in parts:
            if _stroke_group_is_oversized(meta, part):
                refined.extend(_subsplit_stroke_group(meta, part, word_space=TIGHT_WORD_SPACE))
            else:
                refined.append(part)
    return refined


def _word_group_metrics(
    meta: list[dict[str, float | int]],
    group: list[int],
) -> tuple[int, int, int, float]:
    start = int(meta[group[0]]["start"])
    end = int(meta[group[-1]]["end"])
    points = sum(int(meta[i]["n"]) for i in group)
    xmin = min(float(meta[i]["xmin"]) for i in group)
    xmax = max(float(meta[i]["xmax"]) for i in group)
    return start, end, points, float(xmax - xmin)


def _group_has_major_stroke(meta: list[dict[str, float | int]], group: list[int]) -> bool:
    for idx in group:
        stroke = meta[idx]
        xspan = float(stroke["xmax"]) - float(stroke["xmin"])
        if int(stroke["n"]) >= MIN_MAJOR_STROKE_POINTS or xspan >= MIN_MAJOR_STROKE_XSPAN:
            return True
    return False


def _group_is_truncated(end: int, length: int, mask: np.ndarray) -> bool:
    return end >= length and bool(mask[length - 1])


def _group_qualifies(
    meta: list[dict[str, float | int]],
    group: list[int],
    *,
    min_points: int,
    max_points: int,
    min_xspan: float,
    max_xspan: float,
) -> bool:
    _start, _end, points, xspan = _word_group_metrics(meta, group)
    if not (min_points <= points <= max_points):
        return False
    if not (min_xspan <= xspan <= max_xspan):
        return False
    return _group_has_major_stroke(meta, group)


def find_single_word_slice(
    kinematics_np: np.ndarray,
    sub_idx: int,
    length: int,
    *,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    pen_mask: np.ndarray,
) -> LoopSlice:
    """
    Crop to one complete ``hračkárstvo`` copy in the clip.

    Brief pen lifts within a word stay in one cluster; only large gaps, horizontal
    spacing, or line resets start a new word. Oversized clusters are subdivided.
    The first complete cluster matching Task-7 word extent is returned.
    """
    mask = np.asarray(pen_mask[:length], dtype=bool)
    strokes = _pen_down_strokes(mask, min_stroke_len=1)
    if not strokes:
        return LoopSlice(0, length)

    if strokes[-1][1] >= length and mask[length - 1]:
        strokes = strokes[:-1]
    if not strokes:
        return LoopSlice(0, length)

    x, _ = denormalize_xy(
        kinematics_np,
        sub_idx,
        length,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
    meta = _stroke_spatial_meta(x, strokes)
    groups = _refine_word_groups(meta, _split_stroke_groups(meta))

    for group in groups:
        start, end, _points, _xspan = _word_group_metrics(meta, group)
        if _group_is_truncated(end, length, mask):
            continue
        if _group_qualifies(
            meta,
            group,
            min_points=MIN_WORD_POINTS,
            max_points=MAX_WORD_POINTS,
            min_xspan=MIN_WORD_XSPAN,
            max_xspan=MAX_WORD_XSPAN,
        ):
            return LoopSlice(start, end)

    for group in groups:
        start, end, _points, _xspan = _word_group_metrics(meta, group)
        if _group_is_truncated(end, length, mask):
            continue
        if _group_qualifies(
            meta,
            group,
            min_points=220,
            max_points=MAX_WORD_POINTS,
            min_xspan=2200.0,
            max_xspan=MAX_WORD_XSPAN,
        ):
            return LoopSlice(start, end)

    best_stroke: tuple[int, int] | None = None
    best_score = -1.0
    for start, end in strokes:
        if _group_is_truncated(end, length, mask):
            continue
        stroke_len = end - start
        xs = x[start:end]
        xspan = float(np.max(xs) - np.min(xs))
        if stroke_len < MIN_MAJOR_STROKE_POINTS and xspan < MIN_MAJOR_STROKE_XSPAN:
            continue
        score = stroke_len + 0.05 * xspan
        if score > best_score:
            best_score = score
            best_stroke = (start, end)
    if best_stroke is not None:
        return LoopSlice(best_stroke[0], best_stroke[1])

    return LoopSlice(int(meta[groups[0][0]]["start"]), int(meta[groups[0][-1]]["end"]))


def _axis_fit_for_segment_mode(segment_mode: str) -> str:
    """Single-word and letter crops use a square zoom frame."""
    return "square"


def _loop_window_score(x: np.ndarray, y: np.ndarray, start: int, end: int) -> float:
    seg_x = x[start:end]
    seg_y = y[start:end]
    n = len(seg_x)
    if n < MIN_LOOP_POINTS:
        return -1.0

    diffs = np.hypot(np.diff(seg_x), np.diff(seg_y))
    arc = float(np.sum(diffs))
    if arc <= 1e-6:
        return -1.0

    closure = float(np.hypot(seg_x[-1] - seg_x[0], seg_y[-1] - seg_y[0]))
    closure_ratio = closure / arc

    cx = float(np.mean(seg_x))
    cy = float(np.mean(seg_y))
    radii = np.hypot(seg_x - cx, seg_y - cy)
    spread = float(np.std(radii))
    if spread <= 1e-6:
        return -1.0

    compactness = spread / (arc + 1e-6)
    score = (1.0 - min(closure_ratio, 1.0)) * spread * (1.0 + compactness * 50.0)
    if n > MAX_LOOP_POINTS:
        score *= MAX_LOOP_POINTS / n
    return score


def find_best_loop_slice(
    kinematics_np: np.ndarray,
    sub_idx: int,
    length: int,
    *,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    pen_mask: np.ndarray,
    min_loop_points: int = MIN_LOOP_POINTS,
    max_loop_points: int = MAX_LOOP_POINTS,
    pad: int = LOOP_PAD,
) -> LoopSlice:
    """
    Pick the most loop-like pen-down window (Jana-style local loop focus).

    Scores sliding windows by closure (start near end), spatial spread, and
    pen-down continuity; falls back to the longest pen-down stroke if needed.
    """
    x, y = denormalize_xy(
        kinematics_np,
        sub_idx,
        length,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
    mask = np.asarray(pen_mask[:length], dtype=bool)

    best_score = -1.0
    best_slice = LoopSlice(0, min(length, max_loop_points))

    for stroke_start, stroke_end in _pen_down_strokes(mask, min_stroke_len=min_loop_points):
        stroke_len = stroke_end - stroke_start
        if stroke_len < min_loop_points:
            continue

        max_win = min(max_loop_points, stroke_len)
        win_step = max(1, max_win // 12)
        for win_len in range(min_loop_points, max_win + 1, win_step):
            step = max(1, win_len // 8)
            for win_start in range(stroke_start, stroke_end - win_len + 1, step):
                win_end = win_start + win_len
                if not np.all(mask[win_start:win_end]):
                    continue
                score = _loop_window_score(x, y, win_start, win_end)
                if score > best_score:
                    best_score = score
                    best_slice = LoopSlice(win_start, win_end)

    if best_score < 0.0:
        strokes = _pen_down_strokes(mask, min_stroke_len=min_loop_points)
        if strokes:
            start, end = max(strokes, key=lambda s: s[1] - s[0])
            if end - start > max_loop_points:
                end = start + max_loop_points
            best_slice = LoopSlice(start, end)
        else:
            best_slice = LoopSlice(0, min(length, max_loop_points))

    start = max(0, best_slice.start - pad)
    end = min(length, best_slice.end + pad)
    return LoopSlice(start, end)


def extract_kinematic_shap_series(
    feat_idx: int,
    sub_idx: int,
    length: int,
    *,
    shap_kinematics: np.ndarray,
    slice_start: int = 0,
    slice_end: int | None = None,
) -> np.ndarray:
    """Return per-timestep SHAP for one kinematic channel; optionally loop-cropped."""
    if not (0 <= feat_idx < N_KINEMATIC):
        raise IndexError(f"Invalid kinematic feature index: {feat_idx}")
    end = length if slice_end is None else slice_end
    return np.asarray(shap_kinematics[sub_idx, feat_idx, slice_start:end], dtype=np.float64)


def _feature_slug(name: str) -> str:
    return name.replace(" ", "_")


def _case_panel_title(case: ExtremeCase, segment: LoopSlice, *, segment_label: str) -> str:
    short = case.title.replace("Extreme ", "")
    rank_note = f" #{case.example_rank}" if case.example_rank > 1 else ""
    return (
        f"{short}{rank_note}\n"
        f"Subject {case.subject_id} (true={case.true_label}, pred={case.pred_prob:.3f})\n"
        f"{segment_label} t={segment.start}:{segment.end}"
    )


def _pen_down_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous pen-down index ranges (inclusive start, exclusive end)."""
    return _pen_down_strokes(np.asarray(mask, dtype=bool), min_stroke_len=1)


def _build_stroke_line_segments(
    x: np.ndarray,
    y: np.ndarray,
    shap_val: np.ndarray,
    pen_on: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    """Line segments and midpoint SHAP for pen-down strokes only."""
    segments: list[np.ndarray] = []
    colors: list[float] = []
    for start, end in _pen_down_runs(pen_on):
        for i in range(start, end - 1):
            segments.append(np.array([[x[i], y[i]], [x[i + 1], y[i + 1]]], dtype=np.float64))
            colors.append(float(0.5 * (shap_val[i] + shap_val[i + 1])))
    return segments, colors


def _plot_spatial_shap_panel(
    ax: plt.Axes,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    shap_val: np.ndarray,
    pen_on: np.ndarray,
    *,
    vmax: float,
    color_gamma: float,
    title: str,
    show_axis_labels: bool = False,
    cmap: mcolors.Colormap | None = None,
    axis_fit: str = "square",
) -> object:
    cmap = shap_colormap() if cmap is None else cmap
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    display_shap = _contrast_enhance_shap(shap_val, vmax, color_gamma)

    segments, seg_colors = _build_stroke_line_segments(raw_x, raw_y, display_shap, pen_on)
    last_artist: object
    if segments:
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=LINE_WIDTH_LOOP,
            capstyle="round",
            joinstyle="round",
            antialiaseds=True,
            rasterized=True,
            zorder=2,
            clip_on=False,
        )
        lc.set_array(np.asarray(seg_colors, dtype=np.float64))
        ax.add_collection(lc)
        last_artist = lc
    else:
        last_artist = ax.scatter([], [], c=[], cmap=cmap, norm=norm)

    ax.set_title(title, fontsize=9)
    _set_spatial_axis_limits(ax, raw_x, raw_y, pen_on=pen_on, axis_fit=axis_fit)
    if show_axis_labels:
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    return last_artist


def _contrast_enhance_shap(values: np.ndarray, vmax: float, gamma: float) -> np.ndarray:
    """Power-law stretch so moderate |SHAP| values map to stronger colormap saturation."""
    if vmax <= 0:
        return values
    scaled = np.clip(values / vmax, -1.0, 1.0)
    return np.sign(scaled) * np.power(np.abs(scaled), gamma) * vmax


def compute_shared_feature_vmax(
    cases: list[ExtremeCase],
    lengths: list[int],
    loop_slices: list[LoopSlice],
    *,
    shap_kinematics: np.ndarray,
    percentile: float = SHAP_CONTRAST_PERCENTILE,
) -> list[float]:
    """Shared symmetric color scale per kinematic feature across cases (loop only)."""
    vmax_values: list[float] = []
    n_cases = len(cases)
    for feat_idx in range(N_KINEMATIC):
        pooled: list[float] = []
        for sub_idx in range(n_cases):
            sl = loop_slices[sub_idx]
            shap_val = extract_kinematic_shap_series(
                feat_idx,
                sub_idx,
                lengths[sub_idx],
                shap_kinematics=shap_kinematics,
                slice_start=sl.start,
                slice_end=sl.end,
            )
            pooled.extend(np.abs(shap_val).tolist())
        if pooled:
            peak = float(np.percentile(pooled, percentile))
        else:
            peak = 0.0
        vmax_values.append(max(peak, 1e-6))
    return vmax_values


def _set_spatial_axis_limits(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    pen_on: np.ndarray | None = None,
    pad_frac: float = AXIS_PAD_FRAC,
    axis_fit: str = "square",
) -> None:
    """
    Set axis limits from pen-down ink.

    ``fit`` uses a tight rectangular bbox (wide Task 7 clips are not clipped by a
    forced square equal-aspect window). ``square`` centers a square zoom window.
    """
    if pen_on is not None:
        mask = np.asarray(pen_on, dtype=bool)
        if not np.any(mask):
            mask = np.ones_like(x, dtype=bool)
        x = x[mask]
        y = y[mask]
    if x.size == 0:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("auto")
        return

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xspan = max(xmax - xmin, 1e-6)
    yspan = max(ymax - ymin, 1e-6)

    if axis_fit == "fit":
        xpad = xspan * pad_frac
        ypad = yspan * pad_frac
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.set_aspect("auto")
        return

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    square_pad = max(pad_frac, 0.12)
    half = 0.5 * max(xspan, yspan) * (1.0 + square_pad)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


def plot_feature_comparison_figure(
    feat_idx: int,
    feat_name: str,
    cases: list[ExtremeCase],
    *,
    kinematics_np: np.ndarray,
    lengths: list[int],
    segment_slices: list[LoopSlice],
    pen_masks: list[np.ndarray],
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    shap_kinematics: np.ndarray,
    vmax: float,
    output_path: Path,
    segment_label: str,
    segment_mode: str,
    color_gamma: float = SHAP_COLOR_GAMMA,
) -> None:
    """Save one figure per feature: 4x2 grid (cols=TP/TN/FP/FN, rows=two examples)."""
    n_rows = max(1, len(cases) // len(CASE_COLUMN_KEYS))
    n_cols = len(CASE_COLUMN_KEYS)
    axis_fit = _axis_fit_for_segment_mode(segment_mode)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(22, 5.5 * n_rows),
        subplot_kw={"box_aspect": 1},
        constrained_layout=True,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.asarray([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    cmap = shap_colormap()
    last_sc = None

    for sub_idx, (case, sl, length, pen_mask) in enumerate(
        zip(cases, segment_slices, lengths, pen_masks)
    ):
        row_idx = sub_idx // n_cols
        col_idx = sub_idx % n_cols
        x_full, y_full = denormalize_xy(
            kinematics_np,
            sub_idx,
            length,
            channel_mean=channel_mean,
            channel_std=channel_std,
        )
        raw_x = x_full[sl.start : sl.end]
        raw_y = y_full[sl.start : sl.end]
        pen_on = np.asarray(pen_mask[sl.start : sl.end], dtype=bool)
        shap_val = extract_kinematic_shap_series(
            feat_idx,
            sub_idx,
            length,
            shap_kinematics=shap_kinematics,
            slice_start=sl.start,
            slice_end=sl.end,
        )

        ax = axes[row_idx, col_idx]
        last_sc = _plot_spatial_shap_panel(
            ax,
            raw_x,
            raw_y,
            shap_val,
            pen_on,
            vmax=vmax,
            color_gamma=color_gamma,
            title=_case_panel_title(case, sl, segment_label=segment_label),
            cmap=cmap,
            axis_fit=axis_fit,
        )

    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes, fraction=0.02, pad=0.02, label="SHAP value")

    fig.suptitle(
        f"Local DeepSHAP — {feat_name} | {segment_label} | shared scale +/-{vmax:.4g}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_jana_velocity_shap_figure(
    cases: list[ExtremeCase],
    *,
    kinematics_np: np.ndarray,
    lengths: list[int],
    segment_slices: list[LoopSlice],
    pen_masks: list[np.ndarray],
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    shap_kinematics: np.ndarray,
    vmax_vx: float,
    vmax_vy: float,
    output_path: Path,
    segment_label: str,
    segment_mode: str,
    color_gamma: float = SHAP_COLOR_GAMMA,
) -> None:
    """Jana-style grid: Vx/Vy rows x example rows, TP/TN/FP/FN columns."""
    n_example_rows = max(1, len(cases) // len(CASE_COLUMN_KEYS))
    n_cols = len(CASE_COLUMN_KEYS)
    n_plot_rows = 2 * n_example_rows
    axis_fit = _axis_fit_for_segment_mode(segment_mode)
    fig, axes = plt.subplots(
        n_plot_rows,
        n_cols,
        figsize=(22, 5.0 * n_plot_rows),
        subplot_kw={"box_aspect": 1},
        constrained_layout=True,
    )
    if n_plot_rows == 1 and n_cols == 1:
        axes = np.asarray([[axes]])
    elif n_plot_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    cmap = shap_colormap()
    feat_specs = [(VX_IDX, "V_x", vmax_vx), (VY_IDX, "V_y", vmax_vy)]
    last_feat_scs: list[object] = []

    for feat_row_idx, (feat_idx, feat_tex, vmax) in enumerate(feat_specs):
        row_sc = None
        for example_row in range(n_example_rows):
            plot_row = feat_row_idx * n_example_rows + example_row
            for col_idx in range(n_cols):
                sub_idx = example_row * n_cols + col_idx
                case = cases[sub_idx]
                sl = segment_slices[sub_idx]
                length = lengths[sub_idx]
                pen_mask = pen_masks[sub_idx]
                x_full, y_full = denormalize_xy(
                    kinematics_np,
                    sub_idx,
                    length,
                    channel_mean=channel_mean,
                    channel_std=channel_std,
                )
                raw_x = x_full[sl.start : sl.end]
                raw_y = y_full[sl.start : sl.end]
                pen_on = np.asarray(pen_mask[sl.start : sl.end], dtype=bool)
                shap_val = extract_kinematic_shap_series(
                    feat_idx,
                    sub_idx,
                    length,
                    shap_kinematics=shap_kinematics,
                    slice_start=sl.start,
                    slice_end=sl.end,
                )
                rank_note = f" #{case.example_rank}" if case.example_rank > 1 else ""
                title = (
                    f"SHAP ${feat_tex}${rank_note}\n"
                    f"{case.title.replace('Extreme ', '')} | Subject {case.subject_id}"
                )
                ax = axes[plot_row, col_idx]
                row_sc = _plot_spatial_shap_panel(
                    ax,
                    raw_x,
                    raw_y,
                    shap_val,
                    pen_on,
                    vmax=vmax,
                    color_gamma=color_gamma,
                    title=title,
                    show_axis_labels=(plot_row == n_plot_rows - 1),
                    cmap=cmap,
                    axis_fit=axis_fit,
                )
        if row_sc is not None:
            last_feat_scs.append(row_sc)

    for feat_row_idx, (feat_tex, _vmax) in enumerate([("V_x", vmax_vx), ("V_y", vmax_vy)]):
        fig.colorbar(
            last_feat_scs[feat_row_idx],
            ax=axes[feat_row_idx * n_example_rows : (feat_row_idx + 1) * n_example_rows, :],
            fraction=0.02,
            pad=0.02,
            label=f"SHAP {feat_tex}",
        )

    fig.suptitle(
        f"Local DeepSHAP — Vx & Vy on {segment_label} (hračkárstvo)",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _segment_slug(segment_mode: str) -> str:
    mode = segment_mode.strip().lower()
    if mode == "letter_h":
        return "h"
    if mode == "full_word":
        return "full_word"
    return mode


def _segment_label(segment_mode: str) -> str:
    mode = segment_mode.strip().lower()
    if mode == "full_word":
        return "single word"
    if mode == "letter_h":
        return 'letter "h"'
    if mode == "loop":
        return "loop zoom"
    return mode


def save_selected_cases_csv(cases: list[ExtremeCase], output_path: Path) -> None:
    """Write the 8-case OOF selection for reproducibility."""
    rows = []
    n_cols = len(CASE_COLUMN_KEYS)
    for sub_idx, case in enumerate(cases):
        rows.append(
            {
                "grid_row": sub_idx // n_cols,
                "grid_col": sub_idx % n_cols,
                "case_key": case.key,
                "example_rank": case.example_rank,
                "subject_id": case.subject_id,
                "true_label": case.true_label,
                "pred_prob": case.pred_prob,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def load_case_pen_masks(
    case_row_indices: list[int],
    sample_df: pd.DataFrame,
    lengths: list[int],
    *,
    xy_filter: str,
) -> list[np.ndarray]:
    pen_masks: list[np.ndarray] = []
    for sub_idx, row_idx in enumerate(case_row_indices):
        row = sample_df.iloc[row_idx]
        pen_masks.append(
            load_pen_down_mask(
                Path(row["filepath"]),
                age=row["age"],
                xy_filter=xy_filter,
                length=lengths[sub_idx],
            )
        )
    return pen_masks


def build_segment_slices(
    segment_mode: str,
    *,
    kinematics_np: np.ndarray,
    lengths: list[int],
    pen_masks: list[np.ndarray],
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    min_loop_points: int,
    max_loop_points: int,
) -> list[LoopSlice]:
    segment_slices: list[LoopSlice] = []
    for sub_idx, pen_mask in enumerate(pen_masks):
        segment_slices.append(
            resolve_trajectory_slice(
                segment_mode,
                kinematics_np,
                sub_idx,
                lengths[sub_idx],
                channel_mean=channel_mean,
                channel_std=channel_std,
                pen_mask=pen_mask,
                min_loop_points=min_loop_points,
                max_loop_points=max_loop_points,
            )
        )
    return segment_slices


def generate_segment_figures(
    segment_mode: str,
    *,
    cases: list[ExtremeCase],
    kinematics_np: np.ndarray,
    lengths: list[int],
    segment_slices: list[LoopSlice],
    pen_masks: list[np.ndarray],
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    shap_kinematics: np.ndarray,
    out_dir: Path,
    contrast_pct: float,
    color_gamma: float,
    skip_jana_vx_vy: bool,
) -> list[float]:
    segment_label = _segment_label(segment_mode)
    segment_slug = _segment_slug(segment_mode)

    for case, sl in zip(cases, segment_slices):
        print(
            f"  {segment_slug} {case.key} subject_id={case.subject_id}: "
            f"t={sl.start}:{sl.end} (n={sl.length})"
        )

    vmax_per_feature = compute_shared_feature_vmax(
        cases,
        lengths,
        segment_slices,
        shap_kinematics=shap_kinematics,
        percentile=contrast_pct,
    )

    for feat_idx, feat_name in enumerate(KINEMATIC_FEATURES):
        out_path = out_dir / f"task7_local_shap_spatial_{segment_slug}_{_feature_slug(feat_name)}.png"
        plot_feature_comparison_figure(
            feat_idx,
            feat_name,
            cases,
            kinematics_np=kinematics_np,
            lengths=lengths,
            segment_slices=segment_slices,
            pen_masks=pen_masks,
            channel_mean=channel_mean,
            channel_std=channel_std,
            shap_kinematics=shap_kinematics,
            vmax=vmax_per_feature[feat_idx],
            output_path=out_path,
            segment_label=segment_label,
            segment_mode=segment_mode,
            color_gamma=color_gamma,
        )
        print(f"Saved feature comparison: {out_path.resolve()}")

    if not skip_jana_vx_vy:
        jana_path = out_dir / f"task7_local_shap_spatial_{segment_slug}_jana_vx_vy.png"
        plot_jana_velocity_shap_figure(
            cases,
            kinematics_np=kinematics_np,
            lengths=lengths,
            segment_slices=segment_slices,
            pen_masks=pen_masks,
            channel_mean=channel_mean,
            channel_std=channel_std,
            shap_kinematics=shap_kinematics,
            vmax_vx=vmax_per_feature[VX_IDX],
            vmax_vy=vmax_per_feature[VY_IDX],
            output_path=jana_path,
            segment_label=segment_label,
            segment_mode=segment_mode,
            color_gamma=color_gamma,
        )
        print(f"Saved Jana-style Vx/Vy figure: {jana_path.resolve()}")

    print(
        f"Shared kinematic color limits on {segment_label} regions "
        f"(p{contrast_pct:g} |SHAP|, gamma={color_gamma:g} across {len(cases)} cases):"
    )
    for name, vmax in zip(KINEMATIC_FEATURES, vmax_per_feature):
        print(f"  {name}: +/-{vmax:.6f}")
    return vmax_per_feature


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local spatial DeepSHAP for extreme OOF cases (Task 7).",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=DEFAULT_PREDICTIONS_CSV,
        help="OOF predictions CSV with subject_id, true_label, pred_prob.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Folder to search for .pt checkpoints (includes subfolders).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint path (default: holdout_fft_age_gender.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for spatial SHAP PNG outputs.",
    )
    parser.add_argument(
        "--n-background",
        type=int,
        default=N_BACKGROUND,
        help="Background samples for DeepExplainer (default: 100).",
    )
    parser.add_argument(
        "--prob-hi",
        type=float,
        default=PROB_HI,
        help="High-confidence threshold for TP/FP (default: 0.90).",
    )
    parser.add_argument(
        "--prob-lo",
        type=float,
        default=PROB_LO,
        help="Low-confidence threshold for TN/FN (default: 0.10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="RNG seed for background subsampling.",
    )
    parser.add_argument(
        "--min-loop-points",
        type=int,
        default=MIN_LOOP_POINTS,
        help="Minimum pen-down points in a candidate loop window.",
    )
    parser.add_argument(
        "--max-loop-points",
        type=int,
        default=MAX_LOOP_POINTS,
        help="Maximum loop window length.",
    )
    parser.add_argument(
        "--contrast-percentile",
        type=float,
        default=SHAP_CONTRAST_PERCENTILE,
        help="Percentile of |SHAP| used for shared color limits (lower = stronger contrast).",
    )
    parser.add_argument(
        "--color-gamma",
        type=float,
        default=SHAP_COLOR_GAMMA,
        help="Power-law gamma for SHAP color saturation (<1 boosts mid-range contrast).",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=N_CASES_PER_CLASS,
        help="Subjects per TP/TN/FP/FN (default: 2 -> 8-case 4x2 grid).",
    )
    parser.add_argument(
        "--segment-modes",
        nargs="+",
        choices=("full_word", "letter_h", "loop"),
        default=list(DEFAULT_SEGMENT_MODES),
        metavar="MODE",
        help=(
            "Segment(s) to plot. full_word = one complete hračkárstvo copy. "
            "Default runs full_word and letter_h (one SHAP run, separate figure sets)."
        ),
    )
    parser.add_argument(
        "--skip-jana-vx-vy",
        action="store_true",
        help="Do not save the Jana-style 2x4 Vx/Vy SHAP figure.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred_csv = (
        args.predictions_csv
        if args.predictions_csv.is_absolute()
        else _HERE / args.predictions_csv
    )
    if not pred_csv.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {pred_csv}")

    ckpt_dir = None
    if args.checkpoint_dir is not None:
        ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else _HERE / args.checkpoint_dir
    user_ckpt = None
    if args.checkpoint is not None:
        user_ckpt = args.checkpoint if args.checkpoint.is_absolute() else _HERE / args.checkpoint
    ckpt_path = resolve_checkpoint_path(_HERE, user_ckpt, checkpoint_dir=ckpt_dir)

    model, meta = load_task7_checkpoint(ckpt_path, device)
    if not (bool(meta["use_age"]) and bool(meta["use_gender"])):
        raise ValueError("Checkpoint must use late-fusion age and gender.")

    xy_filter = str(meta.get("xy_filter", XAI_XY_FILTER))
    channel_mean = np.asarray(meta["channel_mean"], dtype=np.float32)
    channel_std = np.asarray(meta["channel_std"], dtype=np.float32)
    age_min = float(meta["age_min"])
    age_max = float(meta["age_max"])

    sample_df = build_sample_table()
    min_clip_length = int(MIN_CLIP_LENGTH)
    seq_lengths = build_subject_sequence_lengths(sample_df, xy_filter=xy_filter)
    eligible_subject_ids = {
        sid for sid, seq_len in seq_lengths.items() if seq_len >= min_clip_length
    }
    excluded = sorted(sid for sid, seq_len in seq_lengths.items() if seq_len < min_clip_length)
    if excluded:
        print(
            f"Excluding {len(excluded)} subject(s) with clip length < {min_clip_length}: "
            f"{excluded}"
        )

    cases = select_extreme_cases(
        pd.read_csv(pred_csv),
        prob_hi=float(args.prob_hi),
        prob_lo=float(args.prob_lo),
        n_per_class=int(args.n_per_class),
        eligible_subject_ids=eligible_subject_ids,
    )
    case_row_indices = [_subject_id_to_row_index(sample_df, c.subject_id) for c in cases]
    n_samples = len(sample_df)

    full_ds = Task5TrajectoryDataset(
        sample_df,
        np.arange(n_samples, dtype=np.int64),
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
    full_batch = next(iter(full_loader))

    rng = np.random.default_rng(args.seed)
    n_bg = min(int(args.n_background), n_samples)
    bg_indices = rng.choice(n_samples, size=n_bg, replace=False)

    test_kinematics = full_batch["x"][case_row_indices].to(device)
    test_age = _normalize_age(full_batch["age"][case_row_indices].to(device), age_min, age_max)
    test_gender = full_batch["gender"][case_row_indices].to(device).float()
    lengths = [int(full_batch["length"][i]) for i in case_row_indices]

    bg_kinematics = full_batch["x"][bg_indices].to(device)
    bg_age = _normalize_age(full_batch["age"][bg_indices].to(device), age_min, age_max)
    bg_gender = full_batch["gender"][bg_indices].to(device).float()

    wrapper_model = SHAPWrapper(model).to(device)
    wrapper_model.eval()

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path.resolve()}")
    print(f"Predictions: {pred_csv.resolve()}")
    print("Selected extreme cases:")
    for i, case in enumerate(cases):
        print(
            f"  [{i}] {case.key}: subject_id={case.subject_id}, "
            f"true={case.true_label}, pred={case.pred_prob:.6f}, length={lengths[i]}"
        )

    background_data = [bg_kinematics, bg_age, bg_gender]
    test_data = [test_kinematics, test_age, test_gender]

    print("Building DeepExplainer (this may take several minutes)...")
    explainer = shap.DeepExplainer(wrapper_model, background_data)
    shap_values = explainer.shap_values(test_data)

    shap_kinematics, shap_age, shap_gender = _unpack_three_input_shap(shap_values)
    shap_kinematics = np.squeeze(np.asarray(shap_kinematics, dtype=np.float64))
    shap_age = np.squeeze(np.asarray(shap_age, dtype=np.float64))
    shap_gender = np.squeeze(np.asarray(shap_gender, dtype=np.float64))
    kinematics_np = test_kinematics.detach().cpu().numpy()

    print(
        "SHAP shapes (raw): "
        f"kin={shap_kinematics.shape}, age={shap_age.shape}, gender={shap_gender.shape}"
    )

    segment_modes = [str(mode) for mode in args.segment_modes]
    contrast_pct = float(args.contrast_percentile)
    color_gamma = float(args.color_gamma)
    out_dir = args.output_dir if args.output_dir.is_absolute() else _HERE / args.output_dir

    save_selected_cases_csv(cases, out_dir / "task7_local_shap_selected_cases.csv")
    print(f"Saved case manifest: {(out_dir / 'task7_local_shap_selected_cases.csv').resolve()}")

    pen_masks = load_case_pen_masks(
        case_row_indices,
        sample_df,
        lengths,
        xy_filter=xy_filter,
    )

    for segment_mode in segment_modes:
        print(f"\n=== Segment mode: {segment_mode} ({_segment_label(segment_mode)}) ===")
        segment_slices = build_segment_slices(
            segment_mode,
            kinematics_np=kinematics_np,
            lengths=lengths,
            pen_masks=pen_masks,
            channel_mean=channel_mean,
            channel_std=channel_std,
            min_loop_points=int(args.min_loop_points),
            max_loop_points=int(args.max_loop_points),
        )

        generate_segment_figures(
            segment_mode,
            cases=cases,
            kinematics_np=kinematics_np,
            lengths=lengths,
            segment_slices=segment_slices,
            pen_masks=pen_masks,
            channel_mean=channel_mean,
            channel_std=channel_std,
            shap_kinematics=shap_kinematics,
            out_dir=out_dir,
            contrast_pct=contrast_pct,
            color_gamma=color_gamma,
            skip_jana_vx_vy=bool(args.skip_jana_vx_vy),
        )


if __name__ == "__main__":
    main()
