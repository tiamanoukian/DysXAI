"""
Train and save the canonical Task 7 checkpoint for explainability (XAI) studies.

Uses the locked preprocessing choice: **12 Hz FFT** on XY (`xy_filter="fft"`),
late-fusion age, and ``best_epoch`` from ``tuned_params_fft.json`` (or legacy
``tuned_params.json``). Trains on the subject-stratified 80% tuning split (same
as ``train_final_evaluation.py``) and writes ``checkpoints/xai_fft_age.pt``.

Examples::

    python DysXAI_task7/train_ab_test.py --all-filters
    python DysXAI_task7/train_xai_checkpoint.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from checkpoint_io import (  # noqa: E402
    XAI_FILTER_LABEL,
    XAI_XY_FILTER,
    default_checkpoint_dir,
    default_xai_checkpoint_path,
)
from dataset import TASK7_DATA_DIR, build_sample_table  # noqa: E402
from train_ab_test import (  # noqa: E402
    RANDOM_STATE,
    set_seed,
    subject_level_80_20_split,
    tuned_params_basename,
)
from train_final_evaluation import load_best_epoch, run_one_condition  # noqa: E402


def resolve_best_epoch(params_dir: Path, *, use_age: bool) -> int:
    per_filter = params_dir / tuned_params_basename(XAI_XY_FILTER, use_age=use_age)
    legacy = params_dir / "tuned_params.json"
    if per_filter.is_file():
        return load_best_epoch(per_filter)
    if legacy.is_file():
        print(
            f"Note: {per_filter.name} not found; using best_epoch from {legacy.name}. "
            f"Run: python train_ab_test.py --all-filters  for per-filter tuning."
        )
        return load_best_epoch(legacy)
    raise FileNotFoundError(
        f"No tuning JSON for FFT. Expected {per_filter} or {legacy}. "
        "Run train_ab_test.py first."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Train XAI checkpoint ({XAI_FILTER_LABEL}, late-fusion age).",
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=_HERE,
        help="Directory with tuned_params_fft.json (default: DysXAI_task7/).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=default_checkpoint_dir(_HERE),
        help="Output directory (default: DysXAI_task7/checkpoints/).",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Override output .pt path (default: checkpoints/xai_fft_age.pt).",
    )
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params_dir = args.params_dir if args.params_dir.is_absolute() else _HERE / args.params_dir
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else _HERE / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = (
        args.checkpoint_out
        if args.checkpoint_out is not None
        else default_xai_checkpoint_path(_HERE, checkpoint_dir=ckpt_dir)
    )
    if not ckpt_path.is_absolute():
        ckpt_path = _HERE / ckpt_path

    best_epoch = resolve_best_epoch(params_dir, use_age=True)
    print(f"Device: {device}")
    print(f"Task 7 data: {TASK7_DATA_DIR}")
    print(f"XAI filter: {XAI_XY_FILTER} ({XAI_FILTER_LABEL})")
    print(f"best_epoch: {best_epoch}")
    print(f"Output: {ckpt_path.resolve()}")

    sample_df = build_sample_table()
    train_df, holdout_df = subject_level_80_20_split(sample_df)
    train_idx = np.arange(len(train_df), dtype=np.int64)
    holdout_idx = np.arange(len(holdout_df), dtype=np.int64)
    print(f"Train/tuning: n={len(train_df)} | Holdout (not used for training): n={len(holdout_df)}")

    metrics = run_one_condition(
        xy_filter=XAI_XY_FILTER,
        train_df=train_df,
        holdout_df=holdout_df,
        train_idx=train_idx,
        test_idx=holdout_idx,
        best_epoch=best_epoch,
        device=device,
        use_late_fusion_age=True,
        checkpoint_path=ckpt_path,
        checkpoint_split="xai",
    )
    print(
        f"Holdout sanity check — AUC={metrics.auc:.4f}  Acc={metrics.accuracy:.4f}  "
        f"Sens={metrics.sensitivity:.4f}  Spec={metrics.specificity:.4f}"
    )
    print(f"XAI checkpoint ready: {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
