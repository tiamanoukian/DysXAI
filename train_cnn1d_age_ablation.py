# # `train_cnn1d_age_ablation.py`
#
# Converted from Python for notebook workflow.
#
# **Module docstring**
#
# 1D CNN training with Task 7 (hračkárstvo) data only and A/B testing for the Age channel.
#
# - Restricts `Config.DATA_ROOT` to ``dysxai_tasks_split/task_7_hrackarstvo`` (isolated task).
# - Toggles `Config.INCLUDE_AGE_AS_CHANNEL` to include or drop the broadcast Age feature.
# - Subject-stratified K-fold: prints per-fold and aggregate **Mean AUC** and **Std** on the test fold.
#
# Run from project root:
#     python train_cnn1d_age_ablation.py
#     python train_cnn1d_age_ablation.py --k 5 --epochs 30 --no-comparison
#     python train_cnn1d_age_ablation.py --seed 123
#
# Requires: same stack as `dysxai_leakage_ablation.py` (torch, sklearn, tqdm).

"""
1D CNN training with Task 7 (hračkárstvo) data only and A/B testing for the Age channel.

- Restricts `Config.DATA_ROOT` to ``dysxai_tasks_split/task_7_hrackarstvo`` (isolated task).
- Toggles `Config.INCLUDE_AGE_AS_CHANNEL` to include or drop the broadcast Age feature.
- Subject-stratified K-fold: prints per-fold and aggregate **Mean AUC** and **Std** on the test fold.

Run from project root:
    python train_cnn1d_age_ablation.py
    python train_cnn1d_age_ablation.py --k 5 --epochs 30 --no-comparison
    python train_cnn1d_age_ablation.py --seed 123

Requires: same stack as `dysxai_leakage_ablation.py` (torch, sklearn, tqdm).
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dysxai_init import Config
from dysxai_kfold_age_common import (
    K_FOLDS_DEFAULT,
    RANDOM_SEED_DEFAULT,
    VAL_FRACTION_DEFAULT,
    print_kfold_summary,
    run_kfold_for_setting,
    task_7_data_root,
)
from dysxai_leakage_ablation import CNN1DModel

# ---------------------------------------------------------------------------
# User toggles (overridden by --no-comparison + CLI when used interactively)
# ---------------------------------------------------------------------------
INCLUDE_AGE_FEATURE: bool = True
RUN_COMPARISON_AB: bool = True

K_FOLDS = K_FOLDS_DEFAULT
VAL_FRACTION = VAL_FRACTION_DEFAULT
RANDOM_SEED = RANDOM_SEED_DEFAULT


def main() -> None:
    p = argparse.ArgumentParser(description="1D CNN K-fold (Task 7) with optional Age A/B comparison.")
    p.add_argument("--k", type=int, default=K_FOLDS, help="Number of subject-stratified folds")
    p.add_argument("--epochs", type=int, default=None, help="Epochs per fold (default: Config.NUM_EPOCHS)")
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION, help="Val share within train+val per fold")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="Base random seed")
    p.add_argument(
        "--comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run both WITH and WITHOUT Age (default: on). Use --no-comparison for a single run.",
    )
    p.add_argument(
        "--include-age",
        dest="include_age",
        action=argparse.BooleanOptionalAction,
        default=INCLUDE_AGE_FEATURE,
        help="When not using comparison: set Age on/off (default: True).",
    )
    args = p.parse_args()

    epochs = args.epochs if args.epochs is not None else Config.NUM_EPOCHS
    device = Config.DEVICE

    print("=" * 80)
    print("1D CNN — Task 7 (hračkárstvo) only — Age channel A/B")
    print("=" * 80)
    print(f"DATA_ROOT: {task_7_data_root()}")
    print(f"K={args.k}  val_fraction={args.val_fraction}  epochs={epochs}  seed={args.seed}  device={device}")
    print("=" * 80)

    def build_model(in_ch: int) -> CNN1DModel:
        return CNN1DModel(in_channels=in_ch)

    if args.comparison:
        for include_age, title in [
            (True, "A: WITH Age (INCLUDE_AGE_AS_CHANNEL=True)"),
            (False, "B: WITHOUT Age (INCLUDE_AGE_AS_CHANNEL=False)"),
        ]:
            print(f"\n>>> {title}\n")
            val_a, test_a = run_kfold_for_setting(
                build_model,
                include_age,
                n_folds=args.k,
                val_fraction=args.val_fraction,
                seed=args.seed,
                num_epochs=epochs,
                device=device,
            )
            print_kfold_summary(title, test_a, val_a)
        print("\n" + "=" * 80)
        print("End of A/B comparison (test metrics are the held-out fold per split).")
        print("=" * 80)
    else:
        include_age = args.include_age
        print(f"\n>>> Single run: INCLUDE_AGE_AS_CHANNEL={include_age}\n")
        val_a, test_a = run_kfold_for_setting(
            build_model,
            include_age,
            n_folds=args.k,
            val_fraction=args.val_fraction,
            seed=args.seed,
            num_epochs=epochs,
            device=device,
        )
        print_kfold_summary("Single run", test_a, val_a)

if __name__ == "__main__":
    main()
