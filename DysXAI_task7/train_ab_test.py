"""
Nested-style epoch tuning for Task 7 with strict subject-level split hygiene.

Pipeline:
1) Global 80/20 subject-stratified split:
   - 80% train_tuning subjects
   - 20% holdout_test subjects (kept untouched in this script)
2) 5-fold StratifiedGroupKFold ONLY on train_tuning data.
3) Record validation AUC each epoch; stop a fold early when val AUC does not improve
   for ``patience`` epochs (after ``min_epochs``).
4) Average AUC per epoch across folds and choose one global best epoch (≤ ``max_epochs``).

Use ``--all-filters`` to tune raw, butterworth, and fft separately (three JSON files).

No final holdout retraining/testing is performed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader

# Package-local imports (works whether cwd is repo root or DysXAI_task7)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dataset import (  # noqa: E402
    TASK7_DATA_DIR,
    TASK7_XY_FILTER_ENV,
    Task5TrajectoryDataset,
    _canonical_xy_filter,
    build_sample_table,
    collate_task7,
    fit_channel_scaling,
)
from model import Task7Conv1dClassifier  # noqa: E402


RANDOM_STATE = 42
N_SPLITS = 5
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7
MIN_EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
AUC_IMPROVE_MIN_DELTA = 1e-4

DEFAULT_EPOCH_CSV = "task7_nested_cv_epoch_mean_auc.csv"

# Canonical filter keys written to tuned_params_<key>.json
FILTER_KEYS = ("raw", "butterworth", "fft")
XY_FILTER_ENV_BY_KEY = {
    "raw": "none",
    "butterworth": "butterworth",
    "fft": "fft",
}


def _demographic_suffix(*, use_age: bool, use_gender: bool) -> str:
    if use_age and use_gender:
        return "_age_gender"
    if use_age:
        return "_age"
    if use_gender:
        return "_gender"
    return "_no_demographics"


def tuned_params_basename(filter_key: str, *, use_age: bool, use_gender: bool = False) -> str:
    suffix = _demographic_suffix(use_age=use_age, use_gender=use_gender)
    return f"tuned_params_{filter_key}{suffix}.json"


def epoch_csv_basename(filter_key: str, *, use_age: bool, use_gender: bool) -> str:
    suffix = _demographic_suffix(use_age=use_age, use_gender=use_gender)
    return f"task7_nested_cv_epoch_mean_auc_{filter_key}{suffix}.csv"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_fold_age_minmax(sample_df, train_idx: np.ndarray) -> tuple[float, float]:
    ages = sample_df.iloc[train_idx]["age"].to_numpy(dtype=np.float64)
    finite = np.isfinite(ages)
    if not finite.any():
        return 0.0, 1.0
    a = ages[finite]
    age_min = float(np.min(a))
    age_max = float(np.max(a))
    return age_min, age_max


def _train_fold_age_stats(sample_df, train_idx: np.ndarray) -> tuple[float, float]:
    """Backward-compatible alias; returns (age_min, age_max)."""
    return _train_fold_age_minmax(sample_df, train_idx)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_age: bool,
    use_gender: bool,
    age_min: float,
    age_max: float,
) -> float:
    model.eval()
    ys: list[float] = []
    scores: list[float] = []
    denom = (age_max - age_min) + 1e-8
    for batch in loader:
        x = batch["x"].to(device)
        lengths = batch["length"].to(device)
        y = batch["y"].cpu().numpy()
        if use_age:
            batch_ages = batch["age"].to(device).float()
            normalized_ages = (batch_ages - age_min) / denom
            if use_gender:
                batch_gender = batch["gender"].to(device).float()
                logits = model(
                    x, lengths=lengths, age=normalized_ages, gender=batch_gender
                ).squeeze(-1).cpu().numpy()
            else:
                logits = model(x, lengths=lengths, age=normalized_ages).squeeze(-1).cpu().numpy()
        else:
            if use_gender:
                batch_gender = batch["gender"].to(device).float()
                logits = model(x, lengths=lengths, gender=batch_gender).squeeze(-1).cpu().numpy()
            else:
                logits = model(x, lengths=lengths).squeeze(-1).cpu().numpy()
        ys.extend(y.astype(float).tolist())
        scores.extend(torch.sigmoid(torch.tensor(logits)).numpy().tolist())
    ys_arr = np.array(ys, dtype=float)
    s_arr = np.array(scores, dtype=float)
    if len(np.unique(ys_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(ys_arr, s_arr))


def train_one_fold(
    sample_df,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    use_age: bool,
    use_gender: bool,
    device: torch.device,
    fold_id: int,
    *,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    min_epochs: int = MIN_EPOCHS,
) -> list[float]:
    mu, sigma = fit_channel_scaling(sample_df, train_idx)
    age_min, age_max = _train_fold_age_minmax(sample_df, train_idx)

    train_ds = Task5TrajectoryDataset(
        sample_df,
        train_idx,
        use_age_channel=use_age,
        channel_mean=mu,
        channel_std=sigma,
    )
    val_ds = Task5TrajectoryDataset(
        sample_df,
        val_idx,
        use_age_channel=use_age,
        channel_mean=mu,
        channel_std=sigma,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_task7,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_task7,
        num_workers=0,
    )

    model = Task7Conv1dClassifier(use_age=use_age, use_gender=use_gender).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()

    label = f"kinematics + demographics(age={use_age}, gender={use_gender})"
    denom = (age_max - age_min) + 1e-8
    epoch_val_aucs: list[float] = []
    best_val_auc = float("-inf")
    epochs_without_improvement = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            lengths = batch["length"].to(device)
            y = batch["y"].to(device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            if use_age:
                batch_ages = batch["age"].to(device).float()
                normalized_ages = (batch_ages - age_min) / denom
                if use_gender:
                    batch_gender = batch["gender"].to(device).float()
                    logits = model(x, lengths=lengths, age=normalized_ages, gender=batch_gender)
                else:
                    logits = model(x, lengths=lengths, age=normalized_ages)
            else:
                if use_gender:
                    batch_gender = batch["gender"].to(device).float()
                    logits = model(x, lengths=lengths, gender=batch_gender)
                else:
                    logits = model(x, lengths=lengths)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        val_auc = evaluate(
            model,
            val_loader,
            device,
            use_age=use_age,
            use_gender=use_gender,
            age_min=age_min,
            age_max=age_max,
        )
        epoch_val_aucs.append(val_auc)
        avg_loss = total_loss / max(n_batches, 1)

        if not np.isnan(val_auc) and val_auc > best_val_auc + AUC_IMPROVE_MIN_DELTA:
            best_val_auc = float(val_auc)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if ep == 1 or ep % 5 == 0 or epochs_without_improvement == 0:
            print(
                f"  Fold {fold_id} [{label}] epoch {ep}/{max_epochs} "
                f"train_loss={avg_loss:.4f} val_auc={val_auc:.4f} "
                f"(best={best_val_auc:.4f}, no_improve={epochs_without_improvement})"
            )

        if ep >= min_epochs and epochs_without_improvement >= patience:
            print(
                f"  Fold {fold_id} early stopping at epoch {ep} "
                f"(patience={patience}, best_val_auc={best_val_auc:.4f})"
            )
            break

    return epoch_val_aucs


def subject_level_80_20_split(sample_df):
    """
    Subject-stratified global split to keep each subject intact.
    Returns (train_tuning_df, holdout_test_df).
    """
    sub_df = (
        sample_df[["subject_id", "label"]]
        .drop_duplicates(subset=["subject_id"])
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    train_subj, holdout_subj = train_test_split(
        sub_df["subject_id"].to_numpy(),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=sub_df["label"].to_numpy(),
    )
    train_set = set(int(x) for x in train_subj.tolist())
    holdout_set = set(int(x) for x in holdout_subj.tolist())
    train_tuning_df = sample_df[sample_df["subject_id"].isin(train_set)].copy().reset_index(drop=True)
    holdout_test_df = sample_df[sample_df["subject_id"].isin(holdout_set)].copy().reset_index(drop=True)
    return train_tuning_df, holdout_test_df


def print_distribution(df, title: str) -> None:
    vc = df["label"].value_counts().sort_index()
    total = int(len(df))
    n_ctrl = int(vc.get(0, 0))
    n_dys = int(vc.get(1, 0))
    print(
        f"{title}: n={total} | Control(label=0)={n_ctrl} ({(n_ctrl / max(total, 1)):.1%}) | "
        f"Dysgraphic(label=1)={n_dys} ({(n_dys / max(total, 1)):.1%})"
    )


def run_nested_cv_epoch_tuning(
    train_tuning_df,
    *,
    use_age: bool,
    use_gender: bool,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    min_epochs: int = MIN_EPOCHS,
) -> tuple[int, float, np.ndarray]:
    y = train_tuning_df["label"].to_numpy()
    groups = train_tuning_df["subject_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_epoch_aucs: list[list[float]] = []
    for fold_id, (tr, va) in enumerate(sgkf.split(np.zeros(len(train_tuning_df)), y, groups), start=1):
        print(f"\n--- Tuning Fold {fold_id}/{N_SPLITS} ---")
        per_epoch = train_one_fold(
            train_tuning_df,
            tr,
            va,
            use_age=use_age,
            use_gender=use_gender,
            device=device,
            fold_id=fold_id,
            max_epochs=max_epochs,
            patience=patience,
            min_epochs=min_epochs,
        )
        fold_epoch_aucs.append(per_epoch)

    max_len = max(len(row) for row in fold_epoch_aucs)
    padded = [
        row + [float("nan")] * (max_len - len(row))
        for row in fold_epoch_aucs
    ]
    epoch_auc_matrix = np.asarray(padded, dtype=np.float64)
    mean_auc_per_epoch = np.nanmean(epoch_auc_matrix, axis=0)
    best_epoch_idx = int(np.nanargmax(mean_auc_per_epoch))
    best_epoch = best_epoch_idx + 1
    best_auc = float(mean_auc_per_epoch[best_epoch_idx])
    return best_epoch, best_auc, mean_auc_per_epoch


def write_epoch_mean_auc_csv(path: Path, mean_auc_per_epoch: np.ndarray) -> None:
    """Write per-epoch mean validation AUC (across folds) for reporting/plots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "mean_val_auc"])
        for ep, auc in enumerate(mean_auc_per_epoch, start=1):
            if np.isnan(auc):
                w.writerow([ep, "nan"])
            else:
                w.writerow([ep, f"{float(auc):.6f}"])


def run_tuning_session(
    train_tuning_df,
    *,
    use_age: bool,
    use_gender: bool,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_epochs: int,
    xy_filter_canonical: str,
    params_out: Path,
    epoch_csv_out: Path,
) -> tuple[int, float]:
    """5-fold CV on 80%% train/tuning split; write best_epoch JSON + epoch-mean-AUC CSV."""
    print(f"\n{'=' * 60}")
    print(
        f"Tuning xy_filter={xy_filter_canonical!r} | use_age={use_age} | use_gender={use_gender}"
    )
    print(f"{'=' * 60}")

    best_epoch, best_auc, mean_auc_per_epoch = run_nested_cv_epoch_tuning(
        train_tuning_df,
        use_age=use_age,
        use_gender=use_gender,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        min_epochs=min_epochs,
    )
    write_epoch_mean_auc_csv(epoch_csv_out, mean_auc_per_epoch)
    print(f"Wrote per-epoch mean val AUC: {epoch_csv_out.resolve()}")

    model_label = f"demographics(use_age={use_age}, use_gender={use_gender})"
    print(
        f"Best epoch = {best_epoch} (mean val AUC = {best_auc:.3f}) | model: {model_label}"
    )

    params = {
        "best_epoch": best_epoch,
        "best_epoch_mean_val_auc": best_auc,
        "xy_filter": xy_filter_canonical,
        "use_age": bool(use_age),
        "use_gender": bool(use_gender),
        "max_epochs": max_epochs,
        "patience": patience,
        "min_epochs": min_epochs,
    }
    params_out.parent.mkdir(parents=True, exist_ok=True)
    with params_out.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Saved: {params_out.resolve()}")
    return best_epoch, best_auc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 7 nested-CV epoch tuning: 80/20 subject split + 5-fold tuning on 80%."
    )
    parser.add_argument(
        "--xy-filter",
        default=None,
        metavar="MODE",
        help="Override env: butterworth | fft | none (raw XY before derivatives)",
    )
    parser.add_argument(
        "--use-age",
        action="store_true",
        help="Use late-fusion age channel during tuning (default: kinematics only).",
    )
    parser.add_argument(
        "--use-gender",
        action="store_true",
        help="Use late-fusion gender scalar during tuning (0=male, 1=female).",
    )
    parser.add_argument(
        "--epoch-csv-out",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"CSV path for per-epoch mean val AUC (default: {DEFAULT_EPOCH_CSV} next to this script).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Upper bound on epochs per fold (default: {MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=(
            f"Stop a fold when val AUC does not improve for this many epochs "
            f"(default: {EARLY_STOPPING_PATIENCE})."
        ),
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=MIN_EPOCHS,
        help=f"Minimum epochs before early stopping can trigger (default: {MIN_EPOCHS}).",
    )
    parser.add_argument(
        "--params-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="best_epoch JSON path (default: tuned_params_<filter>.json or legacy tuned_params.json).",
    )
    parser.add_argument(
        "--all-filters",
        action="store_true",
        help="Tune raw, butterworth, and fft separately (writes three JSON + three epoch CSVs).",
    )
    args = parser.parse_args()

    if args.max_epochs < 1:
        parser.error("--max-epochs must be >= 1")
    if args.patience < 1:
        parser.error("--patience must be >= 1")
    if args.min_epochs < 1:
        parser.error("--min-epochs must be >= 1")
    if args.min_epochs > args.max_epochs:
        parser.error("--min-epochs cannot exceed --max-epochs")

    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task 7 data: {TASK7_DATA_DIR}")

    print(
        f"Early stopping: max_epochs={args.max_epochs}, patience={args.patience}, "
        f"min_epochs={args.min_epochs}"
    )
    sample_df = build_sample_table()
    print(f"Samples (matched .svc × metadata): {len(sample_df)}")
    train_tuning_df, holdout_test_df = subject_level_80_20_split(sample_df)
    print_distribution(train_tuning_df, "Train/Tuning split (80%)")
    print_distribution(holdout_test_df, "Holdout split (20%) — not used during tuning")

    use_age = bool(args.use_age)
    use_gender = bool(args.use_gender)

    if args.all_filters:
        if args.xy_filter is not None:
            parser.error("--all-filters cannot be combined with --xy-filter")
        if args.params_out is not None:
            parser.error("--all-filters cannot be combined with --params-out")
        if args.epoch_csv_out is not None:
            parser.error("--all-filters cannot be combined with --epoch-csv-out")
        for filter_key in FILTER_KEYS:
            os.environ[TASK7_XY_FILTER_ENV] = XY_FILTER_ENV_BY_KEY[filter_key]
            canonical, _ = _canonical_xy_filter(None)
            params_path = _HERE / tuned_params_basename(
                filter_key, use_age=use_age, use_gender=use_gender
            )
            csv_path = _HERE / epoch_csv_basename(
                filter_key, use_age=use_age, use_gender=use_gender
            )
            run_tuning_session(
                train_tuning_df,
                use_age=use_age,
                use_gender=use_gender,
                device=device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                min_epochs=args.min_epochs,
                xy_filter_canonical=canonical,
                params_out=params_path,
                epoch_csv_out=csv_path,
            )
        print("\nAll-filter tuning complete.")
        return

    if args.xy_filter is not None:
        os.environ[TASK7_XY_FILTER_ENV] = args.xy_filter.strip()
    canonical, _ = _canonical_xy_filter(None)
    print(f"XY pre-filter ({TASK7_XY_FILTER_ENV}): {canonical}")

    if args.params_out is not None:
        tuned_path = args.params_out if args.params_out.is_absolute() else _HERE / args.params_out
    else:
        tuned_path = _HERE / tuned_params_basename(
            canonical, use_age=use_age, use_gender=use_gender
        )

    if args.epoch_csv_out is not None:
        csv_path = args.epoch_csv_out if args.epoch_csv_out.is_absolute() else _HERE / args.epoch_csv_out
    else:
        csv_path = _HERE / epoch_csv_basename(
            canonical, use_age=use_age, use_gender=use_gender
        )

    run_tuning_session(
        train_tuning_df,
        use_age=use_age,
        use_gender=use_gender,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        xy_filter_canonical=canonical,
        params_out=tuned_path,
        epoch_csv_out=csv_path,
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
