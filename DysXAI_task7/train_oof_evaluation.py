"""
5-fold stratified group out-of-fold (OOF) evaluation on the full Task 7 cohort.

Each subject appears in the outer test fold exactly once. Within each outer fold,
a 4-fold inner StratifiedGroupKFold on the 80% training pool selects the optimal
epoch count (no outer-test leakage). A fresh model is then trained on the full
80% pool for that fixed epoch count and evaluated once on the 20% outer test fold.

After all outer folds, outer-test predictions for the full cohort are pooled
and global metrics plus a confusion matrix (threshold 0.5) are reported per
XY filter (AUC, accuracy, sensitivity, specificity).

Explainability (XAI) uses the **FFT** filter only; see ``checkpoint_io.XAI_XY_FILTER``
and ``train_xai_checkpoint.py``.

Examples::

    python DysXAI_task7/train_oof_evaluation.py
    python DysXAI_task7/train_oof_evaluation.py --max-epochs 40

Checkpoints (default ``DysXAI_task7/checkpoints/``)::

    oof_<filter>_age_fold01.pt … fold05.pt   # one per outer fold

Load for explainability::

    from checkpoint_io import load_task7_checkpoint
    model, meta = load_task7_checkpoint(path, device)
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dataset import (  # noqa: E402
    TASK7_DATA_DIR,
    Task5TrajectoryDataset,
    build_sample_table,
    collate_task7,
    fit_channel_scaling,
)
from model import Task5Conv1dClassifier  # noqa: E402
from checkpoint_io import (  # noqa: E402
    checkpoint_basename,
    default_checkpoint_dir,
    save_task7_checkpoint,
)
from train_ab_test import (  # noqa: E402
    BATCH_SIZE,
    LR,
    RANDOM_STATE,
    WEIGHT_DECAY,
    _train_fold_age_minmax,
    evaluate,
    set_seed,
)

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4
MAX_EPOCHS = 40
FREQ_GRID_HZ = (8.0, 10.0, 12.0, 15.0)

FILTER_ORDER = ["raw", "butterworth", "fft"]
FILTER_LABELS = {
    "raw": "Raw (No Filter)",
    "butterworth": "Tuned Butterworth",
    "fft": "Tuned FFT",
}

DEFAULT_RESULTS_CSV = "task7_oof_results.csv"
DEFAULT_DEMOGRAPHIC_ABLATION_CSV = "task7_demographic_ablation_results.csv"
DEFAULT_ABLATION_FOLD_HPARAMS_CSV = "task7_ablation_oof_fold_hparams.csv"
CLASSIFICATION_THRESHOLD = 0.5


@dataclass(frozen=True)
class OOFMetrics:
    auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    n_samples: int
    n_dysgraphic: int
    n_control: int


def metrics_from_oof_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> OOFMetrics:
    """Global metrics on pooled out-of-fold predictions (threshold 0.5 for accuracy/sens/spec)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    n = len(y_true)

    if len(np.unique(y_true)) < 2:
        return OOFMetrics(
            auc=float("nan"),
            accuracy=float("nan"),
            sensitivity=float("nan"),
            specificity=float("nan"),
            n_samples=n,
            n_dysgraphic=n_pos,
            n_control=n_neg,
        )

    y_pred = (y_prob >= CLASSIFICATION_THRESHOLD).astype(np.int64)
    auc = float(roc_auc_score(y_true, y_prob))
    accuracy = float(accuracy_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return OOFMetrics(
        auc=auc,
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        n_samples=n,
        n_dysgraphic=n_pos,
        n_control=n_neg,
    )


def validate_oof_pooling(
    sample_df,
    y_true_pooled: np.ndarray,
    y_prob_pooled: np.ndarray,
    *,
    n_outer_folds: int = N_OUTER_SPLITS,
) -> None:
    """
    Confirm pooled OOF arrays cover the full cohort exactly once.

    Predictions must come only from outer test folds (enforced by callers).
    StratifiedGroupKFold guarantees each subject appears in exactly one test fold.
    """
    n_cohort = len(sample_df)
    n_true = len(y_true_pooled)
    n_prob = len(y_prob_pooled)
    if n_true != n_cohort or n_prob != n_cohort:
        raise ValueError(
            f"OOF pooling integrity failed: expected {n_cohort} pooled predictions "
            f"(one per cohort row), got y_true={n_true}, y_prob={n_prob}. "
            f"Each of {n_outer_folds} outer test folds must contribute unseen "
            f"test_idx predictions only."
        )
    if n_true != n_prob:
        raise ValueError(
            f"OOF label/probability length mismatch: y_true={n_true}, y_prob={n_prob}"
        )
    n_subjects = sample_df["subject_id"].nunique()
    print(
        f"  OOF pooling OK: n={n_cohort} cohort rows, "
        f"{n_subjects} unique subjects, {n_outer_folds} outer folds"
    )


def print_global_oof_confusion_matrix(
    y_true_pooled: np.ndarray,
    y_prob_pooled: np.ndarray,
    *,
    filter_label: str,
) -> None:
    """Global confusion matrix on pooled out-of-fold predictions (threshold 0.5)."""
    y_true = np.asarray(y_true_pooled, dtype=np.int64)
    y_prob = np.asarray(y_prob_pooled, dtype=np.float64)
    y_pred_pooled = (y_prob >= CLASSIFICATION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_pooled, labels=[0, 1]).ravel()
    n = len(y_true)
    print()
    print("=" * 60)
    print(f"GLOBAL OUT-OF-FOLD CONFUSION MATRIX ({filter_label}, N = {n})")
    print("=" * 60)
    print("Predicted Control (0)  |  Predicted Dysgraphic (1)")
    print(f"Actual Control (0):     TN: {tn:3d}  |  FP: {fp:3d}")
    print(f"Actual Dysgraphic (1):  FN: {fn:3d}  |  TP: {tp:3d}")
    print("=" * 60)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_age: bool,
    use_gender: bool,
    age_min: float,
    age_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys: list[float] = []
    probs: list[float] = []
    subject_ids: list[int] = []
    denom = (age_max - age_min) + 1e-8
    for batch in loader:
        x = batch["x"].to(device)
        lengths = batch["length"].to(device)
        y = batch["y"].cpu().numpy()
        subj = batch["subject_id"].cpu().numpy()
        age_tensor = None
        gender_tensor = None
        if use_age:
            batch_ages = batch["age"].to(device).float()
            age_tensor = (batch_ages - age_min) / denom
        if use_gender:
            gender_tensor = batch["gender"].to(device).float()
        logits = model(
            x, lengths=lengths, age=age_tensor, gender=gender_tensor
        ).squeeze(-1).cpu().numpy()
        ys.extend(y.astype(float).tolist())
        probs.extend(torch.sigmoid(torch.tensor(logits)).numpy().tolist())
        subject_ids.extend(subj.astype(int).tolist())
    return (
        np.asarray(ys, dtype=np.float64),
        np.asarray(probs, dtype=np.float64),
        np.asarray(subject_ids, dtype=np.int64),
    )


def _make_loaders(
    sample_df,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    *,
    xy_filter: str,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    shuffle_train: bool,
    use_age: bool,
) -> tuple[DataLoader, DataLoader, float, float]:
    age_min, age_max = _train_fold_age_minmax(sample_df, train_idx)
    train_ds = Task5TrajectoryDataset(
        sample_df,
        train_idx,
        use_age_channel=use_age,
        channel_mean=channel_mean,
        channel_std=channel_std,
        xy_filter=xy_filter,
    )
    eval_ds = Task5TrajectoryDataset(
        sample_df,
        eval_idx,
        use_age_channel=use_age,
        channel_mean=channel_mean,
        channel_std=channel_std,
        xy_filter=xy_filter,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle_train,
        collate_fn=collate_task7,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_task7,
        num_workers=0,
    )
    return train_loader, eval_loader, age_min, age_max


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    opt: torch.optim.Optimizer,
    crit: nn.Module,
    device: torch.device,
    *,
    use_age: bool,
    use_gender: bool,
    age_min: float,
    age_max: float,
) -> None:
    model.train()
    denom = (age_max - age_min) + 1e-8
    for batch in train_loader:
        x = batch["x"].to(device)
        lengths = batch["length"].to(device)
        y = batch["y"].to(device).unsqueeze(1)
        age_tensor = None
        gender_tensor = None
        if use_age:
            batch_ages = batch["age"].to(device).float()
            age_tensor = (batch_ages - age_min) / denom
        if use_gender:
            gender_tensor = batch["gender"].to(device).float()
        opt.zero_grad(set_to_none=True)
        logits = model(x, lengths=lengths, age=age_tensor, gender=gender_tensor)
        loss = crit(logits, y)
        loss.backward()
        opt.step()


def _inner_fold_val_auc_curve(
    sample_df,
    inner_train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    *,
    xy_filter: str,
    device: torch.device,
    outer_fold_id: int,
    inner_fold_id: int,
    max_epochs: int,
    use_age: bool,
    use_gender: bool,
) -> list[float]:
    """Train fresh model on inner train; return validation AUC after each epoch."""
    mu, sigma = fit_channel_scaling(sample_df, inner_train_idx, xy_filter=xy_filter)
    train_loader, val_loader, age_min, age_max = _make_loaders(
        sample_df,
        inner_train_idx,
        inner_val_idx,
        xy_filter=xy_filter,
        channel_mean=mu,
        channel_std=sigma,
        shuffle_train=True,
        use_age=use_age,
    )

    model = Task5Conv1dClassifier(use_age=use_age, use_gender=use_gender).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()

    val_aucs: list[float] = []
    for ep in range(1, max_epochs + 1):
        _train_one_epoch(
            model,
            train_loader,
            opt,
            crit,
            device,
            use_age=use_age,
            use_gender=use_gender,
            age_min=age_min,
            age_max=age_max,
        )
        val_auc = evaluate(
            model,
            val_loader,
            device,
            use_age=use_age,
            use_gender=use_gender,
            age_min=age_min,
            age_max=age_max,
        )
        val_aucs.append(val_auc)
        if ep == 1 or ep % 5 == 0 or ep == max_epochs:
            print(
                f"      Outer {outer_fold_id} inner {inner_fold_id}/{N_INNER_SPLITS} "
                f"epoch {ep}/{max_epochs} val_auc={val_auc:.4f}"
            )
    return val_aucs


def _fmt_freq_token(freq_hz: float) -> str:
    value = float(freq_hz)
    return str(int(value)) if value.is_integer() else str(value)


def select_best_epoch_inner_cv(
    sample_df,
    outer_train_idx: np.ndarray,
    *,
    xy_filter: str,
    device: torch.device,
    outer_fold_id: int,
    max_epochs: int,
    use_age: bool,
    use_gender: bool,
) -> int:
    """
    4-fold inner CV on the outer training pool.
    Returns 1-based best_epoch with highest mean inner validation AUC.
    """
    train_df = sample_df.iloc[outer_train_idx]
    y = train_df["label"].to_numpy()
    groups = train_df["subject_id"].to_numpy()
    inner_sgkf = StratifiedGroupKFold(
        n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    inner_curves: list[list[float]] = []
    for inner_fold_id, (inner_tr_rel, inner_val_rel) in enumerate(
        inner_sgkf.split(np.zeros(len(train_df)), y, groups), start=1
    ):
        inner_train_idx = outer_train_idx[inner_tr_rel]
        inner_val_idx = outer_train_idx[inner_val_rel]
        print(
            f"    Inner fold {inner_fold_id}/{N_INNER_SPLITS}: "
            f"train n={len(inner_train_idx)}, val n={len(inner_val_idx)}"
        )
        curve = _inner_fold_val_auc_curve(
            sample_df,
            inner_train_idx,
            inner_val_idx,
            xy_filter=xy_filter,
            device=device,
            outer_fold_id=outer_fold_id,
            inner_fold_id=inner_fold_id,
            max_epochs=max_epochs,
            use_age=use_age,
            use_gender=use_gender,
        )
        inner_curves.append(curve)

    mean_auc_by_epoch = np.nanmean(np.asarray(inner_curves, dtype=np.float64), axis=0)
    best_epoch = int(np.nanargmax(mean_auc_by_epoch)) + 1
    best_mean_auc = float(mean_auc_by_epoch[best_epoch - 1])
    print(
        f"    Inner CV selected best_epoch={best_epoch} "
        f"(mean inner val AUC={best_mean_auc:.4f})"
    )
    return best_epoch


def select_best_filter_hparams_inner_cv(
    sample_df,
    outer_train_idx: np.ndarray,
    *,
    base_filter: str,
    device: torch.device,
    outer_fold_id: int,
    max_epochs: int,
    freq_grid_hz: tuple[float, ...] = FREQ_GRID_HZ,
    use_age: bool = True,
    use_gender: bool = False,
) -> tuple[float, int, float]:
    """
    Inner CV joint search over frequency cutoff and epoch count.
    Returns ``(best_freq_hz, best_epoch)`` maximizing mean inner validation AUC.
    """
    best_freq = float(freq_grid_hz[0])
    best_epoch = 1
    best_auc = -np.inf
    for freq_hz in freq_grid_hz:
        xy_filter = f"{base_filter}_{_fmt_freq_token(freq_hz)}"
        print(f"    Trying {xy_filter} (inner CV over epochs 1..{max_epochs})")
        train_df = sample_df.iloc[outer_train_idx]
        y = train_df["label"].to_numpy()
        groups = train_df["subject_id"].to_numpy()
        inner_sgkf = StratifiedGroupKFold(
            n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        inner_curves: list[list[float]] = []
        for inner_fold_id, (inner_tr_rel, inner_val_rel) in enumerate(
            inner_sgkf.split(np.zeros(len(train_df)), y, groups), start=1
        ):
            inner_train_idx = outer_train_idx[inner_tr_rel]
            inner_val_idx = outer_train_idx[inner_val_rel]
            print(
                f"    Inner fold {inner_fold_id}/{N_INNER_SPLITS}: "
                f"train n={len(inner_train_idx)}, val n={len(inner_val_idx)}"
            )
            curve = _inner_fold_val_auc_curve(
                sample_df,
                inner_train_idx,
                inner_val_idx,
                xy_filter=xy_filter,
                device=device,
                outer_fold_id=outer_fold_id,
                inner_fold_id=inner_fold_id,
                max_epochs=max_epochs,
                use_age=use_age,
                use_gender=use_gender,
            )
            inner_curves.append(curve)
        mean_auc_by_epoch = np.nanmean(np.asarray(inner_curves, dtype=np.float64), axis=0)
        freq_best_epoch = int(np.nanargmax(mean_auc_by_epoch)) + 1
        freq_best_auc = float(mean_auc_by_epoch[freq_best_epoch - 1])
        print(
            f"    {xy_filter} -> best_epoch={freq_best_epoch}, "
            f"mean inner val AUC={freq_best_auc:.4f}"
        )
        if freq_best_auc > best_auc:
            best_auc = freq_best_auc
            best_freq = float(freq_hz)
            best_epoch = int(freq_best_epoch)
    return best_freq, best_epoch, float(best_auc)


def train_outer_fold_and_evaluate(
    sample_df,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    xy_filter: str,
    device: torch.device,
    outer_fold_id: int,
    best_epoch: int,
    use_age: bool,
    use_gender: bool,
    checkpoint_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fresh model on full outer train pool for exactly best_epoch epochs;
    single evaluation on outer test fold.
    """
    mu, sigma = fit_channel_scaling(sample_df, train_idx, xy_filter=xy_filter)
    train_loader, test_loader, age_min, age_max = _make_loaders(
        sample_df,
        train_idx,
        test_idx,
        xy_filter=xy_filter,
        channel_mean=mu,
        channel_std=sigma,
        shuffle_train=True,
        use_age=use_age,
    )

    model = Task5Conv1dClassifier(use_age=use_age, use_gender=use_gender).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()

    for ep in range(1, best_epoch + 1):
        _train_one_epoch(
            model,
            train_loader,
            opt,
            crit,
            device,
            use_age=use_age,
            use_gender=use_gender,
            age_min=age_min,
            age_max=age_max,
        )
        if ep == 1 or ep % 5 == 0 or ep == best_epoch:
            print(f"    Outer fold {outer_fold_id} retrain epoch {ep}/{best_epoch}")

    if checkpoint_dir is not None:
        ckpt_name = checkpoint_basename(
            split="oof",
            xy_filter=xy_filter,
            use_age=use_age,
            use_gender=use_gender,
            outer_fold_id=outer_fold_id,
        )
        ckpt_path = save_task7_checkpoint(
            checkpoint_dir / ckpt_name,
            model,
            xy_filter=xy_filter,
            use_age=use_age,
            use_gender=use_gender,
            best_epoch=best_epoch,
            channel_mean=mu,
            channel_std=sigma,
            age_min=age_min,
            age_max=age_max,
            split="oof",
            outer_fold_id=outer_fold_id,
            random_state=RANDOM_STATE,
            extra={"n_train": len(train_idx), "n_test": len(test_idx)},
        )
        print(f"    Saved checkpoint: {ckpt_path}")

    return collect_predictions(
        model,
        test_loader,
        device,
        use_age=use_age,
        use_gender=use_gender,
        age_min=age_min,
        age_max=age_max,
    )


def run_oof_for_filter(
    sample_df,
    *,
    xy_filter: str,
    device: torch.device,
    max_epochs: int,
    use_age: bool,
    use_gender: bool,
    checkpoint_dir: Path | None = None,
    configuration_label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OOFMetrics, list[FoldHyperparamRow]]:
    """Pool outer-test-fold predictions only; validate full-cohort coverage."""
    y_true_pooled: list[float] = []
    y_prob_pooled: list[float] = []
    subj_pooled: list[int] = []
    fold_rows: list[FoldHyperparamRow] = []

    y = sample_df["label"].to_numpy()
    groups = sample_df["subject_id"].to_numpy()
    outer_sgkf = StratifiedGroupKFold(
        n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    for outer_fold_id, (train_idx, test_idx) in enumerate(
        outer_sgkf.split(np.zeros(len(sample_df)), y, groups), start=1
    ):
        n_train = len(train_idx)
        n_test = len(test_idx)
        print(
            f"  Outer fold {outer_fold_id}/{N_OUTER_SPLITS}: "
            f"train n={n_train}, test n={n_test}"
        )
        tuned_xy_filter = xy_filter
        if xy_filter == "raw":
            print("  Inner CV (epoch selection on outer train pool only)...")
            best_epoch = select_best_epoch_inner_cv(
                sample_df,
                train_idx,
                xy_filter=xy_filter,
                device=device,
                outer_fold_id=outer_fold_id,
                max_epochs=max_epochs,
                use_age=use_age,
                use_gender=use_gender,
            )
            print(
                f"  Outer Fold {outer_fold_id}: Tuned Hyperparameters -> "
                f"Filter: raw, Epoch: {best_epoch}"
            )
            selected_freq_hz = float("nan")
            inner_best_mean_auc = float("nan")
        elif xy_filter in ("butterworth", "fft"):
            print(
                "  Inner CV (joint selection of cutoff frequency and epoch "
                f"on outer train pool only)... freq_grid={list(FREQ_GRID_HZ)}"
            )
            best_freq, best_epoch, inner_best_mean_auc = select_best_filter_hparams_inner_cv(
                sample_df,
                train_idx,
                base_filter=xy_filter,
                device=device,
                outer_fold_id=outer_fold_id,
                max_epochs=max_epochs,
                use_age=use_age,
                use_gender=use_gender,
            )
            tuned_xy_filter = f"{xy_filter}_{_fmt_freq_token(best_freq)}"
            print(
                f"  Outer Fold {outer_fold_id}: Tuned Hyperparameters -> "
                f"Freq: {_fmt_freq_token(best_freq)}Hz, Epoch: {best_epoch}"
            )
            selected_freq_hz = float(best_freq)
        else:
            raise ValueError(
                f"Unsupported base filter {xy_filter!r}; choose from {FILTER_ORDER}"
            )

        print(
            f"  Retraining on full outer train using {tuned_xy_filter} "
            f"for {best_epoch} epoch(s)..."
        )
        # Predictions strictly from outer test_idx (unseen during train/inner CV).
        fold_y_true, fold_y_prob, fold_subject_ids = train_outer_fold_and_evaluate(
            sample_df,
            train_idx,
            test_idx,
            xy_filter=tuned_xy_filter,
            device=device,
            outer_fold_id=outer_fold_id,
            best_epoch=best_epoch,
            use_age=use_age,
            use_gender=use_gender,
            checkpoint_dir=checkpoint_dir,
        )
        if len(fold_y_true) != len(test_idx):
            raise ValueError(
                f"Outer fold {outer_fold_id}: expected {len(test_idx)} test predictions, "
                f"got {len(fold_y_true)}"
            )
        if len(fold_subject_ids) != len(test_idx):
            raise ValueError(
                f"Outer fold {outer_fold_id}: expected {len(test_idx)} subject IDs, "
                f"got {len(fold_subject_ids)}"
            )
        fold_rows.append(
            FoldHyperparamRow(
                filter_key=xy_filter,
                outer_fold=outer_fold_id,
                selected_filter=tuned_xy_filter,
                selected_freq_hz=selected_freq_hz,
                selected_epoch=best_epoch,
                inner_best_mean_auc=inner_best_mean_auc,
                n_train=n_train,
                n_test=n_test,
                configuration=configuration_label,
                use_age=use_age,
                use_gender=use_gender,
            )
        )
        y_true_pooled.extend(fold_y_true.tolist())
        y_prob_pooled.extend(fold_y_prob.tolist())
        subj_pooled.extend(fold_subject_ids.tolist())

    y_arr = np.asarray(y_true_pooled, dtype=np.float64)
    p_arr = np.asarray(y_prob_pooled, dtype=np.float64)
    s_arr = np.asarray(subj_pooled, dtype=np.int64)
    validate_oof_pooling(sample_df, y_arr, p_arr)
    metrics = metrics_from_oof_predictions(y_arr, p_arr)
    return y_arr, p_arr, s_arr, metrics, fold_rows


def _fmt_metric(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "—"


def print_metrics_summary(
    xy_filter: str,
    metrics: OOFMetrics,
    y_true_pooled: np.ndarray,
    y_prob_pooled: np.ndarray,
) -> None:
    filter_label = FILTER_LABELS.get(xy_filter, xy_filter)
    print(
        f"\nGlobal OOF ({xy_filter}, n={metrics.n_samples}, "
        f"dys={metrics.n_dysgraphic}, ctrl={metrics.n_control}):"
    )
    print(f"  AUC          {_fmt_metric(metrics.auc)}")
    print(f"  Accuracy     {_fmt_metric(metrics.accuracy)}")
    print(f"  Sensitivity  {_fmt_metric(metrics.sensitivity)}  (dysgraphic recall)")
    print(f"  Specificity  {_fmt_metric(metrics.specificity)}  (control recall)")
    print_global_oof_confusion_matrix(
        y_true_pooled,
        y_prob_pooled,
        filter_label=filter_label,
    )


@dataclass(frozen=True)
class FilterResultRow:
    filter_key: str
    filter_label: str
    metrics: OOFMetrics


@dataclass(frozen=True)
class FoldHyperparamRow:
    filter_key: str
    outer_fold: int
    selected_filter: str
    selected_freq_hz: float
    selected_epoch: int
    inner_best_mean_auc: float
    n_train: int
    n_test: int
    configuration: str = ""
    use_age: bool = False
    use_gender: bool = False


@dataclass(frozen=True)
class DemographicAblationRow:
    configuration: str
    metrics: OOFMetrics


def print_markdown_table(rows: list[FilterResultRow]) -> None:
    print()
    print(
        "| XY filter | Label | AUC | Accuracy | Sensitivity | Specificity | n |"
    )
    print(
        "|-----------|-------|-----|----------|-------------|-------------|---|"
    )
    for row in rows:
        m = row.metrics
        print(
            f"| {row.filter_key} | {row.filter_label} | "
            f"{_fmt_metric(m.auc)} | {_fmt_metric(m.accuracy)} | "
            f"{_fmt_metric(m.sensitivity)} | {_fmt_metric(m.specificity)} | {m.n_samples} |"
        )
    print()


def print_demographic_ablation_table(rows: list[DemographicAblationRow]) -> None:
    print()
    print("| Configuration | Global AUC | Accuracy | Sensitivity | Specificity |")
    print("|---------------|------------|----------|-------------|-------------|")
    for row in rows:
        m = row.metrics
        print(
            f"| {row.configuration} | {_fmt_metric(m.auc)} | {_fmt_metric(m.accuracy)} | "
            f"{_fmt_metric(m.sensitivity)} | {_fmt_metric(m.specificity)} |"
        )
    print()


def write_results_csv(path: Path, rows: list[FilterResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "xy_filter",
                "filter_label",
                "n",
                "n_dysgraphic",
                "n_control",
                "auc",
                "accuracy",
                "sensitivity",
                "specificity",
            ]
        )
        for row in rows:
            m = row.metrics
            w.writerow(
                [
                    row.filter_key,
                    row.filter_label,
                    m.n_samples,
                    m.n_dysgraphic,
                    m.n_control,
                    f"{m.auc:.6f}" if np.isfinite(m.auc) else "nan",
                    f"{m.accuracy:.6f}" if np.isfinite(m.accuracy) else "nan",
                    f"{m.sensitivity:.6f}" if np.isfinite(m.sensitivity) else "nan",
                    f"{m.specificity:.6f}" if np.isfinite(m.specificity) else "nan",
                ]
            )


def write_demographic_ablation_csv(path: Path, rows: list[DemographicAblationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "auc", "accuracy", "sensitivity", "specificity", "n"])
        for row in rows:
            m = row.metrics
            w.writerow(
                [
                    row.configuration,
                    f"{m.auc:.6f}" if np.isfinite(m.auc) else "nan",
                    f"{m.accuracy:.6f}" if np.isfinite(m.accuracy) else "nan",
                    f"{m.sensitivity:.6f}" if np.isfinite(m.sensitivity) else "nan",
                    f"{m.specificity:.6f}" if np.isfinite(m.specificity) else "nan",
                    m.n_samples,
                ]
            )


def write_fold_hparams_csv(path: Path, rows: list[FoldHyperparamRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "configuration",
                "xy_filter",
                "use_age",
                "use_gender",
                "outer_fold",
                "selected_filter",
                "selected_freq_hz",
                "selected_epoch",
                "inner_best_mean_auc",
                "n_train",
                "n_test",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row.configuration,
                    row.filter_key,
                    int(row.use_age),
                    int(row.use_gender),
                    row.outer_fold,
                    row.selected_filter,
                    f"{row.selected_freq_hz:.6f}" if np.isfinite(row.selected_freq_hz) else "nan",
                    row.selected_epoch,
                    f"{row.inner_best_mean_auc:.6f}" if np.isfinite(row.inner_best_mean_auc) else "nan",
                    row.n_train,
                    row.n_test,
                ]
            )


def write_oof_predictions_csv(
    path: Path,
    *,
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    if len(subject_ids) != len(y_true) or len(y_true) != len(y_prob):
        raise ValueError(
            "OOF prediction export length mismatch: "
            f"subject_ids={len(subject_ids)}, y_true={len(y_true)}, y_prob={len(y_prob)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "true_label", "pred_prob"])
        for sid, yt, yp in zip(subject_ids, y_true, y_prob):
            w.writerow([int(sid), int(round(float(yt))), float(yp)])


def _fold_hparams_sort_key(row: FoldHyperparamRow) -> tuple:
    cfg = row.configuration or row.filter_key
    try:
        filt_idx = FILTER_ORDER.index(row.filter_key)
    except ValueError:
        filt_idx = len(FILTER_ORDER)
    return (cfg, filt_idx, row.outer_fold)


def print_fold_hparams_table(rows: list[FoldHyperparamRow]) -> None:
    if not rows:
        return
    has_configuration = any(r.configuration for r in rows)
    print()
    print("Tuned Hyperparameters Per Outer Fold")
    print("-" * 110)
    if has_configuration:
        print(
            "| Configuration | Filter | Fold | Selected filter | Freq (Hz) | Epoch | Inner mean AUC |"
        )
        print(
            "|---------------|--------|------|-----------------|-----------|-------|----------------|"
        )
    else:
        print(
            "| Base filter | Fold | Selected filter | Freq (Hz) | Epoch | Inner mean AUC |"
        )
        print(
            "|-------------|------|-----------------|-----------|-------|----------------|"
        )
    for row in sorted(rows, key=_fold_hparams_sort_key):
        freq_txt = (
            _fmt_freq_token(row.selected_freq_hz) if np.isfinite(row.selected_freq_hz) else "—"
        )
        auc_txt = f"{row.inner_best_mean_auc:.4f}" if np.isfinite(row.inner_best_mean_auc) else "—"
        if has_configuration:
            print(
                f"| {row.configuration:<29} | {row.filter_key:<6} | {row.outer_fold:>4} | "
                f"{row.selected_filter:<15} | {freq_txt:>9} | {row.selected_epoch:>5} | {auc_txt:>14} |"
            )
        else:
            print(
                f"| {row.filter_key:<11} | {row.outer_fold:>4} | "
                f"{row.selected_filter:<15} | {freq_txt:>9} | {row.selected_epoch:>5} | {auc_txt:>14} |"
            )
    print("-" * 110)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "5-fold nested stratified group OOF evaluation "
            "(inner 4-fold epoch selection, full cohort, 3 filters)."
        ),
    )
    parser.add_argument(
        "--filters",
        type=str,
        default=",".join(FILTER_ORDER),
        help="Comma-separated: raw,butterworth,fft",
    )
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument(
        "--use-age",
        action="store_true",
        help="Use late-fusion min-max normalized age feature.",
    )
    parser.add_argument(
        "--use-gender",
        action="store_true",
        help="Use late-fusion gender feature (0=male, 1=female).",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=_HERE / DEFAULT_RESULTS_CSV,
    )
    parser.add_argument(
        "--fold-hparams-csv",
        type=Path,
        default=None,
        help=(
            "Per-outer-fold tuned hyperparameters CSV. "
            "Default: task7_ablation_oof_fold_hparams.csv (ablation) or "
            "task7_oof_fold_hparams.csv (standard OOF)."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=default_checkpoint_dir(_HERE),
        help="Directory for per-fold OOF checkpoints (default: DysXAI_task7/checkpoints/).",
    )
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="Skip writing model checkpoints (evaluation metrics only).",
    )
    parser.add_argument(
        "--ablation-demographics",
        action="store_true",
        help=(
            "Run the 4-way demographic ablation study (None, Age, Gender, Both) "
            "using the FFT filter."
        ),
    )
    args = parser.parse_args()

    if args.max_epochs < 1:
        parser.error("--max-epochs must be >= 1")

    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    for f in filters:
        if f not in FILTER_ORDER:
            parser.error(f"Unknown filter {f!r}; choose from {FILTER_ORDER}")

    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task 7 data: {TASK7_DATA_DIR}")
    print(
        f"Nested CV: outer StratifiedGroupKFold(n_splits={N_OUTER_SPLITS}), "
        f"inner n_splits={N_INNER_SPLITS}, random_state={RANDOM_STATE}"
    )
    print(f"Epoch grid: 1..{args.max_epochs} (inner mean val AUC)")
    print(f"Late fusion: use_age={bool(args.use_age)}, use_gender={bool(args.use_gender)}")
    ckpt_dir: Path | None = None
    if not args.no_save_checkpoints:
        ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else _HERE / args.checkpoint_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints: {ckpt_dir.resolve()}")
    else:
        print("Checkpoints: disabled (--no-save-checkpoints)")

    sample_df = build_sample_table()
    n = len(sample_df)
    print(f"Full cohort: n={n} samples")
    if n == 0:
        raise SystemExit("No samples in sample table.")

    if args.fold_hparams_csv is None:
        fold_hparams_default_name = (
            DEFAULT_ABLATION_FOLD_HPARAMS_CSV
            if args.ablation_demographics
            else "task7_oof_fold_hparams.csv"
        )
        fold_hparams_csv_path = _HERE / fold_hparams_default_name
    else:
        fold_hparams_csv_path = (
            args.fold_hparams_csv
            if args.fold_hparams_csv.is_absolute()
            else _HERE / args.fold_hparams_csv
        )

    if args.ablation_demographics:
        ablation_configs = [
            ("Baseline (Kinematics Only)", False, False),
            ("Baseline + Age", True, False),
            ("Baseline + Gender", False, True),
            ("Baseline + Age + Gender", True, True),
        ]
        ablation_rows: list[DemographicAblationRow] = []
        ablation_fold_hparams_rows: list[FoldHyperparamRow] = []
        print("\n" + "=" * 60)
        print("Demographic Ablation Study (FFT only)")
        print("=" * 60)
        for label, use_age_cfg, use_gender_cfg in ablation_configs:
            print(f"\n--- {label} | xy_filter=fft ---")
            y_true_pooled, y_prob_pooled, subj_pooled, metrics, fold_rows = run_oof_for_filter(
                sample_df,
                xy_filter="fft",
                device=device,
                max_epochs=args.max_epochs,
                use_age=use_age_cfg,
                use_gender=use_gender_cfg,
                checkpoint_dir=ckpt_dir,
                configuration_label=label,
            )
            ablation_fold_hparams_rows.extend(fold_rows)
            print_metrics_summary("fft", metrics, y_true_pooled, y_prob_pooled)
            pred_csv_path = _HERE / f"oof_predictions_fft_{label.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')}.csv"
            write_oof_predictions_csv(
                pred_csv_path,
                subject_ids=subj_pooled,
                y_true=y_true_pooled,
                y_prob=y_prob_pooled,
            )
            print(f"Wrote: {pred_csv_path.resolve()}")
            ablation_rows.append(DemographicAblationRow(configuration=label, metrics=metrics))

        print_demographic_ablation_table(ablation_rows)
        ablation_csv_path = _HERE / DEFAULT_DEMOGRAPHIC_ABLATION_CSV
        write_demographic_ablation_csv(ablation_csv_path, ablation_rows)
        print(f"Wrote: {ablation_csv_path.resolve()}")

        write_fold_hparams_csv(fold_hparams_csv_path, ablation_fold_hparams_rows)
        print(f"Wrote: {fold_hparams_csv_path.resolve()}")
        print_fold_hparams_table(ablation_fold_hparams_rows)
        return

    table_rows: list[FilterResultRow] = []
    fold_hparams_rows: list[FoldHyperparamRow] = []

    for xy_filter in filters:
        print(f"\n{'=' * 60}")
        print(f"OOF evaluation: {FILTER_LABELS[xy_filter]} ({xy_filter})")
        print(f"{'=' * 60}")
        y_true_pooled, y_prob_pooled, subj_pooled, metrics, fold_rows = run_oof_for_filter(
            sample_df,
            xy_filter=xy_filter,
            device=device,
            max_epochs=args.max_epochs,
            use_age=bool(args.use_age),
            use_gender=bool(args.use_gender),
            checkpoint_dir=ckpt_dir,
        )
        print_metrics_summary(xy_filter, metrics, y_true_pooled, y_prob_pooled)
        pred_csv_path = _HERE / f"oof_predictions_{xy_filter}.csv"
        write_oof_predictions_csv(
            pred_csv_path,
            subject_ids=subj_pooled,
            y_true=y_true_pooled,
            y_prob=y_prob_pooled,
        )
        print(f"Wrote: {pred_csv_path.resolve()}")
        fold_hparams_rows.extend(fold_rows)
        table_rows.append(
            FilterResultRow(
                filter_key=xy_filter,
                filter_label=FILTER_LABELS[xy_filter],
                metrics=metrics,
            )
        )

    print_markdown_table(table_rows)
    csv_path = args.results_csv if args.results_csv.is_absolute() else _HERE / args.results_csv
    write_results_csv(csv_path, table_rows)
    print(f"Wrote: {csv_path.resolve()}")
    write_fold_hparams_csv(fold_hparams_csv_path, fold_hparams_rows)
    print(f"Wrote: {fold_hparams_csv_path.resolve()}")
    print_fold_hparams_table(fold_hparams_rows)


if __name__ == "__main__":
    main()
