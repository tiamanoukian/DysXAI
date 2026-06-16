"""
Permutation feature ablation on saved FFT OOF checkpoints (age + gender late fusion).

For each outer fold, loads a saved OOF checkpoint and scores that fold's **outer
test split only** (20% of the full cohort, ~24 clips per fold; subjects never in
that fold's training pool). Permutation ablation shuffles each feature independently;
records delta = baseline - ablated for **AUC** and **accuracy** (threshold 0.5).

Aggregates mean +/- std across 5 folds; writes CSV + horizontal bar chart per metric.

Does **not** train or rerun ``train_oof_evaluation.py`` — only loads existing ``.pt``
files from your demographic ablation run (default: ``Baseline + Age + Gender``).

Examples::

    python DysXAI_task7/explain_oof_ablation.py
    python DysXAI_task7/explain_oof_ablation.py --checkpoint-dir checkpoints
    python DysXAI_task7/explain_oof_ablation.py --configuration "Baseline + Age + Gender"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from checkpoint_io import default_checkpoint_dir, load_task7_checkpoint  # noqa: E402
from dataset import (  # noqa: E402
    TASK7_DATA_DIR,
    Task5TrajectoryDataset,
    build_sample_table,
    collate_task7,
)
from train_ab_test import BATCH_SIZE, RANDOM_STATE, set_seed  # noqa: E402

N_OUTER_SPLITS = 5
KINEMATIC_FEATURES = [
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
]
DEMOGRAPHIC_FEATURES = ["Age", "Gender"]
FEATURE_NAMES = KINEMATIC_FEATURES + DEMOGRAPHIC_FEATURES
N_KINEMATIC = len(KINEMATIC_FEATURES)

DEFAULT_RESULTS_CSV = "task7_oof_feature_importance.csv"
DEFAULT_RESULTS_PNG = "task7_oof_feature_importance.png"
DEFAULT_RESULTS_CSV_ACC = "task7_oof_feature_importance_accuracy.csv"
DEFAULT_RESULTS_PNG_ACC = "task7_oof_feature_importance_accuracy.png"
DEFAULT_PROTOCOL_TXT = "task7_oof_feature_importance_protocol.txt"
DEFAULT_RESULTS_DIR = _HERE / "XAI results" / "results feature importance"
DEFAULT_FOLD_HPARAMS_CSV = "task7_ablation_oof_fold_hparams.csv"
DEFAULT_CONFIGURATION = "Baseline + Age + Gender"
CLASSIFICATION_THRESHOLD = 0.5
KINEMATIC_COLOR = "#1f77b4"
DEMOGRAPHIC_COLOR = "#ff7f0e"

EVALUATION_PROTOCOL = """\
Task 7 OOF permutation feature importance — evaluation protocol
============================================================

Data: full Task 7 cohort (n=120 handwriting clips, subject-stratified).

Split (same as train_oof_evaluation.py):
  - 5-fold outer StratifiedGroupKFold (shuffle=True, random_state=42).
  - Each subject appears in exactly ONE outer test fold.
  - Per fold: ~80% train (used when the .pt checkpoint was trained) / ~20% test.

What this script evaluates:
  - ONLY the outer test fold for each saved checkpoint (inference only; no retraining).
  - Scaling (channel mean/std) and age min/max come from the checkpoint metadata
    (fitted on that fold's outer-train pool during the original OOF run).

Models:
  - Pretrained OOF checkpoints from the demographic ablation (default configuration:
    "Baseline + Age + Gender", FFT filter with per-fold tuned cutoff).

Metrics:
  - AUC: ROC-AUC on test-fold probabilities.
  - Accuracy: fraction correct at probability threshold 0.5.
  - Feature importance: mean delta metric across 5 folds, where
      delta = baseline(test fold) - ablated(test fold)
    and ablation = shuffle that feature across batch items (permutation).

Not evaluated here:
  - Holdout 20% subject split (train_final_evaluation.py).
  - Inner-CV validation sets used only for epoch/frequency tuning during training.
"""


def feature_type(name: str) -> str:
    return "Demographic" if name in DEMOGRAPHIC_FEATURES else "Kinematic"


def load_checkpoint_manifest(
    hparams_csv: Path,
    checkpoint_dir: Path,
    configuration: str,
) -> dict[int, Path]:
    """
    Map outer_fold -> checkpoint path using ``task7_ablation_oof_fold_hparams.csv``.

    Avoids guessing filenames when several ablation configs share the same folder.
    """
    if not hparams_csv.is_file():
        raise FileNotFoundError(f"Fold hparams CSV not found: {hparams_csv}")
    df = pd.read_csv(hparams_csv)
    if "configuration" not in df.columns or "checkpoint_file" not in df.columns:
        raise ValueError(
            f"{hparams_csv.name} must include 'configuration' and 'checkpoint_file' columns."
        )
    sub = df[df["configuration"].astype(str) == configuration]
    if sub.empty:
        available = sorted(df["configuration"].astype(str).unique())
        raise ValueError(
            f"No rows for configuration {configuration!r} in {hparams_csv}. "
            f"Available: {available}"
        )
    manifest: dict[int, Path] = {}
    for _, row in sub.iterrows():
        fold_id = int(row["outer_fold"])
        ckpt_path = checkpoint_dir / str(row["checkpoint_file"])
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint missing for fold {fold_id}: {ckpt_path}\n"
                f"Expected file from column checkpoint_file in {hparams_csv.name}."
            )
        manifest[fold_id] = ckpt_path
    if len(manifest) != N_OUTER_SPLITS:
        raise ValueError(
            f"Expected {N_OUTER_SPLITS} checkpoints for {configuration!r}, found {len(manifest)}"
        )
    return manifest


def find_oof_checkpoint(
    checkpoint_dir: Path,
    outer_fold_id: int,
    *,
    require_age: bool = True,
    require_gender: bool = True,
) -> Path:
    """Resolve per-fold FFT OOF checkpoint (supports tuned names like fft_12)."""
    pattern = f"oof_fft_*_fold{outer_fold_id:02d}.pt"
    candidates = sorted(checkpoint_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching {pattern!r} in {checkpoint_dir}"
        )
    for path in candidates:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        use_age = bool(payload.get("use_age", payload.get("use_late_fusion_age", False)))
        use_gender = bool(payload.get("use_gender", False))
        if use_age == require_age and use_gender == require_gender:
            return path
    raise FileNotFoundError(
        f"No FFT OOF checkpoint for fold {outer_fold_id} with "
        f"use_age={require_age}, use_gender={require_gender} in {checkpoint_dir}"
    )


def _perm_generator(
    seed: int,
    outer_fold_id: int,
    feature_idx: int,
    batch_idx: int,
    device: torch.device,
) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed + outer_fold_id * 10_000 + feature_idx * 100 + batch_idx)
    return g


@torch.no_grad()
def evaluate_test_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_age: bool,
    use_gender: bool,
    age_min: float,
    age_max: float,
    ablate_feature: int | None = None,
    outer_fold_id: int = 1,
    perm_seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    """
    ROC-AUC and accuracy (threshold 0.5) on the outer test loader.

    ``ablate_feature``: 0–11 kinematic channel, 12 Age, 13 Gender, or None (baseline).
    """
    model.eval()
    ys: list[float] = []
    scores: list[float] = []
    denom = (age_max - age_min) + 1e-8

    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        lengths = batch["length"].to(device)
        y = batch["y"].cpu().numpy()
        B = x.size(0)

        age_tensor = None
        gender_tensor = None
        if use_age:
            ages = batch["age"].to(device).float()
            if ablate_feature == 12:
                perm = torch.randperm(
                    B,
                    device=device,
                    generator=_perm_generator(
                        perm_seed, outer_fold_id, 12, batch_idx, device
                    ),
                )
                ages = ages[perm]
            age_tensor = (ages - age_min) / denom
        if use_gender:
            genders = batch["gender"].to(device).float()
            if ablate_feature == 13:
                perm = torch.randperm(
                    B,
                    device=device,
                    generator=_perm_generator(
                        perm_seed, outer_fold_id, 13, batch_idx, device
                    ),
                )
                genders = genders[perm]
            gender_tensor = genders

        if ablate_feature is not None and 0 <= ablate_feature < N_KINEMATIC:
            perm = torch.randperm(
                B,
                device=device,
                generator=_perm_generator(
                    perm_seed, outer_fold_id, ablate_feature, batch_idx, device
                ),
            )
            x = x.clone()
            x[:, ablate_feature, :] = x[perm, ablate_feature, :]

        logits = model(
            x, lengths=lengths, age=age_tensor, gender=gender_tensor
        ).squeeze(-1).cpu().numpy()
        ys.extend(y.astype(float).tolist())
        scores.extend(torch.sigmoid(torch.tensor(logits)).numpy().tolist())

    ys_arr = np.asarray(ys, dtype=np.float64)
    s_arr = np.asarray(scores, dtype=np.float64)
    if len(np.unique(ys_arr)) < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(ys_arr, s_arr))
    y_pred = (s_arr >= CLASSIFICATION_THRESHOLD).astype(np.int64)
    acc = float(accuracy_score(ys_arr.astype(np.int64), y_pred))
    return auc, acc


def run_fold_ablation(
    sample_df,
    test_idx: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
    *,
    outer_fold_id: int,
    perm_seed: int,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    model, meta = load_task7_checkpoint(checkpoint_path, device)
    use_age = bool(meta["use_age"])
    use_gender = bool(meta["use_gender"])
    xy_filter = str(meta["xy_filter"])
    channel_mean = np.asarray(meta["channel_mean"], dtype=np.float32)
    channel_std = np.asarray(meta["channel_std"], dtype=np.float32)
    age_min = float(meta["age_min"])
    age_max = float(meta["age_max"])

    test_ds = Task5TrajectoryDataset(
        sample_df,
        test_idx,
        use_age_channel=use_age,
        channel_mean=channel_mean,
        channel_std=channel_std,
        xy_filter=xy_filter,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_task7,
        num_workers=0,
    )

    baseline_auc, baseline_acc = evaluate_test_metrics(
        model,
        test_loader,
        device,
        use_age=use_age,
        use_gender=use_gender,
        age_min=age_min,
        age_max=age_max,
        ablate_feature=None,
        outer_fold_id=outer_fold_id,
        perm_seed=perm_seed,
    )

    delta_auc: dict[str, float] = {}
    delta_acc: dict[str, float] = {}
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        if feat_name == "Age" and not use_age:
            delta_auc[feat_name] = float("nan")
            delta_acc[feat_name] = float("nan")
            continue
        if feat_name == "Gender" and not use_gender:
            delta_auc[feat_name] = float("nan")
            delta_acc[feat_name] = float("nan")
            continue
        ablated_auc, ablated_acc = evaluate_test_metrics(
            model,
            test_loader,
            device,
            use_age=use_age,
            use_gender=use_gender,
            age_min=age_min,
            age_max=age_max,
            ablate_feature=feat_idx,
            outer_fold_id=outer_fold_id,
            perm_seed=perm_seed,
        )
        delta_auc[feat_name] = baseline_auc - ablated_auc
        delta_acc[feat_name] = baseline_acc - ablated_acc

    return baseline_auc, baseline_acc, delta_auc, delta_acc


def aggregate_fold_deltas(
    fold_records: list[dict[str, object]],
    *,
    deltas_key: str,
    mean_col: str,
    std_col: str,
) -> pd.DataFrame:
    """Mean/std delta metric per feature; sort by descending mean."""
    rows: list[dict[str, object]] = []
    for feat in FEATURE_NAMES:
        deltas = [
            float(r[deltas_key][feat])
            for r in fold_records
            if deltas_key in r
            and feat in r[deltas_key]
            and np.isfinite(float(r[deltas_key][feat]))
        ]
        if not deltas:
            continue
        rows.append(
            {
                "feature": feat,
                "feature_type": feature_type(feat),
                mean_col: float(np.mean(deltas)),
                std_col: float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "n_folds": len(deltas),
            }
        )
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(mean_col, ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def write_results_csv(
    path: Path,
    summary: pd.DataFrame,
    fold_records: list[dict[str, object]],
    *,
    deltas_key: str,
    mean_col: str,
    baseline_key: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wide = summary.copy()
    for rec in fold_records:
        fold_id = int(rec["outer_fold"])
        col = f"fold_{fold_id:02d}_delta"
        wide[col] = [
            float(rec[deltas_key].get(feat, float("nan"))) for feat in wide["feature"]
        ]
        wide[f"fold_{fold_id:02d}_baseline"] = float(rec[baseline_key])

    with path.open("w", newline="", encoding="utf-8") as f:
        wide.to_csv(f, index=False)


def plot_feature_importance(
    path: Path,
    fold_records: list[dict[str, object]],
    summary: pd.DataFrame,
    *,
    deltas_key: str,
    mean_col: str,
    xlabel: str,
    title: str,
) -> None:
    long_rows: list[dict[str, object]] = []
    for rec in fold_records:
        fold_id = int(rec["outer_fold"])
        for feat in FEATURE_NAMES:
            val = rec[deltas_key].get(feat)
            if val is None or not np.isfinite(float(val)):
                continue
            long_rows.append(
                {
                    "feature": feat,
                    "feature_type": feature_type(feat),
                    "delta": float(val),
                    "outer_fold": fold_id,
                }
            )
    long_df = pd.DataFrame(long_rows)
    order = summary["feature"].tolist()

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=long_df,
        y="feature",
        x="delta",
        hue="feature_type",
        order=order,
        hue_order=["Kinematic", "Demographic"],
        palette={"Kinematic": KINEMATIC_COLOR, "Demographic": DEMOGRAPHIC_COLOR},
        dodge=False,
        errorbar="sd",
        capsize=0.08,
        err_kws={"linewidth": 1.2, "color": "0.25"},
        ax=ax,
        legend=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title)
    ax.axvline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=labels, title="", loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_protocol_txt(path: Path, *, configuration: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = EVALUATION_PROTOCOL + f"\nConfiguration used for this run: {configuration}\n"
    path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOF permutation ablation: rank 12 kinematics + age + gender.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=default_checkpoint_dir(_HERE),
        help="Folder containing saved OOF .pt files (default: DysXAI_task7/checkpoints/).",
    )
    parser.add_argument(
        "--fold-hparams-csv",
        type=Path,
        default=_HERE / DEFAULT_FOLD_HPARAMS_CSV,
        help=(
            "CSV with per-fold checkpoint_file names from the ablation run "
            f"(default: {DEFAULT_FOLD_HPARAMS_CSV})."
        ),
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default=DEFAULT_CONFIGURATION,
        help=(
            "Which ablation row set to use in --fold-hparams-csv "
            f"(default: {DEFAULT_CONFIGURATION!r})."
        ),
    )
    parser.add_argument(
        "--no-fold-hparams-csv",
        action="store_true",
        help="Discover checkpoints by glob instead of reading the hparams CSV.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory (default: DysXAI_task7/XAI results/results feature importance/).",
    )
    parser.add_argument(
        "--perm-seed",
        type=int,
        default=RANDOM_STATE,
        help="Base seed for batch-wise permutation (reproducible ablations).",
    )
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else _HERE / args.checkpoint_dir
    results_dir = args.results_dir if args.results_dir.is_absolute() else _HERE / args.results_dir
    csv_path = results_dir / DEFAULT_RESULTS_CSV
    png_path = results_dir / DEFAULT_RESULTS_PNG
    csv_acc_path = results_dir / DEFAULT_RESULTS_CSV_ACC
    png_acc_path = results_dir / DEFAULT_RESULTS_PNG_ACC
    protocol_path = results_dir / DEFAULT_PROTOCOL_TXT

    hparams_csv = (
        args.fold_hparams_csv
        if args.fold_hparams_csv.is_absolute()
        else _HERE / args.fold_hparams_csv
    )
    checkpoint_manifest: dict[int, Path] | None = None
    if not args.no_fold_hparams_csv:
        checkpoint_manifest = load_checkpoint_manifest(
            hparams_csv, ckpt_dir, args.configuration
        )

    print(f"Device: {device}")
    print(f"Task 7 data: {TASK7_DATA_DIR}")
    print(f"Checkpoints: {ckpt_dir.resolve()}")
    if checkpoint_manifest:
        print(f"Using fold manifest: {hparams_csv.name} ({args.configuration})")
    else:
        print("Checkpoint discovery: glob (age + gender)")
    print(f"Results: {results_dir.resolve()}")
    print()
    print("Evaluation: outer OOF test folds only (not holdout, not inner-CV val).")
    print(f"  Threshold for accuracy: {CLASSIFICATION_THRESHOLD}")
    print(f"  Configuration: {args.configuration}")

    sample_df = build_sample_table()
    y = sample_df["label"].to_numpy()
    groups = sample_df["subject_id"].to_numpy()
    outer_sgkf = StratifiedGroupKFold(
        n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    fold_records: list[dict[str, object]] = []
    for outer_fold_id, (_train_idx, test_idx) in enumerate(
        outer_sgkf.split(np.zeros(len(sample_df)), y, groups), start=1
    ):
        if checkpoint_manifest is not None:
            ckpt_path = checkpoint_manifest[outer_fold_id]
        else:
            ckpt_path = find_oof_checkpoint(ckpt_dir, outer_fold_id)
        print(
            f"\nOuter fold {outer_fold_id}/{N_OUTER_SPLITS}: "
            f"test n={len(test_idx)} | {ckpt_path.name}"
        )
        baseline_auc, baseline_acc, deltas_auc, deltas_acc = run_fold_ablation(
            sample_df,
            test_idx,
            ckpt_path,
            device,
            outer_fold_id=outer_fold_id,
            perm_seed=args.perm_seed,
        )
        print(
            f"  Baseline test  AUC: {baseline_auc:.4f}  |  "
            f"Accuracy: {baseline_acc:.4f}"
        )
        rec: dict[str, object] = {
            "outer_fold": outer_fold_id,
            "baseline_auc": baseline_auc,
            "baseline_accuracy": baseline_acc,
            "delta_auc": deltas_auc,
            "delta_accuracy": deltas_acc,
            "checkpoint": ckpt_path.name,
        }
        fold_records.append(rec)
        top_auc = max(deltas_auc.items(), key=lambda kv: kv[1])
        top_acc = max(deltas_acc.items(), key=lambda kv: kv[1])
        print(f"  Largest delta AUC this fold: {top_auc[0]} ({top_auc[1]:.4f})")
        print(f"  Largest delta Acc this fold: {top_acc[0]} ({top_acc[1]:.4f})")

    summary_auc = aggregate_fold_deltas(
        fold_records,
        deltas_key="delta_auc",
        mean_col="mean_delta_auc",
        std_col="std_delta_auc",
    )
    summary_acc = aggregate_fold_deltas(
        fold_records,
        deltas_key="delta_accuracy",
        mean_col="mean_delta_accuracy",
        std_col="std_delta_accuracy",
    )
    write_results_csv(
        csv_path,
        summary_auc,
        fold_records,
        deltas_key="delta_auc",
        mean_col="mean_delta_auc",
        baseline_key="baseline_auc",
    )
    write_results_csv(
        csv_acc_path,
        summary_acc,
        fold_records,
        deltas_key="delta_accuracy",
        mean_col="mean_delta_accuracy",
        baseline_key="baseline_accuracy",
    )
    plot_feature_importance(
        png_path,
        fold_records,
        summary_auc,
        deltas_key="delta_auc",
        mean_col="mean_delta_auc",
        xlabel="Mean delta AUC (baseline - ablated)",
        title="OOF permutation feature importance - AUC",
    )
    plot_feature_importance(
        png_acc_path,
        fold_records,
        summary_acc,
        deltas_key="delta_accuracy",
        mean_col="mean_delta_accuracy",
        xlabel="Mean delta accuracy (baseline - ablated)",
        title="OOF permutation feature importance - accuracy",
    )
    write_protocol_txt(protocol_path, configuration=args.configuration)

    print("\nGlobal feature ranking - AUC (mean delta +/- std across folds):")
    print("-" * 56)
    for _, row in summary_auc.iterrows():
        print(
            f"  {int(row['rank']):2d}. {row['feature']:<12} "
            f"{row['mean_delta_auc']:.4f} +/- {row['std_delta_auc']:.4f}  "
            f"({row['feature_type']})"
        )
    print("-" * 56)
    print("\nGlobal feature ranking - accuracy (mean delta +/- std across folds):")
    print("-" * 56)
    for _, row in summary_acc.iterrows():
        print(
            f"  {int(row['rank']):2d}. {row['feature']:<12} "
            f"{row['mean_delta_accuracy']:.4f} +/- {row['std_delta_accuracy']:.4f}  "
            f"({row['feature_type']})"
        )
    print("-" * 56)
    print(f"Wrote: {csv_path.resolve()}")
    print(f"Wrote: {png_path.resolve()}")
    print(f"Wrote: {csv_acc_path.resolve()}")
    print(f"Wrote: {png_acc_path.resolve()}")
    print(f"Wrote: {protocol_path.resolve()}")


if __name__ == "__main__":
    main()
