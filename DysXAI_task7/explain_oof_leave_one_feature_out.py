"""
Leave-one-feature-out OOF ablation focused on accuracy drop.

For each outer fold, this script loads the saved OOF checkpoint and evaluates the
locked outer test split:
  - baseline: all features available
  - LOO ablation: one feature neutralized ("feature removed") at a time

Importance is reported as:
    delta_accuracy = baseline_accuracy - loo_accuracy

This is inference-only and does not retrain models.
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
from sklearn.metrics import accuracy_score
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
CLASSIFICATION_THRESHOLD = 0.5

DEFAULT_FOLD_HPARAMS_CSV = "task7_ablation_oof_fold_hparams.csv"
DEFAULT_CONFIGURATION = "Baseline + Age + Gender"
DEFAULT_OUT_CSV = "task7_oof_leave_one_out_accuracy.csv"
DEFAULT_OUT_PNG = "task7_oof_leave_one_out_accuracy.png"
DEFAULT_RESULTS_DIR = _HERE / "XAI results" / "results ablation critical"

KINEMATIC_COLOR = "#1f77b4"
DEMOGRAPHIC_COLOR = "#ff7f0e"


def feature_type(name: str) -> str:
    return "Demographic" if name in DEMOGRAPHIC_FEATURES else "Kinematic"


def load_checkpoint_manifest(
    hparams_csv: Path, checkpoint_dir: Path, configuration: str
) -> dict[int, Path]:
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
    out: dict[int, Path] = {}
    for _, row in sub.iterrows():
        fold_id = int(row["outer_fold"])
        ckpt = checkpoint_dir / str(row["checkpoint_file"])
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Checkpoint missing for fold {fold_id}: {ckpt}"
            )
        out[fold_id] = ckpt
    if len(out) != N_OUTER_SPLITS:
        raise ValueError(
            f"Expected {N_OUTER_SPLITS} checkpoints, found {len(out)} for {configuration!r}"
        )
    return out


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_age: bool,
    use_gender: bool,
    age_min: float,
    age_max: float,
    ablate_feature: int | None = None,
    age_neutral: float = 0.5,
    gender_neutral: float = 0.5,
) -> float:
    """
    Accuracy on the outer test loader.

    Leave-one-feature-out is implemented by neutralization:
      - kinematic channels: set channel to 0.0 (z-score neutral)
      - age: set normalized age to ``age_neutral`` (default 0.5)
      - gender: set to ``gender_neutral`` (default 0.5)
    """
    model.eval()
    ys: list[int] = []
    probs: list[float] = []
    denom = (age_max - age_min) + 1e-8

    for batch in loader:
        x = batch["x"].to(device)
        lengths = batch["length"].to(device)
        y = batch["y"].cpu().numpy().astype(np.int64)

        if ablate_feature is not None and 0 <= ablate_feature < N_KINEMATIC:
            x = x.clone()
            x[:, ablate_feature, :] = 0.0

        age_tensor = None
        gender_tensor = None

        if use_age:
            ages = batch["age"].to(device).float()
            age_tensor = (ages - age_min) / denom
            if ablate_feature == 12:
                age_tensor = torch.full_like(age_tensor, float(age_neutral))
        if use_gender:
            gender_tensor = batch["gender"].to(device).float()
            if ablate_feature == 13:
                gender_tensor = torch.full_like(gender_tensor, float(gender_neutral))

        logits = model(
            x, lengths=lengths, age=age_tensor, gender=gender_tensor
        ).squeeze(-1)
        p = torch.sigmoid(logits).cpu().numpy().astype(np.float64)

        ys.extend(y.tolist())
        probs.extend(p.tolist())

    y_arr = np.asarray(ys, dtype=np.int64)
    p_arr = np.asarray(probs, dtype=np.float64)
    y_pred = (p_arr >= CLASSIFICATION_THRESHOLD).astype(np.int64)
    return float(accuracy_score(y_arr, y_pred))


def run_fold_loo(
    sample_df,
    test_idx: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
    *,
    age_neutral: float,
    gender_neutral: float,
) -> tuple[float, dict[str, float]]:
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

    baseline_acc = evaluate_accuracy(
        model,
        test_loader,
        device,
        use_age=use_age,
        use_gender=use_gender,
        age_min=age_min,
        age_max=age_max,
        ablate_feature=None,
        age_neutral=age_neutral,
        gender_neutral=gender_neutral,
    )

    deltas: dict[str, float] = {}
    for idx, feat in enumerate(FEATURE_NAMES):
        if feat == "Age" and not use_age:
            deltas[feat] = float("nan")
            continue
        if feat == "Gender" and not use_gender:
            deltas[feat] = float("nan")
            continue
        loo_acc = evaluate_accuracy(
            model,
            test_loader,
            device,
            use_age=use_age,
            use_gender=use_gender,
            age_min=age_min,
            age_max=age_max,
            ablate_feature=idx,
            age_neutral=age_neutral,
            gender_neutral=gender_neutral,
        )
        deltas[feat] = baseline_acc - loo_acc
    return baseline_acc, deltas


def aggregate_summary(fold_rows: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feat in FEATURE_NAMES:
        vals = [
            float(r["delta_accuracy"][feat])
            for r in fold_rows
            if np.isfinite(float(r["delta_accuracy"].get(feat, float("nan"))))
        ]
        if not vals:
            continue
        rows.append(
            {
                "feature": feat,
                "feature_type": feature_type(feat),
                "mean_delta_accuracy": float(np.mean(vals)),
                "std_delta_accuracy": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_folds": len(vals),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("mean_delta_accuracy", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def write_csv(path: Path, summary: pd.DataFrame, fold_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wide = summary.copy()
    for rec in fold_rows:
        fold_id = int(rec["outer_fold"])
        wide[f"fold_{fold_id:02d}_delta_accuracy"] = [
            float(rec["delta_accuracy"].get(feat, float("nan"))) for feat in wide["feature"]
        ]
        wide[f"fold_{fold_id:02d}_baseline_accuracy"] = float(rec["baseline_accuracy"])
    wide.to_csv(path, index=False)


def write_plot(path: Path, summary: pd.DataFrame, fold_rows: list[dict[str, object]]) -> None:
    long_rows: list[dict[str, object]] = []
    for rec in fold_rows:
        fold_id = int(rec["outer_fold"])
        for feat in FEATURE_NAMES:
            val = rec["delta_accuracy"].get(feat, float("nan"))
            if not np.isfinite(float(val)):
                continue
            long_rows.append(
                {
                    "feature": feat,
                    "feature_type": feature_type(feat),
                    "delta_accuracy": float(val),
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
        x="delta_accuracy",
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
    ax.set_xlabel("Mean delta accuracy (all 14 features - leave-one-out)")
    ax.set_ylabel("")
    ax.set_title("OOF leave-one-feature-out importance (accuracy)")
    ax.axvline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=labels, title="", loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOF leave-one-feature-out ablation (accuracy residuals)."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=default_checkpoint_dir(_HERE),
        help="Directory containing OOF checkpoint .pt files.",
    )
    parser.add_argument(
        "--fold-hparams-csv",
        type=Path,
        default=_HERE / DEFAULT_FOLD_HPARAMS_CSV,
        help=f"Manifest CSV with checkpoint_file names (default: {DEFAULT_FOLD_HPARAMS_CSV}).",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default=DEFAULT_CONFIGURATION,
        help=f"Manifest row set to use (default: {DEFAULT_CONFIGURATION!r}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory (default: DysXAI_task7/XAI results/results ablation critical/).",
    )
    parser.add_argument(
        "--age-neutral",
        type=float,
        default=0.5,
        help="Neutral normalized age value for LOO age ablation (default: 0.5).",
    )
    parser.add_argument(
        "--gender-neutral",
        type=float,
        default=0.5,
        help="Neutral gender value for LOO gender ablation (default: 0.5).",
    )
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else _HERE / args.checkpoint_dir
    hparams_csv = args.fold_hparams_csv if args.fold_hparams_csv.is_absolute() else _HERE / args.fold_hparams_csv
    results_dir = args.results_dir if args.results_dir.is_absolute() else _HERE / args.results_dir

    out_csv = results_dir / DEFAULT_OUT_CSV
    out_png = results_dir / DEFAULT_OUT_PNG

    manifest = load_checkpoint_manifest(hparams_csv, ckpt_dir, args.configuration)

    print(f"Device: {device}")
    print(f"Task 7 data: {TASK7_DATA_DIR}")
    print(f"Checkpoints: {ckpt_dir.resolve()}")
    print(f"Manifest: {hparams_csv.name} ({args.configuration})")
    print(f"Results: {results_dir.resolve()}")
    print(f"LOO neutral values: age={args.age_neutral}, gender={args.gender_neutral}")

    sample_df = build_sample_table()
    y = sample_df["label"].to_numpy()
    groups = sample_df["subject_id"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_rows: list[dict[str, object]] = []
    for fold_id, (_train_idx, test_idx) in enumerate(
        outer.split(np.zeros(len(sample_df)), y, groups), start=1
    ):
        ckpt = manifest[fold_id]
        baseline_acc, deltas = run_fold_loo(
            sample_df,
            test_idx,
            ckpt,
            device,
            age_neutral=args.age_neutral,
            gender_neutral=args.gender_neutral,
        )
        fold_rows.append(
            {
                "outer_fold": fold_id,
                "checkpoint": ckpt.name,
                "baseline_accuracy": baseline_acc,
                "delta_accuracy": deltas,
            }
        )
        top = max(deltas.items(), key=lambda kv: kv[1])
        print(
            f"Fold {fold_id}: baseline_acc={baseline_acc:.4f} | "
            f"largest_delta={top[0]} ({top[1]:.4f})"
        )

    summary = aggregate_summary(fold_rows)
    write_csv(out_csv, summary, fold_rows)
    write_plot(out_png, summary, fold_rows)

    print("\nGlobal LOO ranking by mean delta accuracy (+/- std):")
    print("-" * 60)
    for _, row in summary.iterrows():
        print(
            f"{int(row['rank']):2d}. {row['feature']:<12} "
            f"{row['mean_delta_accuracy']:.4f} +/- {row['std_delta_accuracy']:.4f} "
            f"({row['feature_type']})"
        )
    print("-" * 60)
    print(f"Wrote: {out_csv.resolve()}")
    print(f"Wrote: {out_png.resolve()}")


if __name__ == "__main__":
    main()
