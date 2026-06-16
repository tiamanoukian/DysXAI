"""

Final holdout evaluation for Task 7 after nested-CV epoch tuning.



Reads per-filter ``best_epoch`` JSON files (from ``train_ab_test.py --all-filters``),

repeats the same subject-stratified 80/20 split, then for each XY mode (``raw``,

``butterworth``, ``fft``) trains on 80% for that filter's tuned epoch count and reports

holdout metrics.



Examples::



    # With age (after: train_ab_test.py --use-age --params-out tuned_params_age.json)

    python train_final_evaluation.py --params-file tuned_params_age.json



    # Kinematics only

    python train_final_evaluation.py --no-age --params-file tuned_params_no_age.json



    # Full comparison table (age + no-age, all filters)

    python train_final_evaluation.py --compare-age \\

        --params-file tuned_params_age.json \\

        --params-file-no-age tuned_params_no_age.json

"""



from __future__ import annotations



import argparse

import csv

import json

import os

import sys

from dataclasses import dataclass

from pathlib import Path



import numpy as np

import torch

import torch.nn as nn

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

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

from checkpoint_io import (  # noqa: E402
    checkpoint_basename,
    default_checkpoint_dir,
    save_task7_checkpoint,
)
from model import Task7Conv1dClassifier  # noqa: E402

from train_ab_test import (  # noqa: E402

    BATCH_SIZE,

    FILTER_KEYS,

    LR,

    RANDOM_STATE,

    WEIGHT_DECAY,

    _train_fold_age_minmax,

    set_seed,

    subject_level_80_20_split,

    tuned_params_basename,

)



DEFAULT_PARAMS_AGE = "tuned_params.json"

DEFAULT_PARAMS_NO_AGE = "tuned_params_no_age.json"

DEFAULT_RESULTS_CSV = "task7_holdout_results.csv"

FILTER_ORDER = ["raw", "butterworth", "fft"]

FILTER_LABELS = {

    "raw": "Raw (No Filter)",

    "butterworth": "12Hz Butterworth",

    "fft": "12Hz FFT",

}





@dataclass(frozen=True)

class HoldoutMetrics:

    auc: float

    accuracy: float

    sensitivity: float

    specificity: float

    n_samples: int

    n_positive: int

    n_negative: int





def load_best_epoch(params_path: Path) -> int:

    if not params_path.is_file():

        raise FileNotFoundError(

            f"{params_path} not found. Run train_ab_test.py with --params-out first."

        )

    with params_path.open(encoding="utf-8") as f:

        data = json.load(f)

    if "best_epoch" not in data:

        raise KeyError(f"{params_path} must contain key 'best_epoch'.")

    ep = int(data["best_epoch"])

    if ep < 1:

        raise ValueError(f"best_epoch must be >= 1, got {ep}.")

    return ep





def tuned_params_path(
    xy_filter: str, *, use_age: bool, use_gender: bool, params_dir: Path
) -> Path:

    return params_dir / tuned_params_basename(
        xy_filter, use_age=use_age, use_gender=use_gender
    )





def load_epochs_by_filter(

    *,

    use_age: bool,
    use_gender: bool,

    params_dir: Path,

    legacy_path: Path | None = None,

) -> dict[str, int]:

    """

    Load best_epoch per XY filter from tuned_params_<filter>_<demographics>.json.

    Falls back to a single legacy JSON only when per-filter files are missing.

    """

    epochs: dict[str, int] = {}

    missing: list[str] = []

    for key in FILTER_KEYS:

        path = tuned_params_path(
            key, use_age=use_age, use_gender=use_gender, params_dir=params_dir
        )

        if path.is_file():

            epochs[key] = load_best_epoch(path)

        else:

            missing.append(key)

    if missing:

        if legacy_path is not None and legacy_path.is_file():

            shared = load_best_epoch(legacy_path)

            print(

                f"Warning: missing per-filter JSON for {missing}; "

                f"using best_epoch={shared} from {legacy_path.name} for those filters."

            )

            for key in missing:

                epochs[key] = shared

        else:

            names = ", ".join(
                tuned_params_basename(k, use_age=use_age, use_gender=use_gender)
                for k in missing
            )

            raise FileNotFoundError(

                f"Missing tuned params: {names}. "

                "Run: python train_ab_test.py --all-filters"

                + (" --use-age" if use_age else "")
                + (" --use-gender" if use_gender else "")

            )

    return epochs





def metrics_from_arrays(y_true: np.ndarray, y_score: np.ndarray) -> HoldoutMetrics:

    y_true = np.asarray(y_true, dtype=np.int64)

    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = int((y_true == 1).sum())

    n_neg = int((y_true == 0).sum())



    if len(np.unique(y_true)) < 2:

        return HoldoutMetrics(

            auc=float("nan"),

            accuracy=float("nan"),

            sensitivity=float("nan"),

            specificity=float("nan"),

            n_samples=len(y_true),

            n_positive=n_pos,

            n_negative=n_neg,

        )



    y_pred = (y_score >= 0.5).astype(np.int64)

    auc = float(roc_auc_score(y_true, y_score))

    accuracy = float(accuracy_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return HoldoutMetrics(

        auc=auc,

        accuracy=accuracy,

        sensitivity=sensitivity,

        specificity=specificity,

        n_samples=len(y_true),

        n_positive=n_pos,

        n_negative=n_neg,

    )





@torch.no_grad()

def collect_holdout_predictions(

    model: nn.Module,

    loader: DataLoader,

    device: torch.device,

    *,

    use_age: bool,
    use_gender: bool,
    age_min: float,
    age_max: float,

) -> tuple[np.ndarray, np.ndarray]:

    model.eval()

    ys: list[float] = []

    scores: list[float] = []

    denom = (age_max - age_min) + 1e-8



    for batch in loader:

        x = batch["x"].to(device)

        lengths = batch["length"].to(device)

        y = batch["y"].cpu().numpy()

        age_tensor = None
        gender_tensor = None
        if use_age:
            batch_ages = batch["age"].to(device).float()
            age_tensor = (batch_ages - age_min) / denom
        if use_gender:
            gender_tensor = batch["gender"].to(device).float()
        logits = model(x, lengths=lengths, age=age_tensor, gender=gender_tensor).squeeze(-1).cpu().numpy()

        ys.extend(y.astype(float).tolist())

        scores.extend(torch.sigmoid(torch.tensor(logits)).numpy().tolist())



    return np.array(ys, dtype=np.float64), np.array(scores, dtype=np.float64)





def train_for_epochs(

    model: nn.Module,

    train_loader: DataLoader,

    device: torch.device,

    *,

    epochs: int,

    use_age: bool,
    use_gender: bool,
    age_min: float,
    age_max: float,

) -> None:

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    crit = nn.BCEWithLogitsLoss()

    denom = (age_max - age_min) + 1e-8



    for _ep in range(1, epochs + 1):

        model.train()

        for batch in train_loader:

            x = batch["x"].to(device)

            lengths = batch["length"].to(device)

            y = batch["y"].to(device).unsqueeze(1)

            opt.zero_grad(set_to_none=True)

            age_tensor = None
            gender_tensor = None
            if use_age:
                batch_ages = batch["age"].to(device).float()
                age_tensor = (batch_ages - age_min) / denom
            if use_gender:
                gender_tensor = batch["gender"].to(device).float()
            logits = model(x, lengths=lengths, age=age_tensor, gender=gender_tensor)

            loss = crit(logits, y)

            loss.backward()

            opt.step()





def run_one_condition(

    *,

    xy_filter: str,

    train_df,

    holdout_df,

    train_idx: np.ndarray,

    test_idx: np.ndarray,

    best_epoch: int,

    device: torch.device,

    use_age: bool,
    use_gender: bool,

    checkpoint_path: Path | None = None,

    checkpoint_split: str = "holdout",

) -> HoldoutMetrics:

    mu, sigma = fit_channel_scaling(train_df, train_idx, xy_filter=xy_filter)

    age_min, age_max = _train_fold_age_minmax(train_df, train_idx)



    train_ds = Task5TrajectoryDataset(

        train_df,

        train_idx,

        use_age_channel=use_age,

        channel_mean=mu,

        channel_std=sigma,

        xy_filter=xy_filter,

    )

    test_ds = Task5TrajectoryDataset(

        holdout_df,

        test_idx,

        use_age_channel=use_age,

        channel_mean=mu,

        channel_std=sigma,

        xy_filter=xy_filter,

    )

    train_loader = DataLoader(

        train_ds,

        batch_size=BATCH_SIZE,

        shuffle=True,

        collate_fn=collate_task7,

        num_workers=0,

    )

    test_loader = DataLoader(

        test_ds,

        batch_size=BATCH_SIZE,

        shuffle=False,

        collate_fn=collate_task7,

        num_workers=0,

    )



    model = Task7Conv1dClassifier(use_age=use_age, use_gender=use_gender).to(device)

    train_for_epochs(

        model,

        train_loader,

        device,

        epochs=best_epoch,

        use_age=use_age,
        use_gender=use_gender,
        age_min=age_min,
        age_max=age_max,

    )

    y_true, y_score = collect_holdout_predictions(

        model,

        test_loader,

        device,

        use_age=use_age,
        use_gender=use_gender,
        age_min=age_min,
        age_max=age_max,

    )

    if checkpoint_path is not None:
        saved = save_task7_checkpoint(
            checkpoint_path,
            model,
            xy_filter=xy_filter,
            use_age=use_age,
            use_gender=use_gender,
            best_epoch=best_epoch,
            channel_mean=mu,
            channel_std=sigma,
            age_min=age_min,
            age_max=age_max,
            split=checkpoint_split,
            random_state=RANDOM_STATE,
            extra={"n_train": len(train_idx), "n_holdout": len(test_idx)},
        )
        print(f"Saved checkpoint: {saved}")

    return metrics_from_arrays(y_true, y_score)





@dataclass(frozen=True)

class ResultRow:

    filter_key: str

    filter_label: str

    use_age: bool
    use_gender: bool

    best_epoch: int

    metrics: HoldoutMetrics





def _fmt_metric(value: float) -> str:

    if np.isnan(value):

        return "—"

    return f"{value:.3f}"





def print_results_table(rows: list[ResultRow], *, holdout_n: int) -> None:

    print()

    print(f"Holdout set: n={holdout_n} subjects (20% split, seed={RANDOM_STATE})")

    print(

        "| XY filter | Use age | Use gender | Epochs | AUC | Accuracy | Sensitivity | Specificity |"

    )

    print(

        "|-----------|---------|------------|--------|-----|----------|-------------|-------------|"

    )

    for row in rows:

        m = row.metrics

        age_str = "Yes" if row.use_age else "No"
        gender_str = "Yes" if row.use_gender else "No"

        print(

            f"| {row.filter_label} | {age_str} | {gender_str} | {row.best_epoch} | "

            f"{_fmt_metric(m.auc)} | {_fmt_metric(m.accuracy)} | "

            f"{_fmt_metric(m.sensitivity)} | {_fmt_metric(m.specificity)} |"

        )

    print()





def write_results_csv(path: Path, rows: list[ResultRow]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:

        w = csv.writer(f)

        w.writerow(

            [

                "xy_filter",

                "filter_label",

                "late_fusion_age",
                "late_fusion_gender",

                "best_epoch",

                "holdout_n",

                "holdout_n_dysgraphic",

                "holdout_n_control",

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

                    int(row.use_age),
                    int(row.use_gender),

                    row.best_epoch,

                    m.n_samples,

                    m.n_positive,

                    m.n_negative,

                    f"{m.auc:.6f}" if np.isfinite(m.auc) else "nan",

                    f"{m.accuracy:.6f}" if np.isfinite(m.accuracy) else "nan",

                    f"{m.sensitivity:.6f}" if np.isfinite(m.sensitivity) else "nan",

                    f"{m.specificity:.6f}" if np.isfinite(m.specificity) else "nan",

                ]

            )





def run_evaluation_suite(

    *,

    train_tuning_df,

    holdout_test_df,

    train_idx: np.ndarray,

    test_idx: np.ndarray,

    device: torch.device,

    use_age: bool,
    use_gender: bool,

    epochs_by_filter: dict[str, int],

    label: str,

    checkpoint_dir: Path | None = None,

) -> list[ResultRow]:

    epoch_summary = ", ".join(f"{k}={epochs_by_filter[k]}" for k in FILTER_ORDER)

    print(f"\n=== {label} | use_age={use_age} use_gender={use_gender} | epochs: {epoch_summary} ===")

    rows: list[ResultRow] = []

    for condition in FILTER_ORDER:

        best_epoch = epochs_by_filter[condition]

        print(f"\n--- {FILTER_LABELS[condition]} (best_epoch={best_epoch}) ---")

        ckpt_path = None
        if checkpoint_dir is not None:
            ckpt_path = checkpoint_dir / checkpoint_basename(
                split="holdout",
                xy_filter=condition,
                use_age=use_age,
                use_gender=use_gender,
            )

        metrics = run_one_condition(

            xy_filter=condition,

            train_df=train_tuning_df,

            holdout_df=holdout_test_df,

            train_idx=train_idx,

            test_idx=test_idx,

            best_epoch=best_epoch,

            device=device,

            use_age=use_age,
            use_gender=use_gender,

            checkpoint_path=ckpt_path,

        )

        print(

            f"AUC={metrics.auc:.4f}  Acc={metrics.accuracy:.4f}  "

            f"Sens={metrics.sensitivity:.4f}  Spec={metrics.specificity:.4f}  "

            f"(n={metrics.n_samples}, dys={metrics.n_positive}, ctrl={metrics.n_negative})"

        )

        rows.append(

            ResultRow(

                filter_key=condition,

                filter_label=FILTER_LABELS[condition],

                use_age=use_age,
                use_gender=use_gender,

                best_epoch=best_epoch,

                metrics=metrics,

            )

        )

    return rows





def main() -> None:

    parser = argparse.ArgumentParser(

        description="Task 7 holdout evaluation: AUC, accuracy, sensitivity, specificity."

    )

    parser.add_argument(

        "--params-dir",

        type=Path,

        default=_HERE,

        help="Directory with tuned_params_<filter>.json files (default: DysXAI_task7/).",

    )

    parser.add_argument(

        "--params-file",

        type=Path,

        default=_HERE / DEFAULT_PARAMS_AGE,

        help=f"Legacy single JSON if per-filter files missing (default: {DEFAULT_PARAMS_AGE}).",

    )

    parser.add_argument(

        "--params-file-no-age",

        type=Path,

        default=_HERE / DEFAULT_PARAMS_NO_AGE,

        help=f"Legacy kinematics-only JSON fallback (default: {DEFAULT_PARAMS_NO_AGE}).",

    )

    parser.add_argument(

        "--no-age",

        action="store_true",

        help="Evaluate kinematics-only model (no late-fusion age).",

    )

    parser.add_argument(
        "--use-gender",
        action="store_true",
        help="Use late-fusion gender feature (0=male, 1=female).",
    )

    parser.add_argument(

        "--compare-age",

        action="store_true",

        help="Run all filters with and without late-fusion age (two param files).",

    )
    parser.add_argument(
        "--compare-gender",
        action="store_true",
        help="Run all filters with and without late-fusion gender in one command.",
    )

    parser.add_argument(

        "--results-csv",

        type=Path,

        default=_HERE / DEFAULT_RESULTS_CSV,

        help=f"Write full results table to CSV (default: {DEFAULT_RESULTS_CSV}).",

    )

    parser.add_argument(

        "--checkpoint-dir",

        type=Path,

        default=default_checkpoint_dir(_HERE),

        help="Directory for holdout-trained checkpoints (default: DysXAI_task7/checkpoints/).",

    )

    parser.add_argument(

        "--no-save-checkpoints",

        action="store_true",

        help="Skip writing model checkpoints.",

    )

    args = parser.parse_args()



    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    print(f"Task 7 data: {TASK7_DATA_DIR}")

    print(f"Random state (split): {RANDOM_STATE}")

    ckpt_dir: Path | None = None
    if not args.no_save_checkpoints:
        ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else _HERE / args.checkpoint_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints: {ckpt_dir.resolve()}")
    else:
        print("Checkpoints: disabled (--no-save-checkpoints)")

    sample_df = build_sample_table()

    train_tuning_df, holdout_test_df = subject_level_80_20_split(sample_df)

    train_idx = np.arange(len(train_tuning_df), dtype=np.int64)

    test_idx = np.arange(len(holdout_test_df), dtype=np.int64)

    holdout_n = len(holdout_test_df)

    print(f"Train/tuning: n={len(train_tuning_df)} | Holdout: n={holdout_n}")



    all_rows: list[ResultRow] = []



    params_dir = args.params_dir if args.params_dir.is_absolute() else _HERE / args.params_dir

    age_legacy = args.params_file if args.params_file.is_absolute() else _HERE / args.params_file

    no_age_legacy = (

        args.params_file_no_age

        if args.params_file_no_age.is_absolute()

        else _HERE / args.params_file_no_age

    )



    use_age_values = [True, False] if args.compare_age else [not args.no_age]
    use_gender_values = [False, True] if args.compare_gender else [bool(args.use_gender)]
    epochs_cache: dict[tuple[bool, bool], dict[str, int]] = {}

    for use_age in use_age_values:
        for use_gender in use_gender_values:
            key = (use_age, use_gender)
            if key in epochs_cache:
                continue
            legacy = age_legacy if use_age else no_age_legacy
            epochs_cache[key] = load_epochs_by_filter(
                use_age=use_age,
                use_gender=use_gender,
                params_dir=params_dir,
                legacy_path=legacy,
            )
            age_label = "late-fusion age" if use_age else "kinematics only (no age)"
            gender_label = "late-fusion gender" if use_gender else "no gender"
            print(
                f"Per-filter best_epoch ({age_label}, {gender_label}):",
                epochs_cache[key],
            )

    for use_age in use_age_values:
        for use_gender in use_gender_values:
            epochs_by_filter = epochs_cache[(use_age, use_gender)]
            label = (
                f"{'Late-fusion age' if use_age else 'Kinematics only (no age)'} + "
                f"{'late-fusion gender' if use_gender else 'no gender'}"
            )
            all_rows.extend(
                run_evaluation_suite(
                    train_tuning_df=train_tuning_df,
                    holdout_test_df=holdout_test_df,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    device=device,
                    use_age=use_age,
                    use_gender=use_gender,
                    epochs_by_filter=epochs_by_filter,
                    label=label,
                    checkpoint_dir=ckpt_dir,
                )
            )



    print_results_table(all_rows, holdout_n=holdout_n)

    csv_path = args.results_csv if args.results_csv.is_absolute() else _HERE / args.results_csv

    write_results_csv(csv_path, all_rows)

    print(f"Wrote results CSV: {csv_path.resolve()}")





if __name__ == "__main__":

    os.environ.setdefault("PYTHONUTF8", "1")

    main()


