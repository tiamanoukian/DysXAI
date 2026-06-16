"""
Critical OOF feature-importance analysis (Task 7).

Adds stronger evidence layers on top of single-run feature importance:
1) Repeated permutation ablation (single features) with fold x repeat raw outputs.
2) Bootstrap confidence intervals + directional probability P(delta > 0).
3) Grouped ablation (feature families).
4) Pairwise interaction ablation checks.
5) Calibration impact via Brier score deltas.
6) Rank-stability comparison across AUC / Accuracy / Brier.

All analyses are run on the same 5-fold OOF outer test protocol:
each subject is in test exactly once across folds.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from checkpoint_io import default_checkpoint_dir, load_task7_checkpoint  # noqa: E402
from dataset import TASK7_DATA_DIR, Task5TrajectoryDataset, build_sample_table, collate_task7  # noqa: E402
from train_ab_test import BATCH_SIZE, RANDOM_STATE, set_seed  # noqa: E402

N_OUTER_SPLITS = 5
CLASSIFICATION_THRESHOLD = 0.5

KIN_FEATURES = [
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
DEMO_FEATURES = ["Age", "Gender"]
FEATURES = KIN_FEATURES + DEMO_FEATURES
FEATURE_TO_IDX = {name: i for i, name in enumerate(FEATURES)}
N_KIN = len(KIN_FEATURES)

GROUPS = {
    "Position(XY)": ["X", "Y"],
    "PenGeometry": ["Pressure", "Azimuth", "Tilt", "PenStatus"],
    "Dynamics(V/A/J)": ["Vx", "Vy", "Ax", "Ay", "Jx", "Jy"],
    "Demographics": ["Age", "Gender"],
}
INTERACTION_PAIRS = [("Y", "Vy"), ("Pressure", "PenStatus"), ("Age", "Gender")]

DEFAULT_FOLD_HPARAMS_CSV = "task7_ablation_oof_fold_hparams.csv"
DEFAULT_CONFIGURATION = "Baseline + Age + Gender"
DEFAULT_RESULTS_DIR = _HERE / "XAI results" / "critical_feature_importance"


def feature_type(name: str) -> str:
    return "Demographic" if name in DEMO_FEATURES else "Kinematic"


def load_checkpoint_manifest(hparams_csv: Path, checkpoint_dir: Path, configuration: str) -> dict[int, Path]:
    if not hparams_csv.is_file():
        raise FileNotFoundError(f"Fold hparams CSV not found: {hparams_csv}")
    df = pd.read_csv(hparams_csv)
    sub = df[df["configuration"].astype(str) == configuration]
    if sub.empty:
        available = sorted(df["configuration"].astype(str).unique())
        raise ValueError(f"No rows for {configuration!r}. Available: {available}")
    out: dict[int, Path] = {}
    for _, row in sub.iterrows():
        fold = int(row["outer_fold"])
        p = checkpoint_dir / str(row["checkpoint_file"])
        if not p.is_file():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {p}")
        out[fold] = p
    if len(out) != N_OUTER_SPLITS:
        raise ValueError(f"Expected {N_OUTER_SPLITS} folds, found {len(out)}")
    return out


def _seeded_perm(n: int, *, seed: int, fold: int, feat_key: int, repeat: int, batch: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed + fold * 100_000 + feat_key * 1_000 + repeat * 100 + batch)
    return torch.randperm(n, device=device, generator=g)


@torch.no_grad()
def evaluate_with_permutation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_age: bool,
    use_gender: bool,
    age_min: float,
    age_max: float,
    ablate_indices: set[int] | None,
    fold_id: int,
    repeat_id: int,
    perm_seed: int,
) -> tuple[float, float, float]:
    ys: list[int] = []
    probs: list[float] = []
    denom = (age_max - age_min) + 1e-8
    active = ablate_indices or set()
    for batch_id, batch in enumerate(loader):
        x = batch["x"].to(device)
        lengths = batch["length"].to(device)
        y = batch["y"].cpu().numpy().astype(np.int64)
        bsz = x.shape[0]

        # Kinematic channels: shuffle selected feature channels across batch.
        kin_idxs = [idx for idx in active if idx < N_KIN]
        if kin_idxs:
            x = x.clone()
            for idx in kin_idxs:
                perm = _seeded_perm(
                    bsz,
                    seed=perm_seed,
                    fold=fold_id,
                    feat_key=idx,
                    repeat=repeat_id,
                    batch=batch_id,
                    device=device,
                )
                x[:, idx, :] = x[perm, idx, :]

        age_tensor = None
        gender_tensor = None
        if use_age:
            age_tensor = (batch["age"].to(device).float() - age_min) / denom
            if 12 in active:
                perm = _seeded_perm(
                    bsz,
                    seed=perm_seed,
                    fold=fold_id,
                    feat_key=12,
                    repeat=repeat_id,
                    batch=batch_id,
                    device=device,
                )
                age_tensor = age_tensor[perm]
        if use_gender:
            gender_tensor = batch["gender"].to(device).float()
            if 13 in active:
                perm = _seeded_perm(
                    bsz,
                    seed=perm_seed,
                    fold=fold_id,
                    feat_key=13,
                    repeat=repeat_id,
                    batch=batch_id,
                    device=device,
                )
                gender_tensor = gender_tensor[perm]

        logits = model(x, lengths=lengths, age=age_tensor, gender=gender_tensor).squeeze(-1)
        p = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        ys.extend(y.tolist())
        probs.extend(p.tolist())

    y_arr = np.asarray(ys, dtype=np.int64)
    p_arr = np.asarray(probs, dtype=np.float64)
    y_hat = (p_arr >= CLASSIFICATION_THRESHOLD).astype(np.int64)
    auc = float(roc_auc_score(y_arr, p_arr)) if np.unique(y_arr).size > 1 else float("nan")
    acc = float(accuracy_score(y_arr, y_hat))
    brier = float(brier_score_loss(y_arr, p_arr))
    return auc, acc, brier


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.95) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    n = vals.size
    for _ in range(n_boot):
        sample = vals[rng.integers(0, n, size=n)]
        boots.append(float(np.mean(sample)))
    lo = float(np.percentile(boots, 100.0 * (1 - alpha) / 2))
    hi = float(np.percentile(boots, 100.0 * (1 + alpha) / 2))
    return lo, hi


def two_sided_z_p(values: np.ndarray) -> float:
    """
    Approximate two-sided p-value for mean(delta) != 0 using z-statistic.
    """
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n == 0:
        return float("nan")
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    if sd <= 1e-12:
        return 1.0 if abs(mean) <= 1e-12 else 1e-12
    z = abs(mean) / (sd / math.sqrt(n))
    # two-sided p from Normal(0,1): p = erfc(z/sqrt(2))
    p = math.erfc(z / math.sqrt(2.0))
    return float(min(max(p, 1e-12), 1.0))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR correction. Returns q-values in original order."""
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    pf = p[finite]
    m = pf.size
    order = np.argsort(pf)
    ranked = pf[order]
    q_sorted = np.empty(m, dtype=np.float64)
    prev = 1.0
    for k in range(m - 1, -1, -1):
        rank = k + 1
        val = ranked[k] * m / rank
        prev = min(prev, val)
        q_sorted[k] = prev
    q_f = np.empty(m, dtype=np.float64)
    q_f[order] = np.clip(q_sorted, 0.0, 1.0)
    q[idx] = q_f
    return q


def summarize_long(df_long: pd.DataFrame, *, metric_col: str, group_col: str, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for key, sub in df_long.groupby(group_col, sort=False):
        vals = sub[metric_col].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed + abs(hash(str(key))) % 10_000)
        p_pos = float(np.mean(vals > 0))
        p_raw = two_sided_z_p(vals)
        if lo > 0:
            verdict = "robust_positive"
        elif hi < 0:
            verdict = "robust_negative"
        else:
            verdict = "uncertain"
        rows.append(
            {
                group_col: key,
                f"mean_{metric_col}": mean,
                f"std_{metric_col}": sd,
                f"ci95_low_{metric_col}": lo,
                f"ci95_high_{metric_col}": hi,
                f"p_pos_{metric_col}": p_pos,
                f"p_raw_{metric_col}": p_raw,
                f"verdict_{metric_col}": verdict,
                "n": int(vals.size),
            }
        )
    out = pd.DataFrame(rows)
    pcol = f"p_raw_{metric_col}"
    qcol = f"q_fdr_{metric_col}"
    sigcol = f"sig_fdr05_{metric_col}"
    out[qcol] = benjamini_hochberg(out[pcol].to_numpy(dtype=np.float64))
    out[sigcol] = out[qcol] <= 0.05
    out = out.sort_values(f"mean_{metric_col}", ascending=False).reset_index(drop=True)
    return out


def run_analysis(
    *,
    checkpoint_manifest: dict[int, Path],
    repeats: int,
    perm_seed: int,
    n_boot: int,
    results_dir: Path,
) -> None:
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_df = build_sample_table()
    y = sample_df["label"].to_numpy()
    groups = sample_df["subject_id"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    single_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    for fold_id, (_train_idx, test_idx) in enumerate(outer.split(np.zeros(len(sample_df)), y, groups), start=1):
        ckpt = checkpoint_manifest[fold_id]
        model, meta = load_task7_checkpoint(ckpt, device)
        use_age = bool(meta["use_age"])
        use_gender = bool(meta["use_gender"])
        age_min = float(meta["age_min"])
        age_max = float(meta["age_max"])
        xy_filter = str(meta["xy_filter"])
        mu = np.asarray(meta["channel_mean"], dtype=np.float32)
        sigma = np.asarray(meta["channel_std"], dtype=np.float32)

        test_ds = Task5TrajectoryDataset(
            sample_df,
            test_idx,
            use_age_channel=use_age,
            channel_mean=mu,
            channel_std=sigma,
            xy_filter=xy_filter,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_task7,
            num_workers=0,
        )

        base_auc, base_acc, base_brier = evaluate_with_permutation(
            model,
            test_loader,
            device,
            use_age=use_age,
            use_gender=use_gender,
            age_min=age_min,
            age_max=age_max,
            ablate_indices=set(),
            fold_id=fold_id,
            repeat_id=0,
            perm_seed=perm_seed,
        )
        print(
            f"Fold {fold_id}: baseline AUC={base_auc:.4f} "
            f"Acc={base_acc:.4f} Brier={base_brier:.4f}"
        )

        # Single-feature repeated permutations.
        for r in range(repeats):
            for feat in FEATURES:
                idx = FEATURE_TO_IDX[feat]
                if feat == "Age" and not use_age:
                    continue
                if feat == "Gender" and not use_gender:
                    continue
                auc, acc, brier = evaluate_with_permutation(
                    model,
                    test_loader,
                    device,
                    use_age=use_age,
                    use_gender=use_gender,
                    age_min=age_min,
                    age_max=age_max,
                    ablate_indices={idx},
                    fold_id=fold_id,
                    repeat_id=r + 1,
                    perm_seed=perm_seed,
                )
                single_rows.append(
                    {
                        "fold": fold_id,
                        "repeat": r + 1,
                        "feature": feat,
                        "feature_type": feature_type(feat),
                        "delta_auc": base_auc - auc,
                        "delta_accuracy": base_acc - acc,
                        "delta_brier": brier - base_brier,  # >0 means worse calibration
                    }
                )

        # Grouped ablation (single draw per fold for speed).
        for group_name, feats in GROUPS.items():
            idxs = {FEATURE_TO_IDX[f] for f in feats if not (f == "Age" and not use_age) and not (f == "Gender" and not use_gender)}
            if not idxs:
                continue
            auc, acc, brier = evaluate_with_permutation(
                model,
                test_loader,
                device,
                use_age=use_age,
                use_gender=use_gender,
                age_min=age_min,
                age_max=age_max,
                ablate_indices=idxs,
                fold_id=fold_id,
                repeat_id=99,
                perm_seed=perm_seed,
            )
            group_rows.append(
                {
                    "fold": fold_id,
                    "group": group_name,
                    "n_features": len(idxs),
                    "delta_auc": base_auc - auc,
                    "delta_accuracy": base_acc - acc,
                    "delta_brier": brier - base_brier,
                }
            )

        # Pairwise interactions (single draw per fold).
        for f1, f2 in INTERACTION_PAIRS:
            if (f1 == "Age" and not use_age) or (f2 == "Age" and not use_age):
                continue
            if (f1 == "Gender" and not use_gender) or (f2 == "Gender" and not use_gender):
                continue
            idxs = {FEATURE_TO_IDX[f1], FEATURE_TO_IDX[f2]}
            auc, acc, brier = evaluate_with_permutation(
                model,
                test_loader,
                device,
                use_age=use_age,
                use_gender=use_gender,
                age_min=age_min,
                age_max=age_max,
                ablate_indices=idxs,
                fold_id=fold_id,
                repeat_id=199,
                perm_seed=perm_seed,
            )
            pair_rows.append(
                {
                    "fold": fold_id,
                    "pair": f"{f1}+{f2}",
                    "delta_auc": base_auc - auc,
                    "delta_accuracy": base_acc - acc,
                    "delta_brier": brier - base_brier,
                }
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    df_single = pd.DataFrame(single_rows)
    df_group = pd.DataFrame(group_rows)
    df_pair = pd.DataFrame(pair_rows)

    df_single.to_csv(results_dir / "task7_critical_single_feature_long.csv", index=False)
    df_group.to_csv(results_dir / "task7_critical_group_ablation_long.csv", index=False)
    df_pair.to_csv(results_dir / "task7_critical_pair_ablation_long.csv", index=False)

    # Summaries + significance labels.
    s_auc = summarize_long(df_single, metric_col="delta_auc", group_col="feature", n_boot=n_boot, seed=perm_seed)
    s_acc = summarize_long(df_single, metric_col="delta_accuracy", group_col="feature", n_boot=n_boot, seed=perm_seed + 7)
    s_brier = summarize_long(df_single, metric_col="delta_brier", group_col="feature", n_boot=n_boot, seed=perm_seed + 13)
    summary = s_auc.merge(s_acc, on="feature", how="outer").merge(s_brier, on="feature", how="outer")
    summary["feature_type"] = summary["feature"].map(feature_type)
    summary["rank_auc"] = summary["mean_delta_auc"].rank(ascending=False, method="min")
    summary["rank_accuracy"] = summary["mean_delta_accuracy"].rank(ascending=False, method="min")
    summary["rank_brier"] = summary["mean_delta_brier"].rank(ascending=False, method="min")
    summary["rank_spread"] = (
        summary[["rank_auc", "rank_accuracy", "rank_brier"]].max(axis=1)
        - summary[["rank_auc", "rank_accuracy", "rank_brier"]].min(axis=1)
    )
    summary = summary.sort_values("mean_delta_auc", ascending=False).reset_index(drop=True)
    summary.to_csv(results_dir / "task7_critical_single_feature_summary.csv", index=False)

    gsum = (
        df_group.groupby("group", as_index=False)[["delta_auc", "delta_accuracy", "delta_brier"]]
        .mean()
        .sort_values("delta_accuracy", ascending=False)
    )
    gsum.to_csv(results_dir / "task7_critical_group_ablation_summary.csv", index=False)
    psum = (
        df_pair.groupby("pair", as_index=False)[["delta_auc", "delta_accuracy", "delta_brier"]]
        .mean()
        .sort_values("delta_accuracy", ascending=False)
    )
    psum.to_csv(results_dir / "task7_critical_pair_ablation_summary.csv", index=False)

    # Fold-level dot plot (accuracy).
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 8))
    order = summary.sort_values("mean_delta_accuracy", ascending=False)["feature"].tolist()
    sns.stripplot(
        data=df_single,
        y="feature",
        x="delta_accuracy",
        order=order,
        hue="fold",
        dodge=False,
        size=4,
        alpha=0.7,
        ax=ax,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1, color="0.5")
    ax.set_title("Fold x repeat stability (delta accuracy)")
    ax.set_xlabel("delta accuracy (baseline - ablated)")
    ax.set_ylabel("")
    ax.legend(title="fold", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(results_dir / "task7_critical_fold_repeat_stability_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # CI bar plot (AUC).
    fig, ax = plt.subplots(figsize=(11, 8))
    plot_df = summary.sort_values("mean_delta_auc", ascending=False).copy()
    ax.barh(plot_df["feature"], plot_df["mean_delta_auc"], color=plot_df["feature"].map(lambda f: "#ff7f0e" if f in DEMO_FEATURES else "#1f77b4"))
    ax.errorbar(
        plot_df["mean_delta_auc"],
        np.arange(len(plot_df)),
        xerr=[
            plot_df["mean_delta_auc"] - plot_df["ci95_low_delta_auc"],
            plot_df["ci95_high_delta_auc"] - plot_df["mean_delta_auc"],
        ],
        fmt="none",
        ecolor="0.2",
        capsize=3,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1, color="0.5")
    ax.set_title("Repeated permutation importance with 95% CI (AUC)")
    ax.set_xlabel("mean delta AUC")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(results_dir / "task7_critical_auc_ci.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Quick textual summary.
    robust_pos = summary[summary["verdict_delta_auc"] == "robust_positive"]["feature"].tolist()
    uncertain = summary[summary["verdict_delta_auc"] == "uncertain"]["feature"].tolist()
    note = pd.DataFrame(
        {
            "section": ["robust_positive_auc", "uncertain_auc", "data_scope"],
            "value": [
                ", ".join(robust_pos) if robust_pos else "none",
                ", ".join(uncertain) if uncertain else "none",
                "5-fold OOF outer test only; each subject tested once across folds.",
            ],
        }
    )
    note.to_csv(results_dir / "task7_critical_readout.csv", index=False)

    print("\nWrote critical analysis outputs:")
    for p in [
        "task7_critical_single_feature_long.csv",
        "task7_critical_single_feature_summary.csv",
        "task7_critical_group_ablation_summary.csv",
        "task7_critical_pair_ablation_summary.csv",
        "task7_critical_fold_repeat_stability_accuracy.png",
        "task7_critical_auc_ci.png",
        "task7_critical_readout.csv",
    ]:
        print(f"  - {(results_dir / p).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Critical OOF feature-importance analysis.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=default_checkpoint_dir(_HERE),
        help="Directory containing OOF checkpoints.",
    )
    parser.add_argument(
        "--fold-hparams-csv",
        type=Path,
        default=_HERE / DEFAULT_FOLD_HPARAMS_CSV,
        help="Manifest CSV with checkpoint_file values.",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default=DEFAULT_CONFIGURATION,
        help="Configuration rows to use from fold-hparams CSV.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory (default: XAI results/critical_feature_importance/).",
    )
    parser.add_argument("--repeats", type=int, default=20, help="Permutation repeats per feature per fold.")
    parser.add_argument("--perm-seed", type=int, default=RANDOM_STATE, help="Base random seed for permutations.")
    parser.add_argument("--n-bootstrap", type=int, default=3000, help="Bootstrap samples for CI.")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else _HERE / args.checkpoint_dir
    hparams_csv = args.fold_hparams_csv if args.fold_hparams_csv.is_absolute() else _HERE / args.fold_hparams_csv
    results_dir = args.results_dir if args.results_dir.is_absolute() else _HERE / args.results_dir

    print(f"Task 7 data: {TASK7_DATA_DIR}")
    print(f"Checkpoints: {ckpt_dir.resolve()}")
    print(f"Manifest: {hparams_csv.name} ({args.configuration})")
    print(f"Results: {results_dir.resolve()}")
    print(f"Protocol: 5-fold OOF outer test; each subject tested once across folds.")

    manifest = load_checkpoint_manifest(hparams_csv, ckpt_dir, args.configuration)
    run_analysis(
        checkpoint_manifest=manifest,
        repeats=args.repeats,
        perm_seed=args.perm_seed,
        n_boot=args.n_bootstrap,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
