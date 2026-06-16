"""
Analyze demographic distribution of OOF errors (FP/FN) for Task 7.

Generates per-scenario:
  - confusion matrix heatmap (TN/FP/FN/TP)
  - gender FP/FN chart (with counts on bars)
  - age error breakdown chart
  - combined age x gender heatmaps + ranked demographic groups
  - merged CSV + stats CSV

Batch modes:
  --batch-ablation   : 4 FFT demographic ablation CSVs
  --batch-filters    : oof_predictions_{raw,butterworth,fft}.csv (if present)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from dataset import build_sample_table

_PKG_DIR = Path(__file__).resolve().parent
THRESHOLD = 0.5

# Match other Task 7 report figures (Control=blue, Dysgraphic=red).
GROUP_PALETTE = {"Control": "#1f77b4", "Dysgraphic": "#d62728"}
ERROR_PALETTE = {
    "False Negative": GROUP_PALETTE["Dysgraphic"],
    "False Positive": GROUP_PALETTE["Control"],
}
OUTCOME_PALETTE = {
    "Correct": "#bdbdbd",
    "False Negative": GROUP_PALETTE["Dysgraphic"],
    "False Positive": GROUP_PALETTE["Control"],
}
GENDER_ORDER = ("Male", "Female")

ABLATION_PRED_FILES: list[tuple[str, Path]] = [
    ("Baseline (Kinematics Only)", _PKG_DIR / "oof_predictions_fft_baseline_kinematics_only.csv"),
    ("Baseline + Age", _PKG_DIR / "oof_predictions_fft_baseline_plus_age.csv"),
    ("Baseline + Gender", _PKG_DIR / "oof_predictions_fft_baseline_plus_gender.csv"),
    ("Baseline + Age + Gender", _PKG_DIR / "oof_predictions_fft_baseline_plus_age_plus_gender.csv"),
]

FILTER_PRED_FILES: list[tuple[str, Path]] = [
    ("Raw (No Filter)", _PKG_DIR / "oof_predictions_raw.csv"),
    ("Tuned Butterworth", _PKG_DIR / "oof_predictions_butterworth.csv"),
    ("Tuned FFT", _PKG_DIR / "oof_predictions_fft.csv"),
]


@dataclass(frozen=True)
class ScenarioContext:
    label: str
    pred_csv: Path
    filter_label: str
    use_age: bool | None
    use_gender: bool | None
    output_dir: Path


def _normalize_gender_value(v: object) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "Unknown"
    if f == 0.0:
        return "Male"
    if f == 1.0:
        return "Female"
    return "Unknown"


def apply_report_style() -> None:
    """Shared fonts/grid for error-analysis figures (matches velocity-age plots)."""
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _slug(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def infer_scenario_from_path(pred_csv: Path) -> tuple[str, str, bool | None, bool | None]:
    """Return (scenario_label, filter_label, use_age, use_gender) from filename."""
    stem = pred_csv.stem
    if stem.startswith("oof_predictions_"):
        stem = stem[len("oof_predictions_") :]

    filter_label = "Unknown filter"
    use_age: bool | None = None
    use_gender: bool | None = None

    if stem == "raw":
        filter_label, label = "Raw (No Filter)", "OOF | Raw filter"
    elif stem == "butterworth":
        filter_label, label = "Tuned Butterworth", "OOF | Butterworth filter"
    elif stem == "fft":
        filter_label, label = "Tuned FFT", "OOF | FFT filter"
    elif stem.startswith("fft_baseline"):
        filter_label = "Tuned FFT (ablation)"
        if "kinematics_only" in stem:
            label, use_age, use_gender = "FFT ablation | Kinematics only", False, False
        elif "plus_age_plus_gender" in stem:
            label, use_age, use_gender = "FFT ablation | Age + Gender", True, True
        elif "plus_age" in stem:
            label, use_age, use_gender = "FFT ablation | + Age", True, False
        elif "plus_gender" in stem:
            label, use_age, use_gender = "FFT ablation | + Gender", False, True
        else:
            label = f"FFT ablation | {stem}"
    else:
        label = f"OOF | {stem.replace('_', ' ')}"
        if "raw" in stem:
            filter_label = "Raw"
        elif "butterworth" in stem:
            filter_label = "Butterworth"
        elif "fft" in stem:
            filter_label = "FFT"

    return label, filter_label, use_age, use_gender


def load_merged_frame(pred_csv: Path) -> pd.DataFrame:
    pred_df = pd.read_csv(pred_csv)
    required = {"subject_id", "true_label", "pred_prob"}
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"{pred_csv} missing required columns: {sorted(missing)}")

    sample_df = build_sample_table()
    demo_df = (
        sample_df[["subject_id", "age", "gender"]]
        .drop_duplicates(subset=["subject_id"])
        .copy()
    )
    demo_df["gender_label"] = demo_df["gender"].map(_normalize_gender_value)

    merged = pred_df.merge(demo_df, on="subject_id", how="left", validate="many_to_one")
    merged["pred_label"] = (merged["pred_prob"] >= THRESHOLD).astype(int)
    merged["error_type"] = "Correct"
    merged.loc[(merged["true_label"] == 0) & (merged["pred_label"] == 1), "error_type"] = "False Positive"
    merged.loc[(merged["true_label"] == 1) & (merged["pred_label"] == 0), "error_type"] = "False Negative"
    return merged


def compute_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    y_true = df["true_label"].to_numpy(dtype=int)
    y_prob = df["pred_prob"].to_numpy(dtype=float)
    y_pred = df["pred_label"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out: dict[str, float | int] = {
        "n": int(len(df)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) < 2:
        out["auc"] = float("nan")
    else:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return out


def _errors_with_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """FP/FN rows with known Male/Female gender and integer age."""
    err = df[df["error_type"].isin(["False Positive", "False Negative"])].copy()
    err = err[err["gender_label"].isin(GENDER_ORDER) & pd.notna(err["age"])].copy()
    err["age_int"] = err["age"].round().astype(int)
    err["demo_label"] = err["gender_label"] + ", age " + err["age_int"].astype(str)
    return err


def _pivot_age_gender_counts(
    err_df: pd.DataFrame, error_type: str, ages: list[int] | None = None
) -> pd.DataFrame:
    if ages is None:
        ages = sorted(err_df["age_int"].unique().tolist())
    sub = err_df[err_df["error_type"] == error_type]
    if sub.empty:
        return pd.DataFrame(0, index=ages, columns=list(GENDER_ORDER))
    pv = (
        sub.groupby(["age_int", "gender_label"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    return pv.reindex(index=ages, columns=list(GENDER_ORDER), fill_value=0)


def summarize_age_gender_errors(err_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (age, gender, error_type) with error count."""
    if err_df.empty:
        return pd.DataFrame(
            columns=["gender_label", "age_int", "demo_label", "error_type", "count"]
        )
    summary = (
        err_df.groupby(["gender_label", "age_int", "demo_label", "error_type"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "age_int"], ascending=[False, True])
    )
    return summary


def _top_demo_caption(err_df: pd.DataFrame, summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No FP/FN errors with known age and gender."
    top = summary.iloc[0]
    return (
        f"Largest group: {top['demo_label']} - {int(top['count'])} "
        f"{top['error_type']}{'s' if top['count'] != 1 else ''} "
        f"({int(err_df.shape[0])} total errors)"
    )


def _demographics_subtitle(ctx: ScenarioContext) -> str:
    if ctx.use_age is None and ctx.use_gender is None:
        return f"Filter: {ctx.filter_label}"
    age_txt = "Age=ON" if ctx.use_age else "Age=OFF"
    gender_txt = "Gender=ON" if ctx.use_gender else "Gender=OFF"
    return f"Filter: {ctx.filter_label} | {age_txt} | {gender_txt}"


def plot_confusion_matrix(df: pd.DataFrame, ctx: ScenarioContext, output_path: Path) -> None:
    y_true = df["true_label"].to_numpy(dtype=int)
    y_pred = df["pred_label"].to_numpy(dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = compute_metrics(df)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Control (0)", "Pred Dysgraphic (1)"],
        yticklabels=["Actual Control (0)", "Actual Dysgraphic (1)"],
        ax=ax,
    )
    ax.set_title(
        f"Confusion Matrix\n{ctx.label}\n{_demographics_subtitle(ctx)}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    caption = (
        f"N={metrics['n']} | AUC={metrics['auc']:.3f} | Acc={metrics['accuracy']:.3f} | "
        f"Sens={metrics['sensitivity']:.3f} | Spec={metrics['specificity']:.3f}\n"
        f"TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}"
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_gender_errors(df: pd.DataFrame, ctx: ScenarioContext, output_path: Path) -> None:
    plot_df = df[df["error_type"].isin(["False Positive", "False Negative"])].copy()
    plot_df = plot_df[plot_df["gender_label"].isin(GENDER_ORDER)]
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No FP/FN errors in this scenario.", ha="center", va="center")
        ax.set_title(f"FP/FN by Gender\n{ctx.label}", fontweight="bold")
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.countplot(
        data=plot_df,
        x="gender_label",
        hue="error_type",
        order=list(GENDER_ORDER),
        hue_order=["False Negative", "False Positive"],
        palette=ERROR_PALETTE,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fontsize=9, padding=2)
    ax.set_title(
        f"FP/FN Counts by Gender\n{ctx.label}\n{_demographics_subtitle(ctx)}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Gender")
    ax.set_ylabel("Error count")
    ax.legend(title="Error type", loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_age_errors(df: pd.DataFrame, ctx: ScenarioContext, output_path: Path) -> None:
    age_df = df[pd.notna(df["age"])].copy()
    age_df["age_int"] = age_df["age"].round().astype(int)

    summary = (
        age_df.groupby(["age_int", "error_type"], observed=True)
        .size()
        .reset_index(name="count")
    )
    order_ages = sorted(age_df["age_int"].unique().tolist())
    hue_order = ["Correct", "False Negative", "False Positive"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(
        data=summary,
        x="age_int",
        y="count",
        hue="error_type",
        order=order_ages,
        hue_order=hue_order,
        palette=OUTCOME_PALETTE,
        ax=ax,
    )
    ax.set_title(
        f"Outcome Counts by Age\n{ctx.label}\n{_demographics_subtitle(ctx)}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.legend(title="Outcome", loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_age_gender_errors(df: pd.DataFrame, ctx: ScenarioContext, output_path: Path) -> pd.DataFrame:
    """
    Combined age x gender view: heatmaps (FN/FP) + ranked bar chart of demo groups.
    Returns summary table for CSV export.
    """
    err_df = _errors_with_demographics(df)
    summary = summarize_age_gender_errors(err_df)
    caption = _top_demo_caption(err_df, summary)

    if err_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No FP/FN errors with known age and gender.", ha="center", va="center")
        ax.set_title(
            f"FP/FN by Age and Gender\n{ctx.label}\n{_demographics_subtitle(ctx)}",
            fontweight="bold",
        )
        fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return summary

    ages = sorted(err_df["age_int"].unique().tolist())
    fn_pv = _pivot_age_gender_counts(err_df, "False Negative", ages)
    fp_pv = _pivot_age_gender_counts(err_df, "False Positive", ages)
    vmax = max(float(fn_pv.values.max()), float(fp_pv.values.max()), 1.0)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.38, wspace=0.28)

    ax_fn = fig.add_subplot(gs[0, 0])
    ax_fp = fig.add_subplot(gs[0, 1])
    sns.heatmap(
        fn_pv,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(GROUP_PALETTE["Dysgraphic"], as_cmap=True),
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Count"},
        ax=ax_fn,
    )
    ax_fn.set_title("False Negative", fontweight="bold")
    ax_fn.set_xlabel("Gender")
    ax_fn.set_ylabel("Age (years)")

    sns.heatmap(
        fp_pv,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(GROUP_PALETTE["Control"], as_cmap=True),
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Count"},
        ax=ax_fp,
    )
    ax_fp.set_title("False Positive", fontweight="bold")
    ax_fp.set_xlabel("Gender")
    ax_fp.set_ylabel("Age (years)")

    ax_bar = fig.add_subplot(gs[1, :])
    bar_df = summary.copy()
    bar_df = bar_df.sort_values("count", ascending=True)
    colors = bar_df["error_type"].map(ERROR_PALETTE)
    y_pos = np.arange(len(bar_df))
    ax_bar.barh(y_pos, bar_df["count"], color=colors, edgecolor="white", linewidth=0.6)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(
        [f"{row.demo_label} ({row.error_type})" for row in bar_df.itertuples()],
        fontsize=9,
    )
    ax_bar.set_xlabel("Error count")
    ax_bar.set_title("All age–gender error groups (ranked)", fontweight="bold")
    for i, (_, row) in enumerate(bar_df.iterrows()):
        ax_bar.text(row["count"] + 0.05, i, str(int(row["count"])), va="center", fontsize=9)
    ax_bar.set_xlim(0, max(bar_df["count"].max() + 1.2, 1.5))

    ax_bar.legend(
        handles=[
            Patch(facecolor=ERROR_PALETTE["False Negative"], label="False Negative"),
            Patch(facecolor=ERROR_PALETTE["False Positive"], label="False Positive"),
        ],
        loc="lower right",
        title="Error type",
    )

    fig.suptitle(
        f"FP/FN by Age and Gender\n{ctx.label}\n{_demographics_subtitle(ctx)}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, style="italic")
    fig.subplots_adjust(top=0.88, bottom=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return summary


def print_error_summary(df: pd.DataFrame, ctx: ScenarioContext) -> None:
    print(f"\n=== {ctx.label} ===")
    print(_demographics_subtitle(ctx))
    metrics = compute_metrics(df)
    print(
        f"Global: N={metrics['n']} AUC={metrics['auc']:.4f} Acc={metrics['accuracy']:.4f} "
        f"Sens={metrics['sensitivity']:.4f} Spec={metrics['specificity']:.4f}"
    )
    print(
        f"Confusion: TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}"
    )
    for et in ("False Negative", "False Positive"):
        sub = df[df["error_type"] == et]
        counts = sub["gender_label"].value_counts()
        male_n = int(counts.get("Male", 0))
        female_n = int(counts.get("Female", 0))
        unknown_n = int(counts.get("Unknown", 0))
        line = f"  {et}s: {male_n} Male, {female_n} Female"
        if unknown_n:
            line += f", {unknown_n} Unknown (excluded from gender plot)"
        print(line)

    err_df = _errors_with_demographics(df)
    ag_summary = summarize_age_gender_errors(err_df)
    if ag_summary.empty:
        print("  Age+gender groups: (none)")
    else:
        print(f"  {_top_demo_caption(err_df, ag_summary)}")
        print("  Top age–gender groups:")
        for _, row in ag_summary.head(5).iterrows():
            print(
                f"    - {row['demo_label']}: {int(row['count'])} {row['error_type']}"
                f"{'s' if row['count'] != 1 else ''}"
            )


def run_statistical_tests(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    gender_df = df[df["gender_label"].isin(["Male", "Female"])].copy()
    for error_type in ("False Negative", "False Positive"):
        sub = gender_df[gender_df["error_type"].isin([error_type, "Correct"])].copy()
        test_name = f"chi2_gender_{error_type.lower().replace(' ', '_')}_vs_correct"
        if sub.empty:
            rows.append(
                {
                    "scenario": scenario,
                    "test_name": test_name,
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "n": 0,
                    "note": "No rows for this comparison.",
                }
            )
            continue
        contingency = pd.crosstab(sub["error_type"], sub["gender_label"])
        if contingency.shape != (2, 2):
            rows.append(
                {
                    "scenario": scenario,
                    "test_name": test_name,
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "n": int(len(sub)),
                    "note": "Need both genders and both classes for chi-square.",
                }
            )
            continue
        chi2, p, _, _ = chi2_contingency(contingency)
        rows.append(
            {
                "scenario": scenario,
                "test_name": test_name,
                "statistic": float(chi2),
                "p_value": float(p),
                "n": int(len(sub)),
                "note": "2x2 chi-square on Male/Female counts.",
            }
        )

    age_df = df[pd.notna(df["age"])].copy()
    for error_type in ("False Negative", "False Positive"):
        a = age_df[age_df["error_type"] == error_type]["age"].to_numpy(dtype=float)
        b = age_df[age_df["error_type"] == "Correct"]["age"].to_numpy(dtype=float)
        test_name = f"mannwhitney_age_{error_type.lower().replace(' ', '_')}_vs_correct"
        if len(a) == 0 or len(b) == 0:
            rows.append(
                {
                    "scenario": scenario,
                    "test_name": test_name,
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "n": int(len(a) + len(b)),
                    "note": "Insufficient age data for Mann-Whitney U test.",
                }
            )
            continue
        u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "scenario": scenario,
                "test_name": test_name,
                "statistic": float(u_stat),
                "p_value": float(p_val),
                "n": int(len(a) + len(b)),
                "note": "Two-sided Mann-Whitney U (error group vs Correct).",
            }
        )

    return pd.DataFrame(rows)


def analyze_scenario(ctx: ScenarioContext) -> dict[str, object]:
    if not ctx.pred_csv.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {ctx.pred_csv}")

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_merged_frame(ctx.pred_csv)
    metrics = compute_metrics(df)

    merged_csv = ctx.output_dir / "predictions_with_errors.csv"
    df.to_csv(merged_csv, index=False)

    cm_png = ctx.output_dir / "confusion_matrix.png"
    gender_png = ctx.output_dir / "error_distribution_gender.png"
    age_png = ctx.output_dir / "error_distribution_age.png"
    age_gender_png = ctx.output_dir / "error_distribution_age_gender.png"
    age_gender_csv = ctx.output_dir / "age_gender_error_counts.csv"
    stats_csv = ctx.output_dir / "error_analysis_stats.csv"
    summary_txt = ctx.output_dir / "summary.txt"

    print_error_summary(df, ctx)
    stats_df = run_statistical_tests(df, ctx.label)
    stats_df.to_csv(stats_csv, index=False)

    plot_confusion_matrix(df, ctx, cm_png)
    plot_gender_errors(df, ctx, gender_png)
    plot_age_errors(df, ctx, age_png)
    ag_summary = plot_age_gender_errors(df, ctx, age_gender_png)
    ag_summary.to_csv(age_gender_csv, index=False)

    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"Scenario: {ctx.label}\n")
        f.write(f"{_demographics_subtitle(ctx)}\n")
        f.write(f"Source CSV: {ctx.pred_csv.name}\n")
        f.write(
            f"N={metrics['n']} AUC={metrics['auc']:.6f} Acc={metrics['accuracy']:.6f} "
            f"Sens={metrics['sensitivity']:.6f} Spec={metrics['specificity']:.6f}\n"
        )
        f.write(f"TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}\n")
        err_df = _errors_with_demographics(df)
        ag_summary = summarize_age_gender_errors(err_df)
        f.write(f"{_top_demo_caption(err_df, ag_summary)}\n")
        if not ag_summary.empty:
            f.write("Age-gender error counts:\n")
            for _, row in ag_summary.iterrows():
                f.write(
                    f"  {row['demo_label']}: {int(row['count'])} {row['error_type']}\n"
                )

    print(f"  Wrote: {ctx.output_dir.resolve()}")
    row = {
        "scenario": ctx.label,
        "filter": ctx.filter_label,
        "use_age": ctx.use_age,
        "use_gender": ctx.use_gender,
        "pred_csv": str(ctx.pred_csv.name),
        **metrics,
    }
    return row


def run_batch(jobs: list[tuple[str, Path]], reports_root: Path, subdir_prefix: str) -> None:
    summary_rows: list[dict[str, object]] = []
    print(f"\n{'=' * 70}")
    print(f"Batch error analysis -> {reports_root.resolve()}")
    print(f"{'=' * 70}")

    for label, pred_csv in jobs:
        if not pred_csv.is_file():
            print(f"Skipping missing file: {pred_csv.name}")
            continue
        inferred_label, filter_label, use_age, use_gender = infer_scenario_from_path(pred_csv)
        scenario_label = label if label else inferred_label
        out_dir = reports_root / f"{subdir_prefix}_{_slug(scenario_label)}"
        ctx = ScenarioContext(
            label=scenario_label,
            pred_csv=pred_csv,
            filter_label=filter_label,
            use_age=use_age,
            use_gender=use_gender,
            output_dir=out_dir,
        )
        summary_rows.append(analyze_scenario(ctx))

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = reports_root / "all_scenarios_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nWrote batch summary: {summary_path.resolve()}")

        print("\n| Scenario | Filter | AUC | Acc | TN | FP | FN | TP |")
        print("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in summary_rows:
            print(
                f"| {row['scenario']} | {row['filter']} | "
                f"{row['auc']:.3f} | {row['accuracy']:.3f} | "
                f"{row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze demographic distribution of OOF errors.")
    parser.add_argument(
        "--pred-csv",
        type=Path,
        default=None,
        help="Single OOF prediction CSV from train_oof_evaluation.py.",
    )
    parser.add_argument(
        "--scenario-label",
        type=str,
        default=None,
        help="Optional explicit scenario label for plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for single-scenario analysis.",
    )
    parser.add_argument(
        "--batch-ablation",
        action="store_true",
        help="Analyze all 4 FFT demographic ablation prediction CSVs.",
    )
    parser.add_argument(
        "--batch-filters",
        action="store_true",
        help="Analyze oof_predictions_raw/butterworth/fft.csv if present.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_PKG_DIR / "error_reports",
        help="Root folder for batch report outputs.",
    )
    args = parser.parse_args()

    apply_report_style()

    if args.batch_ablation and args.batch_filters:
        parser.error("Use only one batch flag at a time (--batch-ablation or --batch-filters).")

    if args.batch_ablation:
        run_batch(ABLATION_PRED_FILES, args.reports_dir / "ablation_demographics", "ablation")
        return

    if args.batch_filters:
        run_batch(FILTER_PRED_FILES, args.reports_dir / "filters", "filter")
        return

    pred_csv = args.pred_csv or (_PKG_DIR / "oof_predictions_fft_baseline_plus_age_plus_gender.csv")
    inferred_label, filter_label, use_age, use_gender = infer_scenario_from_path(pred_csv)
    label = args.scenario_label or inferred_label
    out_dir = args.output_dir or (args.reports_dir / f"single_{_slug(label)}")

    ctx = ScenarioContext(
        label=label,
        pred_csv=pred_csv,
        filter_label=filter_label,
        use_age=use_age,
        use_gender=use_gender,
        output_dir=out_dir,
    )
    analyze_scenario(ctx)


if __name__ == "__main__":
    main()
