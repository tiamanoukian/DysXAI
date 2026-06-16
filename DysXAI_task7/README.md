# Task 7 — Dysgraphia classification from *hračkárstvo* kinematics

Final research pipeline for the DysXAI project: classify dysgraphia from tablet pen dynamics on **Task 7** (*hračkárstvo*, the complex-word writing task from the Drotár protocol), with rigorous subject-level cross-validation, FFT/Butterworth preprocessing comparisons, demographic late-fusion ablations, and multi-layer explainability (permutation ablation, DeepSHAP, spatial case studies).

**Cohort:** 120 subjects (57 dysgraphic, 63 control) · **Clip:** one `.svc` per subject in `../dysxai_tasks_split/task_7_hrackarstvo/`


---

## Table of contents

1. [Scientific goal](#scientific-goal)
2. [Repository layout](#repository-layout)
3. [Model and inputs](#model-and-inputs)
4. [Preprocessing](#preprocessing)
5. [Evaluation protocols](#evaluation-protocols)
6. [Recommended run order](#recommended-run-order)
7. [Script reference](#script-reference)
8. [Outputs and artifacts](#outputs-and-artifacts)
9. [Explainability (XAI)](#explainability-xai)
10. [Environment variables](#environment-variables)
11. [Dependencies](#dependencies)
12. [Troubleshooting](#troubleshooting)

---

## Scientific goal

1. **Classification** — Can a 1D CNN on multichannel kinematics discriminate dysgraphic vs. control children on Task 7?
2. **Preprocessing** — Does XY low-pass filtering (none vs. Butterworth vs. FFT) matter, and should cutoff frequency be tuned rather than fixed at 12 Hz?
3. **Demographics** — Does late-fusion **age** and/or **gender** improve ranking (AUC) or only shift sensitivity–specificity?
4. **Interpretability** — Which channels drive predictions globally (DeepSHAP) and locally (spatial SHAP on strokes)? How do signed model velocities relate to clinical directional speed?

Primary reported metrics use **5-fold stratified group out-of-fold (OOF)** predictions pooled over the full cohort. A supplementary **80/20 holdout** (`train_final_evaluation.py`) is used for single-model explainability checkpoints.

---

## Repository layout

```
DysXAI_task7/
├── README.md                          ← this file
├── task7_results_summary_for_paper.md ← results paragraphs, tables, figure list
│
├── dataset.py                         ← Task 7 data loading, filters, derivatives, scaling
├── model.py                           ← Task7Conv1dClassifier (alias Task5Conv1dClassifier)
├── checkpoint_io.py                   ← save/load .pt checkpoints + XAI defaults
│
├── train_ab_test.py                   ← nested epoch tuning on 80% pool (holdout untouched)
├── train_oof_evaluation.py            ← main OOF evaluation + demographic ablation
├── train_final_evaluation.py          ← holdout test after tuning JSON
├── train_xai_checkpoint.py            ← train canonical XAI checkpoint (FFT + age)
├── rebuild_ablation_fold_hparams_from_checkpoints.py
│
├── explain_oof_ablation.py            ← permutation feature importance (AUC + accuracy)
├── explain_oof_leave_one_feature_out.py
├── critical_feature_importance_analysis.py  ← repeated perm + bootstrap CI + groups/pairs
├── explain_deepshap_global.py
├── explain_deepshap_local_spatial.py
├── analyze_errors.py                  ← FP/FN demographics + statistical tests
├── analyze_directional_velocity_age.py
├── analyze_directional_velocity_predictions.py
│
├── visualize_model_architecture.py
├── visualize_filter_overlay.py
├── visualize_fft_cutoff_sweep.py
├── visualize_colored_trajectory.py
│
├── checkpoints/                       ← per-fold OOF + holdout .pt (created at train time)
├── architecture_outputs/
├── XAI results/                         ← figures, CSVs from explainability runs
├── error_reports/                     ← from analyze_errors.py --batch-ablation
├── feature ablation results/
│
├── task7_oof_results.csv              ← filter comparison (raw / butterworth / fft)
├── task7_demographic_ablation_results.csv
├── task7_ablation_oof_fold_hparams.csv
├── task7_holdout_results.csv
└── tuned_params_*.json                ← from train_ab_test.py (per filter, per demographics)
```

All scripts are intended to run from the **repository root** or from `DysXAI_task7/` (imports resolve both).

---

## Model and inputs

**Architecture:** `Task7Conv1dClassifier` in `model.py`

- **Encoder:** 3× Conv1d blocks (64 → 128 → 256) + BatchNorm + ReLU + MaxPool(2) ×2
- **Pooling:** masked global average over valid timesteps (padding ignored)
- **Head:** Dropout → Linear(256+demo → 128) → ReLU → Dropout → Linear(128 → 1) → sigmoid at inference
- **Late fusion:** optional scalar **age** (min–max normalized per fold) and/or **gender** (0=male, 1=female) concatenated after pooling — not broadcast into the conv stack

**Kinematic channels (12)** after preprocessing: X, Y, Pressure, Azimuth, Tilt, PenStatus, Vx, Vy, Ax, Ay, Jx, Jy. Raw **Time** is dropped before the network.

Diagrams: run `visualize_model_architecture.py` → `architecture_outputs/`.

---

## Preprocessing

Implemented in `dataset.py` (reuses `dysxai_init.load_metadata` / `load_raw_timeseries` from the parent repo).

| Step | Detail |
|------|--------|
| XY filter | `raw` (none), `butterworth[_hz]`, or `fft[_hz]` on X/Y **before** derivatives |
| Sampling | 133 Hz nominal; default cutoff 12 Hz when frequency omitted |
| Derivatives | Vx, Vy, Ax, Ay, Jx, Jy via per-sample timestamps (`safe_dt` handles bad gaps) |
| Sequence | Pad/truncate to `MAX_SEQ_LEN = 2000` |
| Scaling | Per-channel z-score (`channel_mean` / `channel_std`) fit on **training indices only** |
| Demographics | Age/gender read from metadata; age min–max normalized per training fold |

**FFT implementation** lives in parent repo: `dysxai_fft_xy_filter.py` (cosine rolloff, optional reflect padding).

**Environment:** `DysXAI_task7_XY_FILTER` selects default filter when scripts do not pass `xy_filter` explicitly. Values: `raw`, `butterworth`, `fft`, or with suffix e.g. `fft_10`.

---

## Evaluation protocols

### A — Out-of-fold (primary)

`train_oof_evaluation.py`

- **Outer:** `StratifiedGroupKFold(n_splits=5)` on 120 subjects — each subject in test exactly once
- **Inner:** 4-fold on the outer **train** pool → selects **(FFT cutoff Hz, epoch)** jointly for ablation mode, or epoch only for standard filter comparison
- **Hz grid (ablation):** {8, 10, 12, 15}
- **Metrics:** AUC, accuracy, sensitivity, specificity at threshold 0.5 on pooled OOF predictions
- **Checkpoints:** `checkpoints/oof_<filter>_<demographics>_foldNN.pt`

```bash
# Compare raw vs Butterworth vs FFT (kinematics only, fixed 12 Hz in filter name)
python DysXAI_task7/train_oof_evaluation.py

# Full demographic ablation (FFT only, nested Hz + epoch tuning)
python DysXAI_task7/train_oof_evaluation.py --ablation-demographics
```

Writes `task7_demographic_ablation_results.csv`, per-config OOF prediction CSVs, and `task7_ablation_oof_fold_hparams.csv`.

### B — Holdout tuning + test (supplementary)

1. `train_ab_test.py` — 80/20 subject split; 5-fold CV on the 80% pool tunes **epoch** only; 20% holdout never seen
2. `train_final_evaluation.py` — retrains on 80% at tuned epoch, evaluates holdout once

```bash
python DysXAI_task7/train_ab_test.py --all-filters
python DysXAI_task7/train_ab_test.py --all-filters --use-age --use-gender
python DysXAI_task7/train_final_evaluation.py --compare-age
python DysXAI_task7/train_final_evaluation.py --use-gender   # age + gender holdout
```

Holdout metrics → `task7_holdout_results.csv`. **Note:** n=24 holdout has high variance; prefer OOF for model comparison.

### C — Inference-only XAI

Scripts under `explain_*` and `critical_*` load saved OOF checkpoints and score **outer test splits only** (~24 clips/fold). They do **not** retrain.

---

## Recommended run order

Full reproduction from a clean checkout (after parent data setup; see root `README.md`):

```bash
# 1) OOF filter comparison (kinematics only)
python DysXAI_task7/train_oof_evaluation.py

# 2) Demographic ablation (saves checkpoints for XAI)
python DysXAI_task7/train_oof_evaluation.py --ablation-demographics

# 3) Error / fairness analysis on ablation predictions
python DysXAI_task7/analyze_errors.py --batch-ablation

# 4) Feature importance (permutation on OOF checkpoints)
python DysXAI_task7/explain_oof_ablation.py --configuration "Baseline + Age + Gender"
python DysXAI_task7/critical_feature_importance_analysis.py

# 5) Holdout tuning (for global DeepSHAP single model)
python DysXAI_task7/train_ab_test.py --all-filters --use-age --use-gender
python DysXAI_task7/train_final_evaluation.py --use-gender
# → checkpoints/holdout_fft_age_gender.pt

# 6) Global DeepSHAP
python DysXAI_task7/explain_deepshap_global.py

# 7) Directional velocity vs predictions (clinical speed alignment)
python DysXAI_task7/analyze_directional_velocity_predictions.py

# 8) Local spatial DeepSHAP (case studies on trajectories)
python DysXAI_task7/explain_deepshap_local_spatial.py

# 9) Optional figures for reports
python DysXAI_task7/visualize_model_architecture.py --use-gender
python DysXAI_task7/visualize_filter_overlay.py --compare-groups
```

If `task7_ablation_oof_fold_hparams.csv` is missing but checkpoints exist:

```bash
python DysXAI_task7/rebuild_ablation_fold_hparams_from_checkpoints.py
```

---

## Script reference

| Script | Purpose |
|--------|---------|
| `dataset.py` | `build_sample_table()`, `Task5TrajectoryDataset`, `load_processed_tensor()`, filter helpers |
| `model.py` | `Task7Conv1dClassifier` |
| `checkpoint_io.py` | `save_task7_checkpoint`, `load_task7_checkpoint`, `XAI_XY_FILTER` constant |
| `train_ab_test.py` | Nested epoch tuning; `--all-filters`, `--use-age`, `--use-gender` |
| `train_oof_evaluation.py` | OOF CV; `--ablation-demographics`, `--use-age`, `--use-gender` |
| `train_final_evaluation.py` | Holdout evaluation; `--compare-age`, `--use-gender` |
| `train_xai_checkpoint.py` | Legacy XAI checkpoint (FFT + age only, from `tuned_params_fft.json`) |
| `explain_oof_ablation.py` | Permutation ablation → `XAI results/results feature importance/` |
| `explain_oof_leave_one_feature_out.py` | Leave-one-feature-out accuracy drop |
| `critical_feature_importance_analysis.py` | Repeated permutation + bootstrap CI + group/pair ablation |
| `explain_deepshap_global.py` | Global beeswarms, mean \|SHAP\|, per-subject CSVs |
| `explain_deepshap_local_spatial.py` | 8 extreme OOF cases, spatial SHAP on word/letter/loop segments |
| `analyze_errors.py` | Confusion matrices, age/gender error charts, χ² / Mann–Whitney |
| `analyze_directional_velocity_age.py` | Population directional speed by label and age |
| `analyze_directional_velocity_predictions.py` | Speed vs OOF dysgraphic probability (Spearman) |
| `visualize_model_architecture.py` | Mermaid + summary diagram for the report |
| `visualize_filter_overlay.py` | Raw vs filtered XY traces |
| `visualize_fft_cutoff_sweep.py` | Multi-cutoff FFT overlay for one control + one dysgraphic clip |
| `visualize_colored_trajectory.py` | Velocity-colored handwriting traces |
| `rebuild_ablation_fold_hparams_from_checkpoints.py` | Recover fold Hz/epoch CSV from `.pt` files |

**Shared training hyperparameters** (`train_ab_test.py`): `RANDOM_STATE=42`, `BATCH_SIZE=16`, `LR=1e-3`, `WEIGHT_DECAY=1e-4`, `MAX_EPOCHS=40`, early stopping patience 7.

---

## Outputs and artifacts

### Key result CSVs (package root)

| File | Content |
|------|---------|
| `task7_oof_results.csv` | Raw / Butterworth / FFT at 12 Hz, kinematics only |
| `task7_demographic_ablation_results.csv` | Four late-fusion configs, FFT + nested Hz |
| `task7_ablation_oof_fold_hparams.csv` | Selected Hz and epoch per outer fold × config |
| `task7_holdout_results.csv` | 20% holdout metrics (all filter × demographic combos) |
| `oof_predictions_fft_*.csv` | Per-subject OOF probabilities (one per ablation config) |

### Published headline results (OOF demographic ablation)

| Configuration | AUC | Accuracy | Sensitivity | Specificity |
|:--------------|----:|---------:|------------:|------------:|
| Kinematics only | **0.955** | 88.3% | 84.2% | 92.1% |
| + Age | 0.942 | 88.3% | 82.5% | **93.7%** |
| + Gender | 0.953 | 88.3% | **87.7%** | 88.9% |
| + Age + Gender | 0.944 | 88.3% | **87.7%** | 88.9% |

FFT cutoff selections are spread across {8, 10, 12, 15} Hz (mean ≈ 10.8 Hz) — see `task7_results_summary_for_paper.md`.

### Figure directories

- `XAI results/DeepSHAP global/` — beeswarms, mean \|SHAP\| bar chart
- `XAI results/DeepSHAP local/` — spatial case-study PNGs + `task7_local_shap_selected_cases.csv`
- `XAI results/Directional velocity vs predictions/` — scatter + correlation CSV
- `XAI results/critical_feature_importance/` — bootstrap ablation summaries
- `error_reports/ablation_demographics/` — per-config confusion matrices and demographic error plots

---

## Explainability (XAI)

### Design choices

1. **Global DeepSHAP** uses one **holdout** model (`holdout_fft_age_gender.pt`) so all 120 clips share the same explainer — required for cohort-level beeswarms.
2. **OOF ablation / critical importance** use **per-fold checkpoints** and only the locked outer-test split — leakage-safe feature ranking.
3. **Demographics vs kinematics panels** are plotted separately because scalar late-fusion features have larger |SHAP| scale than time-pooled kinematics.
4. **Directional velocity decomposition** (Vx_Pos, Vx_Neg, …) resolves the paradox between signed model channels and clinical pen-on speed (controls write faster; ρ ≈ −0.6 with dysgraphic probability).

### Canonical XAI defaults (`checkpoint_io.py`)

```python
XAI_XY_FILTER = "fft"
XAI_USE_AGE = True
XAI_USE_GENDER = False   # global script uses age+gender holdout checkpoint by default
XAI_FILTER_LABEL = "12 Hz FFT"
```

`explain_deepshap_global.py` defaults to `checkpoints/holdout_fft_age_gender.pt`. Override with `--checkpoint`.

### Local spatial DeepSHAP

Selects 8 extreme OOF cases (2× TP, TN, FP, FN), crops segments (`full_word`, `letter_h`, `loop`), overlays SHAP-colored pen traces.

```bash
python DysXAI_task7/explain_deepshap_local_spatial.py
python DysXAI_task7/explain_deepshap_local_spatial.py --segment-modes full_word letter_h
```

---

## Environment variables

| Variable | Effect |
|----------|--------|
| `DysXAI_task7_XY_FILTER` | Default XY filter in `dataset.py` (`raw`, `butterworth`, `fft`, or `fft_10`, etc.) |

Set before importing `dataset` or calling scripts that respect the env default:

```bash
# PowerShell
$env:DysXAI_task7_XY_FILTER = "fft_12"

# bash
export DysXAI_task7_XY_FILTER=fft_12
```

---

## Dependencies

Same as parent project, plus **SHAP** for DeepSHAP scripts:

```
torch, numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, tqdm, shap
```

OpenPyXL is required for metadata Excel I/O (via `dysxai_init`).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No samples in sample table` | Run `dysxai_task_splitter.py` from repo root; confirm `dysxai_tasks_split/task_7_hrackarstvo/` exists |
| `No OOF FFT checkpoints` | Run `train_oof_evaluation.py --ablation-demographics` first |
| `holdout_fft_age_gender.pt not found` | Run `train_ab_test.py --all-filters --use-age --use-gender` then `train_final_evaluation.py --use-gender` |
| CUDA OOM | Reduce `BATCH_SIZE` in `train_ab_test.py` |
| Explain script can't find checkpoint | Pass `--checkpoint-dir DysXAI_task7/checkpoints` or `--checkpoint <path>` |
| Import errors from repo root | Run as `python DysXAI_task7/<script>.py` or `cd DysXAI_task7` |

---


