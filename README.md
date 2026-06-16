# DysXAI — Explainable dysgraphia detection from handwriting kinematics

Reproducible deep-learning pipeline for classifying dysgraphia from WACOM tablet pen dynamics, built on the public [Drotár / SciRep dataset](https://www.nature.com/articles/s41598-020-61929-y). The project combines model comparison notebooks, rigorous leakage checks, per-task data splitting, exploratory kinematic analysis, and a **final Task 7 pipeline** with nested cross-validation and multi-layer explainability.

| Document | Audience |
|----------|----------|
| **This file** | Project overview, notebooks, dataset tooling, analysis scripts |
| [`DysXAI_task7/README.md`](DysXAI_task7/README.md) | **Complete guide** to the final Task 7 classifier, CV, and XAI |
| [`DysXAI_task7/task7_results_summary_for_paper.md`](DysXAI_task7/task7_results_summary_for_paper.md) | Copy-paste results text and tables |
| [`Report for M1 Internship/`](Report%20for%20M1%20Internship/) | M1 internship report (IEEE Overleaf template + writing guide) |
| [`DysXAI_Research_Project_Report/`](DysXAI_Research_Project_Report/) | NeurIPS-style research paper draft (separate from internship report) |

---

## Project structure

```
DysXAI/
├── README.md                          ← you are here
├── dysxai_init.py                     ← Config, loaders, HandwritingDataset (canonical)
│
├── 00_initialization.ipynb            ┐
├── 01_model_cnn1d.ipynb               │ Exploration phase (notebooks)
├── 02_model_tcn.ipynb                 │
├── 03_model_cnnlstm.ipynb             │
├── 04_feature_selection.ipynb         │
├── 05_explainability.ipynb            │
├── 06_padding_ablation.ipynb          │
├── 07_cv_evaluation.ipynb             │
├── 08_analyze_true_duration.ipynb     ┘
│
├── DysXAI_task7/                      ← Final pipeline (see dedicated README)
│
├── dysxai_task_splitter.py            ┐ Dataset & exploratory analysis
├── validate_task_split.py             │
├── generate_images.py                 │
├── analyze_velocity_age.py            │
├── analyze_filtered_velocity_age.py   │
├── calculate_age_compensation.py      │
├── analyze_true_duration.py           │
├── dysxai_fft_xy_filter.py            │
├── task7_meeting_visualizations.py    ┘
│
├── train_cnn1d_age_ablation.py        ┐ Scripted K-fold age ablations
├── train_tcn_age_ablation.py          │ (Task 5, mirror notebook models)
├── train_cnnlstm_age_ablation.py      ┘
├── dysxai_kfold_age_common.py
├── dysxai_tcn_cnnlstm_models.py
├── dysxai_leakage_ablation.py
├── dysxai_cv_evaluation.py
│
├── dysxai_tasks_split/                ← Per-task .svc clips (generated)
├── dataSciRep_public/                 ← Raw dataset (unzip before use)
├── data2_SciRep_pub.xlsx              ← Subject metadata
│
├── Report for M1 Internship/          ← M1 internship LaTeX report
└── DysXAI_Research_Project_Report/    ← Research paper draft
```

---

## Quick start

### 1. Data setup

1. Unzip `dataSciRep_public.zip` so recordings live under `dataSciRep_public/dataSciRep_public/`.
2. Point paths in `dysxai_init.Config` (or override in `00_initialization.ipynb`):
   - `DATA_ROOT` — raw `.svc` session files
   - `META_XLSX` — `data2_SciRep_pub.xlsx`
3. Split sessions into eight task clips:

```bash
python dysxai_task_splitter.py
python validate_task_split.py
```

This creates `dysxai_tasks_split/task_1_l_normal/` … `task_8_sentence/`.

### 2. Exploration notebooks (phase 1)

Always run **`00_initialization.ipynb`** first. Then run model notebooks in any order:

| Notebook | Model | Notes |
|----------|-------|-------|
| `01_model_cnn1d.ipynb` | 1D CNN | Fast; best early benchmark (~92% acc) |
| `02_model_tcn.ipynb` | TCN | Long-range temporal patterns |
| `03_model_cnnlstm.ipynb` | CNN–LSTM | Hybrid local + sequential |
| `04_feature_selection.ipynb` | — | Forward feature selection (CNN1D) |
| `05_explainability.ipynb` | — | Ablation, SHAP, counterfactuals |
| `06_padding_ablation.ipynb` | — | Time-channel / padding leakage |
| `07_cv_evaluation.ipynb` | — | Subject-stratified K-fold AUC |
| `08_analyze_true_duration.ipynb` | — | True writing duration vs label |

Each notebook has a `.py` mirror (`00_initialization.py`, etc.) for non-Jupyter runs.

### 3. Final Task 7 pipeline (phase 2)

All classification, nested CV, demographic ablation, and publication XAI live in **`DysXAI_task7/`**.

```bash
# Primary evaluation
python DysXAI_task7/train_oof_evaluation.py --ablation-demographics

# Explainability (after checkpoints exist)
python DysXAI_task7/explain_deepshap_global.py
```

**Full run order, script reference, and outputs:** [`DysXAI_task7/README.md`](DysXAI_task7/README.md)

---

## Core library (`dysxai_init.py`)

Shared by notebooks and scripts:

| Component | Description |
|-----------|-------------|
| `Config` | Paths, `MAX_LEN`, batch size, CV settings, sampling rate |
| `load_metadata` / `load_raw_timeseries` | Excel + `.svc` parsing |
| `load_and_process_timeseries` | Butterworth on X/Y, derivatives, optional Age channel |
| `HandwritingDataset` | PyTorch dataset with padding mask |
| `subject_independent_split` | Split by subject ID (no leakage) |
| `fit_scaler_on_train` / `apply_scaler` | Mixed MinMax (XY) + StandardScaler (kinematics) |

**Typical model input:** 13 channels (7 base + 6 derivatives + Age; Time dropped) in the notebook pipeline. Task 7 uses 12 kinematic channels + late-fusion demographics (see Task 7 README).

---

## Dataset splitting (`dysxai_task_splitter.py`)

Splits each subject's full tablet session into **eight Drotár-protocol tasks**:

1. *l* normal · 2. *l* fast · 3. *le* normal · 4. *le* fast · 5. *leto* · 6. *lamoken* · 7. *hračkárstvo* · 8. sentence

Boundaries are detected by clustering pen-down Y positions into horizontal rows, then cutting at each row's first pen-down. Output: one `.svc` per task per subject under `dysxai_tasks_split/`.

| Script | Purpose |
|--------|---------|
| `dysxai_task_splitter.py` | Generate task clips |
| `validate_task_split.py` | Verify 8 tasks/subject, pen-on masks, sample counts |
| `generate_images.py` | Export handwriting trace PNGs for all tasks |

---

## Exploratory kinematic analysis (root scripts)

These scripts motivated preprocessing and feature choices for Task 7. They operate on **Task 7 clips** unless noted.

| Script | Question addressed |
|--------|-------------------|
| `analyze_velocity_age.py` | Raw absolute pen speed vs age and dysgraphia label |
| `analyze_filtered_velocity_age.py` | Same analysis after 12 Hz Butterworth on X/Y |
| `calculate_age_compensation.py` | Fit control age→velocity regression; residual deviation per subject |
| `analyze_true_duration.py` | True writing duration (timestamp span) vs label — motivates dropping Time channel |
| `dysxai_fft_xy_filter.py` | FFT-domain low-pass (cosine rolloff) as alternative to Butterworth |
| `task7_meeting_visualizations.py` | Signed velocity plots + raw/Butterworth/FFT overlays for meetings |

```bash
python analyze_velocity_age.py
python analyze_filtered_velocity_age.py
python calculate_age_compensation.py
python analyze_true_duration.py
python task7_meeting_visualizations.py --all
```

---

## Leakage and validation scripts

| Script | Notebook | Purpose |
|--------|----------|---------|
| `dysxai_leakage_ablation.py` | `06_padding_ablation.ipynb` | Padding / time-only channel ablations |
| `dysxai_cv_evaluation.py` | `07_cv_evaluation.ipynb` | Repeated subject-stratified K-fold test AUC |
| `train_cnn1d_age_ablation.py` | — | K-fold: model with vs without Age channel (1D CNN) |
| `train_tcn_age_ablation.py` | — | Same for TCN |
| `train_cnnlstm_age_ablation.py` | — | Same for CNN–LSTM |

Shared K-fold loop: `dysxai_kfold_age_common.py` · TCN/CNN-LSTM defs: `dysxai_tcn_cnnlstm_models.py`

---

## Task 7 summary (pointer)

The final deliverable classifies dysgraphia from **Task 7 (*hračkárstvo*)** only:

- **Model:** 1D CNN with masked global pooling + optional late-fusion age/gender
- **CV:** 5-fold stratified group OOF with inner 4-fold tuning of FFT cutoff {8,10,12,15} Hz and epoch
- **Best OOF AUC:** 0.955 (kinematics only, FFT)
- **XAI:** permutation ablation, bootstrap critical importance, global + local DeepSHAP, directional velocity analysis

Do not duplicate run instructions here — see **[`DysXAI_task7/README.md`](DysXAI_task7/README.md)**.

---

## Reports

| Folder | Format | Use |
|--------|--------|-----|
| [`Report for M1 Internship/`](Report%20for%20M1%20Internship/) | IEEE `bare_jrnl.tex` | **M1 internship report** — see `SECTION_GUIDE.md` |
| [`DysXAI_Research_Project_Report/`](DysXAI_Research_Project_Report/) | NeurIPS `neurips_2026.tex` | Research paper / Master project write-up |

The internship report should emphasize your 3-month progression (notebooks → Task 7 → XAI). The research draft is a separate, submission-style document.

---

## Configuration

Edit `dysxai_init.Config` (or override in `00_initialization.ipynb`):

```python
class Config:
    DATA_ROOT = "path/to/dataSciRep_public/dataSciRep_public"
    META_XLSX = "path/to/data2_SciRep_pub.xlsx"
    MAX_LEN = 2000
    USE_DERIVATIVES = True
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LR = 1e-3
    SAMPLING_RATE_HZ = 133.0
    # Subject-independent splits
    TRAIN_SUBJECT_RATIO = 0.8
    NUM_REPEATED_SPLITS = 5
```

Task 7 uses its own constants in `DysXAI_task7/dataset.py` (`FS_HZ`, `CUTOFF_HZ`, `MAX_SEQ_LEN`) but shares metadata loaders from `dysxai_init`.

---

## Dependencies

```
torch
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
tqdm
openpyxl          # Excel metadata
shap              # Task 7 DeepSHAP only
TSInterpret       # optional, notebook 05
```

<<<<<<< HEAD
### Data Splitting
- `subject_independent_split(meta_df, train_ratio, random_state)`: Split by subjects (no leakage!)

## 🎯 Best Practices

1. **Always run initialization first**: `00_initialization.ipynb` must be run before any model training
2. **Subject-independent splits**: Critical for avoiding data leakage
3. **Fit scaler on training data only**: Never fit on test data!
4. **Use early stopping**: Prevents overfitting
5. **Class weights**: Automatically computed for imbalanced data

## 🐛 Troubleshooting

### Common Issues

**"NameError: name 'Config' is not defined"**
- Solution: Run `00_initialization.ipynb` first

**"FileNotFoundError: Could not find raw file"**
- Solution: Check `Config.DATA_ROOT` path and ensure data is unzipped

**"CUDA out of memory"**
- Solution: Reduce `BATCH_SIZE` in Config class

**"No module named 'TSInterpret'"**
- Solution: Install with `pip install TSInterpret` (see `04_explainability.ipynb`)

## 📝 Notes

- All notebooks are designed to be run independently (after initialization)
- Functions are separated into different cells for clarity
- Markdown cells provide explanations and guidelines
- Code is well-commented for presentation purposes

## 🔗 Dependencies

Required packages:
- torch
- numpy
- pandas
- scikit-learn
- matplotlib
- tqdm
- seaborn (for visualizations)
- TSInterpret (optional, for explainability)

Install with:
=======
>>>>>>> fa3b6e8 (Major pipeline update: Add nested OOF CV, demographic ablation, explainability scripts, and refine repository structure)
```bash
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn tqdm openpyxl shap
pip install TSInterpret   # optional
```
<<<<<<< HEAD
=======

---

## Best practices

1. **Run `00_initialization.ipynb` (or split tasks) before training.**
2. **Always split by subject ID** — never put the same child in train and test.
3. **Fit scalers on training data only.**
4. **Prefer OOF metrics** (Task 7) over single holdout splits for reporting.
5. **Treat FFT cutoff as a hyperparameter** — do not assume 12 Hz is universally optimal.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Config` / `meta_df` not defined | Run `00_initialization.ipynb` |
| `FileNotFoundError` for `.svc` | Check `Config.DATA_ROOT`, unzip dataset, run `dysxai_task_splitter.py` |
| CUDA OOM | Lower `BATCH_SIZE` in `Config` or Task 7 `train_ab_test.py` |
| Task 7 empty cohort | Confirm `dysxai_tasks_split/task_7_hrackarstvo/` exists |
| SHAP import error | `pip install shap` |

---

## Citation

If you use this code or the Drotár dataset, cite the original dataset paper (Drotár & Dobeš, *Scientific Reports*, 2020) and acknowledge the DysXAI pipeline.
>>>>>>> fa3b6e8 (Major pipeline update: Add nested OOF CV, demographic ablation, explainability scripts, and refine repository structure)
