# DysXAI Report – Section-by-Section Writing Guide

This guide summarizes **what to write** (brief comments) and **drafted content** for each section of the NeurIPS-style report. The main `.tex` file has in-file comments; this document serves as a quick reference.

---

## Abstract (~150–200 words)

**What to write:**
1. Problem: dysgraphia detection, need for objective screening.
2. Data: Drotar/SciRep handwriting dataset (pen dynamics).
3. Approach: deep learning (1D CNN, TCN, CNN-LSTM) + XAI (ablation, SHAP).
4. Results: best model ≈92% acc, AUC ≈0.96; important features (velocity, time, acceleration).
5. Implication: interpretable models can support clinicians.

**Draft:** Provided in `neurips_2026.tex`.

---

## 1. Introduction (~1 page)

**What to write:**
1. Dysgraphia: definition, prevalence, impact; clinical screening challenges.
2. Role of digitized handwriting and pen dynamics.
3. Gap: need for interpretable ML for clinical trust.
4. Contributions: DysXAI pipeline, model comparison, rigorous evaluation, XAI.

**Draft:** Provided. Add your own citations and adjust wording.

---

## 2. Related Work (~0.5 page)

**What to write:**
1. Prior work on dysgraphia from handwriting (Drotar, Asselborn).
2. Deep learning for time-series and handwriting.
3. XAI for medical models.

**Draft:** Provided. Expand with more specific references as needed.

---

## 3. Methods (~1.5 pages)

**What to write:**
1. **Dataset:** Drotar structure (121 samples, 120 subjects, 7 base + 6 derivative channels).
2. **Preprocessing:** derivative features, MixedFeatureScaler, padding, subject-independent split.
3. **Models:** 1D CNN, TCN, CNN-LSTM (short architecture descriptions).
4. **Evaluation:** 5 repeated splits, 70/15/15, metrics (Acc, Sens, Spec, AUC).

**Draft:** Provided. Add architecture or pipeline figures if desired.

---

## 4. Experiments (~1 page)

**What to write:**
1. Setup: hyperparameters, hardware, training details.
2. Results: table comparing CNN1D, TCN, CNN-LSTM.
3. Best model: CNN1D with reported metrics.

**Results table (from your runs):**
| Model    | Accuracy | Sensitivity | Specificity | AUC   |
|----------|----------|-------------|------------|-------|
| 1D CNN   | 0.922±0.057 | 0.871±0.119 | 0.971±0.057 | 0.963±0.046 |
| TCN      | 0.889    | --          | --         | 0.938 |
| CNN-LSTM | 0.833    | --          | --         | 0.963 |

---

## 5. Explainability Analysis (~1 page)

**What to write:**
1. **Feature ablation:** which channels, when removed, drop performance most.
2. **Forward feature selection:** Phase 1 rank, Phase 2 add, Phase 3 test.
3. **Deep SHAP:** global feature importance per channel.
4. **Clinical interpretation:** velocity, acceleration, time as key features.

**Findings:**
- Phase 1 (single-feature) rank: Time, Velocity_X, Acceleration_Y, Acceleration_X, Velocity_Y.
- Ablation: velocity, acceleration, jerk, time are most important.
- SHAP: consistent with ablation; kinematic features rank highest.

---

## 6. Discussion (~0.5 page)

**What to write:**
1. **Limitations:** small sample size, single dataset, possible overfitting on validation.
2. **Broader impact:** positive (screening aid) vs. risks (misuse, over-reliance).
3. **Future work:** larger cohorts, longitudinal data, clinician studies.

**Draft:** Provided.

---

## 7. Conclusion (~0.25 page)

**What to write:**
- Summarize contributions, main findings, and future directions in 2–3 sentences.

**Draft:** Provided.

---

## Appendix

**What to write:**
- Per-run metrics (provided in tex).
- Reproducibility: code structure, dependencies, split seeds.
- Optional: architecture diagrams, training curves, full ablation tables.

---

## Checklist (NeurIPS)

- Replace `\answerTODO{}` and `\justificationTODO{}` in `checklist.tex` with your answers.
- See NeurIPS guidelines for each item.

---

## Quick Reference – Your Project Facts

| Item               | Value                                                                 |
|--------------------|-----------------------------------------------------------------------|
| Dataset            | Drotar (dataSciRep_public), 121 samples, 120 subjects                 |
| Classes            | Dysgraphic (57) vs. non-dysgraphic (63)                                |
| Input channels     | 13 (7 base + 6 derivatives: vx, vy, ax, ay, jx, jy)                   |
| Max sequence length| 2000                                                                  |
| Split              | 70% train, 15% val, 15% test (subject-independent)                    |
| Repeated splits   | 5 (seeds: 42, 123, 456, 789, 1011)                                    |
| Best model         | 1D CNN (92.2% acc, AUC 0.963)                                         |
| Notebooks          | 00_init, 01_cnn1d, 02_tcn, 03_cnnlstm, 04_feature_selection, 05_explainability |
