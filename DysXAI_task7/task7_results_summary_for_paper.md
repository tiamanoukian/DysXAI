# Task 7 — Results summary (paper & slides)

Copy-paste ready text and tables from your completed OOF demographic ablation (`n = 120`, 5-fold subject-level CV).

---

## Results paragraph (paper)

We evaluated FFT-filtered kinematic trajectories for dysgraphia classification using nested cross-validation on 120 participants (57 dysgraphic, 63 control). The low-pass cutoff frequency was treated as a hyperparameter, searched over {8, 10, 12, 15} Hz on each outer fold’s training set (with training epochs co-tuned by inner validation AUC). Selected cutoffs varied across folds (8–15 Hz; mean 10.8 Hz over 20 fold–model selections), indicating no single frequency was uniformly optimal, although performance remained high throughout the grid. Out-of-fold (OOF) discrimination was strong for all models (AUC 0.942–0.955). A demographic ablation compared kinematics-only late fusion against models augmented with min–max normalized age and/or binary gender. Overall accuracy was identical across configurations (88.3%, 106/120 correct), but error profiles differed: kinematics-only achieved the highest OOF AUC (0.955) and the most balanced specificity (92.1%); adding gender increased sensitivity (87.7% vs 84.2%) at the cost of specificity (88.9%); adding age alone increased specificity (93.7%) but reduced sensitivity (82.5%). Post-hoc error analysis for the kinematics-only model found no significant association between misclassifications and gender (χ², *p* > 0.23) or age (Mann–Whitney, *p* > 0.18). Together, these results support treating filter cutoff as a tunable hyperparameter rather than a fixed constant, and suggest that trajectory features carry most of the predictive signal in this cohort, with demographic fusion shifting the decision boundary more than improving global ranking performance.

### Short version (abstract / slide)

Nested CV with FFT cutoffs {8, 10, 12, 15} Hz yielded strong OOF dysgraphia classification (AUC up to 0.955). Optimal Hz varied by fold (mean ~11 Hz), so cutoff should be tuned, not fixed. Late-fusion age/gender did not improve AUC; all configs reached 88.3% accuracy with different sensitivity–specificity trade-offs. Errors were not significantly associated with age or gender.

---

## Table 1 — Demographic ablation (OOF, FFT, per-fold tuned Hz)

| Configuration | AUC | Accuracy | Sensitivity | Specificity |
|:--------------|----:|---------:|------------:|------------:|
| Kinematics only | **0.955** | 88.3% | 84.2% | 92.1% |
| + Age | 0.942 | 88.3% | 82.5% | **93.7%** |
| + Gender | 0.953 | 88.3% | **87.7%** | 88.9% |
| + Age + Gender | 0.944 | 88.3% | **87.7%** | 88.9% |

*Source: `task7_demographic_ablation_results.csv`. All models: 120 OOF predictions, subject-level 5-fold CV, FFT filter with nested (Hz, epoch) selection per fold.*

**Suggested caption:** Out-of-fold performance for FFT-based models with demographic late-fusion ablation. Cutoff frequency and epoch were selected independently on each fold’s training data.

---

## Table 2 — FFT cutoff frequency selected by nested CV

### Pooled across all ablation configurations (20 fold-level selections)

| Cutoff (Hz) | Times selected | Share |
|------------:|---------------:|------:|
| 8 | 6 | 30% |
| 10 | 5 | 25% |
| 12 | 6 | 30% |
| 15 | 3 | 15% |

**Mean selected Hz:** 10.8 (SD ≈ 2.4)

### By configuration (per outer fold)

| Configuration | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|:----------------|-------:|-------:|-------:|-------:|-------:|
| Kinematics only | 12 | 8 | 12 | 8 | 15 |
| + Age | 10 | 8 | 12 | 15 | 8 |
| + Gender | 8 | 15 | 10 | 12 | 10 |
| + Age + Gender | 10 | 10 | 8 | 12 | 12 |

*Source: `task7_ablation_oof_fold_hparams.csv`*

**Suggested caption:** FFT low-pass cutoff (Hz) chosen by inner cross-validation on each outer fold’s training subjects. Selections are distributed across the search grid rather than collapsing to a single value.

**Talking point:** Treating 12 Hz as the only cutoff is methodologically weak; treating it as one point in a tuned grid is supported. The spread also implies differences among 8–15 Hz are modest relative to overall model quality.

---

## Table 3 — Error analysis (kinematics-only baseline)

| Test | Statistic | *p*-value | Interpretation |
|:-----|----------:|----------:|:---------------|
| Gender × false negatives | χ² = 1.41 | 0.235 | No evidence FN rate differs by gender |
| Gender × false positives | χ² = 0.08 | 0.774 | No evidence FP rate differs by gender |
| Age: FN vs correct | Mann–Whitney | 0.185 | No evidence FN group differs in age |
| Age: FP vs correct | Mann–Whitney | 0.920 | No evidence FP group differs in age |

*Source: `error_reports/ablation_demographics/ablation_baseline_kinematics_only/error_analysis_stats.csv`*

---

## Slide bullets (5–6 slides)

**Slide 1 — Setup**
- Task 7: classify dysgraphia from tablet kinematics (120 subjects, 5-fold OOF CV)
- FFT low-pass filter; cutoff ∈ {8, 10, 12, 15} Hz tuned per fold (nested CV)
- Ablation: kinematics only vs +age vs +gender vs +both (late fusion)

**Slide 2 — Main result**
- Strong OOF performance (AUC 0.94–0.96)
- Best ranking: **kinematics only** (AUC 0.955)
- Same accuracy (88.3%) for all four configs → different kinds of errors, not fewer errors

**Slide 3 — Demographics**
- +Gender → higher sensitivity, lower specificity (more dysgraphic detected, more control false alarms)
- +Age → higher specificity, lower sensitivity (fewer false alarms, more missed dysgraphia)
- Demographics shift the boundary; they do not clearly improve AUC here

**Slide 4 — Filter cutoff (Hz)**
- No single “winner”: 8 Hz and 12 Hz each selected 6/20 times; 15 Hz only 3/20
- Mean ~11 Hz across selections
- **Message:** cutoff is a hyperparameter; report tuning, not only a fixed 12 Hz default

**Slide 5 — Fairness / errors**
- ~14 misclassifications per model (120 − 106)
- No significant age or gender association with errors (baseline model)
- Caveat: few errors → limited statistical power

**Slide 6 — Takeaways**
1. Trajectory features are highly informative.
2. Tune FFT cutoff (8–15 Hz); do not assume 12 Hz is always optimal.
3. Age/gender fusion: ablation useful, clear AUC gain not shown on this cohort.
4. Report OOF metrics for model comparison; holdout (n=24) is supplementary.

---

## Figure suggestions

| Figure | Content | File / data |
|:-------|:--------|:------------|
| Fig. A | Bar chart: AUC by demographic configuration | Table 1 |
| Fig. B | Bar chart: Hz selection counts (8/10/12/15) | Table 2 |
| Fig. C | Sensitivity vs specificity dot plot (4 configs) | Table 1 |
| Fig. D | Confusion matrices (4 ablation scenarios) | `error_reports/ablation_demographics/` |

---

## Methods one-liner (if needed)

Hyperparameters (FFT cutoff and training epoch) were selected by inner 4-fold subject-level cross-validation on each outer fold’s training set (96 subjects), then the model was retrained and evaluated on the held-out 24 subjects; OOF predictions aggregate all outer folds.

---

## Explainability — Global DeepSHAP (Age + Gender model)

Copy-paste ready text for the XAI / interpretability subsection. Based on the holdout FFT model with late-fusion age and gender (`explain_deepshap_global.py`, *n* = 120).

### Results paragraph (paper)

We applied global DeepSHAP to the age+gender late-fusion classifier to attribute out-of-fold dysgraphic logits to input features. Because demographic inputs are scalar whereas kinematics are time series, we report **separate beeswarm panels** for demographics and for kinematics. Gender showed the largest per-feature attributions (mean |SHAP| ≈ 0.059), with high encoded gender values associated with lower dysgraphic logits and low values with higher logits; age had a weaker but monotonic effect (older → higher dysgraphic logit; mean |SHAP| ≈ 0.013). Kinematic attributions were smaller in magnitude after mean-pooling over timesteps but were structured: directional velocity rows (Vx\_Pos, Vy\_Pos, Vy\_Neg) ranked highest among handwriting channels, and positive horizontal/vertical acceleration (Ax\_Pos, Ay\_Pos) tended toward negative SHAP (control-like). To align explainability with clinical kinematics, we decomposed velocity, acceleration, and jerk into positive- and negative-direction subsets and colored velocity rows by cohort-standardized clinical speed (FFT 12 Hz, pen-on). Complementary analysis showed controls had higher mean directional speed in all four axes (~3×) and that clinical speed correlated negatively with dysgraphic probability (Spearman ρ ≈ −0.58 to −0.63). Thus SHAP explains **this model’s decision logic**—including strong gender leverage at the fusion layer—while population speed analysis supports the clinical finding that faster writing is associated with control status, highlighting that signed raw-velocity attributions alone can be misleading without directional and magnitude-aware summaries.

### Short version (abstract / slide)

Global DeepSHAP on the age+gender model: gender dominates logit attributions; age has a smaller effect. Directional velocity and positive acceleration are the main kinematic signals after time pooling. Clinical speed (directional magnitude) is higher in controls and anti-correlates with dysgraphic probability—use directional panels, not signed mean Vx/Vy alone, when interpreting velocity.

### Key observations (bullets)

**Demographics panel**
- Gender: SHAP ≈ −0.11 to +0.05; clear split (high gender value → negative SHAP; low → positive).
- Age: SHAP ≈ −0.03 to +0.03; younger → slightly control-like, older → dysgraphic-like.
- Mean |SHAP|: Gender (~0.059) ≫ Age (~0.013).

**Full kinematics panel (18 rows: X–PenStatus + directional Vx/Vy/Ax/Ay/Jx/Jy)**
- Top kinematic rows: Vx\_Pos, Vy\_Pos, Vy\_Neg, Vx\_Neg (mean |SHAP| ~0.0006–0.0014).
- Vx\_Pos / Vy\_Pos: high cohort z-scored speed → positive SHAP in this attribution view.
- Vy\_Neg: pattern **reverses by direction** (more negative Vy → positive SHAP).
- Ax\_Pos / Ay\_Pos: high positive acceleration → **negative** SHAP (toward control).
- X, Y, pressure, azimuth, most jerk rows: smaller spread; PenStatus and Jx near zero.

**Clinical speed vs model (Stage B)**
- Control mean speed higher in all four directions (D/C ≈ 0.29–0.33).
- Speed vs OOF dysgraphic probability: ρ ≈ −0.58 to −0.63 (faster → less dysgraphic).

### Critical analysis (bullets)

- **Split panels are necessary:** on one shared x-axis, gender compresses kinematic SHAP to a near-zero line; separation reflects architecture (scalar late fusion vs pooled time series), not absence of kinematic information.
- **Large gender SHAP is model behavior, not etiology:** consistent with the demographic ablation (gender shifts sensitivity/specificity); interpret as “what the classifier uses,” not proof that gender causes dysgraphia.
- **Directional decomposition resolves the velocity paradox:** signed mean Vx/Vy in model space suggested dysgraphic-associated “high velocity”; clinical directional speed shows the opposite at the population level.
- **Pooling attenuates kinematic |SHAP|:** mean-pooling over thousands of timesteps (and Pos/Neg subsets) shrinks per-row values; use mean |SHAP| bar charts for ranking alongside beeswarms.
- **SHAP ≠ clinical speed:** the model ingests signed velocities on all timesteps (different preprocessing than pen-on clinical speed); attributions and descriptive speed can diverge without either being “wrong” for its purpose.
- **Limitations:** holdout explainer on 120 subjects; global pooling does not localize effects within strokes (see local spatial DeepSHAP); directional rows are post-hoc summaries, not separate trained inputs.

### Slide bullets (1–2 slides)

**Slide — Global DeepSHAP setup**
- Holdout FFT + age + gender model; DeepSHAP; mean-pooled kinematic SHAP
- Two panels: demographics | directional full kinematics (18 features)
- Positive SHAP → higher dysgraphic logit

**Slide — What we learn**
- Gender strongest single attribution; age weaker
- Kinematics: directional velocity + positive acceleration matter most
- Controls faster clinically (ρ ≈ −0.6 with dysgraphic prob); interpret velocity via directional speed, not signed mean alone

### Figure suggestions (XAI)

| Figure | Content | File |
|:-------|:--------|:-----|
| Fig. E | Demographics beeswarm (Age, Gender) | `XAI results/DeepSHAP global/task7_deepshap_global_demographics_age_gender.png` |
| Fig. F | Full kinematics beeswarm (directional Vx/Vy/acc/jerk) | `XAI results/DeepSHAP global/task7_deepshap_global_kinematics_full_age_gender.png` |
| Fig. G | Mean \|SHAP\| ranking (all 14 model inputs) | `XAI results/DeepSHAP global/task7_deepshap_global_mean_abs_shap_age_gender.png` |
| Fig. H | Directional speed vs dysgraphic probability (2×2 scatter) | `XAI results/Directional velocity vs predictions/directional_velocity_vs_predictions_scatter_fft.png` |

**Suggested caption (Fig. F):** Global DeepSHAP for kinematic inputs. Velocity, acceleration, and jerk are split into positive- and negative-direction rows (SHAP mean-pooled by sign). Velocity rows are colored by clinical mean speed magnitude; other rows use cohort z-scores. Demographics are shown in a separate panel.

### Methods one-liner (XAI)

Global DeepSHAP (DeepExplainer, 100 background samples) was applied to the holdout age+gender model on all 120 clips. Kinematic SHAP values were mean-pooled over time; velocity, acceleration, and jerk were additionally summarized by mean SHAP on timesteps with positive vs negative signed channel values. Clinical directional speeds used FFT 12 Hz filtering, pen-on masking, robust Δt, and outlier filtering consistent with population velocity analysis.

---

## Related files in this folder

| File | Description |
|:-----|:------------|
| `task7_demographic_ablation_results.csv` | Table 1 metrics |
| `task7_ablation_oof_fold_hparams.csv` | Per-fold Hz and epoch |
| `oof_predictions_fft_baseline_*.csv` | Per-subject OOF predictions |
| `error_reports/ablation_demographics/` | Confusion matrices & plots |
| `task7_oof_results.csv` | Separate OOF run (raw / butterworth / fft at fixed 12 Hz) |
| `task7_holdout_results.csv` | Held-out 20% test (fixed 12 Hz from `train_ab_test`) |
| `XAI results/DeepSHAP global/` | Global beeswarms, mean \|SHAP\| bar chart, per-subject CSVs |
| `XAI results/Directional velocity vs predictions/` | Speed–probability correlations and scatter figure |
