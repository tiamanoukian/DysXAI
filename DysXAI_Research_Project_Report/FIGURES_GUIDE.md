# Figures to Add for ~10-Page Report

To reach approximately 10 pages including figures, save the following outputs from your notebooks. Place saved files in `DysXAI_Research_Project_Report/figures/`.

## Recommended Figures (in order of impact)

### 1. Confusion Matrix (Experiments section)
- **Source:** `01_model_cnn1d.ipynb`, `02_model_tcn.ipynb`, or `03_model_cnnlstm.ipynb`
- **Where:** After the cell that plots the confusion matrix
- **Action:** Add `plt.savefig("DysXAI_Research_Project_Report/figures/confusion_matrix.pdf", bbox_inches='tight')` before `plt.show()`
- **LaTeX:** Insert in Experiments section after Table 1

### 2. Training/Validation Loss Curves (Experiments section)
- **Source:** Same model notebooks—the cell that plots `avg_train_loss` and `avg_val_loss`
- **Action:** Add `plt.savefig("DysXAI_Research_Project_Report/figures/loss_curves.pdf", bbox_inches='tight')`
- **LaTeX:** Insert in Experiments section

### 3. Feature Ablation Bar Chart (Explainability section)
- **Source:** `05_explainability.ipynb`—create or extend the ablation results visualization
- **Content:** Bar chart showing AUC drop when each of the 13 features is removed (Time, Azimuth, Pressure, Velocity_Y, etc. should stand out)
- **Action:** Add a plotting cell if missing; save as `ablation_bars.pdf`

### 4. Deep SHAP Global Importance (Explainability section)
- **Source:** `05_explainability.ipynb`—SHAP summary/bar plot
- **Content:** Mean |SHAP| per channel
- **Action:** Save as `shap_importance.pdf`

### 5. Pipeline Diagram (Methods section, optional)
- **Source:** Create in PowerPoint, Draw.io, or similar
- **Content:** Data (.svc) → Preprocessing (derivatives, MixedFeatureScaler) → Models (CNN1D, TCN, CNN-LSTM) → Explainability (ablation, Deep SHAP)
- **Action:** Export as `pipeline.pdf`

### 6. Histogram of Total Writing Duration (Discussion, optional)
- **Source:** Add cell in `00_initialization.ipynb` to compute total duration from timestamps, split by dysgraphic vs. non-dysgraphic
- **Content:** Overlapping histograms to check if dysgraphic children write slower
- **Action:** Save as `duration_histogram.pdf`

## LaTeX snippet to add (adjust paths as needed)

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.7\linewidth]{figures/confusion_matrix.pdf}
  \caption{Confusion matrix for 1D CNN (representative run).}
  \label{fig:confusion}
\end{figure}
```

Repeat for other figures with appropriate captions and labels.
