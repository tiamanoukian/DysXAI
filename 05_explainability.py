# # Explainability and Feature Importance Analysis
#
# ## Overview
#
# This notebook performs **explainability analysis** on trained models to understand:
# 1. **Feature Importance**: Which features (channels) are most critical for predictions
# 2. **Feature Ablation Study**: Performance degradation when features are removed
# 3. **TSInterpret Integration**: Advanced explainability methods (if available)
#
# ### What This Notebook Does:
# - **Feature Ablation**: Systematically remove each feature and measure performance drop
# - **Feature Importance Ranking**: Identify which handwriting features matter most
# - **TSInterpret**: Use advanced XAI methods for time-series explanations (optional)
#
# ### Prerequisites:
# 1. Run `00_initialization.ipynb` first
# 2. Train at least one model (from `01`, `02`, or `03` notebooks)
# 3. Have a trained model available for analysis

# ## 1. Setup and Imports
#
# Import necessary libraries and prepare for explainability analysis.

# Import necessary modules (assumes 00_initialization.ipynb was run)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Metrics function
def compute_metrics(y_true, y_pred_probs):
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = np.nan
    return {"accuracy": acc, "sensitivity": sensitivity, "specificity": specificity, "auc": auc, "confusion_matrix": cm}

print("✓ Imports complete")

# Load initialization (run 00_initialization.ipynb first, or use dysxai_init)
import os, sys

for _d in [os.getcwd(),
           os.path.dirname(os.path.abspath("05_explainability.ipynb")),
           r"C:\Users\tiama\OneDrive\Desktop\period 2 courses\Reserach Project\DysXAI"]:
    if _d and os.path.exists(os.path.join(_d, "dysxai_init.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break

if 'meta_df' not in globals():
    import dysxai_init
    for name in ['Config', 'load_raw_timeseries', 'load_and_process_timeseries', 'model_channel_names', 'discover_files_and_map_subjects',
                 'compute_derivatives', 'pad_truncate', 'MixedFeatureScaler',
                 'fit_scaler_on_train', 'fit_scaler_from_train_meta', 'get_model_input_channel_count', 'apply_scaler', 'HandwritingDataset',
                 'subject_independent_split']:
        globals()[name] = getattr(dysxai_init, name)
    meta_df = dysxai_init.run_init(verbose=True)
    globals()['meta_df'] = meta_df
    print("Loaded initialization from dysxai_init.py")
else:
    print("meta_df and helpers already available.")

# ## 2. Feature Ablation Study
#
# This section performs a **feature ablation study** to determine which features are most important for dysgraphia detection.
#
# **Method**: For each feature channel, we zero it out and measure the performance degradation. Features with larger performance drops are more important.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Prepare data for ablation study
print("=" * 60)
print("FEATURE ABLATION STUDY")
print("=" * 60)

# Split data
train_meta_abl, _, test_meta_abl = subject_independent_split(
    meta_df,
    train_ratio=Config.TRAIN_SUBJECT_RATIO,
    val_ratio=Config.VAL_SUBJECT_RATIO,
    random_state=Config.RANDOM_STATE,
)

# Fit scaler
scaler_abl = fit_scaler_from_train_meta(train_meta_abl, Config.DATA_ROOT)
in_channels_abl = get_model_input_channel_count()

print(f"Input channels: {in_channels_abl}")

# Create datasets
train_dataset_abl = HandwritingDataset(
    train_meta_abl, Config.DATA_ROOT, scaler_abl, Config.MAX_LEN, Config.USE_DERIVATIVES
)
test_dataset_abl = HandwritingDataset(
    test_meta_abl, Config.DATA_ROOT, scaler_abl, Config.MAX_LEN, Config.USE_DERIVATIVES
)

train_loader_abl = DataLoader(train_dataset_abl, batch_size=Config.BATCH_SIZE, shuffle=True)
test_loader_abl = DataLoader(test_dataset_abl, batch_size=Config.BATCH_SIZE, shuffle=False)

# Class weights
y_train_abl = train_meta_abl['label'].values
class_weights_abl = compute_class_weight(
    class_weight='balanced', classes=np.array([0, 1]), y=y_train_abl
)

print("✓ Data prepared for ablation study")

# ### 2.1 Train Model for Ablation
#
# Train a model to use for the ablation study. You can also use a pre-trained model from previous notebooks.

# Import model class (use CNN-LSTM as example, or import from other notebooks)
# You can also use a pre-trained model from 01, 02, or 03 notebooks

# For this example, we'll train a CNN-LSTM model
# (You can replace this with loading a pre-trained model)
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Define CNN-LSTM model (or import from 03_model_cnnlstm.ipynb)
class CNNLSTMModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, lstm_hidden_size: int = 64, lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)
    def forward(self, x, lengths=None):
        features = self.cnn_extractor(x)
        features_seq = features.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(features_seq)
        hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        out = self.dropout(hidden)
        logits = self.classifier(out)
        return logits, features

# Train model (or load pre-trained)
print("\nTraining model for ablation study...")
ablation_model = CNNLSTMModel(in_channels=in_channels_abl, num_classes=2, dropout=0.3)
ablation_model.to(Config.DEVICE)

weight_tensor = torch.tensor(class_weights_abl, dtype=torch.float32, device=Config.DEVICE)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = Adam(ablation_model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

# Quick training (fewer epochs for ablation)
num_epochs_abl = 20
for epoch in range(num_epochs_abl):
    ablation_model.train()
    for batch in train_loader_abl:
        x, y, lengths = batch["x"].to(Config.DEVICE), batch["y"].to(Config.DEVICE), batch["length"].to(Config.DEVICE)
        optimizer.zero_grad()
        logits, _ = ablation_model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

print("✓ Model trained for ablation study")


# Baseline performance (all features)
ablation_model.eval()
all_probs_baseline, all_labels_baseline = [], []
with torch.no_grad():
    for batch in test_loader_abl:
        x, y, lengths = batch['x'].to(Config.DEVICE), batch['y'].to(Config.DEVICE), batch['length'].to(Config.DEVICE)
        logits, _ = ablation_model(x, lengths)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs_baseline.append(probs.cpu().numpy())
        all_labels_baseline.append(y.cpu().numpy())

all_probs_baseline = np.concatenate(all_probs_baseline)
all_labels_baseline = np.concatenate(all_labels_baseline)
baseline_metrics = compute_metrics(all_labels_baseline, all_probs_baseline)
baseline_auc = baseline_metrics['auc'] if not np.isnan(baseline_metrics['auc']) else baseline_metrics['accuracy']
baseline_acc = baseline_metrics['accuracy']

print(f"\nBaseline Performance (All Features):")
print(f"  Accuracy: {baseline_acc:.4f}")
print(f"  AUC: {baseline_auc:.4f}")

# ### 2.3 Perform Ablation
#
# Remove each feature channel one at a time and measure performance drop.

feature_names = model_channel_names()
assert len(feature_names) == in_channels_abl

# Ablation: Remove each feature channel
ablation_results = []

print("\nPerforming feature ablation...")
for channel_idx in range(in_channels_abl):
    feature_name = feature_names[channel_idx] if channel_idx < len(feature_names) else f'Channel_{channel_idx}'
    print(f"Ablating channel {channel_idx} ({feature_name})...", end=" ")
    
    # Create modified test dataset with this channel zeroed out
    all_probs_abl, all_labels_abl = [], []
    
    with torch.no_grad():
        for batch in test_loader_abl:
            x = batch['x'].to(Config.DEVICE).clone()
            y = batch['y'].to(Config.DEVICE)
            lengths = batch['length'].to(Config.DEVICE)
            
            # Zero out the specific channel
            x[:, channel_idx, :] = 0.0
            
            logits, _ = ablation_model(x, lengths)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs_abl.append(probs.cpu().numpy())
            all_labels_abl.append(y.cpu().numpy())
    
    all_probs_abl = np.concatenate(all_probs_abl)
    all_labels_abl = np.concatenate(all_labels_abl)
    metrics_abl = compute_metrics(all_labels_abl, all_probs_abl)
    
    auc_abl = metrics_abl['auc'] if not np.isnan(metrics_abl['auc']) else metrics_abl['accuracy']
    acc_abl = metrics_abl['accuracy']
    
    # Calculate performance drop
    auc_drop = baseline_auc - auc_abl
    acc_drop = baseline_acc - acc_abl
    
    ablation_results.append({
        'channel': channel_idx,
        'feature_name': feature_name,
        'baseline_auc': baseline_auc,
        'ablated_auc': auc_abl,
        'auc_drop': auc_drop,
        'baseline_acc': baseline_acc,
        'ablated_acc': acc_abl,
        'acc_drop': acc_drop,
    })
    
    print(f"AUC: {auc_abl:.4f} (drop: {auc_drop:.4f})")

# Sort by importance (largest drop = most important)
ablation_results.sort(key=lambda x: x['auc_drop'], reverse=True)

print("\n✓ Feature ablation complete")

# ### 2.4 Visualize Feature Importance
#
# Plot the results of the ablation study.

# Plot feature importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (based on AUC drop)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(12, 6))
features = [r['feature_name'] for r in ablation_results]
auc_drops = [r['auc_drop'] for r in ablation_results]

bars = ax.barh(features, auc_drops, color='steelblue')
ax.set_xlabel('Performance Drop (AUC)', fontsize=12)
ax.set_ylabel('Feature Channel', fontsize=12)
ax.set_title('Feature Importance: Performance Drop When Feature is Removed', fontsize=14)
ax.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar, drop) in enumerate(zip(bars, auc_drops)):
    ax.text(drop + 0.001, i, f'{drop:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()

# Print summary
print("\nFeature Importance Ranking (Most Important First):")
for i, result in enumerate(ablation_results[:10], 1):  # Top 10
    print(f"{i:2d}. {result['feature_name']:20s} - AUC Drop: {result['auc_drop']:.4f}, Acc Drop: {result['acc_drop']:.4f}")

# ## 3. Deep SHAP Explainability (Global Feature Importance)
#
# This section computes **Deep SHAP feature attributions** for the best model selected in our evaluation.
#
# - Uses the **CNN–LSTM ablation model** trained above as the current “best” model  
#   (replace `ablation_model` with your own loaded best model if needed).
# - Uses **all 13 channels** as inputs (including per-timestep **Age**; **Time** dropped at load if `Config.DROP_TIME_CHANNEL`).
# - Aggregates SHAP values across time and samples to get a **global feature importance ranking**.

# 3. Deep SHAP Explainability (using ablation_model as best model)

# Ensure shap is installed
try:
    import shap
except ImportError:
    print("⚠ shap not found. Installing...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "--quiet"])
    import shap
    print("✓ shap installed successfully!")

# Use the CNN-LSTM ablation model as the current 'best' model
# If you have a separately saved best model, load it here instead of ablation_model.
model_for_shap = ablation_model.to(Config.DEVICE)
model_for_shap.eval()

print("\nPreparing background and evaluation samples for Deep SHAP...")

def collect_batch(loader, max_samples):
    xs, ys = [], []
    for batch in loader:
        xs.append(batch["x"])
        ys.append(batch["y"])
        if len(torch.cat(xs, dim=0)) >= max_samples:
            break
    X = torch.cat(xs, dim=0)[:max_samples]
    y = torch.cat(ys, dim=0)[:max_samples]
    return X.to(Config.DEVICE), y.to(Config.DEVICE)

# Background: a subset of training data
background_X, _ = collect_batch(train_loader_abl, max_samples=64)

# Evaluation: a subset of test data
eval_X, eval_y = collect_batch(test_loader_abl, max_samples=128)

print(f"Background shape: {background_X.shape} (B, C, T)")
print(f"Eval shape:       {eval_X.shape} (B, C, T)")

# Wrap model so DeepExplainer sees probabilities for both classes
def model_forward_for_shap(x):
    logits, _ = model_for_shap(x)
    # Return probabilities; DeepExplainer will handle class-wise SHAP values
    return torch.softmax(logits, dim=1)

print("\nBuilding DeepExplainer (this may take a moment)...")
explainer = shap.DeepExplainer(model_forward_for_shap, background_X)
shap_values = explainer.shap_values(eval_X)  # list: one array per class

# For binary classification, shap_values[1] corresponds to the positive (dysgraphic) class
if isinstance(shap_values, list) and len(shap_values) == 2:
    sv_class1 = shap_values[1]  # shape: (n_samples, C, T)
else:
    # Fallback if shap returns a single array
    sv_class1 = shap_values

sv_abs = np.abs(sv_class1)  # (N, C, T)

# Global importance: mean |SHAP| across samples and time for each channel
global_importance = sv_abs.mean(axis=(0, 2))  # shape: (C,)

feature_names_shap = model_channel_names()
assert len(feature_names_shap) == sv_abs.shape[1]

# Pair names with importance and rank
shap_ranking = sorted(
    zip(feature_names_shap, global_importance),
    key=lambda x: x[1],
    reverse=True
)

print("\n" + "=" * 60)
print("DEEP SHAP GLOBAL FEATURE IMPORTANCE (mean |SHAP| per channel)")
print("=" * 60)
for rank, (name, val) in enumerate(shap_ranking, start=1):
    print(f"{rank:2d}. {name:15s}  mean |SHAP| = {val:.6f}")

# Optional: bar plot of SHAP-based importance
plt.figure(figsize=(10, 5))
names, vals = zip(*shap_ranking)
plt.bar(names, vals, color="cornflowerblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean |SHAP value|")
plt.title("Deep SHAP Global Feature Importance (Best Model, All 13 Channels)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ## 4. TSInterpret Integration (Optional)
#
# This section integrates TSInterpret for advanced explainability methods.
#
# **Note**: TSInterpret installation is optional. If not available, this section will be skipped.

# Try to import TSInterpret
TSINTERPRET_AVAILABLE = False
try:
    import TSInterpret
    TSINTERPRET_AVAILABLE = True
    print("✓ TSInterpret is available")
    print("You can use advanced explainability methods from TSInterpret")
except ImportError:
    print("⚠ TSInterpret not available")
    print("To install: pip install TSInterpret")
    print("This section will be skipped")

if TSINTERPRET_AVAILABLE:
    print("\nTSInterpret methods available:")
    print("- Feature attribution methods")
    print("- Counterfactual explanations")
    print("- Saliency maps")
    print("\nSee TSInterpret documentation for usage examples")
else:
    print("\nSkipping TSInterpret integration")
    print("Feature ablation study above provides feature importance analysis")

# ---
#
# ## ✅ Explainability Analysis Complete!
#
# ### Summary
#
# This notebook has performed:
# 1. **Feature Ablation Study**: Identified which features are most important
# 2. **Feature Importance Ranking**: Ranked features by their impact on model performance
# 3. **Deep SHAP**: Global channel attributions for the CNN–LSTM
# 4. **TSInterpret Integration**: (Optional) Advanced explainability methods
#
# ### Key Findings
#
# The feature ablation study shows which handwriting features (velocity, pressure, jerk, etc.) are most critical for dysgraphia detection. Features with larger performance drops when removed are more important.
#
# ### Next Steps
#
# - Use these insights to understand what the model learns
# - Focus on important features for clinical interpretation
# - Compare feature importance across different models (CNN, TCN, CNN-LSTM)
