# Dysgraphia Detection Project - Organized Notebook Structure

## 📁 Project Structure

This project has been organized into separate, interconnected notebooks for better clarity and maintainability:

```
DysXAI/
├── 00_initialization.ipynb      # Setup, data loading, preprocessing
├── 01_model_cnn1d.ipynb         # 1D CNN model training
├── 02_model_tcn.ipynb           # TCN model training
├── 03_model_cnnlstm.ipynb       # CNN-LSTM hybrid model training
├── 04_explainability.ipynb      # Feature importance & XAI analysis
├── README.md                     # This file
├── data2_SciRep_pub.xlsx         # Metadata file
└── dataSciRep_public.zip         # Raw data (unzip before use)
```

## 🚀 Quick Start Guide

### Step 1: Run Initialization
**Always start here!**

1. Open `00_initialization.ipynb`
2. Update paths in the `Config` class (Cell 4):
   - `DATA_ROOT`: Path to your unzipped data folder
   - `META_XLSX`: Path to your Excel metadata file
3. Run all cells in order
4. This sets up:
   - All data loading functions
   - Preprocessing utilities
   - Dataset class
   - Data preparation

### Step 2: Train Models
Run the model notebooks **in any order** (they are independent):

- **`01_model_cnn1d.ipynb`**: Fast, good for local patterns
- **`02_model_tcn.ipynb`**: Captures long-term dependencies
- **`03_model_cnnlstm.ipynb`**: Hybrid approach (CNN + LSTM)

**Note**: Each model notebook is self-contained and can be run independently.

### Step 3: Explainability Analysis
Run `04_explainability.ipynb` after training at least one model to:
- Analyze feature importance
- Generate counterfactual explanations
- Understand model decisions

## 📚 Notebook Details

### 00_initialization.ipynb
**Purpose**: Setup and data preparation

**Key Components**:
- Configuration class (`Config`)
- Data loading functions (`load_metadata`, `load_raw_timeseries`)
- Feature engineering (`compute_derivatives`)
- Preprocessing (`pad_truncate`, `fit_scaler_on_train`, `apply_scaler`)
- Dataset class (`HandwritingDataset`)
- Data splitting (`subject_independent_split`)

**Output**: `meta_df` DataFrame with all data mappings

### 01_model_cnn1d.ipynb
**Purpose**: Train 1D CNN model

**Architecture**:
- 3 convolutional blocks (64 → 128 → 256 channels)
- Global average pooling (masked)
- Binary classification head

**Best For**: Capturing local temporal patterns, fast training

### 02_model_tcn.ipynb
**Purpose**: Train Temporal Convolutional Network

**Architecture**:
- Dilated causal convolutions
- Residual connections
- Multi-level temporal receptive fields

**Best For**: Long-term temporal dependencies

### 03_model_cnnlstm.ipynb
**Purpose**: Train CNN-LSTM hybrid model

**Architecture**:
- CNN block for local feature extraction
- Bidirectional LSTM for sequence modeling
- Classification head

**Best For**: Combining local and long-term patterns

### 04_explainability.ipynb
**Purpose**: Model interpretability and feature analysis

**Features**:
- Feature ablation study
- TSInterpret integration (if available)
- Counterfactual explanations
- Feature importance visualization

## 🔧 Configuration

All hyperparameters are defined in `Config` class in `00_initialization.ipynb`:

```python
class Config:
    # Data paths (UPDATE THESE!)
    DATA_ROOT = "path/to/data"
    META_XLSX = "path/to/metadata.xlsx"
    
    # Model settings
    MAX_LEN = 2000
    USE_DERIVATIVES = True
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LR = 1e-3
    
    # Cross-validation
    TRAIN_SUBJECT_RATIO = 0.8
    NUM_REPEATED_SPLITS = 5
    INNER_CV_FOLDS = 2
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_ENABLED = True
```

## 📊 Key Functions Reference

### Data Loading
- `load_metadata(meta_path)`: Load Excel metadata
- `load_raw_timeseries(filepath)`: Load single time-series file
- `discover_files_and_map_subjects(data_root, subject_meta)`: Map files to subjects

### Preprocessing
- `compute_derivatives(sample)`: Add velocity, acceleration, jerk
- `pad_truncate(ts, max_len)`: Standardize sequence length
- `fit_scaler_on_train(X_list)`: Fit scaler (training data only!)
- `apply_scaler(X, scaler)`: Normalize features

### Training
- `train_one_model(...)`: Generic training loop with early stopping
- `compute_metrics(y_true, y_pred_probs)`: Calculate accuracy, sensitivity, specificity, AUC

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
```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm seaborn
pip install TSInterpret  # Optional, for explainability
```
