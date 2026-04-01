"""
Shared initialization module for DysXAI project.
Import this when running model notebooks without having run 00_initialization.ipynb first.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# --- Config ---
class Config:
    """Global configuration for the dysgraphia detection project."""
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(_PROJECT_ROOT, "dataSciRep_public")
    META_XLSX = os.path.join(_PROJECT_ROOT, "data2_SciRep_pub.xlsx")
    MAX_LEN = 2000
    USE_DERIVATIVES = True
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_SUBJECT_RATIO = 0.7
    VAL_SUBJECT_RATIO = 0.15
    TEST_SUBJECT_RATIO = 0.15
    RANDOM_STATE = 42
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_ENABLED = True


def load_metadata(meta_path: str) -> pd.DataFrame:
    """Load metadata from Excel file."""
    df = pd.read_excel(meta_path)
    df['subject_id'] = df['ID'].astype(int)
    df['label'] = (df['diag'].astype(str).str.upper() == 'DYSGR').astype(int)
    df['file_name'] = df['subject_id'].astype(str)
    return df[['file_name', 'subject_id', 'label']].copy()


def load_raw_timeseries(filepath: str) -> np.ndarray:
    """Load a single handwriting sample from a file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                row = [float(v) for v in parts]
                data.append(row)
            except ValueError:
                continue
    if len(data) == 0:
        raise ValueError(f"No numeric data found in file: {filepath}")
    lengths = [len(r) for r in data]
    uniq, counts = np.unique(lengths, return_counts=True)
    target_len = int(uniq[np.argmax(counts)])
    filtered = [r for r in data if len(r) == target_len]
    if len(filtered) == 0:
        raise ValueError(f"All rows in {filepath} had inconsistent lengths")
    arr = np.array(filtered, dtype=np.float32)
    EXPECTED_BASE_CHANNELS = 7
    current_channels = arr.shape[1]
    if current_channels > EXPECTED_BASE_CHANNELS:
        arr = arr[:, :EXPECTED_BASE_CHANNELS]
    elif current_channels < EXPECTED_BASE_CHANNELS:
        padding_needed = EXPECTED_BASE_CHANNELS - current_channels
        arr = np.pad(arr, ((0, 0), (0, padding_needed)), 'constant', constant_values=0)
    return arr


def discover_files_and_map_subjects(data_root: str, subject_meta: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Discover all .svc/.txt files and map to subject IDs."""
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    file_records = []
    subject_ids = set(subject_meta['subject_id'].tolist())
    if verbose:
        print(f"Looking for files in: {data_root}")
    for root, dirs, files in os.walk(data_root):
        for fname in files:
            if not fname.lower().endswith(('.svc', '.txt')):
                continue
            base = os.path.splitext(fname)[0]
            ints_in_name = [int(m.group()) for m in re.finditer(r'\d+', base)]
            if not ints_in_name:
                continue
            matched = False
            for sid in ints_in_name:
                if sid in subject_ids:
                    rel_path = os.path.relpath(os.path.join(root, fname), data_root)
                    file_records.append({'subject_id': sid, 'file_name': rel_path})
                    matched = True
                    break
            if not matched:
                for sid in subject_ids:
                    for num in ints_in_name:
                        sid_str = str(sid)
                        if sid_str in base or f"{sid:05d}" in base or f"{sid:04d}" in base or f"{sid:03d}" in base:
                            rel_path = os.path.relpath(os.path.join(root, fname), data_root)
                            file_records.append({'subject_id': sid, 'file_name': rel_path})
                            matched = True
                            break
                    if matched:
                        break
    file_df = pd.DataFrame(file_records)
    if file_df.empty:
        raise RuntimeError(f"No .svc/.txt files in {data_root} matched any subject IDs")
    valid_records = []
    for idx, row in file_df.iterrows():
        full_path = os.path.join(data_root, row['file_name'])
        try:
            ts = load_raw_timeseries(full_path)
            if ts.shape[1] >= 3:
                valid_records.append(row)
        except Exception:
            continue
    file_df = pd.DataFrame(valid_records)
    if file_df.empty:
        raise RuntimeError("No valid files found after validation")
    return file_df.merge(subject_meta, on='subject_id', how='inner')


def compute_derivatives(sample: np.ndarray, x_idx=0, y_idx=1, t_idx=2) -> np.ndarray:
    """Compute velocity, acceleration, jerk for x and y."""
    T, C = sample.shape
    if max(x_idx, y_idx, t_idx) >= C:
        raise ValueError(f"Not enough channels ({C})")
    if T < 2:
        return np.concatenate([sample, np.zeros((T, 6), dtype=np.float32)], axis=-1)
    x, y, t = sample[:, x_idx], sample[:, y_idx], sample[:, t_idx]
    dt = np.diff(t)
    dt[dt == 0] = 1e-6
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])
    ax = np.diff(vx) / dt
    ay = np.diff(vy) / dt
    ax = np.concatenate([[ax[0]], ax])
    ay = np.concatenate([[ay[0]], ay])
    jx = np.diff(ax) / dt
    jy = np.diff(ay) / dt
    jx = np.concatenate([[jx[0]], jx])
    jy = np.concatenate([[jy[0]], jy])
    return np.concatenate([sample, np.stack([vx, vy, ax, ay, jx, jy], axis=-1)], axis=-1)


def pad_truncate(ts: np.ndarray, max_len: int) -> Tuple[np.ndarray, int]:
    """Pad or truncate time series to fixed length."""
    T, C = ts.shape
    length = min(T, max_len)
    padded = np.zeros((max_len, C), dtype=np.float32)
    padded[:length] = ts[:length]
    return padded, length


class MixedFeatureScaler:
    """MinMax for X/Y, StandardScaler for rest."""
    def __init__(self, x_idx: int = 0, y_idx: int = 1):
        self.x_idx, self.y_idx = x_idx, y_idx
        self.x_min = self.x_max = self.y_min = self.y_max = None
        self.standard_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_list: List[np.ndarray]):
        concat = np.concatenate(X_list, axis=0)
        num_features = concat.shape[1]
        self.x_min, self.x_max = float(concat[:, self.x_idx].min()), float(concat[:, self.x_idx].max())
        self.y_min, self.y_max = float(concat[:, self.y_idx].min()), float(concat[:, self.y_idx].max())
        if self.x_max == self.x_min:
            self.x_max = self.x_min + 1e-6
        if self.y_max == self.y_min:
            self.y_max = self.y_min + 1e-6
        other_indices = [i for i in range(num_features) if i not in [self.x_idx, self.y_idx]]
        if other_indices:
            self.standard_scaler.fit(concat[:, other_indices])
        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform!")
        X_scaled = X.copy()
        X_scaled[:, self.x_idx] = (X[:, self.x_idx] - self.x_min) / (self.x_max - self.x_min)
        X_scaled[:, self.y_idx] = (X[:, self.y_idx] - self.y_min) / (self.y_max - self.y_min)
        other_indices = [i for i in range(X.shape[1]) if i not in [self.x_idx, self.y_idx]]
        if other_indices:
            X_scaled[:, other_indices] = self.standard_scaler.transform(X[:, other_indices])
        return X_scaled


def fit_scaler_on_train(X_list: List[np.ndarray]) -> MixedFeatureScaler:
    scaler = MixedFeatureScaler(x_idx=0, y_idx=1)
    scaler.fit(X_list)
    return scaler


def apply_scaler(X: np.ndarray, scaler) -> np.ndarray:
    if isinstance(scaler, MixedFeatureScaler):
        return scaler.transform(X)
    T, C = X.shape
    return scaler.transform(X.reshape(-1, C)).reshape(T, C)


class HandwritingDataset(Dataset):
    """PyTorch Dataset for handwriting time-series data."""
    def __init__(self, meta_df: pd.DataFrame, data_root: str, scaler, max_len: int, use_derivatives: bool = True):
        self.meta_df = meta_df.reset_index(drop=True)
        self.data_root = data_root
        self.max_len = max_len
        self.use_derivatives = use_derivatives
        self.scaler = scaler

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        file_name, label, subject_id = row['file_name'], int(row['label']), row['subject_id']
        candidates = [
            os.path.join(self.data_root, f"{file_name}.svc"),
            os.path.join(self.data_root, f"{file_name}.txt"),
            os.path.join(self.data_root, str(file_name)),
        ]
        filepath = next((cp for cp in candidates if os.path.exists(cp)), None)
        if filepath is None:
            raise FileNotFoundError(f"Could not find raw file for sample '{file_name}'")
        ts = load_raw_timeseries(filepath)
        if self.use_derivatives:
            ts = compute_derivatives(ts)
        ts = apply_scaler(ts, self.scaler)
        ts_padded, length = pad_truncate(ts, self.max_len)
        ts_padded = ts_padded.T
        return {
            "x": torch.tensor(ts_padded, dtype=torch.float32),
            "y": torch.tensor(label, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "subject_id": subject_id,
        }


def subject_independent_split(meta_df: pd.DataFrame, train_ratio: float = 0.7,
                              val_ratio: float = 0.15, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """3-way subject-independent split."""
    subjects = meta_df['subject_id'].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(subjects)
    n_train = int(len(subjects) * train_ratio)
    n_val = int(len(subjects) * val_ratio)
    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])
    test_subjects = set(subjects[n_train + n_val:])
    return (
        meta_df[meta_df['subject_id'].isin(train_subjects)].copy(),
        meta_df[meta_df['subject_id'].isin(val_subjects)].copy(),
        meta_df[meta_df['subject_id'].isin(test_subjects)].copy(),
    )


def run_init(verbose: bool = True) -> pd.DataFrame:
    """
    Run full initialization: load metadata, discover files, create meta_df.
    Returns meta_df. Also injects Config, meta_df, and all helper functions into caller's globals
    when used via: dysxai_init.run_init(globals())
    """
    subject_meta = load_metadata(Config.META_XLSX)[['subject_id', 'label']].copy()
    meta_df = discover_files_and_map_subjects(Config.DATA_ROOT, subject_meta, verbose=verbose)
    if verbose:
        print(f"✓ Loaded {len(meta_df)} samples from {meta_df['subject_id'].nunique()} subjects")
    return meta_df
