# # dataset.py
#
# Notebook mirror of dataset.py (run this cell to load the module in Jupyter).
#
# **Module docstring**
#
# Task 7 (hračkárstvo) data: load raw .svc, Butterworth XY low-pass before derivatives,
# then velocity / acceleration / jerk; optional broadcast Age channel.
#
# Paths are anchored to this package folder; data lives at ../dysxai_tasks_split/task_7_hrackarstvo
# relative to DysXAI_task7/.

"""
Task 7 (hračkárstvo) data: load raw .svc, Butterworth XY low-pass before derivatives,
then velocity / acceleration / jerk; optional broadcast Age channel.

Paths are anchored to this package folder; data lives at ../dysxai_tasks_split/task_7_hrackarstvo
relative to DysXAI_task7/.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from scipy import signal

# Parent project root (DysXAI): reuse Excel + raw .svc line parser only.
try:
    _PKG_DIR = Path(__file__).resolve().parent
except NameError:  # Jupyter / REPL: __file__ is undefined
    _cwd = Path.cwd().resolve()
    if (_cwd / "dataset.py").is_file() or (_cwd / "dataset.ipynb").is_file():
        _PKG_DIR = _cwd
    else:
        _nested = _cwd / "DysXAI_task7"
        _PKG_DIR = (
            _nested.resolve()
            if _nested.is_dir()
            and (
                (_nested / "dataset.py").is_file()
                or (_nested / "dataset.ipynb").is_file()
            )
            else _cwd
        )


def _find_dysxai_project_root(pkg_dir: Path) -> Path:
    """Directory that contains dysxai_init.py (repo root)."""
    pkg_dir = pkg_dir.resolve()
    for base in (pkg_dir, Path.cwd().resolve()):
        for p in [base, *base.parents]:
            if (p / "dysxai_init.py").is_file():
                return p.resolve()
    return pkg_dir.parent.resolve()


_PROJECT_ROOT = _find_dysxai_project_root(_PKG_DIR)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dysxai_init import Config, load_metadata, load_raw_timeseries  # noqa: E402

# --- Paths (requirements: ../dysxai_tasks_split/task_7_hrackarstvo from this folder)
TASK7_DATA_REL = Path("..") / "dysxai_tasks_split" / "task_7_hrackarstvo"
TASK7_DATA_DIR = (_PKG_DIR / TASK7_DATA_REL).resolve()

# Signal processing (requirements)
FS_HZ = 133.0
CUTOFF_HZ = 12.0
BUTTER_ORDER = 2
TIME_CHANNEL_INDEX = 2
PRESSURE_CHANNEL_INDEX = 3
PEN_STATUS_CHANNEL_INDEX = 6

MAX_SEQ_LEN = 2000

TASK7_XY_FILTER_ENV = "DysXAI_task7_XY_FILTER"


def subject_id_from_svc_path(filepath: str | Path) -> str | None:
    base = Path(filepath).stem
    m = re.match(r"^u0*(\d+)_", base)
    if m:
        return str(int(m.group(1)))
    head = base.split("_", 1)[0]
    if head.isdigit():
        return str(int(head))
    nums = [int(g) for g in re.findall(r"\d+", base)]
    return str(nums[0]) if nums else None


def butterworth_lowpass_xy(ts: np.ndarray, cutoff_hz: float = CUTOFF_HZ) -> np.ndarray:
    """2nd-order Butterworth on X/Y, zero-phase filtfilt, before derivatives."""
    T = ts.shape[0]
    if T < 2 or ts.shape[1] < 3:
        return ts
    nyq = 0.5 * FS_HZ
    normal_cutoff = float(cutoff_hz) / nyq
    if not (0 < normal_cutoff < 1.0):
        return ts
    b, a = signal.butter(BUTTER_ORDER, normal_cutoff, btype="low", analog=False)
    x = np.ascontiguousarray(ts[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(ts[:, 1], dtype=np.float64)
    out = np.asarray(ts, dtype=np.float32, copy=True)
    try:
        out[:, 0] = signal.filtfilt(b, a, x).astype(np.float32)
        out[:, 1] = signal.filtfilt(b, a, y).astype(np.float32)
    except ValueError:
        pass
    return out


def pen_on_mask(ts: np.ndarray) -> np.ndarray:
    """True where the pen is on the page (length T). Matches project-wide pen-up rule."""
    if ts.ndim != 2 or ts.shape[1] <= PEN_STATUS_CHANNEL_INDEX:
        raise ValueError(f"Expected at least {PEN_STATUS_CHANNEL_INDEX + 1} channels, got {ts.shape}")
    pressure = np.asarray(ts[:, PRESSURE_CHANNEL_INDEX], dtype=np.float64)
    pen_status = np.asarray(ts[:, PEN_STATUS_CHANNEL_INDEX], dtype=np.float64)
    return ~((pen_status == 0.0) & (pressure <= 0.0))


def safe_dt(t: np.ndarray) -> np.ndarray:
    """
    Robust per-step dt for discrete derivatives.

    Replaces non-positive or extreme gaps with the clip median so velocities do not
    explode when tablet timestamps stall or repeat (common at pen lifts).
    """
    t = np.asarray(t, dtype=np.float64)
    if t.size < 2:
        return np.array([], dtype=np.float64)
    dt = np.diff(t)
    good = dt[np.isfinite(dt) & (dt > 0)]
    typical = float(np.median(good)) if good.size else 1.0
    bad = ~np.isfinite(dt) | (dt <= 0) | (dt < 0.25 * typical) | (dt > 4.0 * typical)
    return np.where(bad, typical, dt).astype(np.float64)


def velocities_xy(
    x: np.ndarray, y: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vx, Vy aligned to length T (forward-fill first sample after diff)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dt = safe_dt(t)
    if dt.size == 0:
        return np.zeros_like(x), np.zeros_like(y)
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])
    return vx, vy


def filter_speed_outliers(
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    percentile: float = 99.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop velocity samples whose magnitude exceeds a per-clip percentile cap."""
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    if vx.size == 0:
        return vx, vy
    mag = np.sqrt(vx * vx + vy * vy)
    cap = float(np.percentile(mag, percentile))
    if cap <= 0:
        return vx, vy
    keep = mag <= cap
    return vx[keep], vy[keep]


def compute_derivatives(sample: np.ndarray, x_idx: int = 0, y_idx: int = 1, t_idx: int = 2) -> np.ndarray:
    """Velocity, acceleration, jerk for x and y; same alignment as dysxai_init.compute_derivatives."""
    T, C = sample.shape
    if max(x_idx, y_idx, t_idx) >= C:
        raise ValueError(f"Not enough channels ({C})")
    if T < 2:
        return np.concatenate([sample, np.zeros((T, 6), dtype=np.float32)], axis=-1)
    x = sample[:, x_idx]
    y = sample[:, y_idx]
    t = sample[:, t_idx]
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
    stacked = np.stack([vx, vy, ax, ay, jx, jy], axis=-1).astype(np.float32)
    return np.concatenate([sample.astype(np.float32), stacked], axis=-1)


def drop_time_channel(ts: np.ndarray) -> np.ndarray:
    """Remove raw Time channel (index 2) before Conv1d; preserves kinematics + pen channels."""
    if ts.ndim != 2 or ts.shape[1] <= TIME_CHANNEL_INDEX:
        return ts
    return np.delete(ts, TIME_CHANNEL_INDEX, axis=1).astype(np.float32)


def broadcast_age_channel(ts: np.ndarray, age: Any, missing_value: float = -1.0) -> np.ndarray:
    """Append one column repeated over T."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        a = missing_value
    else:
        try:
            if bool(pd.isna(age)):
                a = missing_value
            else:
                a = float(age)
        except (ValueError, TypeError):
            a = missing_value
    if not np.isfinite(a):
        a = missing_value
    T = ts.shape[0]
    age_col = np.full((T, 1), a, dtype=np.float32)
    return np.concatenate([ts.astype(np.float32), age_col], axis=-1)


def _canonical_xy_filter(xy_filter: str | None) -> tuple[str, float]:
    """
    Resolve preprocessing mode and optional cutoff: ``butterworth[_hz]``,
    ``fft[_hz]``, or ``raw``.

    If ``xy_filter`` is None, read ``TASK7_XY_FILTER_ENV`` (default butterworth).
    Legacy env aliases ``none`` / ``off`` / ``no_filter`` / ``unfiltered`` map to ``raw``.
    Default cutoff is 12 Hz when not specified.
    """
    if xy_filter is not None:
        key = str(xy_filter).strip().lower() or "butterworth"
    else:
        key = (os.environ.get(TASK7_XY_FILTER_ENV) or "butterworth").strip().lower() or "butterworth"
    if key in ("none", "off", "no_filter", "unfiltered"):
        return "raw", CUTOFF_HZ
    if key == "raw":
        return "raw", CUTOFF_HZ
    if "_" not in key and key in ("butterworth", "fft"):
        return key, CUTOFF_HZ
    if "_" in key:
        base, freq_str = key.split("_", 1)
        if base in ("butterworth", "fft"):
            try:
                freq_hz = float(freq_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid xy_filter frequency in {key!r}; expected e.g. 'butterworth_12' or 'fft_10'."
                ) from exc
            if not np.isfinite(freq_hz) or freq_hz <= 0.0:
                raise ValueError(
                    f"Invalid xy_filter frequency {freq_hz!r}; frequency must be > 0."
                )
            return base, freq_hz
    raise ValueError(
        f"Invalid xy_filter {key!r}. Use 'butterworth', 'fft', 'raw', "
        f"'butterworth_<hz>' or 'fft_<hz>' "
        f"(or set env {TASK7_XY_FILTER_ENV} to one of those, or legacy none/off)."
    )


def load_processed_tensor(
    filepath: Path,
    age: Any,
    append_age_channel: bool,
    xy_filter: str | None = None,
) -> np.ndarray:
    """
    Raw .svc → XY pre-filter (``butterworth`` | ``fft`` | ``raw``) → derivatives → drop Time.

    When ``xy_filter`` is ``'raw'``, skip low-pass filtering and pass raw coordinates into
    ``compute_derivatives``. When ``xy_filter`` is None, the mode is taken from
    ``TASK7_XY_FILTER_ENV`` (see ``_canonical_xy_filter``).

    Optional ``append_age_channel`` adds a broadcast Age column (legacy); the Conv1d model
    uses 12 kinematic channels only and passes Age separately when late-fusion training.
    """
    raw = load_raw_timeseries(str(filepath))
    mode, cutoff_hz = _canonical_xy_filter(xy_filter)
    if mode == "fft":
        from dysxai_fft_xy_filter import lowpass_filter_xy_fft

        filt = lowpass_filter_xy_fft(np.asarray(raw, dtype=np.float32), cutoff_hz=cutoff_hz)
    elif mode == "raw":
        filt = np.asarray(raw, dtype=np.float32)
    else:
        filt = butterworth_lowpass_xy(raw, cutoff_hz=cutoff_hz)
    deriv = compute_derivatives(filt)
    no_time = drop_time_channel(deriv)
    if append_age_channel:
        return broadcast_age_channel(no_time, age)
    return no_time


def pad_truncate(ts: np.ndarray, max_len: int = MAX_SEQ_LEN) -> tuple[np.ndarray, int]:
    T, C = ts.shape
    length = min(T, max_len)
    padded = np.zeros((max_len, C), dtype=np.float32)
    padded[:length] = ts[:length]
    return padded, length


def build_sample_table() -> pd.DataFrame:
    """
    Rows: filepath, subject_id, label (0 Control, 1 Dysgraphic), age.
    """
    if not TASK7_DATA_DIR.is_dir():
        raise FileNotFoundError(f"Task 7 data directory not found: {TASK7_DATA_DIR}")

    meta = load_metadata(Config.META_XLSX)
    meta_raw = pd.read_excel(Config.META_XLSX)
    if "ID" not in meta_raw.columns:
        raise ValueError("Metadata file must contain 'ID' column for subject mapping.")
    meta_raw["subject_id"] = pd.to_numeric(meta_raw["ID"], errors="coerce").astype("Int64")
    gender_col = None
    for cand in ("gender", "Gender", "sex", "Sex"):
        if cand in meta_raw.columns:
            gender_col = cand
            break
    if gender_col is None:
        raise ValueError("Metadata must include a gender/sex column.")

    def _map_gender_to_float(value: Any) -> float:
        if value is None or bool(pd.isna(value)):
            raise ValueError("Missing gender value in metadata; expected Male/Female.")
        v = str(value).strip().lower()
        if v in ("m", "male", "man", "boy", "0"):
            return 0.0
        if v in ("f", "female", "woman", "girl", "1"):
            return 1.0
        raise ValueError(f"Unrecognized gender value {value!r}; expected Male/Female.")

    gender_df = meta_raw[["subject_id", gender_col]].copy()
    gender_df["gender"] = gender_df[gender_col].map(_map_gender_to_float)
    gender_map = (
        gender_df.dropna(subset=["subject_id"])
        .drop_duplicates(subset=["subject_id"])
        .set_index("subject_id")["gender"]
        .to_dict()
    )
    sid_set = set(int(x) for x in meta["subject_id"].tolist())

    rows: list[dict[str, Any]] = []
    for fp in sorted(glob.glob(str(TASK7_DATA_DIR / "*.svc"))):
        sid_str = subject_id_from_svc_path(fp)
        if sid_str is None:
            continue
        sid = int(sid_str)
        if sid not in sid_set:
            continue
        hit = meta[meta["subject_id"] == sid]
        if hit.empty:
            continue
        age_val = hit["age"].iloc[0]
        label = int(hit["label"].iloc[0])
        gender_val = gender_map.get(sid, float("nan"))
        rows.append(
            {
                "filepath": Path(fp),
                "subject_id": sid,
                "label": label,
                "age": age_val,
                "gender": gender_val,
            }
        )

    if not rows:
        raise RuntimeError(f"No matched Task 7 .svc samples under {TASK7_DATA_DIR}")
    out = pd.DataFrame(rows).drop_duplicates(subset=["subject_id", "filepath"]).reset_index(drop=True)
    out = out.sort_values(["subject_id", "filepath"]).reset_index(drop=True)
    return out


class Task5TrajectoryDataset(Dataset):
    """One handwriting clip per row in ``sample_rows`` filtered by indices."""

    def __init__(
        self,
        sample_rows: pd.DataFrame,
        indices: Sequence[int],
        use_age_channel: bool,
        channel_mean: np.ndarray | None = None,
        channel_std: np.ndarray | None = None,
        max_seq_len: int = MAX_SEQ_LEN,
        xy_filter: str | None = None,
    ):
        self.df = sample_rows.iloc[list(indices)].reset_index(drop=True)
        self.use_age_channel = bool(use_age_channel)
        self.max_seq_len = max_seq_len
        self.mu = channel_mean
        self.sigma = channel_std
        self.xy_filter = xy_filter

    def __len__(self) -> int:
        return len(self.df)

    def _standardize_channels(self, x: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sigma is None:
            return x
        return ((x.astype(np.float32) - self.mu) / self.sigma).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        path = row["filepath"]
        age = row["age"]
        y = int(row["label"])
        sid = int(row["subject_id"])

        feats = load_processed_tensor(path, age=age, append_age_channel=False, xy_filter=self.xy_filter)
        full, length = pad_truncate(feats, self.max_seq_len)
        full = self._standardize_channels(full)

        xt = torch.from_numpy(full.T).clone()
        out: dict[str, Any] = {
            "x": xt,
            "y": torch.tensor(y, dtype=torch.float32),
            "length": length,
            "subject_id": sid,
        }
        if self.use_age_channel:
            try:
                age_f = float(age) if age is not None and bool(pd.notna(age)) else float("nan")
            except (TypeError, ValueError):
                age_f = float("nan")
            if not np.isfinite(age_f):
                age_f = -1.0
            out["age"] = torch.tensor(age_f, dtype=torch.float32)
        gender = row.get("gender", float("nan"))
        try:
            gender_f = float(gender) if gender is not None and bool(pd.notna(gender)) else float("nan")
        except (TypeError, ValueError):
            gender_f = float("nan")
        if not np.isfinite(gender_f) or gender_f not in (0.0, 1.0):
            raise ValueError(f"Invalid gender value {gender!r}; expected encoded 0.0/1.0.")
        out["gender"] = torch.tensor(gender_f, dtype=torch.float32)
        return out


def fit_channel_scaling(
    sample_rows: pd.DataFrame,
    train_indices: Sequence[int],
    max_seq_len: int = MAX_SEQ_LEN,
    xy_filter: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std over valid timesteps in training folds (12 kinematics channels; age not z-scored)."""
    idxs = list(train_indices)
    chunks: list[np.ndarray] = []
    for i in idxs:
        row = sample_rows.iloc[i]
        raw = load_processed_tensor(
            Path(row["filepath"]), age=row["age"], append_age_channel=False, xy_filter=xy_filter
        )
        # raw 12-channel after drop Time
        p, ln = pad_truncate(raw, max_seq_len)
        chunks.append(p[:ln])
    concat = np.vstack(chunks).astype(np.float64)
    mu = concat.mean(axis=0).astype(np.float32)
    sd = concat.std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return mu, sd


def collate_task7(batch: list[dict[str, Any]], min_timesteps: int = 8) -> dict[str, torch.Tensor]:
    """Pad batch to common time length; x shape (N, C, T)."""
    lengths = [int(b["length"]) for b in batch]
    max_t = max(max(lengths), min_timesteps)
    c = batch[0]["x"].shape[0]
    out_x = torch.zeros(len(batch), c, max_t, dtype=torch.float32)
    for i, b in enumerate(batch):
        L = int(b["length"])
        seg = b["x"][:, :L]
        out_x[i, :, :L] = seg
    out: dict[str, torch.Tensor] = {
        "x": out_x,
        "y": torch.stack([b["y"] for b in batch]),
        "length": torch.tensor(lengths, dtype=torch.long),
        "subject_id": torch.tensor([b["subject_id"] for b in batch], dtype=torch.long),
    }
    if "age" in batch[0]:
        out["age"] = torch.stack([b["age"] for b in batch])
    if "gender" in batch[0]:
        out["gender"] = torch.stack([b["gender"] for b in batch])
    return out


def channel_counts(use_age_channel: bool) -> int:
    """Conv1d input channels (kinematics only); Age uses late fusion, not extra conv channels."""
    _ = use_age_channel
    return 12  # X, Y, Pressure, Azimuth, Altitude, Pen + 6 deriv (no Time)


# Aliases for Task 7 study package naming
Task7TrajectoryDataset = Task5TrajectoryDataset
