# # `analyze_filtered_velocity_age.py`
#
# Converted from notebook workflow — Butterworth XY low-pass before speed derivatives.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import glob
import os
import re
from tqdm import tqdm

from dysxai_init import load_metadata, Config, load_raw_timeseries

# Use the exact same directory targeting as analyze_velocity_age.py
TASK_DIR = os.path.join(Config._PROJECT_ROOT, "dysxai_tasks_split", "task_7_hrackarstvo")

# Tablet sampling rate and Butterworth low-pass (before discrete derivatives)
FS = 133.0
CUTOFF_HZ = 12.0
NYQ = 0.5 * FS
NORMAL_CUTOFF = CUTOFF_HZ / NYQ
_BUTTER_B, _BUTTER_A = signal.butter(2, NORMAL_CUTOFF, btype="low", analog=False)


def subject_id_from_svc_filename(filename: str) -> str | None:
    """Map basename like u00006_task_7_hrackarstvo.svc or 50_task_5.svc to canonical subject id string."""
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^u0*(\d+)_", base)
    if m:
        return str(int(m.group(1)))
    head = base.split("_", 1)[0]
    if head.isdigit():
        return str(int(head))
    nums = [int(g) for g in re.findall(r"\d+", base)]
    return str(nums[0]) if nums else None


def _filtfilt_zero_phase_safe(b, a, coord: np.ndarray) -> np.ndarray:
    """Zero-phase Butterworth via filtfilt; on too-short segments return signal unchanged."""
    x = np.asarray(coord, dtype=np.float64)
    try:
        return signal.filtfilt(b, a, x)
    except ValueError:
        return x.copy()


def process_file_filtered_absolute_speed(filepath: str):
    """
    Loads raw SVC, low-pass filters X/Y (2nd-order Butterworth), then absolute speed magnitudes.
    """
    try:
        data = load_raw_timeseries(filepath)
        x_raw = np.asarray(data[:, 0], dtype=np.float64)
        y_raw = np.asarray(data[:, 1], dtype=np.float64)
        time = data[:, 2]
        pressure = data[:, 3]
        pen_status = data[:, 6]

        x = _filtfilt_zero_phase_safe(_BUTTER_B, _BUTTER_A, x_raw)
        y = _filtfilt_zero_phase_safe(_BUTTER_B, _BUTTER_A, y_raw)

        time_diff = np.diff(time)
        time_diff[time_diff <= 0] = 1

        vx_raw = np.diff(x) / time_diff
        vy_raw = np.diff(y) / time_diff

        pen_status = pen_status[1:]
        pressure = pressure[1:]

        speed_x = np.abs(vx_raw)
        speed_y = np.abs(vy_raw)

        pen_up = (pen_status == 0.0) & (pressure <= 0.0)
        valid_indices = ~pen_up
        speed_x = speed_x[valid_indices]
        speed_y = speed_y[valid_indices]

        return speed_x, speed_y

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None


def main():
    print("Loading Metadata...")
    meta_df = load_metadata(Config.META_XLSX)

    subject_info = {}
    for _, row in meta_df.iterrows():
        sid = str(int(row["subject_id"]))
        age = row["age"]
        if pd.isna(age):
            continue
        group = "Dysgraphic" if int(row["label"]) == 1 else "Control"
        subject_info[sid] = (age, group)

    print("Extracting Filtered Kinematics (Butterworth 12 Hz low-pass @ 133 Hz, before derivatives)...")
    svc_files = glob.glob(os.path.join(TASK_DIR, "*.svc"))

    all_data = []

    for file in tqdm(svc_files):
        subject_id = subject_id_from_svc_filename(file)
        if subject_id is None or subject_id not in subject_info:
            continue

        age, group = subject_info[subject_id]

        speed_x, speed_y = process_file_filtered_absolute_speed(file)

        if speed_x is not None:
            num_points = len(speed_x)
            if num_points == 0:
                continue
            sample_size = max(1, int(num_points * 0.05))
            indices = np.random.choice(num_points, sample_size, replace=False)

            sampled_x = speed_x[indices]
            sampled_y = speed_y[indices]

            for sx, sy in zip(sampled_x, sampled_y):
                all_data.append(
                    {
                        "Age": age,
                        "Group": group,
                        "Absolute_Speed_X": sx,
                        "Absolute_Speed_Y": sy,
                    }
                )

    df = pd.DataFrame(all_data)
    if df.empty:
        print("No data collected (check TASK_DIR, metadata overlap, and .svc files). Exiting.")
        return

    print(f"Total data points collected (subsampled 5%): {len(df)}")

    print("Generating Line Plots...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.lineplot(
        data=df,
        x="Age",
        y="Absolute_Speed_X",
        hue="Group",
        errorbar="sd",
        ax=axes[0],
        palette={"Control": "tab:blue", "Dysgraphic": "tab:red"},
    )
    axes[0].set_title("Filtered Absolute Speed X vs Age")
    axes[0].set_ylabel("Filtered speed magnitude X")

    sns.lineplot(
        data=df,
        x="Age",
        y="Absolute_Speed_Y",
        hue="Group",
        errorbar="sd",
        ax=axes[1],
        palette={"Control": "tab:blue", "Dysgraphic": "tab:red"},
    )
    axes[1].set_title("Filtered Absolute Speed Y vs Age")
    axes[1].set_ylabel("Filtered speed magnitude Y")

    plt.tight_layout()
    output_path = "filtered_speed_curve_task5.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved cleanly formatted visualization to {output_path}")


if __name__ == "__main__":
    main()
