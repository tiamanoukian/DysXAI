# # `analyze_velocity_age.py`
#
# Converted from Python for notebook workflow.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
from tqdm import tqdm

from dysxai_init import load_metadata, Config, load_raw_timeseries

# Use the exact same directory targeting you've been using
TASK_DIR = os.path.join(Config._PROJECT_ROOT, "dysxai_tasks_split", "task_7_hrackarstvo")


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


def process_file_absolute_speed(filepath):
    """
    Loads raw SVC, calculates the absolute speed (magnitude) for X and Y 
    using timestamps, and returns a dataframe of raw data points.
    """
    try:
        # Same loader as dysxai_init: skips comments / header junk, keeps 7 base channels
        data = load_raw_timeseries(filepath)
        x = data[:, 0]
        y = data[:, 1]
        time = data[:, 2]
        pressure = data[:, 3]
        pen_status = data[:, 6]

        # Ensure time is strictly increasing to avoid division by zero
        time_diff = np.diff(time)
        time_diff[time_diff <= 0] = 1 # Prevent zero division, set tiny timestep
        
        # Calculate discrete derivatives (dx/dt)
        vx_raw = np.diff(x) / time_diff
        vy_raw = np.diff(y) / time_diff

        # We must align lengths (diff reduces length by 1)
        pen_status = pen_status[1:]
        pressure = pressure[1:]

        # 1. Take ABSOLUTE VALUE to get raw SPEED magnitude (no negative "reverse" velocities)
        speed_x = np.abs(vx_raw)
        speed_y = np.abs(vy_raw)

        # 2. Keep points where pen is on paper (same rule as visualize_colored_trajectory)
        pen_up = (pen_status == 0.0) & (pressure <= 0.0)
        valid_indices = ~pen_up
        speed_x = speed_x[valid_indices]
        speed_y = speed_y[valid_indices]

        # 3. Handle the 1e8 scaling issue (convert to a manageable unit if necessary)
        # Assuming timestamps might be in milliseconds, converting to seconds
        # You may need to adjust this scaling factor based on your specific dataset's raw units
        # For now, we will leave it raw but absolute. 

        return speed_x, speed_y

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

def main():
    print("Loading Metadata...")
    meta_df = load_metadata(Config.META_XLSX)
    
    # Dictionary for fast lookup: subject_id -> (age, group)
    # load_metadata() exposes subject_id, label (0=control, 1=dysgraphic), age (lowercase)
    subject_info = {}
    for _, row in meta_df.iterrows():
        sid = str(int(row["subject_id"]))
        age = row["age"]
        if pd.isna(age):
            continue
        group = "Dysgraphic" if int(row["label"]) == 1 else "Control"
        subject_info[sid] = (age, group)

    print("Extracting Raw Kinematics...")
    svc_files = glob.glob(os.path.join(TASK_DIR, "*.svc"))
    
    all_data = []

    for file in tqdm(svc_files):
        subject_id = subject_id_from_svc_filename(file)
        if subject_id is None or subject_id not in subject_info:
            continue
            
        age, group = subject_info[subject_id]
        
        speed_x, speed_y = process_file_absolute_speed(file)
        
        if speed_x is not None:
            # Subsample 5% of the data points to prevent memory crashes and overly dense plots
            # Since a single word can have thousands of data points
            num_points = len(speed_x)
            if num_points == 0:
                continue
            sample_size = max(1, int(num_points * 0.05))
            indices = np.random.choice(num_points, sample_size, replace=False)
            
            sampled_x = speed_x[indices]
            sampled_y = speed_y[indices]
            
            for sx, sy in zip(sampled_x, sampled_y):
                all_data.append({
                    'Age': age,
                    'Group': group,
                    'Absolute_Speed_X': sx,
                    'Absolute_Speed_Y': sy
                })

    df = pd.DataFrame(all_data)
    if df.empty:
        print("No data collected (check TASK_DIR, metadata overlap, and .svc files). Exiting.")
        return

    # Optional: Log scale transformation if outliers are still too extreme
    # df['Absolute_Speed_X'] = np.log1p(df['Absolute_Speed_X'])
    # df['Absolute_Speed_Y'] = np.log1p(df['Absolute_Speed_Y'])

    print(f"Total raw data points collected (subsampled): {len(df)}")

    print("Generating Boxplots...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Absolute Speed X
    sns.boxplot(
        data=df, 
        x="Age", 
        y="Absolute_Speed_X", 
        hue="Group", 
        ax=axes[0], 
        palette={"Control": "tab:blue", "Dysgraphic": "tab:red"},
        showfliers=False # Hides extreme outlier dots for cleaner view
    )
    axes[0].set_title("Distribution of Raw Absolute Speed X by Age")
    axes[0].set_ylabel("Raw Speed Magnitude X")

    # Plot 2: Absolute Speed Y
    sns.boxplot(
        data=df, 
        x="Age", 
        y="Absolute_Speed_Y", 
        hue="Group", 
        ax=axes[1], 
        palette={"Control": "tab:blue", "Dysgraphic": "tab:red"},
        showfliers=False
    )
    axes[1].set_title("Distribution of Raw Absolute Speed Y by Age")
    axes[1].set_ylabel("Raw Speed Magnitude Y")

    plt.tight_layout()
    output_path = "raw_speed_distribution_task5.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved cleanly formatted visualization to {output_path}")

if __name__ == "__main__":
    main()
