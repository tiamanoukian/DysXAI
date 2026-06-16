# # Padding / Time-channel leakage ablations
#
# This notebook runs the same experiments as `dysxai_leakage_ablation.py`:
#
# 1. **Time-only model** (channel 2) with **zero** vs **replicate** tail padding (`HandwritingDataset(..., pad_mode=...)`).
# 2. **All channels except Time** (13 inputs: base + derivatives + **Age**) with default zero padding — kinematics without the clock.
#
# `replicate` fills the padded tail with the **last valid timestep** per channel instead of zeros, which weakens a simple "long run of zeros = length" cue.
#
# **Using in feature selection (`04_feature_selection.ipynb`):** rebuild datasets with `pad_mode='replicate'`:
#
# ```python
# train_dataset = HandwritingDataset(
#     train_meta, Config.DATA_ROOT, scaler, Config.MAX_LEN,
#     use_derivatives=Config.USE_DERIVATIVES, pad_mode='replicate',
# )
# ```
#
# **Drop Time without changing indices elsewhere:** use `get_model_input_channel_count()` (typically **13**) and `feature_indices=[i for i in range(in_channels) if i != 2]` when Time is present in the tensor.
#
# **Stronger evidence (implemented in repo):**
# - `dysxai_leakage_ablation.py` — by default reports **Val + Test** AUC on the same 70/15/15 subject split as `04_feature_selection`.
# - `dysxai_cv_evaluation.py` — **subject-stratified K-fold** with a held-out **test** fold per iteration; prints mean ± std **test** AUC. Example: `python dysxai_cv_evaluation.py --k 5 --epochs 50`.
#
# **Variable-length (batch padding):** `HandwritingDataset(..., padding="batch")` + `make_collate_fn(pad_mode)` (see `dysxai_init.py`). Convs still see padded tails within each batch; true masked convolutions would require a deeper architecture change (e.g. masked depthwise conv or attention); masked GAP at the end already uses `length`.

import os, sys
root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)

from dysxai_leakage_ablation import run_ablation

# Full runs: run_ablation()  # includes held-out test AUC (same 70/15/15 split as notebooks)
# Faster:      run_ablation(quick=True)
# Custom:      run_ablation(num_epochs=30, padding="fixed")  # padding="batch" for variable-length batch padding
# Val only:    run_ablation(include_test=False)

# K-fold (subject-stratified, reports mean test AUC): run in terminal:
#   python dysxai_cv_evaluation.py --k 5 --epochs 50 --scenarios time_zero,time_rep,no_time
run_ablation(quick=True)
