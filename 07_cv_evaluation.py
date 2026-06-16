# # 07 — Subject-stratified K-fold evaluation
#
# Wraps **`dysxai_cv_evaluation.py`** (source below). Reports mean ± std **test** AUC across folds for scenarios such as `time_zero`, `time_rep`, `no_time`.
#
# **Requires:** run **`00_initialization.ipynb`** first (or ensure `dysxai_init` paths are valid).

# ## `dysxai_cv_evaluation.py` — full file (reference)

import os
import sys
from pathlib import Path

from IPython.display import Code, display

root = os.path.abspath(os.getcwd())
if root not in sys.path:
    sys.path.insert(0, root)

display(Code(filename=str(Path(root) / "dysxai_cv_evaluation.py"), language="python"))

# ## Run K-fold evaluation

import os
import sys

root = os.path.abspath(os.getcwd())
if root not in sys.path:
    sys.path.insert(0, root)

from dysxai_cv_evaluation import run_kfold

# Example (adjust flags as needed):
run_kfold(n_folds=5, quick=True, scenarios="no_time", output_csv="kfold_leakage_results.csv")
