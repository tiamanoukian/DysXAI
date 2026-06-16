# # 08 — True writing duration analysis
#
# Wraps **`analyze_true_duration.py`** (source below): per-subject total duration from raw timestamps, group tests, KDE overlap, age scatter.
#
# **Requires:** valid `Config.DATA_ROOT` and `Config.META_XLSX` in `dysxai_init`.

# ## `analyze_true_duration.py` — full file (reference)

import os
import sys
from pathlib import Path

from IPython.display import Code, display

root = os.path.abspath(os.getcwd())
if root not in sys.path:
    sys.path.insert(0, root)

display(Code(filename=str(Path(root) / "analyze_true_duration.py"), language="python"))

# ## Run analysis (writes CSV + PNGs to project root)

import os
import sys

root = os.path.abspath(os.getcwd())
if root not in sys.path:
    sys.path.insert(0, root)

from analyze_true_duration import main

main()
