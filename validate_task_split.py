"""
Validate dysxai_tasks_split: eight tasks per subject (one line each).

Run from repo root after dysxai_task_splitter.py::

    python validate_task_split.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dysxai_task_splitter import NUM_TASKS, TASKS, pen_on_mask  # noqa: E402

try:
    from dysxai_init import load_raw_timeseries
except ImportError:
    load_raw_timeseries = None

SPLIT_ROOT = _REPO / "dysxai_tasks_split"


def _subject_id_from_name(name: str) -> int | None:
    m = re.match(r"^u0*(\d+)_", name)
    return int(m.group(1)) if m else None


def _load_svc(path: Path) -> np.ndarray:
    if load_raw_timeseries is not None:
        return load_raw_timeseries(str(path))
    return np.atleast_2d(np.loadtxt(path, skiprows=1))


def validate_split(split_root: Path = SPLIT_ROOT) -> int:
    if not split_root.is_dir():
        print(f"Missing split folder: {split_root}")
        return 1

    rows: list[dict[str, object]] = []
    for task_dir in TASKS:
        folder = split_root / task_dir
        if not folder.is_dir():
            print(f"Missing task folder: {folder}")
            return 1
        for fp in sorted(folder.glob("*.svc")):
            sid = _subject_id_from_name(fp.name)
            if sid is None:
                continue
            data = _load_svc(fp)
            rows.append(
                {
                    "subject_id": sid,
                    "task": task_dir,
                    "pen_on": int(pen_on_mask(data).sum()),
                }
            )

    if not rows:
        print("No .svc files found under dysxai_tasks_split.")
        return 1

    by_subject: dict[int, dict[str, int]] = {}
    for r in rows:
        by_subject.setdefault(int(r["subject_id"]), {})[str(r["task"])] = int(r["pen_on"])

    n_subjects = len(by_subject)
    n_complete = sum(1 for tmap in by_subject.values() if len(tmap) == NUM_TASKS)
    issues: list[str] = []

    print(f"Subjects with split files: {n_subjects}")
    print(f"Subjects with all {NUM_TASKS} tasks: {n_complete}/{n_subjects}")
    print()
    print(f"{'Task':<28} {'pen_on median':>14} {'pen_on min':>12} {'pen_on max':>12}")
    print("-" * 70)

    for task in TASKS:
        vals = [tmap[task] for tmap in by_subject.values() if task in tmap]
        if not vals:
            issues.append(f"No files for {task}")
            continue
        print(
            f"{task:<28} {np.median(vals):>14.0f} {min(vals):>12} {max(vals):>12}"
        )

    print()
    t7_ok = 0
    for sid, tmap in sorted(by_subject.items()):
        if len(tmap) != NUM_TASKS:
            issues.append(f"subject {sid}: only {len(tmap)}/{NUM_TASKS} tasks")
            continue
        counts = [tmap[task] for task in TASKS]
        t5, t6, t7 = counts[4], counts[5], counts[6]
        # leto < lamoken < hračkárstvo is typical (longer words → more ink)
        if t7 > t5 and t7 > t6:
            t7_ok += 1
        else:
            issues.append(
                f"subject {sid}: task 7 shorter than leto/lamoken (t5,t6,t7 = {t5},{t6},{t7})"
            )

    print(f"Subjects with t7 > t5 and t7 > t6 (pen-on): {t7_ok}/{n_complete}")
    print()

    missing = [m for m in issues if "only" in m or "No files" in m]
    order_warn = [m for m in issues if m not in missing]

    if missing:
        print("Errors:")
        for msg in missing:
            print(f"  - {msg}")
        return 1

    if order_warn:
        print("Ordering warnings:")
        for msg in order_warn[:20]:
            print(f"  - {msg}")
        if len(order_warn) > 20:
            print(f"  ... and {len(order_warn) - 20} more")

    print("Split structure OK (all subjects have eight task files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate_split())
