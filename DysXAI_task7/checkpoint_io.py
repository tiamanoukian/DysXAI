"""
Save / load Task 7 Conv1d classifier checkpoints for evaluation and explainability.

Each checkpoint bundles model weights plus preprocessing metadata (channel scaling,
age normalization) required to rebuild datasets and run inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from model import Task7Conv1dClassifier

DEFAULT_CHECKPOINT_DIR_NAME = "checkpoints"

# Canonical preprocessing + checkpoint for Task 7 explainability (XAI) studies.
XAI_XY_FILTER = "fft"
XAI_USE_AGE = True
XAI_USE_GENDER = False
XAI_FILTER_LABEL = "12 Hz FFT"


def default_checkpoint_dir(pkg_dir: Path) -> Path:
    return pkg_dir / DEFAULT_CHECKPOINT_DIR_NAME


def checkpoint_search_roots(pkg_dir: Path, *extra_dirs: Path | None) -> list[Path]:
    """
    Ordered directories to search for ``.pt`` files (direct path, then recursive).

    Includes ``extra_dirs`` (e.g. ``--checkpoint-dir``), ``checkpoints/``, and the
    package root so nested layouts like ``checkpoints/ablation/holdout_*.pt`` work.
    """
    ordered: list[Path] = []
    seen: set[Path] = set()
    for raw in (*extra_dirs, default_checkpoint_dir(pkg_dir), pkg_dir):
        if raw is None:
            continue
        root = Path(raw)
        if not root.is_absolute():
            root = pkg_dir / root
        try:
            key = root.resolve()
        except OSError:
            key = root
        if key in seen:
            continue
        seen.add(key)
        ordered.append(root)
    return ordered


def find_checkpoint_file(basename: str, search_roots: Sequence[Path]) -> Path | None:
    """Find ``basename`` under any search root (top level or nested subfolders)."""
    name = Path(basename).name
    for root in search_roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file():
            return direct.resolve()
        nested = sorted(root.rglob(name))
        if nested:
            return nested[0].resolve()
    return None


def list_matching_checkpoints(
    search_roots: Sequence[Path],
    glob_pattern: str,
    *,
    limit: int = 20,
) -> list[Path]:
    """Collect unique checkpoint paths matching ``glob_pattern`` under search roots."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob(glob_pattern)):
            if not path.is_file():
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            found.append(key)
        for path in sorted(root.rglob(glob_pattern)):
            if not path.is_file():
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            found.append(key)
            if len(found) >= limit:
                return found
    return found


def xai_checkpoint_basename(*, use_age: bool = XAI_USE_AGE, use_gender: bool = XAI_USE_GENDER) -> str:
    """Stable filename for the explainability model (FFT filter)."""
    age_tag = "age" if use_age else "no_age"
    gender_tag = "gender" if use_gender else "no_gender"
    return f"xai_{XAI_XY_FILTER}_{age_tag}_{gender_tag}.pt"


def default_xai_checkpoint_path(
    pkg_dir: Path,
    *,
    use_age: bool = XAI_USE_AGE,
    use_gender: bool = XAI_USE_GENDER,
    checkpoint_dir: Path | None = None,
) -> Path:
    root = checkpoint_dir if checkpoint_dir is not None else default_checkpoint_dir(pkg_dir)
    return root / xai_checkpoint_basename(use_age=use_age, use_gender=use_gender)


def checkpoint_basename(
    *,
    split: str,
    xy_filter: str,
    use_age: bool,
    use_gender: bool = False,
    outer_fold_id: int | None = None,
) -> str:
    """Build a stable filename for a saved checkpoint."""
    age_tag = "age" if use_age else "no_age"
    gender_tag = "gender" if use_gender else "no_gender"
    if split == "oof":
        if outer_fold_id is None:
            raise ValueError("outer_fold_id is required for OOF checkpoints")
        return f"oof_{xy_filter}_{age_tag}_{gender_tag}_fold{outer_fold_id:02d}.pt"
    if split == "holdout":
        return f"holdout_{xy_filter}_{age_tag}_{gender_tag}.pt"
    if split == "xai":
        if xy_filter != XAI_XY_FILTER:
            raise ValueError(
                f"XAI checkpoints use xy_filter={XAI_XY_FILTER!r}, got {xy_filter!r}"
            )
        return xai_checkpoint_basename(use_age=use_age, use_gender=use_gender)
    raise ValueError(f"Unknown split tag: {split!r}")


def save_task7_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    xy_filter: str,
    use_age: bool,
    use_gender: bool = False,
    best_epoch: int,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    age_min: float = 0.0,
    age_max: float = 1.0,
    split: str,
    outer_fold_id: int | None = None,
    random_state: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist model state and training-time preprocessing statistics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "xy_filter": xy_filter,
        "use_age": use_age,
        "use_gender": use_gender,
        "best_epoch": int(best_epoch),
        "channel_mean": np.asarray(channel_mean, dtype=np.float64),
        "channel_std": np.asarray(channel_std, dtype=np.float64),
        "age_min": float(age_min),
        "age_max": float(age_max),
        "split": split,
        "outer_fold_id": outer_fold_id,
        "random_state": int(random_state),
        "model_class": "Task7Conv1dClassifier",
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path.resolve()


def load_task7_checkpoint(
    path: Path,
    device: torch.device | str,
) -> tuple[Task7Conv1dClassifier, dict[str, Any]]:
    """
    Load a checkpoint and return (model, metadata).

    ``metadata`` includes scaling arrays and training settings; use them to
    build ``Task5TrajectoryDataset`` with matching ``channel_mean`` / ``channel_std``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Checkpoints store numpy scaling arrays; requires weights_only=False (PyTorch >= 2.6).
    payload = torch.load(path, map_location=device, weights_only=False)
    use_age = bool(payload.get("use_age", payload.get("use_late_fusion_age", False)))
    use_gender = bool(payload.get("use_gender", False))
    model = Task7Conv1dClassifier(use_age=use_age, use_gender=use_gender)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload
