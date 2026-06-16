"""
Rebuild task7_ablation_oof_fold_hparams.csv from saved OOF checkpoints.

Use when ablation finished but fold-hparams CSV was not written yet.
Each checkpoint stores: xy_filter, best_epoch, use_age, use_gender, outer_fold_id.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import torch

_PKG_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = _PKG_DIR / "checkpoints"
OUT_CSV = _PKG_DIR / "task7_ablation_oof_fold_hparams.csv"

CONFIG_BY_FLAGS = {
    (False, False): "Baseline (Kinematics Only)",
    (True, False): "Baseline + Age",
    (False, True): "Baseline + Gender",
    (True, True): "Baseline + Age + Gender",
}


def configuration_label(use_age: bool, use_gender: bool) -> str:
    return CONFIG_BY_FLAGS[(bool(use_age), bool(use_gender))]


def parse_freq_from_xy_filter(xy_filter: str) -> float:
    m = re.match(r"^fft_(\d+(?:\.\d+)?)$", str(xy_filter).strip().lower())
    if m:
        return float(m.group(1))
    if xy_filter in ("fft", "butterworth"):
        return 12.0
    return float("nan")


def main() -> None:
    ckpt_paths = sorted(CHECKPOINT_DIR.glob("oof_fft_*_fold*.pt"))
    if not ckpt_paths:
        raise SystemExit(f"No OOF FFT checkpoints found in {CHECKPOINT_DIR}")

    rows: list[dict[str, object]] = []
    for path in ckpt_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        use_age = bool(payload["use_age"])
        use_gender = bool(payload["use_gender"])
        xy_filter = str(payload["xy_filter"])
        rows.append(
            {
                "configuration": configuration_label(use_age, use_gender),
                "xy_filter": "fft",
                "use_age": int(use_age),
                "use_gender": int(use_gender),
                "outer_fold": int(payload["outer_fold_id"]),
                "selected_filter": xy_filter,
                "selected_freq_hz": parse_freq_from_xy_filter(xy_filter),
                "selected_epoch": int(payload["best_epoch"]),
                "inner_best_mean_auc": "",
                "n_train": int(payload.get("extra", {}).get("n_train", 0)),
                "n_test": int(payload.get("extra", {}).get("n_test", 0)),
                "checkpoint_file": path.name,
            }
        )

    rows.sort(key=lambda r: (r["configuration"], int(r["outer_fold"])))
    fieldnames = [
        "configuration",
        "xy_filter",
        "use_age",
        "use_gender",
        "outer_fold",
        "selected_filter",
        "selected_freq_hz",
        "selected_epoch",
        "inner_best_mean_auc",
        "n_train",
        "n_test",
        "checkpoint_file",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {OUT_CSV.resolve()}")
    print()
    print("| Configuration | Fold | Freq (Hz) | Epoch | Checkpoint |")
    print("|---|---|---:|---:|---|")
    for r in rows:
        freq = r["selected_freq_hz"]
        freq_txt = f"{float(freq):g}" if freq != "" else "—"
        print(
            f"| {r['configuration']} | {r['outer_fold']} | {freq_txt} | "
            f"{r['selected_epoch']} | {r['checkpoint_file']} |"
        )


if __name__ == "__main__":
    main()
