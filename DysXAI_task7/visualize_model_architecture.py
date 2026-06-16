"""
Publication architecture figures for Task7Conv1dClassifier (PyTorch).

Default (report bundle only)::
    python DysXAI_task7/visualize_model_architecture.py

Writes:
  - task7_model_architecture_multimodal.png / .pdf
  - task7_model_architecture_paper_style.png / .pdf
  - task7_model_architecture.mmd
  - task7_model_architecture_mermaid.svg / .pdf / .png  (vector SVG preferred; PNG rendered at high scale)

Legacy extras (v2 vertical, early-vs-late, graphviz, text summary)::
    python DysXAI_task7/visualize_model_architecture.py --all
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from model import KIN_CHANNELS, Task7Conv1dClassifier  # noqa: E402

DEFAULT_OUT_DIR = _HERE / "architecture_outputs"

# 12 kinematic channel names (Time channel dropped before Conv1d)
CHANNEL_LABELS = (
    "X",
    "Y",
    "Pressure",
    "Azimuth",
    "Tilt",
    "PenStatus",
    "Vx",
    "Vy",
    "Ax",
    "Ay",
    "Jx",
    "Jy",
)

# Colors (colorblind-friendly)
C_PRE = "#E8E8E8"
C_KIN = "#4C78A8"
C_POOL = "#9D7BB8"
C_DEMO = "#F58518"
C_FUSE = "#EECA3B"
C_HEAD = "#59A14F"
C_WARN = "#FFF3CD"
C_BORDER = "#2F2F2F"
C_TEXT = "#1A1A1A"


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def head_in_features(use_age: bool, use_gender: bool) -> int:
    return 256 + int(use_age) + int(use_gender)


def time_after_encoder(t: int) -> tuple[int, int, int]:
    """T at input and after each MaxPool1d(stride=2)."""
    t1 = max(t // 2, 1)
    t2 = max(t1 // 2, 1)
    return t, t1, t2


def try_torchinfo_summary(
    model: Task7Conv1dClassifier,
    *,
    batch_size: int,
    seq_len: int,
    use_age: bool,
    use_gender: bool,
    out_path: Path,
) -> bool:
    try:
        import torch
        from torchinfo import summary
    except ImportError:
        return False

    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    inputs: list = [torch.randn(batch_size, KIN_CHANNELS, seq_len), lengths]
    dtypes: list = [torch.float32, torch.long]
    if use_age:
        inputs.append(torch.rand(batch_size))
        dtypes.append(torch.float32)
    if use_gender:
        inputs.append(torch.randint(0, 2, (batch_size,)).float())
        dtypes.append(torch.float32)

    text = str(summary(model, input_data=tuple(inputs), dtypes=dtypes, verbose=0))
    out_path.write_text(text, encoding="utf-8")
    return True


def write_text_summary(
    path: Path,
    *,
    use_age: bool,
    use_gender: bool,
    seq_len: int,
    total_params: int,
) -> None:
    t0, t1, t2 = time_after_encoder(seq_len)
    fused = head_in_features(use_age, use_gender)
    ch = ", ".join(CHANNEL_LABELS)
    lines = [
        "Task 7 — Task7Conv1dClassifier (PyTorch)",
        "=" * 72,
        "",
        "PREPROCESSING (before the neural network)",
        "  Raw .svc clip",
        "    -> FFT low-pass on X,Y (per-fold tuned cutoff, e.g. 8-15 Hz)",
        "    -> compute Vx,Vy,Ax,Ay,Jx,Jy; drop Time channel",
        "    -> z-score each of 12 kinematic channels (fold train mean/std)",
        "",
        f"  Conv1d input channels ({KIN_CHANNELS}): {ch}",
        "  Age/Gender are scalars per subject clip — not Conv1d channels.",
        "",
        "FORWARD PASS (example time length T)",
        f"  x: (B, 12, {t0})  +  lengths (B,) for masked pooling",
    ]
    if use_age:
        lines.append("  age_norm: (B,)  = (age - age_min) / (age_max - age_min)  [fold-specific]")
    if use_gender:
        lines.append("  gender: (B,)  in {0, 1}")
    lines.extend(
        [
            "",
            "ENCODER (kinematics only)",
            f"  Conv1d(12->64, k=3, pad=1) + BN + ReLU + MaxPool(2)   -> (B, 64, {t1})",
            f"  Conv1d(64->128, k=3, pad=1) + BN + ReLU + MaxPool(2) -> (B, 128, {t2})",
            f"  Conv1d(128->256, k=3, pad=1) + BN + ReLU              -> (B, 256, {t2})",
            "  MaskedGlobalAvgPool1d (valid timesteps only)           -> (B, 256)",
            "",
            "LATE FUSION",
            f"  h_fused = concat[h_kin | age? | gender?]  -> (B, {fused})",
            "  Demographics are appended AFTER temporal encoding.",
            "",
            "CLASSIFIER HEAD",
            f"  Dropout(0.5) -> Linear({fused}->128) -> ReLU",
            "  Dropout(0.5) -> Linear(128->1) -> logit",
            "  Training: BCEWithLogitsLoss; inference: sigmoid -> P(dysgraphic)",
            "",
            f"Parameters: {total_params:,} trainable",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Matplotlib drawing helpers
# ---------------------------------------------------------------------------


def _stage_box(
    ax,
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    title: str,
    lines: list[str],
    facecolor: str,
    title_size: float = 10,
    line_size: float = 8.5,
    edge: str = C_BORDER,
    linewidth: float = 1.4,
) -> None:
    x = cx - w / 2
    y = cy - h / 2
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.06",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(cx, cy + h * 0.28, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=C_TEXT)
    body = "\n".join(lines)
    ax.text(cx, cy - h * 0.08, body, ha="center", va="center", fontsize=line_size, color=C_TEXT, linespacing=1.25)


def _arrow_v(ax, x: float, y_top: float, y_bot: float, *, color: str = C_BORDER) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x, y_top),
            (x, y_bot),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _arrow_h(ax, x_left: float, x_right: float, y: float, *, color: str = C_BORDER) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x_left, y),
            (x_right, y),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _arrow_to(
    ax,
    x_from: float,
    y_from: float,
    x_to: float,
    y_to: float,
    *,
    color: str = C_BORDER,
    style: str = "straight",
) -> None:
    """Draw an arrow between two points; ``elbow`` routes via one corner."""
    if style == "elbow_hv":
        mid = ((x_from + x_to) / 2, y_from)
        ax.plot([x_from, mid[0]], [y_from, mid[1]], color=color, linewidth=1.5, zorder=1)
        ax.add_patch(
            FancyArrowPatch(
                mid,
                (x_to, y_to),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.5,
                color=color,
                shrinkA=0,
                shrinkB=2,
                connectionstyle="arc3,rad=0",
            )
        )
        return
    if style == "elbow_vh":
        ax.plot([x_from, x_from], [y_from, y_to], color=color, linewidth=1.5, zorder=1)
        ax.add_patch(
            FancyArrowPatch(
                (x_from, y_to),
                (x_to, y_to),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.5,
                color=color,
                shrinkA=0,
                shrinkB=2,
            )
        )
        return
    ax.add_patch(
        FancyArrowPatch(
            (x_from, y_from),
            (x_to, y_to),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def draw_matplotlib_full(
    path_base: Path,
    *,
    use_age: bool,
    use_gender: bool,
    seq_len: int,
    total_params: int,
) -> None:
    """Two-branch layout: kinematics encoder (left) + demographic scalars (right) -> concat -> MLP."""
    t0, t1, t2 = time_after_encoder(seq_len)
    fused = head_in_features(use_age, use_gender)
    has_demo = use_age or use_gender

    fig_h = 14.5 if has_demo else 12.5
    fig = plt.figure(figsize=(11.5, fig_h), facecolor="white")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.9])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

    cx_kin = 3.2
    cx_demo = 7.8
    cx_merge = 5.0
    w_kin = 4.4
    w_demo = 2.9
    h_box = 1.22
    gap = 0.38

    ax.text(
        5.0,
        15.55,
        "Task 7 — Conv1d dysgraphia classifier with late-fusion demographics",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=C_TEXT,
    )
    ax.text(
        5.0,
        15.05,
        f"Task7Conv1dClassifier  |  {total_params:,} params  |  example T={seq_len}",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555555",
    )

    # --- Preprocessing (full width) ---
    y = 14.0
    _stage_box(ax, 1.8, y, 2.6, 0.95, title="Raw input", lines=["Tablet .svc clip"], facecolor=C_PRE, title_size=9)
    _arrow_h(ax, 3.15, 3.85, y)
    _stage_box(
        ax,
        4.7,
        y,
        3.0,
        0.95,
        title="Signal prep",
        lines=["FFT low-pass X,Y", "V/A/J derivatives", "drop Time"],
        facecolor=C_PRE,
        title_size=9,
        line_size=8,
    )
    _arrow_h(ax, 6.25, 6.95, y)
    _stage_box(
        ax,
        7.9,
        y,
        2.8,
        0.95,
        title="Normalize",
        lines=[f"z-score 12 ch.", f"(B, 12, {t0})"],
        facecolor=C_KIN,
        title_size=9,
        line_size=8,
    )

    y -= 1.15
    ch_row1 = "  ".join(CHANNEL_LABELS[:6])
    ch_row2 = "  ".join(CHANNEL_LABELS[6:])
    ax.text(
        5.0,
        y,
        f"12 kinematic Conv1d channels (demographics are NOT channels):\n{ch_row1}\n{ch_row2}",
        ha="center",
        va="center",
        fontsize=7.8,
        color=C_TEXT,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0F4FA", edgecolor=C_KIN, linewidth=1),
    )

    # --- Branch labels ---
    y -= 0.85
    ax.text(cx_kin, y, "Kinematics branch", ha="center", fontsize=10, fontweight="bold", color=C_KIN)
    if has_demo:
        ax.text(cx_demo, y, "Demographics branch", ha="center", fontsize=10, fontweight="bold", color="#B45309")
        ax.text(
            cx_demo,
            y - 0.42,
            "scalars per clip — never broadcast over T",
            ha="center",
            fontsize=8,
            color="#B45309",
        )

    # --- Kinematics encoder stack ---
    kin_stages = [
        (
            "Conv block 1",
            [f"Conv1d 12→64, k=3", "BN + ReLU + MaxPool(2)", f"→ (B, 64, {t1})"],
            C_KIN,
        ),
        (
            "Conv block 2",
            [f"Conv1d 64→128, k=3", "BN + ReLU + MaxPool(2)", f"→ (B, 128, {t2})"],
            C_KIN,
        ),
        (
            "Conv block 3",
            [f"Conv1d 128→256, k=3", "BN + ReLU", f"→ (B, 256, {t2})"],
            C_KIN,
        ),
        (
            "Masked global avg pool",
            ["avg over valid timesteps", "ignores padding", "h_kin  (B, 256)"],
            C_POOL,
        ),
    ]

    y_kin_top = y - 0.55
    kin_centers: list[float] = []
    for i, (title, lines, color) in enumerate(kin_stages):
        cy = y_kin_top - i * (h_box + gap)
        kin_centers.append(cy)
        _stage_box(ax, cx_kin, cy, w_kin, h_box, title=title, lines=lines, facecolor=color, title_size=9, line_size=8)
        if i == 0:
            _arrow_v(ax, 5.0, y - 0.95, cy + h_box / 2 + 0.05, color=C_KIN)
        else:
            prev = kin_centers[i - 1]
            _arrow_v(ax, cx_kin, prev - h_box / 2 - 0.04, cy + h_box / 2 + 0.04, color=color)

    pool_cy = kin_centers[-1]

    # --- Demographics stack (parallel path) ---
    demo_centers: list[float] = []
    if has_demo:
        demo_items: list[tuple[str, list[str]]] = []
        if use_age:
            demo_items.append(
                ("Age (scalar)", ["(B,) per subject clip", "fold min-max norm", "→ (B, 1)"])
            )
        if use_gender:
            demo_items.append(
                ("Gender (scalar)", ["(B,) binary 0/1", "male=0, female=1", "→ (B, 1)"])
            )
        demo_start = y_kin_top - 0.2
        for i, (title, lines) in enumerate(demo_items):
            cy = demo_start - i * (h_box + gap + 0.15)
            demo_centers.append(cy)
            _stage_box(
                ax,
                cx_demo,
                cy,
                w_demo,
                h_box,
                title=title,
                lines=lines,
                facecolor=C_DEMO,
                title_size=9,
                line_size=8,
            )
            if i > 0:
                _arrow_v(ax, cx_demo, demo_centers[i - 1] - h_box / 2 - 0.04, cy + h_box / 2 + 0.04, color=C_DEMO)

    # --- Merge / fusion ---
    fuse_y = pool_cy - (h_box + gap + 0.55)
    fuse_parts = ["h_kin (256)"]
    if use_age:
        fuse_parts.append("age (1)")
    if use_gender:
        fuse_parts.append("gender (1)")

    _arrow_v(ax, cx_kin, pool_cy - h_box / 2 - 0.05, fuse_y + 0.72, color=C_POOL)
    if has_demo:
        for dcy in demo_centers:
            _arrow_to(
                ax,
                cx_demo - w_demo / 2,
                dcy,
                cx_merge + 1.35,
                fuse_y + 0.15,
                color=C_DEMO,
                style="elbow_vh",
            )

    fuse_w = 5.6 if has_demo else w_kin
    _stage_box(
        ax,
        cx_merge,
        fuse_y,
        fuse_w,
        1.35,
        title="Late fusion — torch.cat(features, dim=1)",
        lines=[f"[{' | '.join(fuse_parts)}]", f"h_fused  (B, {fused})"],
        facecolor=C_FUSE,
        title_size=9.5,
        line_size=8.5,
    )

    # --- Classifier head ---
    head_y = fuse_y - (h_box + gap + 0.7)
    _arrow_v(ax, cx_merge, fuse_y - 0.68, head_y + h_box / 2 + 0.05, color=C_FUSE)
    _stage_box(
        ax,
        cx_merge,
        head_y,
        fuse_w,
        1.45,
        title="MLP classifier head",
        lines=[
            f"Dropout(0.5) → Linear({fused}→128) → ReLU",
            "Dropout(0.5) → Linear(128→1)",
            "logit → σ → P(dysgraphic)",
        ],
        facecolor=C_HEAD,
        title_size=9.5,
        line_size=8.5,
    )

    # --- Footnote ---
    note = (
        "Late fusion: demographics are concatenated only after the 12-channel trajectory "
        "is encoded to a 256-d vector.\n"
        "Early fusion (not used) would broadcast age/gender across all T timesteps as extra Conv1d channels."
    )
    ax.text(
        5.0,
        0.55,
        note,
        ha="center",
        va="center",
        fontsize=8.5,
        color=C_TEXT,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=C_WARN, edgecolor="#D4A106", linewidth=1.1),
    )

    legend = [
        mpatches.Patch(facecolor=C_PRE, edgecolor=C_BORDER, label="Preprocessing"),
        mpatches.Patch(facecolor=C_KIN, edgecolor=C_BORDER, label="Conv1d encoder"),
        mpatches.Patch(facecolor=C_POOL, edgecolor=C_BORDER, label="Masked pooling"),
        mpatches.Patch(facecolor=C_DEMO, edgecolor=C_BORDER, label="Demographics"),
        mpatches.Patch(facecolor=C_FUSE, edgecolor=C_BORDER, label="Concatenation"),
        mpatches.Patch(facecolor=C_HEAD, edgecolor=C_BORDER, label="Classifier"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=3, fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.01))

    for ext in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(ext), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def draw_matplotlib_late_fusion_detail(
    path_base: Path,
    *,
    use_age: bool,
    use_gender: bool,
    seq_len: int,
) -> None:
    """Compact figure focused on the late-fusion junction (good for reports / supervisor slides)."""
    fused = head_in_features(use_age, use_gender)
    _, _, t2 = time_after_encoder(seq_len)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.set_title(
        "Late fusion in Task 7: demographics join after kinematic encoding",
        fontsize=12,
        fontweight="bold",
        pad=10,
        color=C_TEXT,
    )

    # Kinematics mini-pipeline (left)
    kin_x, kin_w = 2.4, 3.6
    _stage_box(
        ax,
        kin_x,
        4.6,
        kin_w,
        1.0,
        title="Conv1d encoder (12 channels)",
        lines=[f"3 blocks + masked pool", f"h_kin  (B, 256)"],
        facecolor=C_KIN,
        title_size=9,
        line_size=8,
    )
    ax.text(kin_x, 3.55, f"Input: (B, 12, T)  e.g. T={seq_len} → (B, 256, {t2}) → pool", ha="center", fontsize=7.5, color="#444")

    # Demographics (right)
    demo_x = 9.0
    demo_y = 4.85
    if use_age:
        _stage_box(
            ax,
            demo_x,
            demo_y,
            2.4,
            0.85,
            title="age_norm",
            lines=["(B, 1)", "min-max per fold"],
            facecolor=C_DEMO,
            title_size=9,
            line_size=8,
        )
        demo_y -= 1.15
    if use_gender:
        _stage_box(
            ax,
            demo_x,
            demo_y,
            2.4,
            0.85,
            title="gender",
            lines=["(B, 1)", "0=male, 1=female"],
            facecolor=C_DEMO,
            title_size=9,
            line_size=8,
        )

    # Fusion (center-bottom)
    fuse_x = 5.5
    fuse_parts = ["256-d h_kin"]
    if use_age:
        fuse_parts.append("1-d age")
    if use_gender:
        fuse_parts.append("1-d gender")
    _stage_box(
        ax,
        fuse_x,
        2.0,
        4.8,
        1.15,
        title="torch.cat(dim=1)",
        lines=[f"{' + '.join(fuse_parts)}", f"→ h_fused (B, {fused})"],
        facecolor=C_FUSE,
        title_size=10,
        line_size=8.5,
    )

    _arrow_to(ax, kin_x + kin_w / 2, 4.1, fuse_x - 1.0, 2.55, color=C_POOL, style="elbow_vh")
    if use_age or use_gender:
        src_y = 4.85 if use_age else demo_y
        _arrow_to(ax, demo_x - 1.2, src_y, fuse_x + 1.2, 2.55, color=C_DEMO, style="elbow_vh")
        if use_age and use_gender:
            _arrow_to(ax, demo_x - 1.2, demo_y, fuse_x + 1.5, 2.35, color=C_DEMO, style="elbow_vh")

    _stage_box(
        ax,
        fuse_x,
        0.55,
        4.2,
        0.9,
        title="MLP head",
        lines=[f"Linear({fused}→128→1) → σ"],
        facecolor=C_HEAD,
        title_size=9,
        line_size=8,
    )
    _arrow_v(ax, fuse_x, 2.0 - 0.58, 0.55 + 0.45, color=C_HEAD)

    ax.text(
        6.0,
        5.55,
        "Demographics never enter Conv1d — they are appended as scalars after temporal encoding.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#B45309",
    )

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(ext), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _paper_rect(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str = "#FFFFFF",
    edgecolor: str = "#333333",
    linewidth: float = 1.2,
    zorder: int = 2,
) -> None:
    ax.add_patch(
        plt.Rectangle(
            (x, y),
            w,
            h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def _paper_label(ax, x: float, y: float, text: str, *, size: float = 8, bold: bool = False) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold" if bold else "normal",
        color=C_TEXT,
        zorder=5,
    )


def _paper_conv_block(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    sublayers: list[str],
    captions: list[str],
) -> None:
    """Stacked CONV/BN/ReLU/MaxPool block in the style of published CNN figures."""
    _paper_rect(ax, x, y, w, h, facecolor="#F7F7F7")
    _paper_label(ax, x + w / 2, y + h + 0.18, title, size=9, bold=True)

    n = len(sublayers)
    pad = 0.08
    inner_h = (h - pad * (n + 1)) / n
    for i, name in enumerate(sublayers):
        iy = y + pad + i * (inner_h + pad)
        _paper_rect(ax, x + 0.08, iy, w - 0.16, inner_h, facecolor="#FFFFFF", linewidth=0.9)
        _paper_label(ax, x + w / 2, iy + inner_h / 2, name, size=7.5)

    cap_y = y - 0.22
    for line in captions:
        _paper_label(ax, x + w / 2, cap_y, line, size=7)
        cap_y -= 0.2


def draw_paper_style_horizontal(
    path_base: Path,
    *,
    use_age: bool,
    use_gender: bool,
    seq_len: int,
    total_params: int,
) -> None:
    """
    Horizontal paper layout: left-to-right Conv stacks, masked pool,
    late-fusion concat, FC head, sigmoid.
    """
    t0, t1, t2 = time_after_encoder(seq_len)
    fused = head_in_features(use_age, use_gender)

    fig, ax = plt.subplots(figsize=(17, 5.2), facecolor="white")
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    ax.text(
        8.75,
        5.25,
        "Fig. Task 7: Conv1d trajectory encoder with late-fusion demographics (Age + Gender)",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=C_TEXT,
    )
    ax.text(
        8.75,
        4.95,
        f"159,809 trainable parameters  |  example T={seq_len}  |  demographics concatenated after pooling (not Conv1d channels)",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    y_main = 1.55
    h_main = 1.55
    arrow_y = y_main + h_main / 2

    # --- Input ---
    x_in = 0.35
    w_in = 1.15
    _paper_rect(ax, x_in, y_main, w_in, h_main, facecolor="#E8F0FE")
    _paper_label(ax, x_in + w_in / 2, y_main + h_main / 2 + 0.22, "Input", size=9, bold=True)
    _paper_label(
        ax,
        x_in + w_in / 2,
        y_main + h_main / 2 - 0.05,
        "Kinematic\nfeatures",
        size=8,
    )
    _paper_label(ax, x_in + w_in / 2, y_main - 0.28, f"(B, 12, {t0})", size=7)

    x = 1.75
    prev_right = x_in + w_in

    # --- Conv blocks ---
    conv_specs = [
        (
            "Conv layer 1",
            ["CONV", "BN", "ReLU", "MaxPool"],
            [f"64 filters, k=3, pad=1", f"MaxPool stride 2  → (B,64,{t1})"],
        ),
        (
            "Conv layer 2",
            ["CONV", "BN", "ReLU", "MaxPool"],
            [f"128 filters, k=3, pad=1", f"MaxPool stride 2  → (B,128,{t2})"],
        ),
        (
            "Conv layer 3",
            ["CONV", "BN", "ReLU"],
            [f"256 filters, k=3, pad=1", f"→ (B,256,{t2})"],
        ),
    ]
    w_conv = 1.55
    gap = 0.28
    for title, subs, caps in conv_specs:
        _arrow_h(ax, prev_right + 0.04, x - 0.04, arrow_y)
        _paper_conv_block(ax, x, y_main, w_conv, h_main, title=title, sublayers=subs, captions=caps)
        prev_right = x + w_conv
        x += w_conv + gap

    # --- Masked global average pool ---
    w_pool = 1.35
    _paper_rect(ax, x, y_main, w_pool, h_main, facecolor="#EDE7F6")
    _paper_label(ax, x + w_pool / 2, y_main + h_main / 2 + 0.22, "Masked", size=9, bold=True)
    _paper_label(ax, x + w_pool / 2, y_main + h_main / 2 - 0.05, "global avg\npool", size=8)
    _paper_label(ax, x + w_pool / 2, y_main - 0.28, "h_kin  (B, 256)", size=7)
    _arrow_h(ax, prev_right + 0.04, x - 0.04, arrow_y)
    prev_right = x + w_pool
    x += w_pool + gap

    # --- Late fusion concat ---
    w_cat = 1.05
    cat_x = x
    _paper_rect(ax, cat_x, y_main, w_cat, h_main, facecolor="#FFF8E1", edgecolor="#C9A000")
    _paper_label(ax, cat_x + w_cat / 2, y_main + h_main / 2 + 0.15, "Concat", size=9, bold=True)
    _paper_label(ax, cat_x + w_cat / 2, y_main + h_main / 2 - 0.12, "dim=1", size=8)
    _paper_label(ax, cat_x + w_cat / 2, y_main - 0.28, f"h_fused (B,{fused})", size=7)
    _arrow_h(ax, prev_right + 0.04, cat_x - 0.04, arrow_y)
    prev_right = cat_x + w_cat

    # Demographics from above (dashed)
    demo_y = 3.85
    demo_h = 0.72
    demo_w = 1.05
    demo_boxes: list[tuple[float, str, str]] = []
    if use_age:
        demo_boxes.append(("Age", "age_norm", "(B, 1)"))
    if use_gender:
        demo_boxes.append(("Gender", "0 / 1", "(B, 1)"))

    n_demo = len(demo_boxes)
    if n_demo:
        span = w_cat + 0.6
        start = cat_x + w_cat / 2 - span / 2 + demo_w / 2
        for i, (title, sub, shape) in enumerate(demo_boxes):
            dx = start + i * (demo_w + 0.25) if n_demo > 1 else cat_x + w_cat / 2 - demo_w / 2
            _paper_rect(ax, dx, demo_y, demo_w, demo_h, facecolor="#FFE8CC", edgecolor="#E07B00")
            _paper_label(ax, dx + demo_w / 2, demo_y + demo_h / 2 + 0.12, title, size=8, bold=True)
            _paper_label(ax, dx + demo_w / 2, demo_y + demo_h / 2 - 0.12, f"{sub}\n{shape}", size=7)
            ax.plot(
                [dx + demo_w / 2, cat_x + w_cat / 2],
                [demo_y, y_main + h_main + 0.04],
                color="#E07B00",
                linewidth=1.2,
                linestyle="--",
                zorder=1,
            )
            ax.add_patch(
                FancyArrowPatch(
                    (cat_x + w_cat / 2, y_main + h_main + 0.04),
                    (cat_x + w_cat / 2, y_main + h_main - 0.02),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.2,
                    color="#E07B00",
                    linestyle="--",
                )
            )

    x = cat_x + w_cat + gap

    # --- MLP head ---
    head_blocks = [
        ("Dropout", "p = 0.5", "#F1F8E9"),
        ("FC", f"{fused} → 128\nReLU", "#F1F8E9"),
        ("Dropout", "p = 0.5", "#F1F8E9"),
        ("FC", "128 → 1", "#F1F8E9"),
    ]
    w_small = 0.82
    for title, sub, color in head_blocks:
        _arrow_h(ax, prev_right + 0.04, x - 0.04, arrow_y)
        _paper_rect(ax, x, y_main, w_small, h_main, facecolor=color)
        _paper_label(ax, x + w_small / 2, y_main + h_main / 2 + 0.12, title, size=9, bold=True)
        _paper_label(ax, x + w_small / 2, y_main + h_main / 2 - 0.18, sub, size=7.5)
        prev_right = x + w_small
        x += w_small + 0.18

    # --- Sigmoid ---
    w_sig = 0.85
    _arrow_h(ax, prev_right + 0.04, x - 0.04, arrow_y)
    _paper_rect(ax, x, y_main, w_sig, h_main, facecolor="#FFFFFF", edgecolor="#C0392B", linewidth=1.6)
    _paper_label(ax, x + w_sig / 2, y_main + h_main / 2, "Sigmoid", size=9, bold=True)
    prev_right = x + w_sig
    x += w_sig + 0.2

    # --- Output ---
    w_out = 1.05
    _arrow_h(ax, prev_right + 0.04, x - 0.04, arrow_y)
    _paper_rect(ax, x, y_main, w_out, h_main, facecolor="#E8F5E9", edgecolor="#2E7D32")
    _paper_label(ax, x + w_out / 2, y_main + h_main / 2 + 0.1, "Output", size=9, bold=True)
    _paper_label(ax, x + w_out / 2, y_main + h_main / 2 - 0.2, "P(dysgraphic)", size=8)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(ext), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_matplotlib_fusion_compare(path_base: Path, *, use_age: bool, use_gender: bool) -> None:
    """Side-by-side early vs late fusion — ideal for supervisor Q&A."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), facecolor="white")

    for ax, title, early in zip(
        axes,
        ["Early fusion (NOT used)", "Late fusion (Task 7)"],
        [True, False],
    ):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color=C_TEXT)

        if early:
            _stage_box(
                ax,
                5,
                8.2,
                7,
                1.05,
                title="Conv1d input",
                lines=["(B, 14, T) = 12 kin + age + gender", "demographics as extra channels"],
                facecolor="#F8D7DA",
                edge="#C0392B",
            )
            _arrow_v(ax, 5, 8.2 - 0.53, 6.95, color=C_KIN)
            _stage_box(
                ax,
                5,
                6.2,
                7,
                1.35,
                title="Conv1d encoder",
                lines=["demographics mixed", "across all timesteps"],
                facecolor=C_KIN,
            )
            _arrow_v(ax, 5, 6.2 - 0.68, 4.85, color="#C0392B")
            ax.text(
                5,
                4.5,
                "Age/gender must be broadcast\nalong every timestep T",
                ha="center",
                fontsize=9,
                color="#C0392B",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEA", edgecolor="#C0392B"),
            )
            ax.text(5, 0.6, "Not implemented in Task 7", ha="center", fontsize=10, color="#C0392B")
        else:
            _stage_box(ax, 5, 8.2, 7, 1.0, title="Conv1d input", lines=["(B, 12, T) kinematics only"], facecolor=C_KIN)
            _arrow_v(ax, 5, 8.2 - 0.5, 7.05, color=C_KIN)
            _stage_box(ax, 5, 6.2, 7, 1.15, title="Encoder + masked pool", lines=["h_kin  (B, 256)"], facecolor=C_POOL)
            demo_lines = []
            if use_age:
                demo_lines.append("age_norm (B,1)")
            if use_gender:
                demo_lines.append("gender (B,1)")
            _stage_box(
                ax,
                7.6,
                4.5,
                2.8,
                1.1,
                title="Scalars",
                lines=demo_lines or ["—"],
                facecolor=C_DEMO,
                title_size=9,
                line_size=8,
            )
            _arrow_v(ax, 5, 6.2 - 0.58, 5.35, color=C_POOL)
            _arrow_to(ax, 5, 5.65, 5, 5.35, color=C_FUSE, style="straight")
            if demo_lines:
                _arrow_to(ax, 7.6 - 1.4, 4.5, 5.8, 4.85, color=C_DEMO, style="elbow_vh")
            _stage_box(
                ax,
                5,
                4.0,
                7,
                1.1,
                title="torch.cat(dim=1) → MLP",
                lines=["h_fused → P(dysgraphic)"],
                facecolor=C_HEAD,
            )
            _arrow_v(ax, 5, 6.2 - 0.58, 4.55, color=C_POOL)
            ax.text(5, 0.6, "Used in DysXAI Task 7", ha="center", fontsize=10, fontweight="bold", color=C_HEAD)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(ext), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


_WINDOWS_GRAPHVIZ_BIN_DIRS = (
    Path(r"C:\Program Files\Graphviz\bin"),
    Path(r"C:\Program Files (x86)\Graphviz\bin"),
)


def resolve_graphviz_bin_dir(explicit: Path | None = None) -> Path | None:
    """Find a folder containing dot.exe (winget often installs without updating PATH)."""
    if explicit is not None:
        dot = explicit / "dot.exe"
        return explicit if dot.is_file() else None

    env_bin = os.environ.get("GRAPHVIZ_BIN")
    if env_bin:
        found = resolve_graphviz_bin_dir(Path(env_bin))
        if found is not None:
            return found

    if shutil.which("dot"):
        return None  # already on PATH

    for candidate in _WINDOWS_GRAPHVIZ_BIN_DIRS:
        if (candidate / "dot.exe").is_file():
            return candidate
    return None


def ensure_graphviz_on_path(bin_dir: Path | None) -> bool:
    if bin_dir is not None:
        bin_str = str(bin_dir)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if bin_str not in path_parts:
            os.environ["PATH"] = bin_str + os.pathsep + os.environ.get("PATH", "")
    return shutil.which("dot") is not None


def draw_graphviz_full(
    path_base: Path,
    *,
    use_age: bool,
    use_gender: bool,
    seq_len: int,
    total_params: int,
    graphviz_bin: Path | None = None,
) -> str | None:
    """Return None on success, or a short reason string if Graphviz output was skipped."""
    try:
        from graphviz import Digraph
        from graphviz.backend.execute import ExecutableNotFound
    except ImportError:
        return "python_package"

    bin_dir = resolve_graphviz_bin_dir(graphviz_bin)
    if not ensure_graphviz_on_path(bin_dir):
        return "system_binary"

    t0, t1, t2 = time_after_encoder(seq_len)
    fused = head_in_features(use_age, use_gender)

    g = Digraph("Task7", format="png")
    g.attr(rankdir="TB", splines="ortho", nodesep="0.35", ranksep="0.45")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10", color=C_BORDER)
    g.attr("edge", color="#555555", penwidth="1.2", arrowsize="0.7")

    g.node("raw", "Raw .svc clip", fillcolor=C_PRE)
    g.node("prep", f"FFT + derivatives + z-score\\n12 channels, shape (B,12,{t0})", fillcolor=C_KIN)
    g.edge("raw", "prep")

    g.node("c1", f"Conv1d 12→64 + BN + ReLU + MaxPool\\n(B, 64, {t1})", fillcolor=C_KIN)
    g.node("c2", f"Conv1d 64→128 + BN + ReLU + MaxPool\\n(B, 128, {t2})", fillcolor=C_KIN)
    g.node("c3", f"Conv1d 128→256 + BN + ReLU\\n(B, 256, {t2})", fillcolor=C_KIN)
    g.node("pool", "Masked Global Avg Pool\\nh_kin (B, 256)", fillcolor=C_POOL)
    g.edge("prep", "c1")
    g.edge("c1", "c2")
    g.edge("c2", "c3")
    g.edge("c3", "pool")

    with g.subgraph(name="cluster_demo") as demo:
        demo.attr(label="Demographics branch (NOT Conv1d channels)", style="dashed", color=C_DEMO)
        if use_age:
            demo.node("age", "Age scalar (B,)\\nfold min-max norm", fillcolor=C_DEMO)
        if use_gender:
            demo.node("gender", "Gender scalar (B,)\\n0=male, 1=female", fillcolor=C_DEMO)

    g.node("fuse", f"Late fusion: torch.cat(dim=1)\\nh_fused (B, {fused})", fillcolor=C_FUSE)
    g.edge("pool", "fuse")
    if use_age:
        g.edge("age", "fuse", style="dashed", color=C_DEMO)
    if use_gender:
        g.edge("gender", "fuse", style="dashed", color=C_DEMO)

    g.node(
        "head",
        f"Dropout → Linear({fused}→128) → ReLU\\nDropout → Linear(128→1) → σ → P(dysgraphic)",
        fillcolor=C_HEAD,
    )
    g.edge("fuse", "head")

    g.attr(label=f"Task 7 Conv1d + late fusion ({total_params:,} params)\\nexample T={seq_len}", fontsize="12")

    out = path_base.with_suffix("")
    try:
        g.render(filename=out.name, directory=str(out.parent), cleanup=True)
        g.format = "pdf"
        g.render(filename=out.name, directory=str(out.parent), cleanup=True)
    except ExecutableNotFound:
        return "system_binary"
    return None


def write_mermaid_architecture(
    path: Path,
    *,
    use_age: bool,
    use_gender: bool,
    seq_len: int,
    total_params: int,
) -> None:
    """Write Mermaid source for draw.io / GitHub / Notion (multimodal late-fusion layout)."""
    fused = head_in_features(use_age, use_gender)
    _, _, t2 = time_after_encoder(seq_len)

    age_node = (
        '        A["Age scalar<br/>(B, 1)<br/>fold min-max norm"]:::input\n'
        if use_age
        else ""
    )
    gender_node = (
        '        G["Gender scalar<br/>(B, 1)<br/>0=male, 1=female"]:::input\n'
        if use_gender
        else ""
    )
    age_edge = "    A -.->|bypass CNN| Cat\n" if use_age else ""
    gender_edge = "    G -.->|bypass CNN| Cat\n" if use_gender else ""

    text = f"""---
title: Task 7 Conv1d + Late Fusion ({total_params:,} params)
---
flowchart LR
    classDef input fill:#e8f4f8,stroke:#2980b9,stroke-width:2px,color:#2c3e50
    classDef conv fill:#fff3e0,stroke:#d35400,stroke-width:2px,color:#2c3e50
    classDef pool fill:#fcf3cf,stroke:#f1c40f,stroke-width:2px,color:#2c3e50
    classDef dense fill:#f4ecf7,stroke:#8e44ad,stroke-width:2px,color:#2c3e50
    classDef fusion fill:#e8f8f5,stroke:#27ae60,stroke-width:3px,color:#2c3e50
    classDef output fill:#fadbd8,stroke:#c0392b,stroke-width:2px,color:#2c3e50

    subgraph Inputs ["Multimodal inputs"]
        direction TB
        K["Kinematics<br/>(B, 12, {seq_len})<br/>12 z-scored channels"]:::input
{age_node}{gender_node}    end

    subgraph CNN ["Temporal encoder (kinematics only)"]
        direction LR
        C1["Conv1d 12→64<br/>BN + ReLU<br/>MaxPool stride 2"]:::conv
        C2["Conv1d 64→128<br/>BN + ReLU<br/>MaxPool stride 2"]:::conv
        C3["Conv1d 128→256<br/>BN + ReLU"]:::conv
        K --> C1 --> C2 --> C3
    end

    subgraph Pooling ["Time-series flattening"]
        direction TB
        MP["Masked global avg pool<br/>ignores padded timesteps"]:::pool
        Flat["h_kin latent vector<br/>(B, 256)"]:::pool
        C3 --> MP --> Flat
    end

    subgraph LateFusion ["Late fusion integration"]
        direction TB
        Cat{{"torch.cat(dim=1)<br/>h_fused (B, {fused})"}}:::fusion
    end

    Flat --> Cat
{age_edge}{gender_edge}
    subgraph MLP ["Classification head"]
        direction LR
        D1["Dropout 0.5<br/>Linear {fused}→128<br/>ReLU"]:::dense
        D2["Dropout 0.5<br/>Linear 128→1"]:::dense
        Cat -->|combined {fused}-d vector| D1 --> D2
    end

    Out["P(dysgraphic)<br/>sigmoid at inference"]:::output
    D2 --> Out
"""
    path.write_text(text, encoding="utf-8")


def _region_frame(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    *,
    edge: str = "#BBBBBB",
    face: str = "#FAFAFA",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
            linestyle="--",
            zorder=0,
        )
    )
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9, fontweight="bold", color="#444444", zorder=1)


def _node_box(
    ax,
    cx: float,
    cy: float,
    w: float,
    h: float,
    lines: list[str],
    *,
    facecolor: str,
    edgecolor: str = C_BORDER,
    fontsize: float = 8,
    bold_first: bool = True,
) -> tuple[float, float, float, float]:
    x, y = cx - w / 2, cy - h / 2
    _paper_rect(ax, x, y, w, h, facecolor=facecolor, edgecolor=edgecolor)
    if not lines:
        return x, y, w, h
    if len(lines) == 1:
        _paper_label(ax, cx, cy, lines[0], size=fontsize, bold=bold_first)
    else:
        _paper_label(ax, cx, cy + h * 0.18, lines[0], size=fontsize, bold=bold_first)
        body = "\n".join(lines[1:])
        ax.text(cx, cy - h * 0.08, body, ha="center", va="center", fontsize=fontsize - 0.5, color=C_TEXT, zorder=5)
    return x, y, w, h


def draw_multimodal_publication(
    path_base: Path,
    *,
    use_age: bool,
    use_gender: bool,
    seq_len: int,
    total_params: int,
) -> None:
    """
    Publication layout: separated input streams, highlighted masked pool,
    demographic bypass arrows into concat (matches Gemini / committee brief).
    """
    fused = head_in_features(use_age, use_gender)
    t0, t1, t2 = time_after_encoder(seq_len)

    fig, ax = plt.subplots(figsize=(15.5, 7.2), facecolor="white")
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(
        7.75,
        6.85,
        "Task 7: Conv1d temporal encoder with late-fusion demographics",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=C_TEXT,
    )
    ax.text(
        7.75,
        6.45,
        f"{total_params:,} trainable parameters  |  example T={seq_len}  |  Age/Gender bypass the CNN and join at concat",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )

    y_lo = 0.55
    lane_y = 3.35

    # --- Region 1: Multimodal inputs ---
    _region_frame(ax, 0.25, y_lo, 2.55, 5.35, "Multimodal inputs", edge="#2980b9", face="#e8f4f8")
    kx, ky = 1.52, 5.15
    _node_box(
        ax,
        kx,
        ky,
        2.0,
        1.05,
        ["Kinematics", f"(B, 12, {t0})", "12 z-scored ch."],
        facecolor="#FFFFFF",
        edgecolor="#2980b9",
    )
    demo_nodes: list[tuple[float, float, str, list[str]]] = []
    if use_age:
        demo_nodes.append((1.52, 3.85, "#E07B00", ["Age scalar", "(B, 1)", "min-max / fold"]))
    if use_gender:
        demo_nodes.append((1.52, 2.55, "#E07B00", ["Gender scalar", "(B, 1)", "0=male, 1=female"]))
    for dx, dy, ec, lines in demo_nodes:
        _node_box(ax, dx, dy, 2.0, 0.95, lines, facecolor="#FFE8CC", edgecolor=ec)

    # --- Region 2: Temporal encoder ---
    _region_frame(ax, 3.05, 1.85, 5.35, 3.55, "Temporal encoder (kinematics only)", edge="#d35400", face="#fff8f0")
    conv_specs = [
        (f"Conv1d 12→64", f"BN+ReLU+MaxPool", f"(B,64,{t1})"),
        (f"Conv1d 64→128", f"BN+ReLU+MaxPool", f"(B,128,{t2})"),
        (f"Conv1d 128→256", f"BN+ReLU", f"(B,256,{t2})"),
    ]
    cx = 3.65
    conv_centers: list[float] = []
    for i, (a, b, c) in enumerate(conv_specs):
        ccx = cx + 1.55 / 2
        conv_centers.append(ccx)
        _node_box(ax, ccx, lane_y, 1.55, 1.55, [a, b, c], facecolor="#FFFFFF", edgecolor="#d35400")
        if i > 0:
            _arrow_h(ax, conv_centers[i - 1] + 0.78, ccx - 0.78, lane_y)
        cx += 1.75
    _arrow_h(ax, kx + 1.0, conv_centers[0] - 0.78, lane_y)

    # --- Region 3: Masked pooling (highlighted novelty) ---
    _region_frame(ax, 8.6, 1.85, 2.45, 3.55, "Time-series flattening", edge="#f1c40f", face="#fffde7")
    pool_cx = 9.82
    _node_box(
        ax,
        pool_cx,
        4.35,
        2.05,
        1.35,
        ["Masked global", "avg pool", "ignores padding"],
        facecolor="#FFF9C4",
        edgecolor="#F1C40F",
        fontsize=8.5,
    )
    flat_cx = 9.82
    _node_box(
        ax,
        flat_cx,
        lane_y,
        2.05,
        1.2,
        ["h_kin", f"(B, 256)"],
        facecolor="#FFF9C4",
        edgecolor="#F1C40F",
        fontsize=9,
    )
    _arrow_h(ax, conv_centers[-1] + 0.78, pool_cx - 1.05, lane_y)
    _arrow_v(ax, pool_cx, 4.35 - 0.68, lane_y + 0.62, color="#F1C40F")

    # --- Region 4: Late fusion ---
    _region_frame(ax, 11.3, 2.15, 2.15, 2.95, "Late fusion", edge="#27ae60", face="#e8f8f5")
    fuse_cx, fuse_cy = 12.37, lane_y
    _node_box(
        ax,
        fuse_cx,
        fuse_cy,
        1.75,
        1.45,
        ["torch.cat", "dim = 1", f"h_fused (B,{fused})"],
        facecolor="#C8E6C9",
        edgecolor="#27ae60",
        fontsize=8.5,
    )
    _arrow_h(ax, flat_cx + 1.03, fuse_cx - 0.88, lane_y)

    # Demographic bypass (dashed, above encoder)
    bypass_y = 5.55
    for dx, dy, _, lines in demo_nodes:
        ax.plot([dx + 1.0, fuse_cx - 0.5], [dy, bypass_y], color="#E07B00", linewidth=1.4, linestyle="--", zorder=1)
        ax.plot([fuse_cx - 0.5, fuse_cx], [bypass_y, fuse_cy + 0.72], color="#E07B00", linewidth=1.4, linestyle="--", zorder=1)
        ax.add_patch(
            FancyArrowPatch(
                (fuse_cx, fuse_cy + 0.72),
                (fuse_cx, fuse_cy + 0.35),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color="#E07B00",
                linestyle="--",
            )
        )
        label = "age" if "Age" in lines[0] else "gender"
        ax.text((dx + fuse_cx) / 2 + 0.3, bypass_y + 0.12, f"bypass CNN ({label})", fontsize=7, color="#E07B00")

    # --- Region 5: Classification head ---
    _region_frame(ax, 13.65, 1.85, 1.55, 3.55, "MLP head", edge="#8e44ad", face="#f9f5fb")
    mlp_x = 14.42
    _node_box(
        ax,
        mlp_x,
        4.35,
        1.2,
        1.35,
        ["Dropout 0.5", f"Lin {fused}→128", "ReLU"],
        facecolor="#FFFFFF",
        edgecolor="#8e44ad",
        fontsize=7.5,
    )
    _node_box(
        ax,
        mlp_x,
        lane_y,
        1.2,
        1.2,
        ["Dropout 0.5", "Lin 128→1"],
        facecolor="#FFFFFF",
        edgecolor="#8e44ad",
        fontsize=7.5,
    )
    _arrow_h(ax, fuse_cx + 0.88, mlp_x - 0.62, 4.35)
    _arrow_v(ax, mlp_x, 4.35 - 0.68, lane_y + 0.62, color="#8e44ad")
    _arrow_h(ax, fuse_cx + 0.88, mlp_x - 0.62, lane_y)

    # Output
    out_cx = 14.42
    out_cy = 1.15
    _node_box(
        ax,
        out_cx,
        out_cy,
        1.35,
        0.85,
        ["P(dysgraphic)", "σ at inference"],
        facecolor="#FADBD8",
        edgecolor="#C0392B",
        fontsize=8,
    )
    _arrow_v(ax, mlp_x, lane_y - 0.62, out_cy + 0.44, color="#C0392B")

    # Presentation callouts
    ax.text(
        9.82,
        1.35,
        "★ Custom layer: averages only valid writing timesteps,\nnot padded zeros (variable clip lengths).",
        ha="center",
        fontsize=7.5,
        color="#7D6608",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFDE7", edgecolor="#F1C40F"),
    )
    ax.text(
        12.37,
        1.35,
        f"★ Fusion: 256 kin + {int(use_age)} age + {int(use_gender)} gender = {fused}-d vector",
        ha="center",
        fontsize=7.5,
        color="#1E8449",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F8F5", edgecolor="#27AE60"),
    )

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(ext), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mermaid_cli_base() -> list[str] | None:
    """Return mermaid-cli argv prefix (mmdc or npx), or None if unavailable."""
    mmdc = shutil.which("mmdc")
    if mmdc:
        return [mmdc]
    npx = shutil.which("npx")
    if npx:
        return [npx, "-y", "@mermaid-js/mermaid-cli"]
    return None


def try_render_mermaid(mmd_path: Path, out_stem: Path) -> list[Path]:
    """
    Render .mmd to SVG (vector), PDF, and high-resolution PNG.

    Default mermaid-cli uses 800px width — too small for wide diagrams and looks
    blurry when zoomed. We use a wide canvas plus scale for crisp PNG export.
    """
    import subprocess

    cli = _mermaid_cli_base()
    if cli is None:
        return []

    # Wide layout + scale 3 ≈ sharp print / slide zoom (avoid default 800px blur).
    render_jobs: list[tuple[Path, list[str]]] = [
        (out_stem.with_suffix(".svg"), []),
        (out_stem.with_suffix(".pdf"), []),
        (out_stem.with_suffix(".png"), ["-w", "3200", "-s", "3", "-b", "white"]),
    ]

    written: list[Path] = []
    for out_path, extra in render_jobs:
        cmd = [*cli, "-i", str(mmd_path), "-o", str(out_path), *extra]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if out_path.is_file():
                written.append(out_path)
        except (subprocess.CalledProcessError, OSError):
            continue
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Publication architecture diagrams for Task 7.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-age", action="store_true")
    parser.add_argument("--no-gender", action="store_true")
    parser.add_argument("--seq-len", type=int, default=200, help="Example sequence length T for shape labels.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also generate legacy diagrams (v2 vertical, early-vs-late, graphviz, text summary, torchinfo).",
    )
    parser.add_argument(
        "--graphviz-bin",
        type=Path,
        default=None,
        help="Folder with dot.exe if not on PATH (only used with --all).",
    )
    args = parser.parse_args()

    use_age = not args.no_age
    use_gender = not args.no_gender
    out_dir = args.out_dir if args.out_dir.is_absolute() else _HERE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Task7Conv1dClassifier(use_age=use_age, use_gender=use_gender)
    total, _trainable = count_parameters(model)

    paper_path = out_dir / "task7_model_architecture_paper_style"
    multimodal_path = out_dir / "task7_model_architecture_multimodal"
    mermaid_path = out_dir / "task7_model_architecture.mmd"
    mermaid_stem = out_dir / "task7_model_architecture_mermaid"

    write_mermaid_architecture(
        mermaid_path, use_age=use_age, use_gender=use_gender, seq_len=args.seq_len, total_params=total
    )
    print(f"Wrote: {mermaid_path}")

    draw_paper_style_horizontal(
        paper_path, use_age=use_age, use_gender=use_gender, seq_len=args.seq_len, total_params=total
    )
    print(f"Wrote: {paper_path.with_suffix('.png')}")
    print(f"Wrote: {paper_path.with_suffix('.pdf')}")

    draw_multimodal_publication(
        multimodal_path, use_age=use_age, use_gender=use_gender, seq_len=args.seq_len, total_params=total
    )
    print(f"Wrote: {multimodal_path.with_suffix('.png')}")
    print(f"Wrote: {multimodal_path.with_suffix('.pdf')}")

    mermaid_outputs = try_render_mermaid(mermaid_path, mermaid_stem)
    if mermaid_outputs:
        for p in mermaid_outputs:
            print(f"Wrote: {p}")
        if mermaid_stem.with_suffix(".svg") in mermaid_outputs:
            print("  Tip: use the .svg or .pdf in your report (vector = never blurry).")
    else:
        print(
            "Mermaid render skipped — paste .mmd into https://mermaid.live "
            "or install: npx -y @mermaid-js/mermaid-cli"
        )

    if args.all:
        arch_path = out_dir / "task7_model_architecture_v2"
        fusion_detail_path = out_dir / "task7_late_fusion_detail"
        compare_path = out_dir / "task7_early_vs_late_fusion"
        summary_path = out_dir / "task7_model_architecture_summary.txt"
        torchinfo_path = out_dir / "task7_model_torchinfo.txt"

        draw_matplotlib_full(arch_path, use_age=use_age, use_gender=use_gender, seq_len=args.seq_len, total_params=total)
        draw_matplotlib_late_fusion_detail(
            fusion_detail_path, use_age=use_age, use_gender=use_gender, seq_len=args.seq_len
        )
        draw_matplotlib_fusion_compare(compare_path, use_age=use_age, use_gender=use_gender)
        print(f"Wrote (legacy): {arch_path.with_suffix('.png')}")
        print(f"Wrote (legacy): {fusion_detail_path.with_suffix('.png')}")
        print(f"Wrote (legacy): {compare_path.with_suffix('.png')}")

        gv_skip = draw_graphviz_full(
            arch_path.with_name(arch_path.stem + "_graphviz"),
            use_age=use_age,
            use_gender=use_gender,
            seq_len=args.seq_len,
            total_params=total,
            graphviz_bin=args.graphviz_bin,
        )
        if gv_skip is None:
            print(f"Wrote (legacy): {arch_path.stem}_graphviz.png / .pdf")

        write_text_summary(summary_path, use_age=use_age, use_gender=use_gender, seq_len=args.seq_len, total_params=total)
        print(f"Wrote (legacy): {summary_path}")

        if try_torchinfo_summary(
            model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            use_age=use_age,
            use_gender=use_gender,
            out_path=torchinfo_path,
        ):
            print(f"Wrote (legacy): {torchinfo_path}")


if __name__ == "__main__":
    main()
