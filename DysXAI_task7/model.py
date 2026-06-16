"""1D CNN for Task 7 trajectories: 12 kinematic channels + optional late-fused demographics."""

from __future__ import annotations

import torch
import torch.nn as nn

KIN_CHANNELS = 12


class MaskedGlobalAvgPool1d(nn.Module):
    """Global average pooling over valid timesteps only (ignores right-padding)."""

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            return x.mean(dim=-1)
        B, C, T = x.shape
        device = x.device
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).float()
        x_masked = x * mask
        denom = lengths.clamp(min=1).view(B, 1).float()
        return x_masked.sum(dim=-1) / denom


class Task7Conv1dClassifier(nn.Module):
    """Conv1d encoder + masked global pool + MLP head (optional late-fusion age/gender)."""

    def __init__(self, use_age: bool = False, use_gender: bool = False):
        super().__init__()
        self.use_age = bool(use_age)
        self.use_gender = bool(use_gender)

        self.encoder = nn.Sequential(
            nn.Conv1d(KIN_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.pool = MaskedGlobalAvgPool1d()

        in_features = 256 + int(self.use_age) + int(self.use_gender)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _lengths_after_encoder(lengths: torch.Tensor, n_maxpool: int = 2) -> torch.Tensor:
        """Account for two stride-2 MaxPool1d layers in ``encoder``."""
        out = lengths
        for _ in range(n_maxpool):
            out = (out // 2).clamp(min=1)
        return out

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        age: torch.Tensor | None = None,
        gender: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(x)
        pool_lengths = self._lengths_after_encoder(lengths) if lengths is not None else None
        h = self.pool(h, pool_lengths)
        h_flat = h.view(h.size(0), -1)
        features = [h_flat]
        if self.use_age and age is not None:
            features.append(age.view(-1, 1) if age.dim() == 1 else age)
        if self.use_gender and gender is not None:
            features.append(gender.view(-1, 1) if gender.dim() == 1 else gender)
        h_fused = torch.cat(features, dim=1)
        return self.head(h_fused)


# Backward-compatible alias used by older script imports
Task5Conv1dClassifier = Task7Conv1dClassifier
