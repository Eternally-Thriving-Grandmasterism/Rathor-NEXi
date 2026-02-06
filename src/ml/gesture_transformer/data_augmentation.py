# src/ml/gesture_transformer/data_augmentation.py – Data Augmentation Pipeline v1.0
# Spatiotemporal augmentations for gesture landmark sequences
# Valence-aware intensity scaling, GPU-accelerated where possible
# MIT License – Autonomicity Games Inc. 2026

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class GestureAugmentation(nn.Module):
    """
    Composable data augmentation pipeline for (B, T, L) gesture sequences
    where T = time/sequence length, L = landmarks × 3 (x,y,z)
    All operations are differentiable (where needed) and GPU-friendly
    """

    def __init__(
        self,
        seq_len: int = 45,
        landmark_dim: int = 225,
        p_spatial_noise: float = 0.4,
        p_temporal_dropout: float = 0.25,
        p_time_warp: float = 0.35,
        p_rotation: float = 0.3,
        p_scaling: float = 0.3,
        p_gaussian_noise: float = 0.5,
        valence_intensity_scale: float = 1.0,   # higher valence → stronger aug
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.seq_len = seq_len
        self.landmark_dim = landmark_dim
        self.device = device

        # Valence modulation factor (higher valence = stronger augmentation)
        self.valence_intensity_scale = valence_intensity_scale

        self.p_spatial_noise = p_spatial_noise
        self.p_temporal_dropout = p_temporal_dropout
        self.p_time_warp = p_time_warp
        self.p_rotation = p_rotation
        self.p_scaling = p_scaling
        self.p_gaussian_noise = p_gaussian_noise

    def forward(self, x: torch.Tensor, valence: float = None) -> torch.Tensor:
        """
        x: (B, T, L) or (T, L) – batch or single sequence
        valence: optional scalar [0,1] – controls augmentation strength
        """
        if valence is None:
            valence = currentValence.get()

        intensity = self.valence_intensity_scale * valence

        if x.dim() == 2:
            x = x.unsqueeze(0)  # add batch dim if needed
            squeeze = True
        else:
            squeeze = False

        B, T, L = x.shape

        # 1. Gaussian noise (per landmark)
        if random.random() < self.p_gaussian_noise * intensity:
            noise = torch.randn_like(x) * 0.015 * intensity
            x = x + noise

        # 2. Spatial noise (per-frame jitter)
        if random.random() < self.p_spatial_noise * intensity:
            jitter = torch.randn(B, T, L) * 0.02 * intensity
            x = x + jitter

        # 3. Temporal dropout (randomly mask frames)
        if random.random() < self.p_temporal_dropout * intensity:
            keep_prob = 0.85 + 0.1 * (1 - intensity)  # less dropout when valence high
            mask = torch.rand(B, T, device=x.device) < keep_prob
            mask = mask.unsqueeze(-1).expand(-1, -1, L)
            x = x * mask.float()

        # 4. Time warping (non-linear time stretch)
        if random.random() < self.p_time_warp * intensity:
            warp_factor = 0.8 + random.random() * 0.4  # 80%–120% speed
            new_t = torch.linspace(0, T-1, int(T * warp_factor), device=x.device)
            new_t = new_t * (T-1) / (T * warp_factor - 1)
            x = torch.nn.functional.interpolate(
                x.permute(0, 2, 1).unsqueeze(-1),
                size=int(T * warp_factor),
                mode='linear',
                align_corners=False
            ).squeeze(-1).permute(0, 2, 1)
            # Pad/crop back to original length
            if x.shape[1] > T:
                x = x[:, :T, :]
            elif x.shape[1] < T:
                x = F.pad(x, (0, 0, 0, T - x.shape[1]))

        # 5. 3D rotation (rigid body augmentation)
        if random.random() < self.p_rotation * intensity:
            angle = random.uniform(-15, 15) * math.pi / 180 * intensity
            axis = torch.randn(3, device=x.device)
            axis = axis / (axis.norm() + 1e-8)
            cos = math.cos(angle)
            sin = math.sin(angle)
            ux, uy, uz = axis
            rot = torch.tensor([
                [cos + ux**2*(1-cos), ux*uy*(1-cos)-uz*sin, ux*uz*(1-cos)+uy*sin],
                [uy*ux*(1-cos)+uz*sin, cos + uy**2*(1-cos), uy*uz*(1-cos)-ux*sin],
                [uz*ux*(1-cos)-uy*sin, uz*uy*(1-cos)+ux*sin, cos + uz**2*(1-cos)]
            ], device=x.device, dtype=x.dtype)
            # Apply rotation to every (x,y,z) triplet
            x_reshaped = x.view(B, T, -1, 3)  # (B,T,N_points,3)
            x_rot = torch.einsum("...ij,jk->...ik", x_reshaped, rot)
            x = x_rot.view(B, T, L)

        # 6. Global scaling
        if random.random() < self.p_scaling * intensity:
            scale = random.uniform(0.85, 1.15)
            x = x * scale

        if squeeze:
            x = x.squeeze(0)

        return x

def sampleFromProbs(probs: torch.Tensor) -> int:
    """Helper to sample index from probability distribution"""
    cum = torch.cumsum(probs, dim=0)
    r = torch.rand(1, device=probs.device)
    return torch.searchsorted(cum, r).item()
