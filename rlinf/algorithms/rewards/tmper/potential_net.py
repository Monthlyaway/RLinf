# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Potential network, loss functions, and pair-sampling dataset for TMPER.

The potential function Phi_theta(s) maps (image, state) -> scalar in (0, 1),
trained offline with pairwise ranking loss to satisfy temporal monotonicity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from rlinf.models.embodiment.modules.compact_encoders import LightweightImageEncoder64


class PotentialNetwork(nn.Module):
    """Vision + state fusion network outputting scalar potential in (0, 1).

    Architecture (late fusion):
        Vision:  RGB (128x128x3) -> resize 64x64 -> LightweightCNN -> latent_dim
        State:   (state_dim,) -> MLP -> state_latent_dim
        Fusion:  concat -> MLP -> Sigmoid -> scalar
    """

    def __init__(
        self,
        state_dim: int = 42,
        image_size: int = 64,
        latent_dim: int = 256,
        state_latent_dim: int = 128,
    ):
        super().__init__()
        self.image_size = image_size

        self.vision_encoder = LightweightImageEncoder64(
            num_images=1, latent_dim=latent_dim, image_size=image_size
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_latent_dim),
            nn.ReLU(),
            nn.Linear(state_latent_dim, state_latent_dim),
            nn.ReLU(),
        )

        fusion_dim = latent_dim + state_latent_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.state_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.fusion_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Compute potential value Phi(s) in (0, 1).

        Args:
            images: uint8 or float tensor. If uint8, expected shape [B, H, W, 3]
                (HWC) and will be normalized. If float, expected [B, 1, 3, h, w].
            states: [B, state_dim] float tensor.

        Returns:
            Scalar potential [B] in (0, 1).
        """
        img_features = self._encode_images(images)
        state_features = self.state_encoder(states)
        fused = torch.cat([img_features, state_features], dim=-1)
        logit = self.fusion_head(fused).squeeze(-1)
        return torch.sigmoid(logit)

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            x = images.float() / 255.0
            if x.dim() == 4 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)  # [B,H,W,3] -> [B,3,H,W]
        else:
            x = images
            if x.dim() == 4 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)

        if x.dim() == 4:
            # [B, 3, H, W] -> resize -> [B, 1, 3, h, w]
            if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
                x = F.interpolate(
                    x,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            x = x.unsqueeze(1)  # [B, 1, 3, h, w]

        return self.vision_encoder(x)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def ranking_loss(
    phi_v: torch.Tensor,
    phi_u: torch.Tensor,
    progress_v: torch.Tensor,
    progress_u: torch.Tensor,
    max_progress: torch.Tensor,
    c: float = 5.0,
) -> torch.Tensor:
    """Pairwise logistic ranking loss with dynamic margin.

    Expects P_v > P_u for all samples in the batch.

    Args:
        phi_v: Potential of the "later" state [B].
        phi_u: Potential of the "earlier" state [B].
        progress_v: Progress label of state v [B].
        progress_u: Progress label of state u [B].
        max_progress: Maximum progress value per trajectory [B].
        c: Margin scaling coefficient.
    """
    margin = c * (progress_v - progress_u).float() / max_progress.float()
    return -F.logsigmoid(phi_v - phi_u - margin).mean()


def boundary_calibration_loss(
    phi_start: torch.Tensor, phi_success: torch.Tensor
) -> torch.Tensor:
    """MSE loss anchoring start -> 0 and success -> 1."""
    return (phi_start**2).mean() + ((phi_success - 1.0) ** 2).mean()


def smoothness_loss(phi_adj_v: torch.Tensor, phi_adj_u: torch.Tensor) -> torch.Tensor:
    """Squared-difference penalty for adjacent frames (|P_u - P_v| == 1)."""
    return ((phi_adj_v - phi_adj_u) ** 2).mean()


# ---------------------------------------------------------------------------
# Pair-sampling dataset
# ---------------------------------------------------------------------------


class PotentialPairDataset(Dataset):
    """Samples (state_u, state_v) pairs from augmented trajectories for training.

    Each item yields a pair with P_v > P_u from the same trajectory, plus
    anchor frames (start and success) and an adjacent pair for smoothness loss.

    Args:
        augmented_dataset: A ``RewindAugmentedDataset`` (or list of traj dicts).
        pairs_per_traj: Number of pairs to sample per trajectory per epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        augmented_dataset,
        pairs_per_traj: int = 50,
        seed: int = 42,
    ):
        self.trajectories = []
        self.success_anchor_cache: dict[int, dict] = {}

        for i in range(len(augmented_dataset)):
            traj = augmented_dataset[i]
            self.trajectories.append(traj)
            if traj["is_success_anchor"].any():
                self.success_anchor_cache[i] = traj

        if not self.success_anchor_cache:
            raise ValueError("No trajectory with is_success_anchor=True found")

        self.pairs_per_traj = pairs_per_traj
        self._seed = seed
        self._rng = torch.Generator().manual_seed(seed)
        self._total = len(self.trajectories) * pairs_per_traj

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict:
        traj_idx = idx // self.pairs_per_traj
        traj = self.trajectories[traj_idx]

        progress = traj["progress_step"]  # [L]
        L = len(progress)

        # Sample two distinct frame indices, ensure P_v > P_u
        u = torch.randint(0, L, (1,), generator=self._rng).item()
        v = torch.randint(0, L, (1,), generator=self._rng).item()
        while v == u:
            v = torch.randint(0, L, (1,), generator=self._rng).item()

        p_u, p_v = progress[u].item(), progress[v].item()
        if p_u > p_v:
            u, v = v, u
            p_u, p_v = p_v, p_u
        elif p_u == p_v:
            # Rare edge case; pick another v
            for offset in range(1, L):
                v2 = (u + offset) % L
                if progress[v2].item() != p_u:
                    if progress[v2].item() > p_u:
                        v = v2
                        p_v = progress[v].item()
                    else:
                        u, v = v2, u
                        p_u, p_v = progress[u].item(), progress[v].item()
                    break

        # Adjacent pair for smoothness loss
        adj_u = torch.randint(0, max(1, L - 1), (1,), generator=self._rng).item()
        adj_v = adj_u + 1 if adj_u + 1 < L else adj_u - 1

        # Start anchor: first frame of this trajectory
        start_idx = 0

        # Success anchor: find a trajectory that has is_success_anchor=True
        success_traj_idx = next(iter(self.success_anchor_cache))
        success_traj = self.success_anchor_cache[success_traj_idx]
        success_idx = int(
            success_traj["is_success_anchor"].nonzero(as_tuple=True)[0][0]
        )

        result = {
            "img_u": traj["images"][u],  # [H, W, 3] uint8
            "state_u": traj["states"][u],  # [state_dim]
            "img_v": traj["images"][v],
            "state_v": traj["states"][v],
            "progress_u": torch.tensor(p_u, dtype=torch.long),
            "progress_v": torch.tensor(p_v, dtype=torch.long),
            "max_progress": torch.tensor(traj["max_progress"], dtype=torch.long),
            # Adjacent pair
            "img_adj_u": traj["images"][adj_u],
            "state_adj_u": traj["states"][adj_u],
            "img_adj_v": traj["images"][adj_v],
            "state_adj_v": traj["states"][adj_v],
            # Anchors
            "img_start": traj["images"][start_idx],
            "state_start": traj["states"][start_idx],
            "img_success": success_traj["images"][success_idx],
            "state_success": success_traj["states"][success_idx],
        }
        return result
