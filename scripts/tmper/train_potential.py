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

"""Offline training script for the TMPER potential network.

Usage:
    python scripts/tmper/train_potential.py \
        --data-dir data/demos/PickCube-v1/raw_pkl \
        --output-dir data/checkpoints/tmper
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rlinf.algorithms.rewards.tmper.potential_net import (
    PotentialNetwork,
    PotentialPairDataset,
    boundary_calibration_loss,
    ranking_loss,
    smoothness_loss,
)
from rlinf.data.rewind_augmentation import RewindAugmentedDataset


def build_datasets(
    data_dir: str,
    num_augmentations: int,
    val_ratio: float,
    pairs_per_traj: int,
    seed: int,
) -> tuple[PotentialPairDataset, PotentialPairDataset, RewindAugmentedDataset]:
    """Build train/val pair datasets with trajectory-level split.

    Splits raw pkl files into train/val *before* augmentation so validation
    trajectories are truly unseen.
    """
    pkl_files = sorted(Path(data_dir).glob("*.pkl"))
    all_trajs = []
    for p in pkl_files:
        with open(p, "rb") as f:
            traj = pickle.load(f)
        if traj.get("success", False):
            all_trajs.append(traj)

    n_total = len(all_trajs)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    print(f"Trajectory split: {n_train} train, {n_val} val (total {n_total})")

    from rlinf.data.rewind_augmentation import rewind_augment_trajectory

    aug_rng = np.random.default_rng(seed)

    def _augment_subset(subset_indices):
        trajectories = []
        for traj_idx in subset_indices:
            augmented = rewind_augment_trajectory(
                all_trajs[traj_idx],
                traj_id=int(traj_idx),
                num_augmentations=num_augmentations,
                rng=aug_rng,
            )
            trajectories.extend(augmented)
        return trajectories

    train_trajs = _augment_subset(train_indices)
    val_trajs = _augment_subset(val_indices)

    class _ListDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    train_ds = PotentialPairDataset(
        _ListDataset(train_trajs), pairs_per_traj=pairs_per_traj, seed=seed
    )
    val_ds = PotentialPairDataset(
        _ListDataset(val_trajs), pairs_per_traj=pairs_per_traj, seed=seed + 1
    )

    full_aug_ds = RewindAugmentedDataset(data_dir, num_augmentations, seed)

    return train_ds, val_ds, full_aug_ds


def evaluate(
    model: PotentialNetwork,
    val_loader: DataLoader,
    device: torch.device,
    c: float,
) -> dict:
    """Compute validation metrics."""
    model.eval()
    total_loss = 0.0
    correct_pairs = 0
    total_pairs = 0

    with torch.no_grad():
        for batch in val_loader:
            img_u = batch["img_u"].to(device)
            state_u = batch["state_u"].to(device)
            img_v = batch["img_v"].to(device)
            state_v = batch["state_v"].to(device)
            prog_u = batch["progress_u"].to(device)
            prog_v = batch["progress_v"].to(device)
            max_prog = batch["max_progress"].to(device)

            phi_u = model(img_u, state_u)
            phi_v = model(img_v, state_v)

            L_rank = ranking_loss(phi_v, phi_u, prog_v, prog_u, max_prog, c)

            phi_start = model(
                batch["img_start"].to(device), batch["state_start"].to(device)
            )
            phi_success = model(
                batch["img_success"].to(device), batch["state_success"].to(device)
            )
            L_bc = boundary_calibration_loss(phi_start, phi_success)

            phi_adj_u = model(
                batch["img_adj_u"].to(device), batch["state_adj_u"].to(device)
            )
            phi_adj_v = model(
                batch["img_adj_v"].to(device), batch["state_adj_v"].to(device)
            )
            L_smooth = smoothness_loss(phi_adj_v, phi_adj_u)

            total_loss += (L_rank + L_bc + 0.1 * L_smooth).item()

            correct_pairs += (phi_v > phi_u).sum().item()
            total_pairs += phi_v.shape[0]

    n_batches = max(1, len(val_loader))
    return {
        "val_loss": total_loss / n_batches,
        "val_pairwise_acc": correct_pairs / max(1, total_pairs),
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, _ = build_datasets(
        args.data_dir,
        args.num_augmentations,
        args.val_ratio,
        args.pairs_per_traj,
        args.seed,
    )

    print(f"Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = PotentialNetwork(
        state_dim=args.state_dim,
        image_size=64,
        latent_dim=256,
        state_latent_dim=128,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rank_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_smooth_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for batch in train_loader:
            img_u = batch["img_u"].to(device)
            state_u = batch["state_u"].to(device)
            img_v = batch["img_v"].to(device)
            state_v = batch["state_v"].to(device)
            prog_u = batch["progress_u"].to(device)
            prog_v = batch["progress_v"].to(device)
            max_prog = batch["max_progress"].to(device)

            phi_u = model(img_u, state_u)
            phi_v = model(img_v, state_v)

            L_rank = ranking_loss(phi_v, phi_u, prog_v, prog_u, max_prog, args.c)

            phi_start = model(
                batch["img_start"].to(device), batch["state_start"].to(device)
            )
            phi_success = model(
                batch["img_success"].to(device), batch["state_success"].to(device)
            )
            L_bc = boundary_calibration_loss(phi_start, phi_success)

            phi_adj_u = model(
                batch["img_adj_u"].to(device), batch["state_adj_u"].to(device)
            )
            phi_adj_v = model(
                batch["img_adj_v"].to(device), batch["state_adj_v"].to(device)
            )
            L_smooth = smoothness_loss(phi_adj_v, phi_adj_u)

            loss = (
                args.lambda_rank * L_rank
                + args.lambda_bc * L_bc
                + args.lambda_smooth * L_smooth
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rank_loss += L_rank.item()
            epoch_bc_loss += L_bc.item()
            epoch_smooth_loss += L_smooth.item()
            epoch_correct += (phi_v > phi_u).sum().item()
            epoch_total += phi_v.shape[0]

        n_batches = max(1, len(train_loader))
        dt = time.time() - t0
        train_acc = epoch_correct / max(1, epoch_total)

        val_metrics = evaluate(model, val_loader, device, args.c)

        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} "
                f"({dt:.1f}s) | "
                f"loss={epoch_loss / n_batches:.4f} "
                f"[rank={epoch_rank_loss / n_batches:.4f} "
                f"bc={epoch_bc_loss / n_batches:.4f} "
                f"smooth={epoch_smooth_loss / n_batches:.4f}] | "
                f"train_acc={train_acc:.3f} | "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"val_acc={val_metrics['val_pairwise_acc']:.3f}"
            )

        if val_metrics["val_pairwise_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_pairwise_acc"]
            ckpt_path = os.path.join(args.output_dir, "potential_phi.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_pairwise_acc": best_val_acc,
                    "config": {
                        "state_dim": args.state_dim,
                        "image_size": 64,
                        "latent_dim": 256,
                        "state_latent_dim": 128,
                        "c": args.c,
                        "lambda_rank": args.lambda_rank,
                        "lambda_bc": args.lambda_bc,
                        "lambda_smooth": args.lambda_smooth,
                    },
                },
                ckpt_path,
            )
            if (epoch + 1) % args.log_interval == 0 or epoch == 0:
                print(f"  -> Saved best checkpoint (val_acc={best_val_acc:.3f})")

    print(f"\nTraining complete. Best val pairwise accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to: {os.path.join(args.output_dir, 'potential_phi.pt')}")


def main():
    parser = argparse.ArgumentParser(
        description="Train TMPER potential network offline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing raw pkl trajectories (with images)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/checkpoints/tmper",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--c", type=float, default=5.0, help="Margin scaling coefficient"
    )
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--num-augmentations", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--pairs-per-traj", type=int, default=50)
    parser.add_argument("--state-dim", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Print log every N epochs"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
