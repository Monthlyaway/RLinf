#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Hyperparameter sweep for TMPER potential network.

Trains multiple configurations in sequence, evaluates each, and saves
plots + a summary CSV for manual comparison.

Usage:
    source .venv/bin/activate
    python scripts/tmper/sweep_potential.py \
        --data-dir data/demos/PickCube-v1/raw_pkl \
        --sweep-dir data/sweep_tmper \
        2>&1 | tee data/sweep_tmper/sweep_stdout.log
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from rlinf.data.rewind_augmentation import rewind_augment_trajectory


def load_raw_trajectories(data_dir: str):
    pkl_files = sorted(Path(data_dir).glob("*.pkl"))
    trajs = []
    for p in pkl_files:
        with open(p, "rb") as f:
            traj = pickle.load(f)
        if traj.get("success", False):
            trajs.append(traj)
    return trajs


def build_train_val(
    all_trajs: list[dict],
    num_augmentations: int,
    val_ratio: float,
    pairs_per_traj: int,
    seed: int,
):
    n_total = len(all_trajs)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

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
    return train_ds, val_ds


def evaluate_model(model, val_loader, device, c):
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


def train_one_config(
    config: dict,
    train_ds: PotentialPairDataset,
    val_ds: PotentialPairDataset,
    device: torch.device,
    output_dir: str,
    run_name: str,
) -> dict:
    """Train one configuration and return metrics."""
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = PotentialNetwork(
        state_dim=config["state_dim"],
        image_size=64,
        latent_dim=256,
        state_latent_dim=128,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    ckpt_dir = os.path.join(output_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = 0
    loss_history = []

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_rank = 0.0
        epoch_bc = 0.0
        epoch_smooth = 0.0
        epoch_correct = 0
        epoch_total = 0

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

            L_rank = ranking_loss(phi_v, phi_u, prog_v, prog_u, max_prog, config["c"])

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
                config["lambda_rank"] * L_rank
                + config["lambda_bc"] * L_bc
                + config["lambda_smooth"] * L_smooth
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rank += L_rank.item()
            epoch_bc += L_bc.item()
            epoch_smooth += L_smooth.item()
            epoch_correct += (phi_v > phi_u).sum().item()
            epoch_total += phi_v.shape[0]

        n_batches = max(1, len(train_loader))
        train_acc = epoch_correct / max(1, epoch_total)
        val_metrics = evaluate_model(model, val_loader, device, config["c"])

        loss_history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss / n_batches,
                "rank": epoch_rank / n_batches,
                "bc": epoch_bc / n_batches,
                "smooth": epoch_smooth / n_batches,
                "train_acc": train_acc,
                "val_acc": val_metrics["val_pairwise_acc"],
            }
        )

        if val_metrics["val_pairwise_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_pairwise_acc"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_pairwise_acc": best_val_acc,
                    "config": {
                        "state_dim": config["state_dim"],
                        "image_size": 64,
                        "latent_dim": 256,
                        "state_latent_dim": 128,
                        "c": config["c"],
                        "lambda_rank": config["lambda_rank"],
                        "lambda_bc": config["lambda_bc"],
                        "lambda_smooth": config["lambda_smooth"],
                    },
                },
                os.path.join(ckpt_dir, "potential_phi.pt"),
            )

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"  [{run_name}] Epoch {epoch + 1:3d}/{config['epochs']} | "
                f"loss={epoch_loss / n_batches:.4f} "
                f"[R={epoch_rank / n_batches:.4f} "
                f"B={epoch_bc / n_batches:.4f} "
                f"S={epoch_smooth / n_batches:.4f}] | "
                f"tacc={train_acc:.3f} vacc={val_metrics['val_pairwise_acc']:.3f}"
            )

    return {
        "run_name": run_name,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "ckpt_dir": ckpt_dir,
        "loss_history": loss_history,
    }


def eval_and_plot(
    ckpt_path: str,
    traj: dict,
    device: torch.device,
    output_dir: str,
    run_name: str,
    config: dict,
) -> dict:
    """Evaluate a checkpoint and save test plots with run_name in filename."""
    from rlinf.algorithms.rewards.tmper.potential_net import PotentialNetwork

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    net_keys = {"state_dim", "image_size", "latent_dim", "state_latent_dim"}
    model = PotentialNetwork(**{k: v for k, v in cfg.items() if k in net_keys})
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    images = traj["images"]
    states = traj["observations"]

    all_phi = []
    n = len(images)
    for start in range(0, n, 32):
        end = min(start + 32, n)
        img_batch = torch.from_numpy(np.stack(images[start:end])).to(device)
        state_batch = torch.from_numpy(np.stack(states[start:end])).float().to(device)
        with torch.no_grad():
            phi = model(img_batch, state_batch)
        all_phi.append(phi.cpu().numpy())
    phi = np.concatenate(all_phi)

    T = len(phi)
    diffs = np.diff(phi)
    real_violations = int((diffs < -1e-3).sum())
    monotonicity = 1.0 - real_violations / max(1, T - 1)

    phi_range = float(phi[-1] - phi[0])
    phi_std_diff = float(np.std(diffs))
    phi_max_jump = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0

    smoothness_score = 1.0 / (1.0 + phi_std_diff * 100)

    # --- Plot ---
    config_str = (
        f"c={config['c']}, λs={config['lambda_smooth']}, "
        f"lr={config['lr']}, λbc={config['lambda_bc']}"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(range(T), phi, "b-o", markersize=2, linewidth=1.5)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Potential Φ")
    ax1.set_title(
        f"{run_name}\n{config_str}\n"
        f"mono={monotonicity:.3f}, smooth={smoothness_score:.3f}, "
        f"range=[{phi[0]:.3f}, {phi[-1]:.3f}], violations={real_violations}"
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    colors = ["green" if d >= 0 else "red" for d in diffs]
    ax2.bar(range(T - 1), diffs, color=colors, width=1.0)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Frame transition")
    ax2.set_ylabel("ΔΦ")
    ax2.set_title("Frame-to-frame potential change")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{run_name}_progress.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()

    return {
        "monotonicity": monotonicity,
        "smoothness_score": smoothness_score,
        "phi_range": phi_range,
        "phi_start": float(phi[0]),
        "phi_end": float(phi[-1]),
        "real_violations": real_violations,
        "phi_std_diff": phi_std_diff,
        "phi_max_jump": phi_max_jump,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep hyperparams for TMPER potential"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--sweep-dir", type=str, default="data/sweep_tmper")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traj-index", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.sweep_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Sweep output: {args.sweep_dir}")

    # Load raw trajectories once
    all_trajs = load_raw_trajectories(args.data_dir)
    print(f"Loaded {len(all_trajs)} successful trajectories")

    # Load eval trajectory
    pkl_files = sorted(Path(args.data_dir).glob("*.pkl"))
    with open(pkl_files[args.traj_index], "rb") as f:
        eval_traj = pickle.load(f)
    print(
        f"Eval trajectory: {pkl_files[args.traj_index].name}, T={eval_traj['elapsed_steps']}"
    )

    # ===== SWEEP GRID =====
    # H1: c controls whether margins are achievable within (0,1) output range
    # H2: lambda_smooth controls resistance to jumps
    # H3: lr controls convergence speed
    sweep_grid = {
        "c": [0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
        "lambda_smooth": [0.1, 0.5, 1.0, 3.0],
        "lr": [5e-4, 1e-3],
    }

    fixed_params = {
        "lambda_rank": 1.0,
        "lambda_bc": 1.0,
        "batch_size": 64,
        "state_dim": 42,
        "num_augmentations": 5,
        "val_ratio": 0.2,
        "pairs_per_traj": 50,
    }

    configs = []
    for c, ls, lr in itertools.product(
        sweep_grid["c"], sweep_grid["lambda_smooth"], sweep_grid["lr"]
    ):
        name = f"c{c}_ls{ls}_lr{lr}"
        cfg = {
            **fixed_params,
            "c": c,
            "lambda_smooth": ls,
            "lr": lr,
            "epochs": args.epochs,
        }
        configs.append((name, cfg))

    total_runs = len(configs)
    print(f"\n{'=' * 70}")
    print(f"Total configurations to sweep: {total_runs}")
    print(
        f"Grid: c={sweep_grid['c']}, λs={sweep_grid['lambda_smooth']}, lr={sweep_grid['lr']}"
    )
    print(
        f"Fixed: epochs={args.epochs}, λr={fixed_params['lambda_rank']}, λbc={fixed_params['lambda_bc']}"
    )
    print(f"{'=' * 70}\n")

    # Build datasets once (they depend only on augmentations and seed)
    train_ds, val_ds = build_train_val(
        all_trajs,
        num_augmentations=fixed_params["num_augmentations"],
        val_ratio=fixed_params["val_ratio"],
        pairs_per_traj=fixed_params["pairs_per_traj"],
        seed=args.seed,
    )
    print(f"Dataset: {len(train_ds)} train pairs, {len(val_ds)} val pairs\n")

    # CSV for summary
    csv_path = os.path.join(args.sweep_dir, "sweep_results.csv")
    csv_fields = [
        "run_name",
        "c",
        "lambda_smooth",
        "lr",
        "best_val_acc",
        "best_epoch",
        "monotonicity",
        "smoothness_score",
        "phi_range",
        "phi_start",
        "phi_end",
        "real_violations",
        "phi_std_diff",
        "phi_max_jump",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    results = []
    for i, (run_name, config) in enumerate(configs):
        print(f"\n{'=' * 70}")
        print(f"[{i + 1}/{total_runs}] {run_name}")
        print(f"  c={config['c']}, λs={config['lambda_smooth']}, lr={config['lr']}")
        print(f"{'=' * 70}")

        t0 = time.time()
        train_result = train_one_config(
            config, train_ds, val_ds, device, args.sweep_dir, run_name
        )
        train_time = time.time() - t0

        ckpt_path = os.path.join(train_result["ckpt_dir"], "potential_phi.pt")
        if os.path.exists(ckpt_path):
            eval_result = eval_and_plot(
                ckpt_path, eval_traj, device, args.sweep_dir, run_name, config
            )
        else:
            eval_result = {
                "monotonicity": 0,
                "smoothness_score": 0,
                "phi_range": 0,
                "phi_start": 0,
                "phi_end": 0,
                "real_violations": -1,
                "phi_std_diff": 0,
                "phi_max_jump": 0,
            }

        row = {
            "run_name": run_name,
            "c": config["c"],
            "lambda_smooth": config["lambda_smooth"],
            "lr": config["lr"],
            "best_val_acc": train_result["best_val_acc"],
            "best_epoch": train_result["best_epoch"],
            **eval_result,
        }
        csv_writer.writerow(row)
        csv_file.flush()
        results.append(row)

        print(
            f"  -> Done in {train_time:.0f}s | "
            f"vacc={train_result['best_val_acc']:.3f} "
            f"mono={eval_result['monotonicity']:.3f} "
            f"smooth={eval_result['smoothness_score']:.3f} "
            f"violations={eval_result['real_violations']}"
        )

    csv_file.close()

    # Print sorted summary
    print(f"\n\n{'=' * 70}")
    print("SWEEP SUMMARY (sorted by smoothness_score descending)")
    print(f"{'=' * 70}")
    sorted_results = sorted(results, key=lambda r: r["smoothness_score"], reverse=True)
    print(
        f"{'Run':<30} {'c':>5} {'λs':>5} {'lr':>8} {'vacc':>6} {'mono':>6} {'smooth':>7} {'viol':>5} {'range':>7}"
    )
    print("-" * 90)
    for r in sorted_results:
        print(
            f"{r['run_name']:<30} "
            f"{r['c']:>5} {r['lambda_smooth']:>5} {r['lr']:>8} "
            f"{r['best_val_acc']:>6.3f} "
            f"{r['monotonicity']:>6.3f} "
            f"{r['smoothness_score']:>7.3f} "
            f"{r['real_violations']:>5} "
            f"{r['phi_range']:>7.3f}"
        )

    print(f"\nResults CSV: {csv_path}")
    print(f"Plots saved in: {args.sweep_dir}/")
    print(f"Each run's checkpoint in: {args.sweep_dir}/<run_name>/potential_phi.pt")


if __name__ == "__main__":
    main()
