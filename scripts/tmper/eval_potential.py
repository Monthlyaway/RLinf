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

"""Evaluate a trained TMPER potential network with three acceptance tests.

Test 1 (Progress Bar): Potential must be strictly increasing on a held-out
    successful trajectory.
Test 2 (Drop Test): Potential should drop when the trajectory is spliced
    with a rewind segment simulating a "drop".
Test 3 (Idle Test): Potential should remain flat near 0 for the first few
    frames where the robot has not yet moved.

Usage:
    python scripts/tmper/eval_potential.py \
        --checkpoint data/checkpoints/tmper/potential_phi.pt \
        --data-dir data/demos/PickCube-v1/raw_pkl \
        --output-dir data/eval/tmper
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rlinf.algorithms.rewards.tmper.potential_net import PotentialNetwork


def load_model(checkpoint_path: str, device: torch.device) -> PotentialNetwork:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    net_keys = {"state_dim", "image_size", "latent_dim", "state_latent_dim"}
    model = PotentialNetwork(**{k: v for k, v in cfg.items() if k in net_keys})
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(
        f"Loaded checkpoint from epoch {ckpt['epoch']}, "
        f"val_acc={ckpt.get('val_pairwise_acc', 'N/A')}"
    )
    return model


def compute_potentials(
    model: PotentialNetwork,
    images: list[np.ndarray],
    states: list[np.ndarray],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Run model inference on a sequence of (image, state) pairs."""
    all_phi = []
    n = len(images)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        img_batch = torch.from_numpy(np.stack(images[start:end])).to(device)
        state_batch = torch.from_numpy(np.stack(states[start:end])).float().to(device)
        with torch.no_grad():
            phi = model(img_batch, state_batch)
        all_phi.append(phi.cpu().numpy())
    return np.concatenate(all_phi)


def test_progress_bar(
    model: PotentialNetwork,
    traj: dict,
    device: torch.device,
    output_dir: str,
    eps: float = 1e-3,
) -> bool:
    """Test 1: Potential must be monotonically increasing on a successful trajectory.

    Args:
        eps: Tolerance for floating-point noise in saturated sigmoid regions.
            Violations smaller than eps are counted separately as numerical noise.
    """
    images = traj["images"]
    states = traj["observations"]
    phi = compute_potentials(model, images, states, device)

    T = len(phi)
    diffs = np.diff(phi)
    all_violations = int((diffs < 0).sum())
    real_violations = int((diffs < -eps).sum())
    noise_violations = all_violations - real_violations
    is_pass = real_violations == 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(range(T), phi, "b-o", markersize=3, label="Φ(s)")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Potential Φ")
    ax1.set_title(
        f"Test 1: Progress Bar — {'PASS' if is_pass else 'FAIL'}\n"
        f"Real violations (>{eps}): {real_violations}, "
        f"Noise (<{eps}): {noise_violations}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    colors = []
    for d in diffs:
        if d >= 0:
            colors.append("green")
        elif d >= -eps:
            colors.append("orange")
        else:
            colors.append("red")
    ax2.bar(range(T - 1), diffs, color=colors)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Frame transition")
    ax2.set_ylabel("ΔΦ")
    ax2.set_title("Frame-to-frame potential change (red=real violation, orange=noise)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test1_progress_bar.png"), dpi=150)
    plt.close()

    print(f"\n{'=' * 60}")
    print("TEST 1: Progress Bar")
    print(f"  Φ range: [{phi[0]:.4f}, {phi[-1]:.4f}]")
    print(f"  Real violations (delta < -{eps}): {real_violations}/{T - 1}")
    print(f"  Noise violations (delta in [-{eps}, 0)): {noise_violations}/{T - 1}")
    print(f"  Result: {'PASS ✓' if is_pass else 'FAIL ✗'}")
    print(f"{'=' * 60}")

    return is_pass


def test_drop(
    model: PotentialNetwork,
    traj: dict,
    device: torch.device,
    output_dir: str,
) -> bool:
    """Test 2: Potential should drop when trajectory is spliced with rewind.

    Constructs a synthetic trajectory that goes forward to ~60% completion,
    then reverses 20 steps to simulate a "drop" event.
    """
    T = len(traj["observations"]) - 1
    cut_point = int(T * 0.6)
    rewind_steps = min(20, cut_point - 1)

    fwd_obs = traj["observations"][: cut_point + 1]
    fwd_img = traj["images"][: cut_point + 1]

    rew_obs = [traj["observations"][cut_point - j] for j in range(1, rewind_steps + 1)]
    rew_img = [traj["images"][cut_point - j] for j in range(1, rewind_steps + 1)]

    all_obs = fwd_obs + rew_obs
    all_img = fwd_img + rew_img
    all_progress = list(range(1, cut_point + 2)) + list(
        range(cut_point, cut_point - rewind_steps, -1)
    )

    phi = compute_potentials(model, all_img, all_obs, device)
    progress = np.array(all_progress)

    peak_idx = cut_point  # 0-indexed position of the cut point in the combined seq
    phi_at_peak = phi[peak_idx]
    end_idx = len(phi) - 1
    phi_at_end = phi[end_idx]
    dropped = phi_at_end < phi_at_peak

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(range(len(phi)), phi, "b-o", markersize=3, label="Φ(s)")
    ax1.axvline(
        x=peak_idx,
        color="red",
        linestyle="--",
        label=f"Cut point (frame {peak_idx})",
    )
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Potential Φ")
    ax1.set_title(
        f"Test 2: Drop Test — {'DROP DETECTED' if dropped else 'NO DROP'}\n"
        f"Φ at peak: {phi_at_peak:.4f}, Φ at rewind end: {phi_at_end:.4f}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(len(progress)), progress, "g-s", markersize=3, label="Progress P")
    ax2.axvline(x=peak_idx, color="red", linestyle="--")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Progress step")
    ax2.set_title("Progress labels (forward then rewind)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test2_drop_test.png"), dpi=150)
    plt.close()

    print(f"\n{'=' * 60}")
    print("TEST 2: Drop Test (soft criterion)")
    print(
        f"  Cut at frame {peak_idx} (~60% of trajectory), rewind {rewind_steps} steps"
    )
    print(f"  Φ at peak: {phi_at_peak:.4f}")
    print(f"  Φ at rewind end: {phi_at_end:.4f}")
    print(f"  Drop detected: {'Yes' if dropped else 'No'}")
    print(f"  Result: {'EXPECTED BEHAVIOR ✓' if dropped else 'NEEDS INVESTIGATION'}")
    print(f"{'=' * 60}")

    return dropped


def test_idle(
    model: PotentialNetwork,
    traj: dict,
    device: torch.device,
    output_dir: str,
    idle_frames: int = 15,
    max_std: float = 0.05,
) -> bool:
    """Test 3: Potential should stay flat near 0 for initial idle frames."""
    n_frames = min(idle_frames, len(traj["observations"]))
    images = traj["images"][:n_frames]
    states = traj["observations"][:n_frames]
    phi = compute_potentials(model, images, states, device)

    phi_std = float(np.std(phi))
    phi_mean = float(np.mean(phi))
    is_flat = phi_std < max_std

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(n_frames), phi, "b-o", markersize=5, label="Φ(s)")
    ax.axhline(y=phi_mean, color="orange", linestyle="--", label=f"mean={phi_mean:.4f}")
    ax.fill_between(
        range(n_frames),
        phi_mean - phi_std,
        phi_mean + phi_std,
        alpha=0.2,
        color="orange",
        label=f"±std={phi_std:.4f}",
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Potential Φ")
    ax.set_title(
        f"Test 3: Idle Test — {'PASS' if is_flat else 'FAIL'}\n"
        f"std={phi_std:.4f} (threshold={max_std})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test3_idle_test.png"), dpi=150)
    plt.close()

    print(f"\n{'=' * 60}")
    print("TEST 3: Idle Test")
    print(f"  First {n_frames} frames: mean={phi_mean:.4f}, std={phi_std:.4f}")
    print(f"  Flatness threshold: std < {max_std}")
    print(f"  Result: {'PASS ✓' if is_flat else 'FAIL ✗'}")
    print(f"{'=' * 60}")

    return is_flat


def main():
    parser = argparse.ArgumentParser(description="Evaluate TMPER potential network")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/checkpoints/tmper/potential_phi.pt",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/demos/PickCube-v1/raw_pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval/tmper",
    )
    parser.add_argument(
        "--traj-index", type=int, default=0, help="Index of the trajectory to evaluate"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, device)

    pkl_files = sorted(Path(args.data_dir).glob("*.pkl"))
    if args.traj_index >= len(pkl_files):
        raise ValueError(
            f"traj_index={args.traj_index} but only {len(pkl_files)} pkls found"
        )

    with open(pkl_files[args.traj_index], "rb") as f:
        traj = pickle.load(f)

    if "images" not in traj:
        raise ValueError(
            "Trajectory does not contain 'images' field. "
            "Re-run convert_h5_to_pkl.py with --rgb-h5-path."
        )

    print(f"\nEvaluating on: {pkl_files[args.traj_index].name}")
    print(f"  T={traj['elapsed_steps']}, success={traj['success']}")

    results = {}
    results["test1_progress"] = test_progress_bar(model, traj, device, args.output_dir)
    results["test2_drop"] = test_drop(model, traj, device, args.output_dir)
    results["test3_idle"] = test_idle(model, traj, device, args.output_dir)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(
        f"  Test 1 (Progress Bar):  {'PASS' if results['test1_progress'] else 'FAIL'}"
    )
    print(
        f"  Test 2 (Drop Test):     {'PASS' if results['test2_drop'] else 'NEEDS WORK'}"
    )
    print(f"  Test 3 (Idle Test):     {'PASS' if results['test3_idle'] else 'FAIL'}")
    print(f"{'=' * 60}")
    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
