# Copyright 2025 The RLinf Authors.
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

"""Render an augmented rewind trajectory as an .mp4 with progress_step overlay.

Loads a raw trajectory pkl (containing env_states) and an augmented trajectory
pkl (containing progress_step labels), then re-renders each frame in ManiSkill
by restoring the physics state corresponding to each progress_step index.

Usage:
    python scripts/tmper/visualize_rewind.py \
        --raw-pkl data/demos/PickCube-v1/raw_pkl/traj_0.pkl \
        --aug-pkl data/demos/PickCube-v1/augmented/aug_traj_0001.pkl \
        --output data/demos/PickCube-v1/debug_rewind.mp4

    # Or generate augmentation on-the-fly from raw pkl:
    python scripts/tmper/visualize_rewind.py \
        --raw-pkl data/demos/PickCube-v1/raw_pkl/traj_0.pkl \
        --aug-index 2 \
        --output data/demos/PickCube-v1/debug_rewind.mp4
"""

import argparse
import os
import pickle
import sys

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rlinf.data.rewind_augmentation import rewind_augment_trajectory


def load_raw_trajectory(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_aug_trajectory(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def build_state_dict(env_states: dict, frame_idx: int) -> dict:
    """Reconstruct a ManiSkill state_dict from the flat env_states storage."""
    state_dict: dict = {"actors": {}, "articulations": {}}
    for flat_key, arr in env_states.items():
        parts = flat_key.split("/")
        group, name = parts[0], parts[1]
        state_dict[group][name] = torch.tensor(
            arr[frame_idx : frame_idx + 1], dtype=torch.float32
        )
    return state_dict


def overlay_text(
    frame: np.ndarray, progress: int, max_progress: int, is_rewinding: bool
) -> np.ndarray:
    """Draw progress info on the top-left of a rendered frame."""
    frame = frame.copy()
    h, w = frame.shape[:2]

    status = "REWIND" if is_rewinding else "FORWARD"
    color = (0, 0, 255) if is_rewinding else (0, 200, 0)

    texts = [
        f"P={progress}/{max_progress}",
        status,
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, h / 800)
    thickness = max(1, int(h / 400))
    y_offset = int(30 * scale)

    for i, text in enumerate(texts):
        y = y_offset + i * int(35 * scale)
        cv2.putText(
            frame, text, (10, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
        )
        cv2.putText(frame, text, (10, y), font, scale, color, thickness, cv2.LINE_AA)

    return frame


def render_augmented_trajectory(
    raw_traj: dict,
    aug_traj: dict,
    output_path: str,
    fps: int = 10,
    resolution: int = 512,
) -> None:
    """Render the augmented trajectory and save as mp4."""
    import mani_skill.envs  # noqa: F401 - register envs

    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="cpu",
        num_envs=1,
        sensor_configs={"shader_pack": "default"},
    )
    env.reset(seed=0)

    env_states = raw_traj["env_states"]
    progress_steps = aug_traj["progress_step"]
    max_progress = aug_traj["max_progress"]

    if isinstance(progress_steps, torch.Tensor):
        progress_steps = progress_steps.tolist()

    frames = []
    prev_progress = 0

    for t, progress in enumerate(progress_steps):
        frame_idx = progress - 1  # progress is 1-indexed

        state_dict = build_state_dict(env_states, frame_idx)
        env.unwrapped.set_state_dict(state_dict)

        rendered = env.render()
        if isinstance(rendered, torch.Tensor):
            rendered = rendered.cpu().numpy()
        frame = rendered[0]  # remove batch dim

        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        if frame.shape[0] != resolution or frame.shape[1] != resolution:
            frame = cv2.resize(frame, (resolution, resolution))

        is_rewinding = progress < prev_progress
        frame = overlay_text(frame, progress, max_progress, is_rewinding)
        frames.append(frame)
        prev_progress = progress

    env.close()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"Saved {len(frames)} frames to {output_path} ({fps} fps)")
    print(f"Progress sequence: {progress_steps}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a rewind-augmented trajectory as mp4"
    )
    parser.add_argument(
        "--raw-pkl",
        type=str,
        required=True,
        help="Path to the raw trajectory pkl (with env_states)",
    )
    parser.add_argument(
        "--aug-pkl",
        type=str,
        default=None,
        help="Path to a saved augmented trajectory pkl. "
        "If not provided, --aug-index is used.",
    )
    parser.add_argument(
        "--aug-index",
        type=int,
        default=1,
        help="Index of augmented variant to visualize "
        "(0=original, 1..N=augmented). Used when "
        "--aug-pkl is not provided.",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=5,
        help="Number of augmentations to generate (only used with --aug-index)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True, help="Output .mp4 path")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=512)

    args = parser.parse_args()

    raw_traj = load_raw_trajectory(args.raw_pkl)

    if args.aug_pkl:
        aug_traj = load_aug_trajectory(args.aug_pkl)
    else:
        rng = np.random.default_rng(args.seed)
        augmented = rewind_augment_trajectory(
            raw_traj,
            traj_id=0,
            num_augmentations=args.num_augmentations,
            rng=rng,
        )
        if args.aug_index >= len(augmented):
            print(f"aug_index {args.aug_index} out of range (max {len(augmented) - 1})")
            sys.exit(1)
        aug_traj = augmented[args.aug_index]

    render_augmented_trajectory(
        raw_traj,
        aug_traj,
        args.output,
        fps=args.fps,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
