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

"""Convert ManiSkill replayed .h5 demo trajectories to per-trajectory .pkl files.

Usage:
    python scripts/tmper/convert_h5_to_pkl.py \
        --h5-path data/demos/PickCube-v1/motionplanning/<name>.state.*.h5 \
        --output-dir data/demos/PickCube-v1/raw_pkl
"""

import argparse
import json
import os
import pickle

import h5py
import numpy as np


def convert_h5_to_pkls(h5_path: str, output_dir: str) -> None:
    json_path = h5_path.replace(".h5", ".json")
    with open(json_path) as f:
        meta = json.load(f)

    episodes = meta["episodes"]
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        for ep in episodes:
            ep_id = ep["episode_id"]
            traj_key = f"traj_{ep_id}"
            traj = f[traj_key]

            obs_all = np.array(traj["obs"])  # (T+1, state_dim)
            actions = np.array(traj["actions"])  # (T, action_dim)
            terminated = np.array(traj["terminated"])  # (T,)
            truncated = np.array(traj["truncated"])  # (T,)
            success_flags = np.array(traj["success"])  # (T,)

            T = actions.shape[0]
            rewards = np.zeros(T, dtype=np.float32)
            if success_flags.any():
                rewards[success_flags] = 1.0

            env_states = {}
            if "env_states" in traj:
                es = traj["env_states"]
                for group_name in es.keys():
                    for actor_name in es[group_name].keys():
                        key = f"{group_name}/{actor_name}"
                        env_states[key] = np.array(es[group_name][actor_name])

            episode_data = {
                "observations": [obs_all[t] for t in range(T + 1)],
                "actions": [actions[t] for t in range(T)],
                "rewards": [rewards[t] for t in range(T)],
                "terminated": [terminated[t] for t in range(T)],
                "truncated": [truncated[t] for t in range(T)],
                "success": bool(ep.get("success", success_flags.any())),
                "episode_id": ep_id,
                "elapsed_steps": ep.get("elapsed_steps", T),
                "env_states": env_states,
            }

            out_path = os.path.join(output_dir, f"traj_{ep_id}.pkl")
            with open(out_path, "wb") as pf:
                pickle.dump(episode_data, pf)

            print(
                f"  traj_{ep_id}: T={T}, state_dim={obs_all.shape[1]}, "
                f"action_dim={actions.shape[1]}, success={episode_data['success']}"
            )

    print(f"\nConverted {len(episodes)} trajectories to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert ManiSkill h5 demos to pkl")
    parser.add_argument(
        "--h5-path",
        type=str,
        required=True,
        help="Path to the replayed .h5 file (with state obs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for per-trajectory .pkl files",
    )
    args = parser.parse_args()
    convert_h5_to_pkls(args.h5_path, args.output_dir)


if __name__ == "__main__":
    main()
