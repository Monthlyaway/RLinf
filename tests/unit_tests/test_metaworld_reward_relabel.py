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

import json
import random

import gymnasium as gym
import torch
from omegaconf import OmegaConf

from rlinf.envs.metaworld import MetaWorldBenchmark
from rlinf.envs.wrappers.frozen_potential_reward import FrozenPotentialReward
from rlinf.reward_relabel.frozen_potential import PotentialMLP
from toolkits.preference_data.build_preferences_from_episodes import (
    EpisodeRecord,
    build_task_pairs,
)


class DummyVectorEnv(gym.Env):
    """Minimal vectorized env for reward relabel testing."""

    def reset(self, *args, **kwargs):
        return {"states": torch.zeros((2, 1), dtype=torch.float32)}, {}

    def step(self, action):
        obs = {"states": torch.ones((2, 1), dtype=torch.float32)}
        reward = torch.ones(2, dtype=torch.float32)
        terminated = torch.zeros(2, dtype=torch.bool)
        truncated = torch.zeros(2, dtype=torch.bool)
        info = {}
        return obs, reward, terminated, truncated, info


def test_metaworld_core4_suite_and_custom_task_override():
    benchmark = MetaWorldBenchmark("metaworld_core4")
    assert benchmark.get_env_names() == [
        "button-press-v3",
        "drawer-open-v3",
        "sweep-into-v3",
        "hammer-v3",
    ]

    custom_benchmark = MetaWorldBenchmark(
        "metaworld_core4",
        task_names=["drawer-open-v3"],
    )
    assert custom_benchmark.get_num_tasks() == 1
    assert custom_benchmark.get_env_names() == ["drawer-open-v3"]


def test_frozen_potential_reward_adds_pbrs(tmp_path):
    model = PotentialMLP(input_dim=1, hidden_dims=(), activation="tanh")
    linear = model.network[0]
    linear.weight.data.fill_(1.0)
    linear.bias.data.zero_()

    model_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": {
                "input_dim": 1,
                "hidden_dims": [],
                "activation": "tanh",
            },
        },
        model_path,
    )
    stats_path = tmp_path / "normalization_stats.json"
    stats_path.write_text(
        json.dumps({"start_mean": 0.0, "goal_mean": 2.0}),
        encoding="utf-8",
    )

    wrapper = FrozenPotentialReward(
        DummyVectorEnv(),
        OmegaConf.create(
            {
                "model_path": str(model_path),
                "normalization_stats_path": str(stats_path),
                "gamma": 1.0,
                "device": "cpu",
                "combine_mode": "add",
            }
        ),
    )
    wrapper.reset()
    _, reward, _, _, info = wrapper.step(None)

    assert torch.allclose(reward, torch.full((2,), 1.5))
    assert torch.allclose(info["shaped_reward"], torch.full((2,), 0.5))


def test_build_task_pairs_emits_all_pair_types():
    episodes = [
        EpisodeRecord(
            task_name="button-press-v3",
            episode_id="success_short",
            success=True,
            states=[[0.0], [0.5], [1.0], [1.5], [2.0]],
            horizon=5,
        ),
        EpisodeRecord(
            task_name="button-press-v3",
            episode_id="success_long",
            success=True,
            states=[[0.0], [0.2], [0.4], [0.6], [0.8], [1.0], [1.2], [1.4]],
            horizon=8,
        ),
        EpisodeRecord(
            task_name="button-press-v3",
            episode_id="failure",
            success=False,
            states=[[0.0], [0.1], [0.2], [0.2]],
            horizon=4,
        ),
    ]

    pairs = build_task_pairs(
        episodes,
        max_pairs_per_type=100,
        min_success_gap=2,
        min_segment_len=2,
        rng=random.Random(0),
    )

    pair_types = {pair["pair_type"] for pair in pairs}
    assert pair_types == {"success_failure", "efficiency", "temporal"}
