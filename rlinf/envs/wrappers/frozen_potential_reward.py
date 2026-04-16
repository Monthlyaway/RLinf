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

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rlinf.reward_relabel import FrozenPotential


class FrozenPotentialReward(gym.Wrapper):
    """Apply frozen PBRS relabeling on top of environment rewards."""

    def __init__(self, env: gym.Env, reward_relabel_cfg):
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        self.cfg = reward_relabel_cfg
        self.combine_mode = getattr(self.cfg, "combine_mode", "add")
        if self.combine_mode not in {"add", "replace"}:
            raise ValueError(
                f"Unsupported combine_mode: {self.combine_mode}. "
                "Supported values are {'add', 'replace'}."
            )

        self.frozen_potential = FrozenPotential(
            model_path=self.cfg.model_path,
            normalization_stats_path=self.cfg.normalization_stats_path,
            gamma=float(self.cfg.gamma),
            device=getattr(self.cfg, "device", "cpu"),
        )
        self._last_obs = None

    @property
    def is_start(self):
        return getattr(self.env, "is_start")

    @is_start.setter
    def is_start(self, value):
        setattr(self.env, "is_start", value)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self._last_obs = self._copy(obs)
        return obs, info

    def step(self, action, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        reward, shaped_reward = self._apply_reward_relabel(obs, reward, info)
        if isinstance(info, dict):
            info["shaped_reward"] = self._convert_like(shaped_reward, reward)
        return obs, reward, terminated, truncated, info

    def chunk_step(self, chunk_actions):
        obs_list, rewards, terminations, truncations, infos_list = self.env.chunk_step(
            chunk_actions
        )
        adjusted_rewards = []
        for step_idx, obs in enumerate(obs_list):
            step_reward = (
                rewards[:, step_idx] if getattr(rewards, "ndim", 1) > 1 else rewards
            )
            updated_reward, shaped_reward = self._apply_reward_relabel(
                obs=obs,
                reward=step_reward,
                info=infos_list[step_idx]
                if isinstance(infos_list, (list, tuple))
                else infos_list,
            )
            adjusted_rewards.append(updated_reward)
            if isinstance(infos_list, (list, tuple)) and isinstance(
                infos_list[step_idx], dict
            ):
                infos_list[step_idx]["shaped_reward"] = self._convert_like(
                    shaped_reward, updated_reward
                )

        if getattr(rewards, "ndim", 1) > 1:
            if isinstance(rewards, torch.Tensor):
                rewards = torch.stack(adjusted_rewards, dim=1)
            else:
                rewards = np.stack(adjusted_rewards, axis=1)
        else:
            rewards = adjusted_rewards[-1]
        return obs_list, rewards, terminations, truncations, infos_list

    def _apply_reward_relabel(self, obs, reward, info):
        if self._last_obs is None:
            self._last_obs = self._copy(obs)
            zero_reward = self._zeros_like_reward(reward)
            return reward, zero_reward

        current_states = self._extract_states(self._last_obs)
        next_states = self._extract_states(obs)
        if isinstance(info, dict) and "final_observation" in info:
            final_states = self._extract_states(info["final_observation"])
            done_mask = self._extract_done_mask(info, next_states.shape[0])
            next_states = next_states.clone()
            next_states[done_mask] = final_states[done_mask]

        shaped_reward = self.frozen_potential.shaping_reward(
            current_states, next_states
        )
        self._last_obs = self._copy(obs)
        return self._combine_rewards(reward, shaped_reward), shaped_reward

    def _combine_rewards(self, reward, shaped_reward):
        shaped_like_reward = self._convert_like(shaped_reward, reward)
        if self.combine_mode == "replace":
            return shaped_like_reward
        return reward + shaped_like_reward

    def _extract_states(self, obs: dict[str, Any]) -> torch.Tensor:
        if not isinstance(obs, dict):
            raise TypeError("FrozenPotentialReward expects dict observations.")
        states = obs.get("states", obs.get("state"))
        if states is None:
            raise KeyError("Observation dict must contain 'states' or 'state'.")
        if isinstance(states, torch.Tensor):
            tensor = states.detach().to(torch.float32)
        else:
            tensor = torch.as_tensor(states, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _extract_done_mask(self, info: dict[str, Any], num_envs: int) -> torch.Tensor:
        done_mask = info.get("_final_observation", info.get("_final_info"))
        if done_mask is None:
            return torch.zeros(num_envs, dtype=torch.bool)
        if isinstance(done_mask, torch.Tensor):
            mask = done_mask.detach().cpu().bool()
        else:
            mask = torch.as_tensor(done_mask, dtype=torch.bool)
        if mask.dim() == 0:
            mask = mask.repeat(num_envs)
        return mask

    def _zeros_like_reward(self, reward):
        if isinstance(reward, torch.Tensor):
            return torch.zeros_like(reward)
        if isinstance(reward, np.ndarray):
            return np.zeros_like(reward)
        return 0.0

    def _convert_like(self, value: torch.Tensor, reward):
        if isinstance(reward, torch.Tensor):
            return value.to(dtype=reward.dtype, device=reward.device)
        if isinstance(reward, np.ndarray):
            return value.detach().cpu().numpy().astype(reward.dtype)
        return float(value.mean().item())

    def _copy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: self._copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._copy(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._copy(item) for item in value)
        return copy.deepcopy(value)
