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
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(name: str) -> type[nn.Module]:
    """Return the activation module class used by the potential MLP."""
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }
    if name not in activations:
        raise ValueError(
            f"Unsupported activation: {name}. Supported activations: {list(activations)}"
        )
    return activations[name]


class PotentialMLP(nn.Module):
    """Small MLP that maps a state vector to a scalar potential."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "tanh",
    ):
        super().__init__()
        activation_cls = _get_activation(activation)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_cls())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Predict scalar potentials for ``states``."""
        return self.network(states).squeeze(-1)


def trajectory_return(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    """Return the trajectory score used by the BT objective."""
    return model(states).sum()


def structure_regularizer(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    """Return the entropy-based increment regularizer for one trajectory."""
    if states.shape[0] < 2:
        return states.new_tensor(0.0)
    potentials = model(states)
    increments = potentials[1:] - potentials[:-1]
    increments = F.softplus(increments)
    probs = increments / increments.sum().clamp_min(1e-6)
    return torch.sum(probs * torch.log(probs.clamp_min(1e-6)))


class FrozenPotential:
    """Load a frozen potential and convert it into PBRS rewards."""

    def __init__(
        self,
        model_path: str,
        normalization_stats_path: str,
        gamma: float,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model_cfg = checkpoint["model_cfg"]
        self.model = PotentialMLP(
            input_dim=model_cfg["input_dim"],
            hidden_dims=tuple(model_cfg["hidden_dims"]),
            activation=model_cfg.get("activation", "tanh"),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.gamma = gamma

        with Path(normalization_stats_path).open("r", encoding="utf-8") as file:
            normalization_stats = json.load(file)
        self.start_mean = float(normalization_stats["start_mean"])
        self.goal_mean = float(normalization_stats["goal_mean"])
        self.denominator = self.goal_mean - self.start_mean
        if abs(self.denominator) < 1e-6:
            self.denominator = 1.0

    def _to_tensor(self, states: torch.Tensor | Sequence[float]) -> torch.Tensor:
        if isinstance(states, torch.Tensor):
            tensor = states.detach().to(self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @torch.inference_mode()
    def potential(self, states: torch.Tensor | Sequence[float]) -> torch.Tensor:
        """Return normalized potential values in approximately ``[0, 1]``."""
        tensor = self._to_tensor(states)
        raw_potential = self.model(tensor)
        return (raw_potential - self.start_mean) / self.denominator

    @torch.inference_mode()
    def shaping_reward(
        self,
        current_states: torch.Tensor | Sequence[float],
        next_states: torch.Tensor | Sequence[float],
    ) -> torch.Tensor:
        """Return ``gamma * phi(s') - phi(s)`` for a batch of transitions."""
        current_phi = self.potential(current_states)
        next_phi = self.potential(next_states)
        return self.gamma * next_phi - current_phi


def collect_successful_endpoints(
    records: Iterable[dict],
) -> tuple[list[list[float]], list[list[float]]]:
    """Collect successful start/end states from pair records."""
    start_states: list[list[float]] = []
    end_states: list[list[float]] = []
    seen_episode_ids: set[str] = set()

    for record in records:
        for prefix in ("winner", "loser"):
            if not record.get(f"{prefix}_success", False):
                continue
            episode_id = f"{record.get('task_name', 'unknown')}::{record.get(f'{prefix}_episode_id', 'unknown')}"
            if episode_id in seen_episode_ids:
                continue
            states = record[f"{prefix}_states"]
            if not states:
                continue
            seen_episode_ids.add(episode_id)
            start_states.append(states[0])
            end_states.append(states[-1])
    return start_states, end_states
