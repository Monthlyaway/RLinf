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

"""TMPER (Temporally Monotonic Potential Energy Reward) module.

Provides an offline-trained potential function Phi(s) for PBRS reward shaping
in embodied RL tasks. This module is used standalone for offline training
(Milestone 2), not through the RLinf reasoning reward worker registry.
"""

from rlinf.algorithms.rewards.tmper.potential_net import (  # noqa: F401
    PotentialNetwork,
    PotentialPairDataset,
    boundary_calibration_loss,
    ranking_loss,
    smoothness_loss,
)

__all__ = [
    "PotentialNetwork",
    "PotentialPairDataset",
    "ranking_loss",
    "boundary_calibration_loss",
    "smoothness_loss",
]
