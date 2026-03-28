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

import logging

_logger = logging.getLogger(__name__)


def register_reward(name: str, reward_class: type):
    assert name not in reward_registry, f"Reward {name} already registered"
    reward_registry[name] = reward_class


def get_reward_class(name: str):
    assert name in reward_registry, f"Reward {name} not found"
    return reward_registry[name]


reward_registry = {}

# Reasoning reward modules have heavy optional dependencies (latex2sympy2, etc.)
# that are not installed in the embodied-only venv.  Wrap in try-except so
# subpackages like tmper can be imported without pulling in all deps.
try:
    from rlinf.algorithms.rewards.math import MathReward

    register_reward("math", MathReward)
except ImportError:
    _logger.debug("MathReward not available (missing dependencies)")

try:
    from rlinf.algorithms.rewards.vqa import VQAReward

    register_reward("vqa", VQAReward)
except ImportError:
    _logger.debug("VQAReward not available (missing dependencies)")

try:
    from rlinf.algorithms.rewards.code import CodeRewardOffline

    register_reward("code_offline", CodeRewardOffline)
except ImportError:
    _logger.debug("CodeRewardOffline not available (missing dependencies)")

try:
    from rlinf.algorithms.rewards.searchr1 import SearchR1Reward

    register_reward("searchr1", SearchR1Reward)
except ImportError:
    _logger.debug("SearchR1Reward not available (missing dependencies)")

try:
    from rlinf.algorithms.rewards.rstar2 import Rstar2Reward

    register_reward("rstar2", Rstar2Reward)
except ImportError:
    _logger.debug("Rstar2Reward not available (missing dependencies)")
