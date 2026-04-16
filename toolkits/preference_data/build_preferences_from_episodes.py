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

"""Build automatic preference pairs from collected episode pickles."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class EpisodeRecord:
    """Compact representation of one collected episode."""

    task_name: str
    episode_id: str
    success: bool
    states: list[list[float]]
    horizon: int


def _extract_state(observation: dict) -> list[float] | None:
    state = observation.get("states", observation.get("state"))
    if state is None:
        return None
    return np.asarray(state, dtype=np.float32).reshape(-1).tolist()


def _extract_task_name(observation: dict) -> str:
    task_name = observation.get("task_names", observation.get("task_name", "unknown"))
    if isinstance(task_name, list):
        return str(task_name[0]) if task_name else "unknown"
    return str(task_name)


def load_episode(path: Path) -> EpisodeRecord | None:
    """Load one episode pickle into an ``EpisodeRecord``."""
    with path.open("rb") as file:
        payload = pickle.load(file)

    observations = payload.get("observations", [])
    if not observations:
        return None

    states = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        state = _extract_state(observation)
        if state is not None:
            states.append(state)

    if len(states) < 2:
        return None

    return EpisodeRecord(
        task_name=_extract_task_name(observations[0]),
        episode_id=path.stem,
        success=bool(payload.get("success", False)),
        states=states,
        horizon=len(states),
    )


def load_episodes(input_dir: Path) -> list[EpisodeRecord]:
    """Load all episode pickles under ``input_dir``."""
    episodes = []
    for path in sorted(input_dir.rglob("*.pkl")):
        episode = load_episode(path)
        if episode is not None:
            episodes.append(episode)
    return episodes


def build_success_failure_pairs(
    successes: list[EpisodeRecord],
    failures: list[EpisodeRecord],
) -> list[dict]:
    """Create ``success > failure`` preference pairs."""
    pairs = []
    for success_episode in successes:
        for failure_episode in failures:
            pairs.append(
                {
                    "task_name": success_episode.task_name,
                    "pair_type": "success_failure",
                    "winner_episode_id": success_episode.episode_id,
                    "loser_episode_id": failure_episode.episode_id,
                    "winner_success": True,
                    "loser_success": False,
                    "winner_states": success_episode.states,
                    "loser_states": failure_episode.states,
                }
            )
    return pairs


def build_efficiency_pairs(
    successes: list[EpisodeRecord],
    min_success_gap: int,
) -> list[dict]:
    """Create ``shorter success > longer success`` pairs."""
    sorted_successes = sorted(successes, key=lambda episode: episode.horizon)
    pairs = []
    for shorter_index, shorter_episode in enumerate(sorted_successes):
        for longer_episode in sorted_successes[shorter_index + 1 :]:
            if longer_episode.horizon - shorter_episode.horizon < min_success_gap:
                continue
            pairs.append(
                {
                    "task_name": shorter_episode.task_name,
                    "pair_type": "efficiency",
                    "winner_episode_id": shorter_episode.episode_id,
                    "loser_episode_id": longer_episode.episode_id,
                    "winner_success": True,
                    "loser_success": True,
                    "winner_states": shorter_episode.states,
                    "loser_states": longer_episode.states,
                }
            )
    return pairs


def build_temporal_pairs(
    successes: list[EpisodeRecord],
    min_segment_len: int,
) -> list[dict]:
    """Create ``later segment > earlier segment`` pairs."""
    pairs = []
    for episode in successes:
        if episode.horizon < max(2 * min_segment_len, 4):
            continue
        segment_len = max(min_segment_len, episode.horizon // 4)
        if 2 * segment_len > episode.horizon:
            segment_len = episode.horizon // 2
        if segment_len < 2:
            continue

        earlier_segment = episode.states[:segment_len]
        later_segment = episode.states[-segment_len:]
        pairs.append(
            {
                "task_name": episode.task_name,
                "pair_type": "temporal",
                "winner_episode_id": f"{episode.episode_id}:later",
                "loser_episode_id": f"{episode.episode_id}:earlier",
                "winner_success": True,
                "loser_success": True,
                "winner_states": later_segment,
                "loser_states": earlier_segment,
            }
        )
    return pairs


def sample_pairs(
    pairs: list[dict],
    limit: int,
    rng: random.Random,
) -> list[dict]:
    """Sample up to ``limit`` pairs."""
    if limit <= 0 or len(pairs) <= limit:
        return pairs
    return rng.sample(pairs, k=limit)


def build_task_pairs(
    task_episodes: Iterable[EpisodeRecord],
    *,
    max_pairs_per_type: int,
    min_success_gap: int,
    min_segment_len: int,
    rng: random.Random,
) -> list[dict]:
    """Build all pair types for one task."""
    episodes = list(task_episodes)
    successes = [episode for episode in episodes if episode.success]
    failures = [episode for episode in episodes if not episode.success]

    all_pairs = []
    all_pairs.extend(
        sample_pairs(
            build_success_failure_pairs(successes, failures),
            max_pairs_per_type,
            rng,
        )
    )
    all_pairs.extend(
        sample_pairs(
            build_efficiency_pairs(successes, min_success_gap),
            max_pairs_per_type,
            rng,
        )
    )
    all_pairs.extend(
        sample_pairs(
            build_temporal_pairs(successes, min_segment_len),
            max_pairs_per_type,
            rng,
        )
    )
    rng.shuffle(all_pairs)
    return all_pairs


def split_pairs(
    pairs: list[dict],
    val_ratio: float,
) -> tuple[list[dict], list[dict]]:
    """Split pairs into train/validation subsets."""
    if not pairs:
        return [], []
    if len(pairs) == 1:
        return pairs, pairs
    val_size = int(round(len(pairs) * val_ratio))
    if 0 < len(pairs) and val_size == 0:
        val_size = 1
    if val_size >= len(pairs):
        val_size = max(1, len(pairs) - 1)
    return pairs[val_size:], pairs[:val_size]


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write ``records`` as JSONL."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def build_summary(episodes: list[EpisodeRecord], pairs: list[dict]) -> dict:
    """Return a per-task summary for debugging and reproducibility."""
    pair_counter = Counter(record["pair_type"] for record in pairs)
    return {
        "num_episodes": len(episodes),
        "num_successes": sum(episode.success for episode in episodes),
        "num_failures": sum(not episode.success for episode in episodes),
        "num_pairs": len(pairs),
        "pair_counts": dict(pair_counter),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-pairs-per-type", type=int, default=2000)
    parser.add_argument("--min-success-gap", type=int, default=5)
    parser.add_argument("--min-segment-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Build preference datasets grouped by task name."""
    args = parse_args()
    rng = random.Random(args.seed)
    episodes = load_episodes(args.input_dir)
    episodes_by_task: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for episode in episodes:
        episodes_by_task[episode.task_name].append(episode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task_name, task_episodes in sorted(episodes_by_task.items()):
        task_pairs = build_task_pairs(
            task_episodes,
            max_pairs_per_type=args.max_pairs_per_type,
            min_success_gap=args.min_success_gap,
            min_segment_len=args.min_segment_len,
            rng=rng,
        )
        train_pairs, val_pairs = split_pairs(task_pairs, args.val_ratio)
        task_dir = args.output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(task_dir / "train.jsonl", train_pairs)
        write_jsonl(task_dir / "val.jsonl", val_pairs)
        with (task_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(build_summary(task_episodes, task_pairs), file, indent=2)


if __name__ == "__main__":
    main()
