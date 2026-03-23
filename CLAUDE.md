# CLAUDE.md — RLinf TMPER Project Context

> Living playbook for AI agents working on the TMPER (Temporally Monotonic Potential Energy Reward) project within the RLinf codebase. Updated incrementally as insights accumulate.

---

## Project Context

### Domain Knowledge

- **TMPER(Temporally Monotonic Potential Energy Reward)** is a reward-shaping framework for embodied RL. It has an offline phase (rewind augmentation + potential function training) and an online phase (PBRS reward injection into SAC/PPO/GRPO).
- The project roadmap lives in `.cursor/one-page.md`; feature specs live in `.cursor/docs/`.
- Currently, the development happens on `feature/rewind-aug` branch. PRs target a `dev` branch, following RLinf conventional commit conventions.
- The user is a Fudan undergrad doing a thesis; clarity and reproducibility are priorities.

### Technical Constraints

- **No GPU cluster access** — single-machine runs only (`cluster.num_nodes: 1`).
- **China network** — all downloads must use domestic mirrors: `--use-mirror` for `install.sh`, `HF_ENDPOINT=https://hf-mirror.com` for HuggingFace, `--index-url https://mirrors.aliyun.com/pypi/simple/` for pip.
- **Virtual env** is pre-installed at `.venv/bin/activate` with ManiSkill3 (3.0.0b22), mplib, imageio, OpenCV, PyTorch.
- The potential function uses **proprioceptive state only** (joints, not images) — see one-page.md C3.

---

## ManiSkill3 Operational Knowledge

### Expert Demo Generation (No Model Needed)

- ManiSkill3 ships built-in motion planning solutions at `mani_skill.examples.motionplanning.panda.solutions`. Use these instead of training an RL model for demo collection.
- Supported tasks: `PickCube-v1`, `StackCube-v1`, `PegInsertionSide-v1`, `PushCube-v1`, `PlugCharger-v1`, and others.
- Command: `python -m mani_skill.examples.motionplanning.panda.run -e "PickCube-v1" -n 10 --only-count-success --obs-mode none -b cpu --record-dir data/demos`
- Output: `.h5` + `.json` in `data/demos/<task>/motionplanning/`.
- Motion planning uses `mplib` and runs on CPU; no GPU required.

### H5 Trajectory Format

- Each trajectory `traj_{id}` contains: `obs` (T+1, dim), `actions` (T, dim), `terminated` (T,), `truncated` (T,), `success` (T,), `env_states` (nested groups).
- **obs has T+1 frames** (includes initial reset observation). Actions have T frames.
- When generated with `--obs-mode none`, obs is empty. Must replay with `-o state` to extract state observations: `python -m mani_skill.trajectory.replay_trajectory --traj-path <h5> -o state --save-traj`

### State Restoration and Rendering

- `env.unwrapped.set_state_dict(state_dict)` restores full physics state for any frame.
- `state_dict` structure: `{"actors": {"table-workspace": Tensor[1,13], "cube": Tensor[1,13], ...}, "articulations": {"panda": Tensor[1,31]}}` — always has batch dimension.
- `env.render()` returns `Tensor[1, H, W, 3]` (uint8) — remove batch dim with `[0]`.
- **Must `import mani_skill.envs`** before `gym.make()` to register environments; otherwise `NameNotFound` error.

### PickCube-v1 Specifics

- State dim: 42 (`obs_mode="state"`, `wrap_obs_mode: simple`).
- Action dim: 8 (`pd_joint_pos` control mode from motion planning), or 4 (`panda-ee-dpos` for SAC).
- Typical episode length: 49–88 steps (motion planning solutions).
- No pretrained checkpoints exist for MLP/SAC on PickCube; RLinf only publishes VLA models for multi-task benchmarks.
- The vulkan fallback warning (`Failed to find system libvulkan`) is harmless — ignore it.

---

## RLinf Data Architecture

### Trajectory and Replay Buffer

- `Trajectory` dataclass (`rlinf/data/embodied_io_struct.py`): stores `[T, B, ...]` tensors for actions, rewards, terminations, truncations, dones, obs dicts.
- `TrajectoryReplayBuffer` (`rlinf/data/replay_buffer.py`): trajectory-level storage with caching, windowed sampling, checkpoint save/load.
- `ReplayBufferDataset` (`rlinf/data/embodied_buffer_dataset.py`): `IterableDataset` wrapper that samples from replay + optional demo buffer.

### CollectEpisode Pickle Format

- `CollectEpisode` wrapper (`rlinf/envs/wrappers/collect_episode.py`) saves episodes as pickle dicts.
- Format: `{"observations": [...], "actions": [...], "rewards": [...], "terminated": [...], "truncated": [...], "infos": [...], "success": bool}`.
- Enabled via `env.train.data_collection.enabled: True` in YAML config.
- Our raw pkl files (`data/demos/PickCube-v1/raw_pkl/`) follow this format plus `env_states` and `episode_id`.

### TMPER Data Pipeline

- **Raw pkl** (`scripts/tmper/convert_h5_to_pkl.py`): converts ManiSkill h5 demos to per-trajectory pkl files with `env_states` for rendering.
- **Augmentation** (`rlinf/data/rewind_augmentation.py`): `rewind_augment_trajectory()` produces forward+rewind sequences; `RewindAugmentedDataset` wraps as PyTorch Dataset.
- **Visualization** (`scripts/tmper/visualize_rewind.py`): renders augmented trajectories as mp4 by restoring env_states via `set_state_dict`, overlays progress_step.
- env_states are **not** carried into augmented trajectories — the visualizer loads the raw pkl separately and indexes into env_states using `progress_step - 1`.

---

## Architecture Decisions

### Rewind Augmentation Design

- Progress labels are **1-indexed** and align with observations (length T+1). Forward: `[1, 2, ..., i+1]`. Rewind: `[i, i-1, ..., i-k+1]`.
- `is_success_anchor` is True **only** on the last frame of the original (un-augmented) trajectory. Augmented copies never have it.
- `traj_id` uses formula `base_traj_idx * (num_augmentations + 1) + offset` to guarantee uniqueness across all originals and augmentations.
- Actions in the rewind segment are the reversed actions from the forward segment near the cut point — this produces physically plausible undo motions for kinematically reversible tasks.

### SAC Training in RLinf

- SAC config: `examples/embodiment/config/maniskill_sac_mlp.yaml` — `loss_type: embodied_sac`, `adv_type: embodied_sac`.
- SAC actor worker: `rlinf/workers/actor/fsdp_sac_policy_worker.py` — owns replay buffer, Q-networks, target network, entropy temperature.
- Reward injection point for PBRS (Milestone 3): `EnvWorker._calc_step_reward()` in `rlinf/workers/env/env_worker.py`.
- `gamma: 0.8` is the default discount; entropy tuning uses `alpha_type: softplus`, `target_entropy: -4`.

---

## Development Guidelines

### File Layout Convention

```
scripts/tmper/           # Pipeline scripts (convert, visualize, train reward, etc.)
rlinf/data/              # Core data modules (augmentation, datasets)
data/demos/              # Generated demo data (gitignored)
.cursor/docs/            # Feature specs and requirements
.cursor/one-page.md      # Project roadmap and analysis
```

### Testing Approach

- No CI pipeline access; each feature must include a self-contained smoke test.
- Smoke tests should validate: field existence, shapes, dtypes, monotonicity of progress labels, anchor correctness, dimension consistency.
- Visualizer mp4 output serves as the human-verifiable acceptance test.

### Checklists for Common Tasks

**Adding a new TMPER pipeline stage:**
1. Write the core module in `rlinf/data/` or `rlinf/algorithms/rewards/`.
2. Write the CLI script in `scripts/tmper/`.
3. Add a smoke test that validates all output fields.
4. Generate a visual artifact (mp4, plot) for human verification.
5. Update `.cursor/one-page.md` milestone checklist.

---

## Anti-patterns and Pitfalls

- **Do not train an RL model just for demo generation** when motion planning is available. ManiSkill3's built-in solvers are faster, deterministic, and 100% success rate.
- **Do not forget `import mani_skill.envs`** before any `gym.make("PickCube-v1", ...)` call.
- **Do not store env_states in augmented trajectories** — they are large and redundant. Use progress_step indices to look up frames in the original raw pkl.
- **Do not use `obs_mode="none"` for state-based pipelines** — always replay with `-o state` to materialize observations.
- **Do not assume T+1 = actions length** — observations have T+1 frames (initial reset obs), actions have T frames.
- **Do not hardcode state dimensions** — PickCube is 42-dim state / 8-dim action with `pd_joint_pos`, but other tasks differ.

---

*Last updated: 2026-03-22 — Milestone 1 (Rewind Augmentation) complete.*
