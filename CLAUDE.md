# CLAUDE.md — RLinf TMPER Project Context

> Living playbook for AI agents working on the TMPER (Temporally Monotonic Potential Energy Reward) project within the RLinf codebase. Updated incrementally as insights accumulate.

---

## Project Context

### Domain Knowledge

- **TMPER(Temporally Monotonic Potential Energy Reward)** is a reward-shaping framework for embodied RL. It has an offline phase (rewind augmentation + potential function training) and an online phase (PBRS reward injection into SAC/PPO/GRPO).
- The project roadmap lives in `.cursor/one-page.md`; feature specs live in `.cursor/docs/`.
- Currently, the development happens on `feature/train-potential` branch (previously `feature/rewind-aug`). PRs target a `dev` branch, following RLinf conventional commit conventions.
- The user is a Fudan undergrad doing a thesis; clarity and reproducibility are priorities.

### Technical Constraints

- **No GPU cluster access** — single-machine runs only (`cluster.num_nodes: 1`).
- **China network** — all downloads must use domestic mirrors: `--use-mirror` for `install.sh`, `HF_ENDPOINT=https://hf-mirror.com` for HuggingFace, `--index-url https://mirrors.aliyun.com/pypi/simple/` for pip.
- **Virtual env** is pre-installed at `.venv/bin/activate` with ManiSkill3 (3.0.0b22), mplib, imageio, OpenCV, PyTorch.
- The potential function uses **visual images (camera) + proprioceptive state** (joints) — see one-page.md C3. No privileged info (object positions, accelerations).

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

### H5 Structure Differs by obs_mode

- **`-o state`** replay: `traj_{id}/obs` is a flat array `(T+1, 42)` float32.
- **`-o rgb`** replay: `traj_{id}/obs/` is a nested HDF5 group:
  - `obs/sensor_data/base_camera/rgb` — `(T+1, 128, 128, 3)` uint8
  - `obs/agent/qpos` — `(T+1, 9)` float32, `obs/agent/qvel` — `(T+1, 9)` float32
  - `obs/extra/is_grasped` — `(T+1,)` bool, `obs/extra/tcp_pose` — `(T+1, 7)` float32, `obs/extra/goal_pos` — `(T+1, 3)` float32
  - `obs/sensor_param/base_camera/{extrinsic_cv, cam2world_gl, intrinsic_cv}`
- **Two separate replays are needed** to get both flat state and RGB images. They cannot be combined into a single `replay_trajectory` call. Merge happens at `convert_h5_to_pkl.py` via `--rgb-h5-path`.
- RGB replay files are named `<timestamp>.rgb.pd_joint_pos.physx_cpu.h5`; state replay files are `<timestamp>.state.pd_joint_pos.physx_cpu.h5`.

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

- **Raw pkl** (`scripts/tmper/convert_h5_to_pkl.py`): converts ManiSkill h5 demos to per-trajectory pkl files with `env_states` for rendering. Supports `--rgb-h5-path` for merging RGB images.
- **Augmentation** (`rlinf/data/rewind_augmentation.py`): `rewind_augment_trajectory()` produces forward+rewind sequences; `RewindAugmentedDataset` wraps as PyTorch Dataset. Supports `images` field (torch.uint8 `[L, H, W, 3]`).
- **Visualization** (`scripts/tmper/visualize_rewind.py`): renders augmented trajectories as mp4 by restoring env_states via `set_state_dict`, overlays progress_step.
- env_states are **not** carried into augmented trajectories — the visualizer loads the raw pkl separately and indexes into env_states using `progress_step - 1`.

### TMPER Potential Network

- **PotentialNetwork** (`rlinf/algorithms/rewards/tmper/potential_net.py`): vision (LightweightImageEncoder64, 256d) + state MLP (128d) → fusion head → Sigmoid → scalar in (0, 1).
- **Loss functions**: `ranking_loss` (pairwise logistic with dynamic margin), `boundary_calibration_loss` (MSE anchoring start→0, success→1), `smoothness_loss` (adjacent frame penalty).
- **PotentialPairDataset**: samples (s_u, s_v) pairs with P_v > P_u from same trajectory, plus anchor frames and adjacent pairs.
- **Training**: `scripts/tmper/train_potential.py` — offline self-supervised, Adam optimizer, trajectory-level train/val split.
- **Evaluation**: `scripts/tmper/eval_potential.py` — three acceptance tests (progress bar monotonicity, drop detection, idle flatness).
- **Checkpoint**: `data/checkpoints/tmper/potential_phi.pt` — contains `model_state_dict`, `config`, `epoch`, `val_pairwise_acc`.

---

## Architecture Decisions

### Rewind Augmentation Design

- Progress labels are **1-indexed** and align with observations (length T+1). Forward: `[1, 2, ..., i+1]`. Rewind: `[i, i-1, ..., i-k+1]`.
- `is_success_anchor` is True **only** on the last frame of the original (un-augmented) trajectory. Augmented copies never have it.
- `traj_id` uses formula `base_traj_idx * (num_augmentations + 1) + offset` to guarantee uniqueness across all originals and augmentations.
- Actions in the rewind segment are the reversed actions from the forward segment near the cut point — this produces physically plausible undo motions for kinematically reversible tasks.

### Potential Network Design

- Uses `LightweightImageEncoder64` from `rlinf/models/embodiment/modules/compact_encoders.py` with `latent_dim=256` for vision, yielding ~8.5M total parameters (dominated by the 32768→256 bottleneck projection). The spec says ~13K for default `latent_dim=64`; 256 is a deliberate trade-off for richer visual features.
- Input images are 128×128 uint8 HWC, resized to 64×64 and normalized to [0,1] float inside `PotentialNetwork._encode_images()`. Handles both uint8 HWC input (from dataset) and float CHW input (if preprocessed externally).
- The rewards `__init__.py` wraps reasoning reward imports in try-except to allow TMPER import without pulling in latex2sympy2 and other reasoning-only deps.
- TMPER module is standalone (not registered in `reward_registry`) — it's an offline model loaded directly, not a standard RLinf reward worker.

### Potential Network Training Insights

- **Boundary calibration converges fast**: `L_bc` drops to near 0 within the first 5 epochs. The ranking loss `L_rank` drives the remaining training.
- **Model learns S-shaped potential curve**: On PickCube-v1, the first ~15 frames (approach) stay near 0, frames 15-30 (grasp) rise sharply, and frames 30+ (lift) saturate near 1. This matches the physical task structure.
- **91% val pairwise accuracy is achievable with 10 trajectories**: With 8 train / 2 val split, `num_augmentations=5`, 200 epochs, and tuned hyperparameters (`c=1.0, lambda_smooth=3.0, lr=5e-4`), best val acc reaches ~0.91.
- **Training takes ~5 min on single GPU**: 200 epochs × ~1.3s/epoch on CUDA with batch_size=64.
- **Evaluation eps=1e-3 tolerance**: Test 1 (progress bar) uses eps=1e-3 to distinguish real monotonicity violations from sigmoid saturation noise. With this tolerance, trained models achieve 0 real violations on in-distribution trajectories.

### Potential Network Hyperparameter Tuning (Sweep Results)

- **Root cause of step-function behavior**: Large margin coefficient `c` creates ranking-loss margin targets (up to `c × 1.0`) that exceed the sigmoid output range (0,1). With `c=5.0`, margin targets reach 3-4, forcing the network to saturate the sigmoid at its extremes (0 or 1), producing a binary step function instead of a smooth curve.
- **Optimal hyperparameters**: `c=1.0, lambda_smooth=3.0, lr=5e-4` — achieves perfect monotonicity (0 violations), smoothness_score=0.243, max_jump=0.164. This was the winning config from a 48-config grid sweep (`c ∈ {0.1,0.3,0.5,1.0,2.0,5.0} × lambda_smooth ∈ {0.1,0.5,1.0,3.0} × lr ∈ {5e-4,1e-3}`).
- **lambda_smooth is the dominant smoothness lever**: Increasing `lambda_smooth` from 0.1 to 3.0 improves smoothness by 55-83% across all `c` values. Effect of `c` and `lr` is secondary.
- **c should be ≤ 1.0**: With `c ∈ {0.1, 0.3, 0.5, 1.0}`, margin targets stay within (0,1), preventing sigmoid saturation. `c=1.0` gives the best monotonicity; lower `c` values can produce slightly smoother but less monotonic curves.
- **Saturation ceiling for vision-based models**: Even with optimal hyperparameters, the potential saturates near 1.0 after frame ~35 (out of 74 for PickCube-v1). The CNN cannot reliably distinguish fine height differences during the lifting phase. This is a data/model capacity limitation, not a hyperparameter issue.
- **Sweep script**: `scripts/tmper/sweep_potential.py` — trains all configs sequentially, saves per-config `*_progress.png` plots and `sweep_results.csv` summary. Use for future hyperparameter exploration.

### SAC Training in RLinf

- SAC config: `examples/embodiment/config/maniskill_sac_mlp.yaml` — `loss_type: embodied_sac`, `adv_type: embodied_sac`.
- SAC actor worker: `rlinf/workers/actor/fsdp_sac_policy_worker.py` — owns replay buffer, Q-networks, target network, entropy temperature.
- Reward injection point for PBRS (Milestone 3): `EnvWorker._calc_step_reward()` in `rlinf/workers/env/env_worker.py`.
- `gamma: 0.8` is the default discount; entropy tuning uses `alpha_type: softplus`, `target_entropy: -4`.

---

## Development Guidelines

### File Layout Convention

```
scripts/tmper/                          # Pipeline scripts (convert, visualize, train, eval)
rlinf/data/                             # Core data modules (augmentation, datasets)
rlinf/algorithms/rewards/tmper/         # Potential network + losses + pair dataset
data/demos/                             # Generated demo data (gitignored)
data/checkpoints/tmper/                 # Trained model weights (gitignored)
data/eval/tmper/                        # Evaluation plots (gitignored)
.cursor/docs/                           # Feature specs and requirements
.cursor/one-page.md                     # Project roadmap and analysis
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

**Regenerating demo data with RGB for potential network training:**
1. Generate demos: `python -m mani_skill.examples.motionplanning.panda.run -e "PickCube-v1" -n 10 --only-count-success --obs-mode none -b cpu --record-dir data/demos`
2. Replay state: `python -m mani_skill.trajectory.replay_trajectory --traj-path <h5> -o state --save-traj`
3. Replay RGB: `python -m mani_skill.trajectory.replay_trajectory --traj-path <h5> -o rgb --save-traj`
4. Convert to pkl: `python scripts/tmper/convert_h5_to_pkl.py --h5-path <state.h5> --rgb-h5-path <rgb.h5> --output-dir data/demos/PickCube-v1/raw_pkl`
5. Verify: `python -c "import pickle; t=pickle.load(open('data/demos/PickCube-v1/raw_pkl/traj_0.pkl','rb')); print(len(t['images']), t['images'][0].shape)"`

**Training the potential network:**
1. Ensure raw pkl files have both `observations` and `images` fields.
2. Run: `PYTHONUNBUFFERED=1 python scripts/tmper/train_potential.py --data-dir data/demos/PickCube-v1/raw_pkl --output-dir data/checkpoints/tmper`
3. Evaluate: `python scripts/tmper/eval_potential.py --checkpoint data/checkpoints/tmper/potential_phi.pt --data-dir data/demos/PickCube-v1/raw_pkl --output-dir data/eval/tmper`
4. Check three plots in `data/eval/tmper/`: `test1_progress_bar.png`, `test2_drop_test.png`, `test3_idle_test.png`.

---

## Anti-patterns and Pitfalls

- **Do not train an RL model just for demo generation** when motion planning is available. ManiSkill3's built-in solvers are faster, deterministic, and 100% success rate.
- **Do not forget `import mani_skill.envs`** before any `gym.make("PickCube-v1", ...)` call.
- **Do not store env_states in augmented trajectories** — they are large and redundant. Use progress_step indices to look up frames in the original raw pkl.
- **Do not use `obs_mode="none"` for state-based pipelines** — always replay with `-o state` to materialize observations.
- **Do not assume T+1 = actions length** — observations have T+1 frames (initial reset obs), actions have T frames.
- **Do not hardcode state dimensions** — PickCube is 42-dim state / 8-dim action with `pd_joint_pos`, but other tasks differ.
- **Do not import `rlinf.algorithms.rewards.tmper.*` through the parent package** directly in environments without reasoning deps. The `rewards/__init__.py` wraps reasoning reward imports in try-except, but downstream code should import the tmper subpackage directly: `from rlinf.algorithms.rewards.tmper.potential_net import PotentialNetwork`.
- **Do not pass extra config keys when instantiating PotentialNetwork from checkpoint** — the saved `config` dict includes training hyperparams (`c`, `lambda_*`). Filter to constructor keys only: `{"state_dim", "image_size", "latent_dim", "state_latent_dim"}`.
- **Do not split train/val after augmentation** — augmented copies share base trajectory data, causing data leakage. Always split raw pkl files first, then augment each subset independently.
- **Do not expect strict float32 monotonicity in sigmoid-saturated regions** — PotentialNetwork outputs near 0 and 1 have tiny violations (1e-5 to 1e-8) due to float32 precision. Use eps=1e-3 tolerance when checking monotonicity.
- **Do not run long training scripts without `PYTHONUNBUFFERED=1`** — Python buffers stdout when not connected to a TTY, making monitoring impossible.
- **Do not set margin coefficient c > 1.0** — values like `c=5.0` create ranking loss margins exceeding the (0,1) sigmoid range, forcing the network into a binary step function. Keep `c ≤ 1.0`.
- **Do not confuse CLI flag names across scripts** — `train_potential.py` uses `--output-dir`, `eval_potential.py` uses `--checkpoint` and `--output-dir`. Always verify argparse definitions before composing multi-script commands.

---

*Last updated: 2026-03-28 — Milestone 2 (Train Potential Model) complete. Hyperparameter sweep debugged step-function behavior; optimal params: c=1.0, λ_smooth=3.0, lr=5e-4.*
