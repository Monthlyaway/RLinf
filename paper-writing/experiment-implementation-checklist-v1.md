# 实验实施清单 v1

这份清单对应论文实验正文的第一版落地方案。目标不是一次把所有扩展都做完，而是把当前仓库里最贴合论文主问题、最容易在 deadline 前跑通的主链路先接完整：

`collect episodes -> build preferences -> train potentials -> export normalization stats -> downstream PPO -> downstream SAC`

## 1. 目标口径

- 主环境：`MetaWorld core4`
- 主任务：`button-press-v3`、`drawer-open-v3`、`sweep-into-v3`、`hammer-v3`
- 主模型：state-only `mlp_policy`
- 主下游算法：`PPO`
- 次要鲁棒性验证：`SAC`
- 主对比：
  - `Sparse`
  - `Practical Shaping`
  - `BT-only Potential + PBRS`
  - `BT+Struct Potential + PBRS`
- 不进入首版主实验正文：
  - `Oracle-lite`
  - `ManiSkill portability`
  - `VLA case study`

## 2. 阶段顺序

### Phase 0：准备配置与最小代码支持

需要先完成以下仓库改动：

- 新增 `metaworld_core4` suite
- 为 MetaWorld 增加 `reward_mode: success | raw`
- 给 MetaWorld 观测补稳定 `task_names`
- 新增 frozen PBRS reward wrapper
- 新增 offline preference / reward model 工具脚本
- 新增实验 config 模板

### Phase 1：收集轨迹

输入：

- `examples/embodiment/config/metaworld_core4_ppo_mlp_collect.yaml`

输出目录：

- `results/metaworld_core4_collect/collected_data/`

期望产物：

- 每个并行环境导出的 episode `.pkl`
- episode 中至少包含：
  - `success`
  - `observations`
  - `rewards`
  - `infos`
  - `task_names`
  - `states`

运行目标：

- 使用 `reward_mode=raw`
- 让 collector 采到混合质量轨迹，而不是只保留成功轨迹

### Phase 2：构造 preference pairs

输入：

- `results/metaworld_core4_collect/collected_data/`

脚本：

- `toolkits/preference_data/build_preferences_from_episodes.py`

输出目录：

- `results/preferences/<task_name>/train.jsonl`
- `results/preferences/<task_name>/val.jsonl`
- `results/preferences/<task_name>/summary.json`

首版 pair 类型只保留三类：

- `success > failure`
- `shorter success > longer success`
- `later segment > earlier segment`

注意：

- 必须按任务拆分，不做四任务混训
- 每条 jsonl 至少带：
  - `task_name`
  - `pair_type`
  - `winner_states`
  - `loser_states`
  - `winner_success`
  - `loser_success`
  - `winner_episode_id`
  - `loser_episode_id`

### Phase 3：离线训练 potential

输入：

- `results/preferences/<task_name>/train.jsonl`
- `results/preferences/<task_name>/val.jsonl`

脚本：

- `toolkits/reward_model/train_potential_from_preferences.py`

输出目录：

- `results/potentials/<task_name>/bt_only/best.pt`
- `results/potentials/<task_name>/bt_only/metrics.json`
- `results/potentials/<task_name>/bt_only/normalization_stats.json`
- `results/potentials/<task_name>/bt_struct/best.pt`
- `results/potentials/<task_name>/bt_struct/metrics.json`
- `results/potentials/<task_name>/bt_struct/normalization_stats.json`

需要支持的参数：

- `--loss-type {bt_only,bt_struct}`
- `--lambda-struct`
- `--epochs`
- `--batch-size`
- `--lr`
- `--device`

离线训练验收标准：

- 两个 loss 分支都能正常收敛并导出 checkpoint
- `metrics.json` 至少包含：
  - `best_val_pair_accuracy`
  - `best_val_pairwise_tau`
  - `best_val_struct_loss`
  - `best_epoch`
- `normalization_stats.json` 至少包含：
  - `start_mean`
  - `goal_mean`
  - `successful_episode_count`

### Phase 4：下游 PPO

输入 config：

- `examples/embodiment/config/metaworld_core4_ppo_mlp.yaml`

运行模式：

- `Sparse`：`reward_mode=success`，`reward_relabel.enabled=false`
- `BT-only`：`reward_mode=success`，`reward_relabel.enabled=true`，路径指向 `bt_only`
- `BT+Struct`：`reward_mode=success`，`reward_relabel.enabled=true`，路径指向 `bt_struct`

输出目录建议：

- `results/downstream_ppo/<task_name>/<condition>/seed_<seed>/`

### Phase 5：下游 SAC

输入 config：

- `examples/embodiment/config/metaworld_core4_sac_mlp.yaml`

运行模式：

- `Practical Shaping`：`reward_mode=raw`，`reward_relabel.enabled=false`
- `BT+Struct`：`reward_mode=success`，`reward_relabel.enabled=true`

输出目录建议：

- `results/downstream_sac/<task_name>/<condition>/seed_<seed>/`

## 3. 新增文件

### 文档

- `paper-writing/experiment-section-v1.md`
- `paper-writing/experiment-implementation-checklist-v1.md`

### 配置

- `examples/embodiment/config/env/metaworld_core4.yaml`
- `examples/embodiment/config/metaworld_core4_ppo_mlp_collect.yaml`
- `examples/embodiment/config/metaworld_core4_ppo_mlp.yaml`
- `examples/embodiment/config/metaworld_core4_sac_mlp.yaml`
- `examples/embodiment/config/reward_relabel/frozen_potential.yaml`

### 脚本 / 模块

- `toolkits/preference_data/build_preferences_from_episodes.py`
- `toolkits/reward_model/train_potential_from_preferences.py`
- `rlinf/reward_relabel/__init__.py`
- `rlinf/reward_relabel/frozen_potential.py`
- `rlinf/envs/wrappers/frozen_potential_reward.py`

## 4. 需要修改的现有模块

### `rlinf/envs/metaworld/metaworld_config.json`

新增：

- `CORE4`

固定任务顺序：

1. `button-press-v3`
2. `drawer-open-v3`
3. `sweep-into-v3`
4. `hammer-v3`

### `rlinf/envs/metaworld/__init__.py`

需要支持：

- `metaworld_core4`
- 可选 `task_names` 覆盖，便于单任务运行

### `rlinf/envs/metaworld/metaworld_env.py`

新增：

- `reward_mode: success | raw`
- 观测字段 `task_names`

行为要求：

- `reward_mode=success` 时返回稀疏成功奖励
- `reward_mode=raw` 时直接使用 MetaWorld 原生 `_reward`
- `use_rel_reward` 继续控制是否对 reward 做相邻差分

### `rlinf/envs/wrappers/__init__.py`

导出：

- `FrozenPotentialReward`

### `rlinf/workers/env/env_worker.py`

按 `reward_relabel.enabled` 自动挂载 frozen reward wrapper。

挂载顺序建议：

1. 原始 env
2. `RecordVideo`
3. `FrozenPotentialReward`
4. `CollectEpisode`

这样 episode 导出时能拿到真正送给 agent 的 reward。

## 5. 新配置默认值

### `metaworld_core4_*`

统一覆盖：

- `obs_dim: 4`
- `action_dim: 4`
- `num_action_chunks: 1`

### `reward_relabel/frozen_potential.yaml`

默认字段：

- `enabled: false`
- `combine_mode: add`
- `device: cpu`
- `gamma: ${algorithm.gamma}`
- `model_path: null`
- `normalization_stats_path: null`

说明：

- `C3/C4` 默认使用 `sparse base reward + PBRS`
- 首版不做 reward replacement

## 6. 最小 run matrix

### Phase A：先把主结论打出来

- 4 tasks
- 3 seeds
- `Sparse / BT-only / BT+Struct`
- `PPO`

目标：

- 证明 `offline ranking good != online RL stable`
- 证明 `BT+Struct` 能降低 seed variance

### Phase B：补工程基线与算法鲁棒性

- 4 tasks
- 3 seeds
- `Practical Shaping / BT+Struct`
- `SAC`

目标：

- 说明收益不是 PPO 特例
- 说明无人工标注的结构势函数能接近工程可用 shaping

## 7. 建议运行命令口径

### 轨迹收集

```bash
source .venv/bin/activate
python examples/embodiment/train_embodied_agent.py --config-name metaworld_core4_ppo_mlp_collect
```

### preference 构造

```bash
python toolkits/preference_data/build_preferences_from_episodes.py \
  --input-dir results/metaworld_core4_collect/collected_data \
  --output-dir results/preferences
```

### BT-only 训练

```bash
python toolkits/reward_model/train_potential_from_preferences.py \
  --train-file results/preferences/button-press-v3/train.jsonl \
  --val-file results/preferences/button-press-v3/val.jsonl \
  --output-dir results/potentials/button-press-v3/bt_only \
  --loss-type bt_only
```

### BT+Struct 训练

```bash
python toolkits/reward_model/train_potential_from_preferences.py \
  --train-file results/preferences/button-press-v3/train.jsonl \
  --val-file results/preferences/button-press-v3/val.jsonl \
  --output-dir results/potentials/button-press-v3/bt_struct \
  --loss-type bt_struct \
  --lambda-struct 0.1
```

### 下游 PPO

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-name metaworld_core4_ppo_mlp \
  env.train.task_names=[button-press-v3] \
  env.eval.task_names=[button-press-v3] \
  reward_relabel.enabled=true \
  reward_relabel.model_path=results/potentials/button-press-v3/bt_struct/best.pt \
  reward_relabel.normalization_stats_path=results/potentials/button-press-v3/bt_struct/normalization_stats.json
```

### 下游 SAC

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-name metaworld_core4_sac_mlp \
  env.train.task_names=[button-press-v3] \
  env.eval.task_names=[button-press-v3] \
  reward_relabel.enabled=true \
  reward_relabel.model_path=results/potentials/button-press-v3/bt_struct/best.pt \
  reward_relabel.normalization_stats_path=results/potentials/button-press-v3/bt_struct/normalization_stats.json
```

## 8. 验收标准

### 文档验收

- 实验正文是英文，可直接并回总稿
- 工程清单把脚本、config、产物和执行顺序写清楚

### 代码验收

- `metaworld_core4` 只采样 4 个目标任务
- `reward_mode=raw` 返回非二值 MetaWorld reward
- episode `.pkl` 能稳定拿到 `task_names`
- preference builder 能按任务输出 `train.jsonl` / `val.jsonl`
- `BT-only` 与 `BT+Struct` 都能导出 checkpoint 与 normalization stats
- frozen PBRS wrapper 能在 PPO/SAC 下工作，不改 actor/rollout 接口

## 9. 首版不做的事

- 不把实验正文直接 merge 回 `paper-draft-v1.md`
- 不做多任务共享 reward model
- 不做 VLA 主实验
- 不做 Oracle 主结果
- 不追求一次把 5 seed、更多任务、更多环境全部跑满
