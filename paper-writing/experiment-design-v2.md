# 实验方案 v2：基于当前 RLinf 仓库与当前机器的可执行版本

## 0. 核心判断

这篇论文的核心问题不是“环境是否足够多”，而是：

1. 纯 ranking 学到的势函数是否会在下游 RL 中出现 **monotonicity trap / increment collapse**。
2. 一个**不依赖人工逐帧标注**、也不依赖特权 oracle progress 的结构约束，能否稳定下游 RL。

因此，实验不必强行铺很多 simulator。更合理的做法是：

- 选 **一个当前仓库可控、任务足够多样** 的仿真家族做主实验。
- 用 **任务多样性** 代替 **环境多样性**。
- 用 **自动构造 preference** 代替 **人工标注 progress**。

我建议主线直接收敛到 **MetaWorld**，必要时再加一个 **ManiSkill** sanity check，而不是继续扩 DMC / 其他 simulator。

---

## 1. 对 v1 的 review

### 1.1 v1 的优点

- 抓住了论文最核心的决定性实验：**离线 ranking 准确，但在线 RL 高方差/崩溃**。
- 有明确的条件对比：`Pure Ranking`、`Structure-Regularized`、`Oracle`。
- 已经意识到只看 Kendall-τ 不够，必须看下游 RL 成败。

### 1.2 v1 的主要问题

#### 问题 1：依赖了当前 repo 不支持的环境组合

v1 里写了 `Meta-World + DMC`。  
但当前仓库 `rlinf/envs/__init__.py` 支持的是：

- `maniskill`
- `libero`
- `metaworld`
- `behavior`
- `calvin`
- `robocasa`
- `frankasim`
- `realworld`
- `robotwin`
- `isaaclab`
- `habitat`

**没有 DMC。**

所以 v1 如果坚持 DMC，要新接一套 env，论文 deadline 下不划算。

#### 问题 2：默认“reward model 接进 RLinf”其实今天不能直接跑

当前 `rlinf/workers/reward/reward_worker.py` 里：

- `use_reward_model=True` 直接 `NotImplementedError`
- `compute_batch_rewards_with_model()` 也没有实现

这说明 v1 那种“先离线训练 reward model，再直接走 RLinf 的 reward worker 接口”并不是现成路径。

#### 问题 3：Oracle progress 作为主基线，不够工程化

v1 的 `Condition 3: Progress Oracle` 直接回归绝对 progress。  
这适合论文 upper bound，但不适合作为主线，因为它：

- 依赖特权信息
- 与“无人工标注/更贴近工程实践”的目标不一致
- 在当前仓库里也没有现成的 progress supervision 流水线

结论：**Oracle 可以保留为 optional upper bound，不应是主实验的中心。**

#### 问题 4：算力预算与当前机器不匹配

当前机器我确认到的是：

- `1 x NVIDIA GeForce RTX 4080 SUPER`
- `32GB VRAM`

v1 里那种：

- 6 个任务
- 5 个随机种子
- 2 个下游 RL 算法
- 3 到 4 个 reward 条件
- 每个条件长 horizon 大预算训练

对这台机器不现实。  
需要先收缩成 **少任务、少 seed、轻模型、分阶段推进**。

#### 问题 5：大模型从随机初始化开始不合理

当前仓库是 **RL 后训练仓库**，不是“大模型从零预训练仓库”。  
如果使用 VLA：

- `OpenVLA`
- `OpenVLA-OFT`
- `π0 / π0.5`

都应该默认从 Hugging Face 上已有的 **SFT / warmup / RL 前权重** 开始，而不是随机初始化。

随机初始化只适合：

- `mlp_policy`
- `cnn_policy`
- `flow_policy`

这类轻量 policy。

---

## 2. 当前 repo 和当前机器的真实状态

## 2.1 当前机器 / 当前环境

我已经验证过：

- GPU: `RTX 4080 SUPER 32GB`
- 当前 `.venv` 使用的是 `bash requirements/install.sh embodied --model openvla --env maniskill_libero --use-mirror --install-rlinf`
- `install.sh` 顶部已导出代理：
  - `https_proxy=http://127.0.0.1:7897`
  - `http_proxy=http://127.0.0.1:7897`
  - `all_proxy=socks5://127.0.0.1:7897`

当前 import 状态：

- `mani_skill`: OK
- `libero`: OK
- `metaworld`: OK
- `mlp_policy`: OK
- `openpi`: 不可用，缺 `jax/openpi` 依赖

这意味着：

- **环境层面**：MetaWorld / ManiSkill / LIBERO 可以作为实验候选。
- **模型层面**：当前最稳的是 `mlp_policy` 和 `openvla` 路线；`openpi` 需要单独补模型依赖，不适合立即当主线。

## 2.2 当前仓库现成可用的训练资产

现有配置里已经有：

- `examples/embodiment/config/metaworld_50_ppo_openpi.yaml`
- `examples/embodiment/config/metaworld_50_ppo_openpi_pi05.yaml`
- `examples/embodiment/config/metaworld_50_grpo_openvlaoft.yaml`
- `examples/embodiment/config/maniskill_ppo_mlp.yaml`
- `examples/embodiment/config/maniskill_sac_mlp.yaml`
- `examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml`
- `examples/embodiment/config/maniskill_ppo_openvlaoft_quickstart.yaml`

其中最重要的事实是：

- MetaWorld 任务集在 repo 里已经配置好了，任务定义见 `rlinf/envs/metaworld/metaworld_config.json`
- `CollectEpisode` wrapper 已经能自动导出 episode 数据
- SAC replay buffer 也能自动落盘
- 这意味着 **“自动收轨迹 -> 自动构 preference 数据 -> 离线训练 reward/potential -> 在线 RL 验证”** 这条链，数据部分已经不需要重造

---

## 3. v2 的总体策略

## 3.1 主实验只用一个 simulator family：MetaWorld

这是 v2 最重要的收缩。

理由：

1. **一个环境家族已经足够多样。**  
   MetaWorld-50 本身就覆盖：
   - button pressing
   - drawer / door
   - pick-place
   - hammer / stick
   - sweep
   - assembly / insertion

2. **和 v1 原始想法一致。**  
   你原来就想用 MetaWorld；现在只是把 DMC 去掉。

3. **和论文主题更对齐。**  
   我们要验证的是 reward 结构问题，不是跨物理引擎泛化问题。

4. **当前 repo 原生支持。**  
   不需要新接 env API。

我的建议是：

- **主实验**：MetaWorld 4 个任务起步，最多扩到 6 个
- **可选补充**：ManiSkill 1 到 2 个任务做 portability sanity check

不要反过来。

## 3.2 任务选择：从 4 个任务起步

直接沿用你 v1 里最有代表性的 4 个任务：

- `button-press-v3`
- `drawer-open-v3`
- `sweep-into-v3`
- `hammer-v3`

这 4 个任务已经覆盖：

- 细粒度接触
- 关节/抽屉类位姿推进
- 长距离连续控制
- 工具使用/非平滑关键动作

如果前 4 个任务结果稳定，再扩两个：

- `assembly-v3`
- `pick-place-wall-v3`

---

## 4. v2 的实验条件

## 4.1 论文主对比，不再用 Oracle 做主线

我建议主表只保留下面 4 组：

### C1. Sparse Reward Only

直接使用 env 当前 reward。

作用：

- 作为“完全不做 reward learning”的最朴素 baseline

### C2. Practical Auto-Shaping Baseline

使用当前 env 已有的工程化 shaping 信号。

对 MetaWorld：

- 直接使用当前 wrapper 的 success-difference reward

对 ManiSkill optional sanity check：

- 使用当前 env wrapper 里的 dense heuristic reward

作用：

- 作为“工程里大家真正会用的手工/半手工 shaping” baseline

### C3. Preference Potential with BT Only

你的核心反方 baseline：

- 用自动 preference 数据训练势函数
- 只用 BT / ranking loss
- 在线时用 PBRS: `F = gamma * phi(s') - phi(s)`

### C4. Preference Potential with BT + Struct

你的方法：

- `L_total = L_BT + lambda * L_struct`

这才是主角。

## 4.2 Oracle 只做 optional upper bound

如果后面时间够，可以再加：

### C5. Oracle-lite / Privileged Progress

但只建议在以下条件成立时加：

- 环境确实能稳定读到 progress proxy
- 不需要额外重接系统
- 不拖主线进度

如果不能满足，就不要做。

这篇论文完全可以靠 `C1-C4` 站住。

---

## 5. 无人工标注的数据构建方案

这里是 v2 的关键：**不用人工逐帧 progress label，也不用 GT 距离。**

## 5.1 数据来源

全部来自当前 repo 自己跑出来的轨迹：

1. 随机初始化 / 早期 checkpoint rollout
2. 中期 checkpoint rollout
3. 后期 checkpoint rollout
4. `CollectEpisode` 导出的离线 episode
5. SAC replay buffer / PPO rollout cache

也就是说，先让 agent 跑起来，再从它自己的轨迹里自动挖 preference。

## 5.2 preference 构造规则

我建议按“置信度从高到低”分 4 类：

### P1. Outcome Pair

- 成功轨迹 `>` 失败轨迹

这是最高置信度，不需要人工标签。

### P2. Efficiency Pair

- 同任务下，成功且更短的轨迹 `>` 成功但更长的轨迹

这条非常符合工程实践，因为真实系统里成功且更快往往更优。

### P3. Checkpoint Pair

- 来自较后 checkpoint 且 rollout return 更高 / success 更高的轨迹 `>` 来自较早 checkpoint 的轨迹

这是非常工程化的自动标注来源，因为它直接利用训练过程中的 policy 改善。

### P4. Intra-Trajectory Temporal Pair

只在高置信度成功轨迹里构造：

- 后段片段 `>` 前段片段

注意：

- 这条只对成功轨迹用
- 不对失败轨迹强行做“后面一定更好”的假设

这样能显著减少错误偏好。

## 5.3 不再依赖的信号

v2 里不把这些当必须项：

- GT 欧氏距离
- 人工 progress label
- 人工 success 中间帧标注
- 倒放轨迹一定更差的强假设

这些都可以作为 optional 数据增强，但不应成为主线依赖。

---

## 6. 下游 RL 如何接到当前 repo

## 6.1 不走 `reward_worker` 主路径

因为当前 `reward_worker` 的 model-based reward 没实现。

所以 v2 最稳的工程路径是：

### 路径 A：env-side relabel

在 `rlinf/workers/env/env_worker.py` 里，在拿到：

- `obs_list`
- `chunk_rewards`
- `chunk_terminations`
- `chunk_truncations`

之后，如果配置打开了 `frozen_potential_reward`，就：

1. 用冻结的 `phi(s)` 计算每一步势值
2. 生成 `gamma * phi(s_{t+1}) - phi(s_t)`
3. 替换或混合原始 `chunk_rewards`

这条路径最贴近当前 embodied pipeline。

### 路径 B：replay / rollout relabel

对 SAC 也可以：

1. 先收原始轨迹
2. 进入 replay buffer 前做 reward relabel

这对离线/半离线对照更方便。

## 6.2 建议新增的最小代码模块

建议只做下面 3 个最小增量：

1. `toolkits/preference_data/build_preferences_from_episodes.py`
   - 从 `CollectEpisode` 导出的 episode 自动构造 pair

2. `toolkits/reward_model/train_potential_from_preferences.py`
   - 训练 `BT-only` 和 `BT+Struct` 两类势函数

3. `rlinf/reward_relabel/frozen_potential.py`
   - 给 env worker / replay buffer 调用的 PBRS relabel 模块

不要先做“通用 reward worker 重构”。  
deadline 下，这个工程面太大，收益太低。

---

## 7. 模型与初始化策略

## 7.1 主实验模型：轻量 policy，不上大 VLA

在当前 `1 x 4080 SUPER 32GB` 上，我建议：

- **主实验用 `mlp_policy`**
- 如果要图像版，再补一个 `cnn_policy`

原因：

1. 论文要先证明 reward 机制成立
2. 轻量 policy 更容易把 variance / convergence 现象跑清楚
3. 可以跑更多 seed 和 task
4. 不会把结论混进大模型训练不稳定里

## 7.2 VLA 只做补充，不做主线

如果后面时间够，补一个 lightweight VLA case：

- `OpenVLA-OFT` on ManiSkill quickstart

但要满足两个前提：

1. 先把轻量 policy 版本结果做扎实
2. 使用 **已有 HF 预训练 / SFT / warmup 权重**

如果要补 VLA case，我建议只考虑这两类起点：

- **MetaWorld follow-up**：
  - `RLinf-OpenVLAOFT-Metaworld-SFT`
- **ManiSkill follow-up**：
  - `gen-robot/openvla-7b-rlvla-warmup`
  - 或 `RLinf/Openvla-oft-SFT-libero10-trajall`

不要把时间花在“随机初始化大模型能不能学起来”上。

## 7.3 关于“从随机开始”还是“从预训练开始”

我的结论是：

### 对轻量 policy

- 可以从随机初始化开始

### 对 VLA

- **必须从 Hugging Face 上已有权重开始**
- 不建议从随机初始化开始

理由很简单：

- 当前 repo 是 RL 后训练仓库
- 当前机器是单卡 4080
- 从随机初始化训练 VLA 会把论文主问题完全淹没

---

## 8. 推荐的分阶段执行顺序

## Phase A：先把主问题跑通

只做：

- MetaWorld 4 任务
- `mlp_policy`
- PPO
- 3 个条件：`Sparse` / `BT-only` / `BT+Struct`
- 3 个 seed

目标：

- 先看到“离线 ranking 相近，但在线 RL 差异明显”

## Phase B：补强稳定性结论

加上：

- SAC
- `Practical Auto-Shaping` baseline
- 更完整的曲线与 AUC

目标：

- 证明不是 PPO 特例

## Phase C：补一个展示性实验

二选一：

- MetaWorld 扩到 6 任务
- ManiSkill 1 到 2 个任务做外推 sanity check

目标：

- 提升论文说服力，但不破坏主线节奏

---

## 9. 资源预算建议

在当前机器上，不建议一上来就跑：

- 5 seed
- 50 task
- 大模型
- PPO + SAC + Oracle 全开

更合理的是：

### 第一轮

- 4 task
- 3 seed
- MLP
- PPO only

### 第二轮

- 4 task
- 3 seed
- MLP
- PPO + SAC

### 最终论文表格

- 只对最关键配置补更多 seed

这比“每个配置都全量跑满”更符合 deadline 工程现实。

---

## 10. 最终建议

如果现在立刻开做，我建议你把论文实验主线改成：

### 主结论

在 **MetaWorld 单一仿真家族** 中，使用 **自动构造的 preference 数据**，比较：

- Sparse reward
- Practical shaping baseline
- BT-only potential
- BT+Struct potential

并在 **PPO + SAC** 下验证：

- offline ranking ≠ online RL stability
- `L_struct` 能显著降低 seed variance
- 不依赖人工 progress 标注，也能稳定提升 sample efficiency

### 主实验设置

- 主环境：MetaWorld
- 主任务：`button-press-v3` / `drawer-open-v3` / `sweep-into-v3` / `hammer-v3`
- 主模型：`mlp_policy`
- 主算法：PPO + SAC
- 主种子数：3

### 补充实验

- optional Oracle-lite
- optional ManiSkill sanity check
- optional VLA case study

---

## 11. 一句话版本

v2 的正确方向不是“继续找更多环境”，而是：

**用 MetaWorld 这一套当前仓库原生支持、任务足够多样的仿真家族，把自动 preference -> 结构约束势函数 -> 下游 RL 稳定性这条链路做扎实。**
