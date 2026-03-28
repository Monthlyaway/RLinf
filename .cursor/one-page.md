# One Page: 时序单调势能奖励

> **论文标题**：时序单调势能奖励：基于倒放增广与比例排序的稠密奖励塑造
>
> **作者**：杨淳瑜（复旦大学 计算与智能创新学院）　导师：吴祖煊
>
> **实现平台**：RLinf (https://github.com/RLinf/RLinf)

---

## Step 1: Background (背景)

### 问题情境

在具身智能强化学习领域，真机长程操作任务面临两个结构性瓶颈：

1. **稀疏奖励**：环境仅在任务终止时给出奖励（如 success=1，其余 step 全为 0），导致绝大多数 transition 的梯度信号为零。
2. **真机采样瓶颈**：物理硬件的采样吞吐量固定且极低，无法像仿真一样大规模并行。

两者叠加使得策略收敛延迟呈指数级增长。

### 当前解决路线

| 路线 | 代表方法 | 核心局限 |
|------|---------|---------|
| 人类在环 (HiL) | HG-DAgger, HIL-SERL/RLPD | 在线人工干预成本极高，策略上限受限于人类标签质量 |
| 手工 Dense Reward | 几何/启发式规则 | 任务耦合，无跨任务泛化 |
| VLM 零样本打分 | VLM-RM | 忽略物理连续性，易 Reward Hacking |
| LLM 代码生成 | Eureka, RF-Agent | 贪心搜索易陷入局部最优 |
| 势能函数拟合 | TimeRewarder, GCR, ReWiND | 理论可行性最高，但存在两个核心缺陷 |

### 势能函数拟合的两个核心缺陷

1. **Reward Hacking**：训练数据仅含成功轨迹，模型未见过失败边界，在 near-miss 状态给出虚高分数。
2. **势能与真实进度脱节**：仅约束开始和结束的数值，未对相邻状态间势能差值与物理时间跨度的比例关系做显式约束，导致中段reward变形。

### 发起人

杨淳瑜（复旦大学计算与智能创新学院本科生），导师吴祖煊。作为本科毕业论文选题，目标在 RLinf 开源框架上实现并验证。

---

## Step 2: Definition (定义)

**本研究是什么**：一种基于演示倒放负样本合成与时序比例排序的势能奖励框架 (Temporally Monotonic Potential Energy Reward, TMPER)，在 RLinf 平台上实现，用于为 SAC/PPO/GRPO在仿真长程操作任务中提供 PBRS 理论保证的稠密探索梯度。简单来说，用倒放样本的方式增广数据集，用比例排序的方式约束reward数值。

### Goals

1. 加速RL训练的收敛速度
2. 证明reward设计的正确性
3. 证明减少了reward hacking 的发生

交付物：

1. 在dev分支上开发，所有的feature都往dev分支合并，dev分支的commit需要符合rlinf仓库的PR规范。我没有rlinf团队的提PR用的ci pipeline资源，所以每一个针对dev的MR都需要写清楚是怎么测试的，自己做基本的ci测试。
2. 实验记录：完整的实验记录，维护在 .cursor/experiments.md 文件中，需要关联 .cursor/one-page.md 文件中的feature。
3. demo 视频
4. 沉淀文档：在 .cursor 目录下，沉淀所有与本项目相关的文档。

Features:

1. **倒放样本增广模块**：从成功轨迹自动合成带物理倒退语义的负样本序列，覆盖失败边界。
2. **训练reward模块**：从少量专家演示中自监督训练严格单调的参数化势能函数 $\Phi_\theta(s) \in [0,1]$，训练目标包含
   1. 时序比例排序损失：后一状态的势能值与前一状态的势能值之差与物理时间跨度之比，应该等于奖励函数中的比例系数。假设越做越好
   2. 边界标定损失：开始和结束的势能值应该分别为0和1。
   3. 平滑约束损失：相邻状态的势能值之差应该尽可能小。
3. **在线 PBRS 接入reward模块**：冻结势能网络，计算 $r_{\text{shape}} = \gamma \Phi(s_{t+1}) - \Phi(s_t)$ 并与环境稀疏奖励叠加，接入 RLinf 的 SAC/PPO/GRPO 训练回路。
4. **实验验证**：在至少一个 ManiSkill/LIBERO 长程稀疏奖励任务上，与 ReWiND、GCR 及纯稀疏奖励基线进行定量对比。

### Non-goals

- 不涉及真机部署（仅仿真验证）
- 不涉及 VLM/LLM 的奖励生成方法
- 不替换 SAC/PPO/GRPO 算法本身（仅增强其奖励信号）
- 不修改 RLinf 核心调度/分布式逻辑
- 不涉及语言条件输入的势能函数——势能函数基于视觉图像与本体感知状态，不使用语言指令

---

## Step 3: Characteristics (特点/优点/缺点)

### 特点 (Neutral Traits)

| # | Characteristic |
|---|---------------|
| C1 | 离线自监督——势能函数的训练不需要人工标量标注，仅需专家轨迹的时序顺序 |
| C2 | 两阶段解耦——离线势能学习与在线 RL 完全解耦，势能网络在线阶段权重冻结，随着训练削减 |
| C3 | 输入为视觉图像（机器人视角摄像头或固定视角摄像头）与本体感知状态（关节角度等），不使用上帝视角特权信息（如物体绝对位置、加速度） |
| C4 | PBRS 理论保证——势能差分形式保证最优策略集不变 |
| C5 | 基于排序学习而非回归——训练目标是相对排序而非绝对值预测 |

### 优点 (Advantages)

| # | Advantage | 来源 |
|---|-----------|------|
| A1 | 通过倒放增广自动获得失败负样本，无需收集真实失败轨迹 | C1 |
| A2 | 时序比例排序使势能梯度与真实任务推进速度对齐，避免中段梯度畸变 | C5 |
| A3 | PBRS 差分形式天然抑制 reward hacking（静态状态自动获得负惩罚） | C4 |
| A4 | 离线训练完成后不引入在线推理延迟（轻量 MLP 前向传播） | C2 |
| A5 | 不依赖 VLM/LLM 推理，计算成本低于视觉语言方法 | C3 |

### 缺点 (Disadvantages)

| # | Disadvantage | 来源 |
|---|-------------|------|
| D1 | 演示倒放假设轨迹可逆——对不可逆物理过程（如切割、倾倒）可能生成非物理负样本 | C1 |
| D2 | 势能函数质量依赖专家演示的覆盖度和多样性 | C1 |
| D3 | PBRS 在 $o_{t+1}=o_t$ 时产生静态惩罚，高势能区域惩罚绝对值大，可能过度抑制必要的等待行为 | C4 |
| D4 | 与 ReWiND/GCR 的公平对比困难——模态和预训练骨干不同 | C3 |

---

## Step 4: Problems It Solves or Alleviates

| # | 问题 | 维度 | 解决/缓解 | 对应优点 |
|---|-----|------|----------|---------|
| S1 | 稀疏奖励导致策略梯度信息密度极低 | Technical | **解决** | A2, A4 |
| S2 | 真机采样效率低下，有限 transition 浪费严重 | Operational | **缓解** | A1, A2 |
| S3 | Reward Hacking：策略停留在视觉/状态相似但未完成任务的状态 | Technical | **缓解** | A1, A3 |
| S4 | 现有势能方法中段梯度畸变（过陡或过平） | Technical | **解决** | A2 |
| S5 | 获取失败负样本需要额外人工操作或在线收集 | Operational | **解决** | A1 |
| S6 | VLM/LLM 奖励方法推理成本高 | Cost | **解决** | A5 |

---

## Step 5: Problems It Introduces

| # | 问题 | 维度 | 来源 | 严重程度 |
|---|-----|------|------|---------|
| I1 | 对 RLinf 代码库需要做二次开发：新增离线训练流程 + 修改奖励注入接口 | Engineering | C2 | 中 |
| I2 | 倒放负样本可能引入非物理状态转移（不可逆任务） | Technical | D2 | 中 |
| I3 | 势能函数泛化到新任务需要重新训练 | Operational | D3 | 低 |
| I4 | 高势能区静态惩罚可能与 SAC 熵正则产生冲突 | Technical | D4 | 中 |
| I5 | 基线对比的公平性：ReWiND 用图像+语言，GCR 用 VIP 骨干，本方法用底层状态 | Experimental | D5 | 高 |
| I6 | 任务选取需要平衡"足够长程"与"状态空间连续可逆"两个约束 | Experimental | D2 | 中 |
| I7 | 超参数敏感性：缩放系数 $c$、$\gamma$、三个损失项权重需要仔细调优 | Technical | C5 | 中 |

---

## Step 6: Evolution Roadmap (演进历程)

### Milestone 0: 代码库理解与任务选定 (2026.02 — 2026.03.01) [已基本完成]

- [x] 安装 RLinf (embodied + openvla + maniskill_libero) 
```bash
bash requirements/install.sh embodied --model openvla --env maniskill_libero --use-mirror --install-rlinf
```
- [ ] 梳理 RLinf SAC 训练回路的完整数据流
- [ ] 确定实验任务（从 ManiSkill 或 LIBERO 中选择稀疏奖励长程任务）
- [ ] 确定评估指标（成功率、收敛步数、样本效率曲线）
- **交付物**：任务选定文档、基线运行脚本

### Milestone 1: 倒放样本增广模块 [已完成]

**目标**：提供一个数据管道，将纯成功的专家演示自动转化为包含"倒放负样本"的 PyTorch Dataset，供下游势能函数网络进行自监督对比学习。

**产出**：

| 产出物 | 路径 | 说明 |
|--------|------|------|
| 格式转换脚本 | `scripts/tmper/convert_h5_to_pkl.py` | 将 ManiSkill3 回放后的 `.h5` 轨迹文件转换为逐条 `.pkl` 文件 |
| 倒放增广核心模块 | `rlinf/data/rewind_augmentation.py` | `rewind_augment_trajectory()` 函数 + `RewindAugmentedDataset` 类 |
| 可视化调试脚本 | `scripts/tmper/visualize_rewind.py` | 将增广轨迹渲染为 `.mp4`，叠加 progress_step 和 FORWARD/REWIND 标识 |
| 专家演示数据 | `data/demos/PickCube-v1/` | 10 条 PickCube-v1 成功轨迹（运动规划生成） |
| 增广数据集 | `data/demos/PickCube-v1/augmented/` | 60 条增广轨迹（10 原始 + 50 倒放变体） |
| 调试视频 | `data/demos/PickCube-v1/debug_rewind_*.mp4` | 原始轨迹 + 多条增广轨迹的渲染视频 |
| 详细使用文档 | [.cursor/docs/1-rewind-aug.md](.cursor/docs/1-rewind-aug.md) | 含需求、原理、验收标准及完整开发教程 |

**进度**：

- [x] 使用 ManiSkill3 运动规划生成 10 条 PickCube-v1 成功轨迹（100% 成功率，平均 72.6 步）
- [x] 回放轨迹提取 state 观测（42 维本体感知状态）
- [x] 实现 `convert_h5_to_pkl.py`：H5 → 逐条 PKL（含 env_states 用于渲染）
- [x] 实现 `rewind_augment_trajectory()`：切断点 + 倒放步数随机采样，生成带 `progress_step` 的增广序列
- [x] 实现 `RewindAugmentedDataset`：封装为标准 `torch.utils.data.Dataset`，输出含全部 10 个必要字段
- [x] 实现 `visualize_rewind.py`：通过 `set_state_dict` 恢复物理状态，渲染 mp4 并叠加进度标签
- [x] 端到端冒烟测试通过：字段完整性、形状一致性、进度标签单调性、锚点正确性、traj_id 唯一性
- [x] pre-commit (ruff lint + format) 通过

### Milestone 2: 训练势能模型 (Train Potential Model) [已完成]

**目标**：训练参数化势能函数 $\Phi_\theta(s) \in (0, 1)$，在任意物理状态下输出单调反映任务完成度的标量分数，满足 PBRS 差分公式。

**产出**：

| 产出物 | 路径 | 说明 |
|--------|------|------|
| 扩展的数据管道 | `scripts/tmper/convert_h5_to_pkl.py` | 支持 `--rgb-h5-path` 存储 RGB 图像 |
| 扩展的增广数据集 | `rlinf/data/rewind_augmentation.py` | 输出含 `images` 字段的增广轨迹 |
| 势能网络定义 | `rlinf/algorithms/rewards/tmper/potential_net.py` | PotentialNetwork + 损失函数 + PotentialPairDataset |
| 训练脚本 | `scripts/tmper/train_potential.py` | 离线自监督训练入口 |
| 评估脚本 | `scripts/tmper/eval_potential.py` | 三项验收测试 + 势能曲线可视化 |
| 模型 checkpoint | `data/checkpoints/tmper/potential_phi.pt` | 冻结权重（val pairwise acc 85.3%） |
| 评估图表 | `data/eval/tmper/*.png` | 进度条测试、掉落测试、静止测试可视化 |
| 详细设计文档 | `.cursor/docs/2-train-potential-model.md` | 含理论推导、网络架构、损失函数设计 |

**进度**：

- [x] 扩展数据管道：replay RGB 观测，修改 `convert_h5_to_pkl.py` 存储 `images` 字段
- [x] 修改 `RewindAugmentedDataset` 支持 `images` 字段透传
- [x] 实现 `PotentialNetwork`：LightweightImageEncoder64 (256d) + 状态 MLP (128d) + 融合头 → Sigmoid
- [x] 实现三个损失函数：L_rank (时序比例排序)、L_bc (边界标定)、L_smooth (平滑约束)
- [x] 实现 `PotentialPairDataset`：同轨迹内配对采样
- [x] 实现训练脚本：train/val 轨迹级拆分、Adam 优化、best checkpoint 保存
- [x] 实现评估脚本：进度条测试 (PASS)、掉落测试 (PASS)、静止测试 (PASS)
- [x] 端到端冒烟测试通过
- [x] pre-commit (ruff lint + format) 通过

### Milestone 3: 在线 PBRS 接入reward模块


### Milestone 4: 总结实验

### Milestone 5: 论文撰写

---

## Step 7: Feasibility & Extension Analysis

### I1: 倒放负样本非物理性 — Trade-off

**分析**：倒放假设状态转移可逆。对于 pick-and-place 类任务（抓取→移动→放置），倒放后（放置→移动→抓取）在运动学上是合理的。但对于切割、倾倒、组装等不可逆操作，倒放会产生非物理状态。

**方案**：在任务选取上主动规避不可逆操作。ManiSkill 的 `PickCube`, `StackCube`, `PegInsertionSide` 以及 LIBERO 的大部分 spatial/goal 任务都是可逆的，适合本方法。接受并文档化此 trade-off。

### I2: 静态惩罚与 SAC 熵正则冲突 — Critical & Solvable

**分析**：当 $o_{t+1} = o_t$ 时，$r_{\text{shape}} = \Phi(o_t)(\gamma - 1) < 0$。在高势能区（接近终点），这个负值更大。同时 SAC 的熵项鼓励探索（包括原地不动），两者可能冲突。

**方案**：
1. 通过缩放系数 $c$ 控制势能函数的动态范围
2. SAC 熵系数 $\alpha$ 的自动调节机制已在 RLinf 中实现（`EntropyTemperature`，支持 softplus/exp/fixed），可通过 `target_entropy` 调节平衡点
3. 作为消融实验的重点项（Milestone 3）


### I6、3: 任务选取 — Low Priority (已有明确方向)

ManiSkill 中的候选任务（按长程性排序）：
- `PickCube`：基础验证任务，episode 较短，用于快速迭代
- `StackCube`：中等长程，需抓取+定位+放置
- `PegInsertionSide`：需精确对位，奖励更稀疏
- `PutCarrotOnPlateInScene` / `PutOnInSceneMulti`：多步操作，RLinf 已有自定义任务实现

LIBERO 中的候选任务：
- LIBERO-Spatial / LIBERO-Goal：多步操作，稀疏奖励

---

## Appendix A: RLinf Architecture Integration Map

```
┌─────────────────────────────────────────────────────┐
│                    OFFLINE PHASE                     │
│                                                     │
│  Expert Demos ──→ Rewinding Augmentation            │
│  (TrajectoryReplayBuffer)    │                      │
│                              ▼                      │
│                    Augmented Dataset                 │
│                    (τ̃ with progress labels P)        │
│                              │                      │
│                              ▼                      │
│                    Potential Network Φ_θ             │
│                    (MLP + Sigmoid → [0,1])           │
│                    L_rank + L_bc + L_smooth          │
│                              │                      │
│                              ▼                      │
│                    Frozen Checkpoint (.pt)           │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                    ONLINE PHASE (RLinf)              │
│                                                     │
│  EnvWorker                                          │
│  ├─ env.chunk_step() → obs, sparse_reward           │
│  ├─ _calc_step_reward() ←── inject Φ here           │
│  │   r_total = r_sparse + γΦ(s') - Φ(s)            │
│  └─ compute_bootstrap_rewards() → ChunkStepResult   │
│                              │                      │
│                              ▼                      │
│  SAC Actor Worker (fsdp_sac_policy_worker.py)       │
│  ├─ ReplayBuffer.add_trajectories()                 │
│  ├─ forward_critic(): Q-target with r_total         │
│  ├─ forward_actor(): π optimization                 │
│  └─ forward_alpha(): entropy tuning                 │
└─────────────────────────────────────────────────────┘
```

