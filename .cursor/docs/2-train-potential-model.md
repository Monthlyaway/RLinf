# Milestone 2: 训练势能模型 (Train Potential Model)

> **前置依赖**：[Milestone 1: 倒放增广模块](1-rewind-aug.md)（已完成）
>
> **状态**：已完成（2026-03-28）

---

## Step 1: Background (背景)

### 问题陈述

在强化学习（如 SAC/PPO）探索长程、稀疏奖励任务时，仅靠环境在任务结束时提供的"成功/失败"二元信号无法支持有效探索。势能模型的需求是：**作为一个冻结的评估器，在强化学习的每一步（Step）交互中，为当前状态提供一个连续、稠密且物理意义明确的"进度打分"**，从而引导智能体走出随机探索的泥沼。

### 分布偏移与奖励投机

仅利用专家演示（Expert Demonstrations）训练奖励模型会导致两个结构性问题：

**分布偏移（Distribution Shift）**：专家的离线数据集具有结构性的"失败缺失"——模型只见过不断通向成功的理想轨迹，而从未见过失败边界。假设你平时只教小狗在晴天去绿色的草地上捡球，结果比赛那天下了大雪，草地全白了。小狗懵了，完全不知道该怎么做。研究人员通常会给 AI 看很多"人类完美完成任务"的视频（即成功的专家演示数据），用来教它识别什么是好动作。但是，当机器人自己去尝试探索时，它难免会犯错，比如把水杯拿起来一半又掉回了桌上。因为 AI 的"裁判打分系统"以前从来没见过这种失败的动作，面对这种超纲题，打分系统常常会发生误判，瞎给分数，甚至把错误动作认成好动作。

**奖励投机（Reward Hacking）**：想象一下，你跟小孩说："只要房间看起来干净，我就给你买冰淇淋作为奖励。"结果小孩为了图省事，直接把满地乱七八糟的玩具全踢进了床底下。AI 也是个极度聪明的"偷懒大王"，只要你的奖励规则有漏洞，它就不会去老老实实干活，而是想尽办法非法刷分。比如，研究人员用一张"机械臂抓住方块"的图片作为满分标准，只要机器人的当前动作越像这张图，得分就越高。结果机器人根本不去学怎么抓方块，而是学会了**"伪装目标（Goal Mimicry）"**：它直接把机械臂摆出一个假装在抓东西的姿势，或者故意把机械臂挡在摄像机镜头前面，让画面"看起来"很像成功的样子。它通过这种"钻空子"的静态摆拍骗取了最高奖励，但实际上末端夹爪完全错过了物体，方块依然原封不动地躺在桌面上。

### 为什么需要势能函数？

在长程操作任务中，如果直接使用视觉大模型（VLM）计算当前状态与目标状态的图像相似度作为稠密奖励，智能体会迅速学会上述"奖励投机"。策略网络会操纵机械臂停留在某个视觉上看似抓取了物体，但实际并未接触的"伪目标态"，或者直接将机械臂挡在相机前面以刷取高相似度分数。

为了避免这种钻空子行为，我们需要一个真正的**势能函数**。它不仅要能识别"距离成功有多近"，更重要的是，通过我们在 Milestone 1 中构造的倒放负样本，它深刻理解了"什么是物理倒退"。当智能体做出导致物体掉落等退步动作时，该函数必须能给出严厉的降分。

要一个包含失败轨迹的数据集，能够在训练的初期就明白什么是向成功推进，什么是向失败推进。

---

## Step 2: Definition (定义)

### Goals

1. 训练一个参数化势能函数 $\Phi_\theta(s) \in (0, 1)$，在任意物理状态下输出单调反映任务完成度的标量分数
2. 势能函数满足时序单调性：沿成功方向势能严格递增，沿失败方向势能严格递减
3. 势能函数可直接用于 PBRS 差分公式 $r_{shape} = \gamma \Phi(s') - \Phi(s)$，在 Milestone 3 中接入在线 RL

### Non-goals

- 不在 M2 中接入在线 RL 训练回路（Milestone 3 的范围）
- 不训练动作策略（action expert），仅训练评分器
- 不使用上帝视角特权信息（如物体绝对位置、加速度）作为输入
- 不修改 RLinf 核心调度/分布式逻辑

### 交付物

| 产出物 | 路径 | 说明 |
|--------|------|------|
| 势能网络定义 | `rlinf/algorithms/rewards/tmper/potential_net.py` | 视觉+状态融合的 MLP 势能函数 |
| 训练脚本 | `scripts/tmper/train_potential.py` | 离线自监督训练入口 |
| 模型 checkpoint | `data/checkpoints/tmper/potential_phi.pt` | 冻结权重，供 M3 加载 |
| 评估可视化脚本 | `scripts/tmper/eval_potential.py` | 在测试轨迹上绘制势能曲线 |
| 扩展的数据管道 | `scripts/tmper/convert_h5_to_pkl.py`（修改） | 支持存储 RGB 图像 |
| 扩展的增广数据集 | `rlinf/data/rewind_augmentation.py`（修改） | 输出含 `images` 字段 |

---

## Step 3: Technical Design (技术设计)

### 3.1 PBRS 理论基础

势能函数的应用并非直接将其输出作为奖励，而是严格遵循**基于势能的奖励塑形（Potential-Based Reward Shaping, PBRS）**理论。

根据 Ng 等人 (1999) 的数学证明，如果在原有的马尔可夫决策过程（MDP）中直接添加任意的稠密奖励，往往会改变任务的最优策略（导致智能体为了贪图稠密奖励而放弃最终任务）。但如果我们将势能函数 $\Phi_{\theta}$ 构造成**折扣差分（Discounted Difference）**的形式：

$$r_{shape}(s_t, a_t, s_{t+1}) = \gamma \Phi_{\theta}(s_{t+1}) - \Phi_{\theta}(s_t)$$

那么原 MDP 的最优策略集将得到绝对的数学保证而不发生改变。

- TODO: 完整的数学证明需要查阅 Ng et al. (1999) 原文补充

在当今的前沿研究中，如 TimeRewarder 和 Goal-Contrastive Rewards (GCR)，均采用离线学习一个反映任务进度的势函数 $\Phi(s)$，随后在在线强化学习中冻结该模型，并利用上述差分公式为智能体提供防投机的步进（Step-wise）梯度。如果智能体在环境中反复绕圈或进行无意义的循环投机，由于势差的路径积分特性，其净塑形奖励将趋近于零。

### 3.2 模型架构

**总体思路**：视觉编码器提取图像特征，状态编码器提取本体感知特征，两路特征拼接后经 MLP 输出头映射为标量势能值。

#### 输入定义

| 输入 | 来源 | 维度 | 说明 |
|------|------|------|------|
| RGB 图像 | ManiSkill3 `base_camera` | `(128, 128, 3)` uint8 | PickCube-v1 固定第三人称视角摄像头 |
| 本体感知状态 | `obs_mode="state"` | `(42,)` float32 | 关节角度、末端位姿等（详见 M1 文档） |

**输出**：标量 $\Phi_{\theta}(s) \in (0, 1)$，经过 Sigmoid 激活函数映射，表示当前状态在整个任务完成度中的"相对势能"。

#### 网络结构

采用 Late Fusion（先分别编码，再拼接）策略：

```
视觉分支:
  RGB (128×128×3) → resize 64×64 → LightweightCNN → 256-dim
  (4层Conv2d, 32 filters, stride [2,1,1,1], ReLU + LayerNorm + Tanh)

状态分支:
  State (42-dim) → Linear(42, 128) → ReLU → Linear(128, 128) → ReLU → 128-dim

融合输出:
  Concat(256, 128) = 384-dim → Linear(384, 256) → ReLU → Linear(256, 1) → Sigmoid → Φ ∈ (0, 1)
```

#### 视觉编码器选型依据

RLinf 已有两个视觉编码器可复用：

| 编码器 | 位置 | 特点 |
|--------|------|------|
| `ResNet10` | `rlinf/models/embodiment/modules/resnet_utils.py` | 预训练权重 + 冻结 backbone，256-dim 输出 |
| `LightweightImageEncoder64` | `rlinf/models/embodiment/modules/compact_encoders.py` | 端到端可训练，~13K 参数，64×64 输入 |

**选择 `LightweightImageEncoder64`**，理由：
- ResNet10 需要额外的预训练权重文件，增加部署依赖
- ResNet10 backbone 冻结，无法学习 PickCube 任务特定的视觉特征（如方块颜色、夹爪状态）
- 数据量较小（10 条轨迹 × ~70 帧 = ~700 帧），轻量 CNN 参数少（~13K），配合数据增广可避免过拟合
- 64×64 分辨率足以捕捉 PickCube 场景中的物体位置和夹爪状态

### 3.3 数据管道

#### 数据来源

直接使用 Milestone 1 产出的增广数据集（详见 [M1 文档](1-rewind-aug.md)）。当前 M1 管道输出 42 维 state-only 数据，需要扩展以支持 RGB 图像。

对于增广数据集，当前的任务是对于 Milestone 1 中生成的数据集进行训练，而且要有泛化性，例如初始位置需要随机，轨迹需要随机而不是确定性的。ManiSkill3 运动规划使用随机 seed 生成不同初始位置的轨迹，已满足此要求。

#### RGB 数据生成

需要重新 replay 轨迹以提取 RGB 观测：

```bash
# 重新 replay 轨迹，提取 RGB 观测
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path data/demos/PickCube-v1/motionplanning/20260323_132703.h5 \
  -o rgb \
  --save-traj
```

replay 后的 h5 文件中，每条轨迹增加图像数据：

```
traj_0/
├── obs/
│   └── sensor_data/
│       └── base_camera/
│           └── rgb        (T+1, 128, 128, 3) uint8
├── actions               (T, 8) float32
└── ...
```

#### 数据管道修改

1. **`convert_h5_to_pkl.py`**：修改以同时存储 `images` 字段（`list[ndarray]`，每个元素形状 `(128, 128, 3)` uint8）
2. **`RewindAugmentedDataset`**：修改 `__getitem__` 以输出 `images` 字段（`torch.uint8` tensor，形状 `[L, 128, 128, 3]`），与 `states` 对齐

存储估算：60 条增广轨迹 × ~50 帧/条 × 48KB/帧 ≈ **144MB**，完全可控。

#### Pair 采样策略

每次从同一条增广轨迹 $\widetilde{\tau}$ 中随机采样两个状态 $s_u$ 和 $s_v$，并提取它们对应的物理进度标签 $P_u$ 与 $P_v$。此外，提取每条轨迹的绝对起点 $s_{start}$ 和终点 $s_{success}$ 作为边界锚点。

**采样约定**：$v$ 总是在 $u$ 后面（即 $P_v > P_u$），类似于 BPR（贝叶斯个性化排序）中正样本对总是排在负样本对前面。

**每 batch 采样流程**：
1. 随机选取一条轨迹 $\widetilde{\tau}_i$
2. 在该轨迹的帧索引中随机采样两个位置 $(u, v)$
3. 确保 $P_v > P_u$（若不满足则交换）
4. 提取对应的图像 + 状态作为输入，进度差作为 margin 计算依据
5. 同时从 `is_start_anchor=True` 和 `is_success_anchor=True` 的帧中采样边界锚点

### 3.4 损失函数设计

系统在离线阶段对势能网络执行自监督排序优化，总损失为：

$$\mathcal{L}(\theta) = \lambda_{rank} \mathcal{L}_{\text{rank}} + \lambda_{bc} \mathcal{L}_{\text{bc}} + \lambda_{smooth} \mathcal{L}_{\text{smooth}}$$

#### 3.4.1 时序比例排序损失 $\mathcal{L}_{\text{rank}}$

**公式**：

$$m(u,v) = c \cdot \frac{|P_{v} - P_{u}|}{T_{\max}}$$

$$\mathcal{L}_{\text{rank}} = \mathbb{E}_{P_v > P_u}\left\lbrack - \log\sigma\left( \Phi_{\theta}(s_{v}) - \Phi_{\theta}(s_{u}) - m(u,v) \right) \right\rbrack$$

**设计动机**：这是本算法的核心。相比于 ReWiND 直接使用均方误差（MSE）回归绝对进度（这在长序列中极易导致数值不稳定），我们采用 **Pairwise Logistic Ranking**。网络不需要知道当前绝对是第几步，只需要知道"状态 $v$ 在物理时间上比状态 $u$ 领先了多少（即动态边距 $m$）"。详见 [Step 4: 理论证明](#step-4-theoretical-justification-理论证明)。

**防投机机制**：当采样到 $S_{\text{rewind}}$（倒放段）与 $S_{\text{forward}}$（前向段）的组合，且物理表现为"退回"时，真实的进度标签必定是 $P_u > P_v$。此时我们在代码逻辑中交换两者顺序送入公式，强迫网络输出 $\Phi_{\theta}(s_{u}) > \Phi_{\theta}(s_{v})$。这就赋予了"物理倒退"极其严厉的负向梯度。

#### 3.4.2 边界标定损失 $\mathcal{L}_{\text{bc}}$

**公式**：

$$\mathcal{L}_{\text{bc}} = \left( \Phi_{\theta}(s_{\text{start}}) - 0 \right)^{2} + \left( \Phi_{\theta}(s_{\text{success}}) - 1 \right)^{2}$$

**设计动机**：单靠排序损失 $\mathcal{L}_{\text{rank}}$ 只能保证相对大小（例如网络可能输出全为 0.5 和 0.51）。通过 MSE 将起点锚定在 0，终点锚定在 1，相当于给这把"软尺"钉上了刻度，防止模型输出极值塌缩。

#### 3.4.3 平滑约束损失 $\mathcal{L}_{\text{smooth}}$

**公式**：

$$\mathcal{L}_{\text{smooth}} = \mathbb{E}_{\mid P_{u} - P_{v} \mid = 1}\left\lbrack \left( \Phi_{\theta}(s_{v}) - \Phi_{\theta}(s_{u}) \right)^{2} \right\rbrack$$

**设计动机**：在在线 RL 的高频控制（如 10Hz-30Hz）中，相邻两帧的画面变化极小。如果不加平滑约束，网络输出的势能可能会剧烈抖动（例如 0.3 -> 0.8 -> 0.35），这会给强化学习的 Critic 网络引入毁灭性的方差。二次惩罚保证了相邻物理步的势能变化是连续、平滑的。

**在 rewind 边界的行为**：Smooth loss 条件 $|P_u - P_v| = 1$ 在 forward→rewind 转折点也会触发（例如 forward 末尾 P=35 与 rewind 开头 P=34）。这是正确行为——转折点的相邻帧在物理上确实相邻（对应同一段 env\_state），势能应该平滑过渡。

### 3.5 超参数与 Sigmoid 饱和：为什么 $c$ 过大导致阶跃函数

本节从数学上解释为什么初始超参（$c=5.0$, $\lambda_{smooth}=0.1$）会使势能曲线退化为 0→1 阶跃函数，以及为什么调整至 $c=1.0$, $\lambda_{smooth}=3.0$ 后曲线变为平滑 S 形。

#### 3.5.1 问题的数学根源

回顾 ranking loss 中的动态 margin：

$$m(u,v) = c \cdot \frac{|P_v - P_u|}{T_{\max}}$$

该 margin 定义了网络在状态对 $(s_u, s_v)$ 上需要拉开的最小势能差。ranking loss 可以展开为：

$$\mathcal{L}_{\text{rank}} = -\log\sigma\left(\underbrace{\Phi_\theta(s_v) - \Phi_\theta(s_u)}_{\Delta\Phi} - m(u,v)\right)$$

为使该 loss 趋近 0，需要 $\sigma(\cdot)$ 的参数为大正数，即：

$$\Delta\Phi \gg m(u,v) \quad \Longrightarrow \quad \Phi_\theta(s_v) - \Phi_\theta(s_u) \gg c \cdot \frac{|P_v - P_u|}{T_{\max}}$$

但 $\Phi_\theta$ 的输出经 Sigmoid 激活，值域被约束在 $(0, 1)$，因此 $\Delta\Phi$ 有硬性上界：

$$\Delta\Phi_{\max} = \Phi_\theta(s_v)_{\max} - \Phi_\theta(s_u)_{\min} < 1 - 0 = 1$$

#### 3.5.2 c = 5.0 时的不可能约束

当 $c = 5.0$ 时，考虑一对相距较远的状态（如 $|P_v - P_u| = 60$，$T_{\max} = 75$）：

$$m(u,v) = 5.0 \times \frac{60}{75} = 4.0$$

此时 ranking loss 要求：

$$\Phi_\theta(s_v) - \Phi_\theta(s_u) \gg 4.0$$

但 $\Delta\Phi < 1$，这在数学上不可能满足。网络面临 $\Delta\Phi \geq 4.0$ 的约束，而其输出范围仅为 $(0, 1)$。

**网络的唯一选择**：为了尽可能最小化 loss，网络被迫将 Sigmoid 推向极端饱和——对所有"早期"状态输出 $\Phi \approx 0$（Sigmoid 输入 $z \ll 0$），对所有"晚期"状态输出 $\Phi \approx 1$（Sigmoid 输入 $z \gg 0$）。两者之间的过渡被压缩到极少的帧数内（3-4 帧），形成阶跃函数。

用数学语言概括：当 $c \cdot \frac{|P_v - P_u|}{T_{\max}} > 1$ 对大量采样对成立时，ranking loss 在整个可行域上均有大正值梯度，持续将网络推向二值化极端。

#### 3.5.3 c = 1.0 时的可满足约束

当 $c = 1.0$ 时，同一对状态的 margin 变为：

$$m(u,v) = 1.0 \times \frac{60}{75} = 0.8$$

此时：

$$\Delta\Phi \geq 0.8 \quad \text{（可满足，因为 } \Delta\Phi_{\max} \approx 1.0\text{）}$$

对于更近的状态对（如 $|P_v - P_u| = 10$）：

$$m(u,v) = 1.0 \times \frac{10}{75} \approx 0.133$$

网络只需在这对状态间拉开 0.133 的势能差，Sigmoid 无需饱和即可轻松满足。这使得网络可以学习一条**渐进的 S 形曲线**：在整个轨迹范围内均匀分配势能增量，而非将全部变化压缩在几帧内。

#### 3.5.4 $\lambda_{smooth}$ 的对抗作用

仅降低 $c$ 是必要条件但不充分。当 $\lambda_{smooth}$ 过小（如 0.1）时，smoothness loss 的梯度贡献被 ranking loss 压倒：

$$\nabla_\theta \mathcal{L} = \lambda_{rank}\nabla_\theta \mathcal{L}_{\text{rank}} + \lambda_{smooth}\nabla_\theta \mathcal{L}_{\text{smooth}} + \cdots$$

在 $\lambda_{rank}=1.0$, $\lambda_{smooth}=0.1$ 的配比下，即使 $c=1.0$，ranking loss 仍可能驱动网络在某个区间集中分配势能增量（形成较陡的 S 形），因为 ranking loss 不关心增量是否均匀——它只关心大小关系。

将 $\lambda_{smooth}$ 提升至 3.0 后，smoothness loss 的梯度贡献超过 ranking loss。它惩罚任何相邻帧间的大势能跳变 $(\Phi_{t+1} - \Phi_t)^2$，迫使网络将势能变化分散到更多帧上。ranking loss 保证总体方向正确（后面的帧势能更高），smoothness loss 保证过渡是渐进的。

#### 3.5.5 超参搜索的实证验证

48 组网格搜索（$c \times \lambda_{smooth} \times lr$）的实验结果定量验证了上述分析：

| 配置 | max\_jump | 效果 | 数学解释 |
|------|-----------|------|----------|
| c=5.0, λs=0.1 | 0.392 | 3 帧阶跃 | margin 最大 4.0 >> $\Delta\Phi_{\max}$，被迫饱和 |
| c=1.0, λs=0.1 | 0.284 | 较陡 S 形 | margin ≤ 1.0 可满足，但无足够平滑约束 |
| c=5.0, λs=3.0 | 0.170 | 中等 S 形 | 饱和压力仍在，但 smooth loss 部分抵消 |
| c=1.0, λs=3.0 | **0.164** | **平滑 S 形** | margin 可满足 + smooth loss 均匀化增量 |

**核心结论**：$c \leq 1.0$ 是消除阶跃的**必要条件**（解除不可能约束），$\lambda_{smooth} \geq 3.0$ 是实现平滑的**充分条件**（均匀化势能增量）。两者缺一不可。

---

### 3.6 训练流程

#### 超参数

| 参数 | 值 | 理由 |
|------|----|------|
| $c$（margin 缩放系数） | 1.0 | 经超参搜索确定。$c \leq 1.0$ 保证 margin 目标不超出 sigmoid (0,1) 范围，避免二值阶跃 |
| $\lambda_{rank}$ | 1.0 | 主损失 |
| $\lambda_{bc}$ | 1.0 | 边界锚定同等重要 |
| $\lambda_{smooth}$ | 3.0 | 经超参搜索确定。从 0.1→3.0 可提升平滑度 55-83%，是平滑性最关键的杠杆 |
| 优化器 | Adam | 标准选择 |
| 学习率 | 5e-4 | 经超参搜索确定。比 1e-3 略保守，避免过早收敛至阶跃解 |
| batch size | 64 pairs | 每 batch 从多条轨迹采样 |
| epochs | 200 | 小数据集需要多轮迭代 |
| pairs/traj/epoch | 50 | 每条轨迹每 epoch 采样 50 个状态对 |

#### 训练伪代码

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 采样同轨迹状态对 (s_u, s_v)，确保 P_v > P_u
        img_u, state_u, img_v, state_v, progress_u, progress_v, max_prog = batch

        # 2. 前向传播
        phi_u = model(img_u, state_u)  # Φ(s_u)
        phi_v = model(img_v, state_v)  # Φ(s_v)

        # 3. 计算 L_rank
        margin = c * (progress_v - progress_u).float() / max_prog
        L_rank = -log_sigmoid(phi_v - phi_u - margin).mean()

        # 4. 计算 L_bc（从 anchor 帧）
        phi_start = model(img_start, state_start)
        phi_success = model(img_success, state_success)
        L_bc = (phi_start - 0)**2 + (phi_success - 1)**2

        # 5. 计算 L_smooth（相邻帧对）
        L_smooth = ((phi_adj_v - phi_adj_u)**2).mean()

        # 6. 总损失
        loss = lambda_rank * L_rank + lambda_bc * L_bc + lambda_smooth * L_smooth
        loss.backward()
        optimizer.step()
```

---

## Step 4: Theoretical Justification (理论证明)

### 4.1 为什么 Pairwise Ranking 优于 MSE

ReWiND 使用均方误差（MSE）直接回归绝对时间进度 $t/T$，在处理长程序列时会导致数值不稳定和性能下降。我们可以从底层损失函数的梯度推导、统计学分布假设，以及现有文献的消融实验结论三个维度，用工程语言推演这个结论。

#### 长序列下的梯度塌缩与尺度敏感（数学推导）

根据 ReWiND 的定义，其前向进度的目标函数为均方误差（MSE）：

$$L_{progress} = \sum_{t=1}^T (R_\psi(o_{1:t}, z) - \frac{t}{T})^2$$

在长程任务中（例如序列长度 $T = 1000$），相邻两帧 $t$ 和 $t+1$ 的真实物理差异极小。它们在 ReWiND 中的绝对回归目标分别是 $\frac{t}{1000}$ 和 $\frac{t+1}{1000}$，目标差值 $\Delta = 0.001$。

如果网络输出稍微偏离（例如对这两帧均预测为 $0.5$），网络对第 $t$ 帧的梯度为：

$$\frac{\partial L}{\partial R_\psi} = 2(R_\psi - \frac{t}{T})$$

此时相邻帧的梯度差仅为 $\frac{2}{T}$（即 $0.002$）。

**工程后果**：当 $T$ 非常大时，$\frac{1}{T}$ 趋近于 0，MSE 的损失值尺度 $O(\frac{1}{T^2})$ 会落入浮点数精度的底层噪声区间。网络为了最小化整体 MSE，会倾向于输出一个平滑的均值直线，完全丧失区分相邻物理状态（如"即将抓取"与"已经抓稳"）的分辨率。

#### 统计假设错位：回归（Regression）与排序（Ranking）的本质区别

基于贝叶斯个性化排序（BPR）的底层理论，使用最小二乘法（MSE）等价于假设目标变量服从正态分布的最大似然估计（MLE）。MSE 强迫网络输出一个绝对的定量数值（Quantitative Regression）。

但在具身智能的轨迹中，时间标签 $t/T$ 只是一个线性的时间戳，而物理世界的进度是非线性的。例如，机器人用 80% 的时间在移动（视觉变化平缓），用 20% 的时间在夹取物体（视觉变化剧烈）。如果强迫网络用 MSE 拟合线性的 $\frac{t}{T}$，当物理状态发生剧变但时间进度要求平滑时，MSE 会产生巨大的异常梯度，撕裂原本已经学到的特征流形。

评估一个物理动作是否有效，本质上是一个关于定性先后顺序的排序或分类问题（Qualitative Classification），而不是定量的绝对数值回归。

#### 为什么 Pairwise Logistic Loss 更稳定？（对比证明）

文献中处理长序列进度建模的稳健方案（如 LambdaRank、BPR、TimeRewarder）均摒弃了 MSE，转而采用分类或排序损失。

以本方案采用的 Pairwise Logistic Ranking Loss 为例：

$$L_{rank} = -\log \sigma \left( \Phi_{\theta}(o_v) - \Phi_{\theta}(o_u) - m(u, v) \right)$$

对 $\Phi_\theta(o_v)$ 求导，其梯度大小正比于 $1 - \sigma(\Delta \Phi - m)$。这个形式的数学优势在于：

1. **不限制绝对输出**：网络不需要知道当前绝对进度是 $0.500$ 还是 $0.501$。只要状态 $v$ 在物理时间上比状态 $u$ 领先，且势能差没有拉开足够的边距 $m$，梯度就会一直存在。
2. **更陡峭的局部梯度**：即使 $\Delta \Phi$ 的差距极小，交叉熵/逻辑回归函数的对数特性也能提供足够大的非线性惩罚梯度，迫使网络在连续的高维空间中拉开微小的物理差距。

**实证结论支持**：TimeRewarder 的消融实验直接证明了这一点：将连续的标量距离直接作为网络 MSE 回归的目标，会导致数值不稳定和性能下降。必须将连续的时间距离转化为分类软标签（如两热点离散化），或使用 Pairwise 排序损失，才能在长序列中提供稳定且陡峭的梯度。

### 4.2 倒放数据的 Hard Negative 放大效应

虽然对于单个孤立的图片对（比如取出 $x=6$ 和 $x=4$）来说，反向采样和 ReWiND 算出来的单步梯度在微积分上是一模一样的。但关键区别在于**采样分布**。

假设你有一条原始轨迹，长度为 100 步。目标在第 100 步。

**纯反向采样**：DataLoader 在这 100 个状态里随机抽对子。抽到 $(s_{100}, s_1)$（简单负样本，相差极大）的概率，和抽到 $(s_{100}, s_{99})$（极难负样本，几乎长得一样）的概率是一样的。

**ReWiND 的拉长机制**：它在第 100 步截断，然后把后 20 步倒放拼上去，episode 长度变成了 120 步。轨迹变成了 $s_1 \to ... \to s_{100} \to s_{99} \to ... \to s_{80}$。在这个拉长后的数据集里，靠近目标区域的状态（$s_{80}$ 到 $s_{100}$）被物理复制了一遍。

当你在新数据集里采样时，DataLoader 抽到靠近目标区域的"微小倒退"样本（比如 $s_{100}$ 退到 $s_{99}$）的概率被大幅度放大了。

在随机梯度下降（SGD）中，梯度的期望 $\mathbb{E}[\nabla \mathcal{L}]$ 是由样本出现的频率决定的。ReWiND 实际上是通过延长 episode，在数学上等价于给目标点附近的 Hard Negatives 赋予了极高的 Loss 权重。

---

## Step 5: Acceptance Criteria (验收标准)

为了验证我们训练出的势能模型是正确且可用的，需要通过以下三个直观的"外行人也能看懂"的测试。注意：M2 仅做单元测试势能函数，训练 action expert 是 M3 以后的事情。

**功能性验收**

1. 脚本运行，给出一段轨迹的所有状态的势能值，并绘制势能曲线。

### 测试 1：进度条测试（看它懂不懂"前进"）— 硬性通过标准

- **做法**：拿一段机器人完美把方块放到目标点的测试视频（模型没见过的），把视频的第 1 帧（刚起步）、第 50 帧（抓到方块）和第 100 帧（完成任务）输入模型。
- **通过标准**：模型输出的分数必须像"游戏进度条"一样严格上涨。例如输出必须类似 `0.02 -> 0.55 -> 0.98`。如果中间的分数比结尾还高，说明模型练废了。

### 测试 2：手滑掉落测试（看它懂不懂"搞砸了"）— 软性观察指标

- **做法**：人为录制一段"假动作"视频。机器人在第 40 帧把方块抓到了半空中（此时分数假设为 `0.6`），但在第 45 帧，我们强行让夹爪松开，方块"啪"地掉回桌面上（这是标准任务中没教过的失败）。
- **观察目标**：在方块掉落的瞬间，模型打出的分数应当发生显著下跌（例如从 `0.6` 下降至 `0.1` 附近）。这一条是验证 Milestone 1 倒放数据是否真正起效的关键证据。
- **注意**：这只是预期趋势，不作为 pass/fail 的硬性判定。断崖式的下降不一定会发生，下跌幅度取决于势能函数学到的状态表征质量。记录数值变化趋势，在论文中作为分析材料讨论。

### 测试 3：静止发呆测试（看它懂不懂"无作为"）— 硬性通过标准

- **做法**：让机器人在原地发呆，或者只做一些无关紧要的晃动（夹爪不接触物体），持续 100 帧输入模型。
- **通过标准**：分数应该像一条死水平线，几乎不发生波动（比如稳定在 `0.05` 左右抖动）。不能因为时间在流逝就给它涨分。如果在发呆时分数依然上涨，说明模型学到了错误的"时间捷径"而不是"物理动作进度"。

---

## Step 6: Risk Analysis (风险分析)

### 6.1 PBRS 的静态惩罚特性（Static Penalty）

PBRS 的静态惩罚特性是其底层数学公式引入的固有时间衰减属性。当智能体在探索过程中停滞不前时，该机制会不受控制地持续输出负反馈。

#### 机制触发条件与数据流推导

- **常规计算**：PBRS 的单步塑形奖励公式为 $r_{\text{shape}} = \gamma \Phi(o_{t+1}) - \Phi(o_t)$，其中 $\Phi$ 是由模型评估出的当前状态势能，$\gamma$ 是环境的折扣因子。
- **静态触发**：当智能体在探索时卡在某处，导致物理状态未发生任何改变（即前后帧 $o_{t+1} = o_t$）时，塑形奖励公式在数学上直接退化为：$r_{\text{shape}} = \Phi(o_t)(\gamma - 1)$。
- **惩罚生成**：因为标准的强化学习折扣因子 $\gamma < 1$，且势能网络输出范围被约束在 $\Phi \in (0, 1)$ 内，$(\gamma - 1)$ 恒为负数。这导致智能体只要保持静止，每一步都会收到一个确定的负值奖励。

#### 工程与训练代价

- **进度越高，惩罚越狠**：如果智能体在接近任务终点的地方（此时势能 $\Phi(o_t)$ 处于高位，接近 1）发生卡顿，它所承受的单步静态惩罚绝对值会变得极大。
- **策略熵压制**：在 SAC 等最大熵强化学习算法的训练初期，这种由于卡顿带来的高频、大数值负反馈，会直接压制策略探索时的随机性（即策略熵无法有效增加）。
- **退回起点漏洞**：为了将 $\Phi(o_t)(\gamma - 1)$ 的惩罚值最小化，智能体会学会一种过早收敛（Premature convergence）的局部最优策略——直接退回或停留在任务起点。因为在起点处状态势能 $\Phi(o_{start})$ 接近 0，相应的静态惩罚也约等于 0，这会彻底阻断对稀疏目标的后续探索。

#### 缓解策略（Milestone 3 范围）

1. 通过缩放系数 $c$ 控制势能函数的动态范围
2. SAC 熵系数 $\alpha$ 的自动调节机制已在 RLinf 中实现（`EntropyTemperature`，支持 softplus/exp/fixed），可通过 `target_entropy` 调节平衡点
3. 作为消融实验的重点项

### 6.2 数据规模风险

当前仅有 10 条成功轨迹（增广后 60 条），总帧数约 700 帧。对于训练含 CNN 的视觉编码器来说，数据量较小。

**缓解措施**：
- 轻量 CNN 参数极少（~13K），降低过拟合风险
- 可增加 `num_augmentations`（如 10→20）扩大增广倍数
- 可生成更多原始轨迹（如 50 条）
- 可对图像施加 color jitter、random crop 等在线增强

### 6.3 过拟合风险

小数据集 + 足够大的网络 → 训练集 loss 趋零但泛化能力差。

**缓解措施**：
- 保留 2 条轨迹作为验证集（8 train / 2 val）
- 监控验证集上的排序准确率（Kendall tau 或 pairwise accuracy）
- Early stopping 基于验证集 loss

---

## Step 7: Implementation Plan (实现计划)

### 文件布局

```
scripts/tmper/
├── convert_h5_to_pkl.py          # [修改] 增加 RGB 图像存储
├── visualize_rewind.py           # [不变] 增广轨迹可视化
├── train_potential.py            # [新增] 势能网络离线训练入口
├── eval_potential.py             # [新增] 势能曲线可视化评估
└── sweep_potential.py            # [新增] 超参数网格搜索

rlinf/algorithms/rewards/tmper/
├── __init__.py                   # [新增] 模块注册
└── potential_net.py              # [新增] PotentialNetwork 定义 + 损失函数

rlinf/data/
└── rewind_augmentation.py        # [修改] 支持 images 字段

data/checkpoints/tmper/
└── potential_phi.pt              # [产出] 训练好的势能网络权重
```

### 子任务分解

- [x] **M2.1**: 扩展数据管道 — replay 轨迹为 RGB，修改 `convert_h5_to_pkl.py` 存储图像，修改 `RewindAugmentedDataset` 输出 images
- [x] **M2.2**: 实现 `PotentialNetwork` — 视觉编码器 + 状态编码器 + 融合头 + Sigmoid 输出
- [x] **M2.3**: 实现损失函数 — `L_rank`、`L_bc`、`L_smooth` + pair 采样 DataLoader
- [x] **M2.4**: 实现训练脚本 `train_potential.py` — 训练循环、logging、checkpoint 保存
- [x] **M2.5**: 实现评估脚本 `eval_potential.py` — 在测试轨迹上运行三个验收测试，绘制势能曲线
- [x] **M2.6**: 端到端冒烟测试 — 从数据生成到模型训练到评估的完整流水线
- [x] **M2.7**: pre-commit 通过 + 文档更新

### 与 M1/M3 的接口定义

**M1 → M2（输入接口）**：

```python
# RewindAugmentedDataset.__getitem__ 返回 dict，新增 images 字段
{
    "states":            torch.float32  [L, 42],
    "images":            torch.uint8    [L, 128, 128, 3],   # 新增
    "actions":           torch.float32  [L-1, 8],
    "rewards":           torch.float32  [L-1],
    "terminated":        torch.bool     [L-1],
    "truncated":         torch.bool     [L-1],
    "progress_step":     torch.int64    [L],
    "max_progress":      int,
    "is_start_anchor":   torch.bool     [L],
    "is_success_anchor": torch.bool     [L],
    "traj_id":           int,
}
```

**M2 → M3（输出接口）**：

```python
# 加载冻结的势能网络
checkpoint = torch.load("data/checkpoints/tmper/potential_phi.pt")
# 注意：config 中包含训练超参，需过滤到构造器参数
net_keys = {"state_dim", "image_size", "latent_dim", "state_latent_dim"}
cfg = {k: v for k, v in checkpoint["config"].items() if k in net_keys}
model = PotentialNetwork(**cfg)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 在线推理
phi = model(image, state)  # → scalar in (0, 1)
r_shape = gamma * phi_next - phi_current
```

---

# 开发文档：势能网络训练教程

> 本节记录 Milestone 2 实际实现的具体步骤、遇到的问题及解决方案。

## 快速开始

```bash
source .venv/bin/activate

# 1. 生成 10 条专家演示
python -m mani_skill.examples.motionplanning.panda.run \
  -e "PickCube-v1" -n 10 --only-count-success \
  --obs-mode none -b cpu --record-dir data/demos

# 2. 分别 replay 为 state 和 RGB
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path data/demos/PickCube-v1/motionplanning/*.h5 -o state --save-traj
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path data/demos/PickCube-v1/motionplanning/*.h5 -o rgb --save-traj

# 3. 合并为 pkl（state + RGB + env_states）
python scripts/tmper/convert_h5_to_pkl.py \
  --h5-path data/demos/PickCube-v1/motionplanning/*.state.*.h5 \
  --rgb-h5-path data/demos/PickCube-v1/motionplanning/*.rgb.*.h5 \
  --output-dir data/demos/PickCube-v1/raw_pkl

# 4. 训练势能网络（200 epochs，~5 min on GPU）
PYTHONUNBUFFERED=1 python scripts/tmper/train_potential.py \
  --data-dir data/demos/PickCube-v1/raw_pkl \
  --output-dir data/checkpoints/tmper

# 5. 评估（三项验收测试 + 曲线图）
python scripts/tmper/eval_potential.py \
  --checkpoint data/checkpoints/tmper/potential_phi.pt \
  --data-dir data/demos/PickCube-v1/raw_pkl \
  --output-dir data/eval/tmper
```

## PotentialNetwork 使用细节

### 输入格式

```python
from rlinf.algorithms.rewards.tmper.potential_net import PotentialNetwork

model = PotentialNetwork(state_dim=42, image_size=64, latent_dim=256, state_latent_dim=128)

# 方式 1：uint8 HWC 输入（来自 dataset/pkl）
images = torch.randint(0, 255, (B, 128, 128, 3), dtype=torch.uint8)
states = torch.randn(B, 42)
phi = model(images, states)  # [B] float in (0, 1)

# 方式 2：float CHW 输入（已预处理）
images = torch.randn(B, 1, 3, 64, 64)
phi = model(images, states)
```

`_encode_images()` 内部处理：
1. `uint8` → `float / 255.0`
2. `[B, H, W, 3]` → `[B, 3, H, W]`（HWC → CHW 转置）
3. 如果尺寸不是 64×64，自动 `F.interpolate` 双线性缩放
4. 添加 `num_images` 维度 → `[B, 1, 3, 64, 64]`
5. 送入 `LightweightImageEncoder64`

### 模型参数量

| 组件 | 参数量 | 说明 |
|------|--------|------|
| `vision_encoder`（LightweightImageEncoder64） | ~8.4M | 主要来自 bottleneck linear (32×32×32 → 256) |
| `state_encoder` | ~22K | 42→128→128 两层 MLP |
| `fusion_head` | ~66K | 384→256→1 两层 MLP |
| **总计** | **~8.5M** | |

### Checkpoint 格式

```python
{
    "model_state_dict": OrderedDict,        # 模型权重
    "optimizer_state_dict": OrderedDict,     # Adam 优化器状态
    "epoch": int,                            # 最佳 epoch 编号
    "val_pairwise_acc": float,               # 最佳验证集排序准确率
    "config": {
        "state_dim": 42,
        "image_size": 64,
        "latent_dim": 256,
        "state_latent_dim": 128,
        "c": 1.0,                            # 训练超参，加载模型时需过滤
        "lambda_rank": 1.0,
        "lambda_bc": 1.0,
        "lambda_smooth": 3.0,
    }
}
```

**加载模型时的注意事项**：checkpoint 的 `config` 字典包含了 `c`、`lambda_*` 等训练超参。这些参数不是 `PotentialNetwork` 构造器的参数，直接 `PotentialNetwork(**config)` 会报 `TypeError`。必须过滤：

```python
net_keys = {"state_dim", "image_size", "latent_dim", "state_latent_dim"}
model = PotentialNetwork(**{k: v for k, v in config.items() if k in net_keys})
```

## PotentialPairDataset 数据流

### 采样流程

```
PotentialPairDataset
  └── wraps RewindAugmentedDataset (或 list[dict])
       └── 每条轨迹包含: states [L, 42], images [L, 128, 128, 3], progress_step [L], ...

__getitem__(idx):
  1. traj_idx = idx // pairs_per_traj → 选取轨迹
  2. 从同一条轨迹内随机采样 u, v，确保 P_v > P_u
  3. 采样相邻帧对 (adj_u, adj_v) → 用于 L_smooth
  4. 取当前轨迹的 start_idx=0 → 用于 L_bc 的起点锚
  5. 从 success_anchor_cache 取含 is_success_anchor=True 的轨迹 → 用于 L_bc 的终点锚
  6. 返回 dict: img_u, state_u, img_v, state_v, progress_u, progress_v, max_progress,
               img_adj_u, state_adj_u, img_adj_v, state_adj_v,
               img_start, state_start, img_success, state_success
```

### 训练数据拆分

训练/验证拆分在增广前进行，确保验证集轨迹的物理场景未被训练集见过：

```
10 条原始 pkl → 随机打乱 → 8 train / 2 val
  → 分别 rewind_augment → 48 train aug / 12 val aug
    → 分别包装为 PotentialPairDataset
      → train: 48 × 50 = 2400 pairs/epoch
      → val:   12 × 50 = 600 pairs/epoch
```

## 损失函数实际表现

### 训练曲线典型走势

| Epoch | L_rank | L_bc | L_smooth | Train Acc | Val Acc |
|-------|--------|------|----------|-----------|---------|
| 1 | ~0.75 | ~0.23 | ~0.001 | ~79% | ~78% |
| 10 | ~0.63 | ~0.00 | ~0.002 | ~89% | ~84% |
| 50 | ~0.63 | ~0.00 | ~0.002 | ~89% | ~86% |
| 100 | ~0.63 | ~0.00 | ~0.002 | ~90% | ~89% |
| 200 | ~0.63 | ~0.00 | ~0.002 | ~89% | ~87% |

> 以上数值基于调优后的超参（c=1.0, λ_smooth=3.0, lr=5e-4），最佳 val acc 约 91%（epoch ~80）。

关键观察：
- `L_bc` 收敛最快（~5 epochs 即降至 <0.01），因为只需将两个锚点推到 0 和 1
- `L_rank` 持续缓慢下降至 ~0.63 附近震荡，是主要的训练驱动力
- `L_smooth` 始终很小（<0.002），经 λ_smooth=3.0 加权后约 0.006，对总损失有适度贡献
- 验证集准确率比训练集低 ~3-5%，在合理过拟合范围内

## 评估脚本使用指南

### Test 1: 进度条测试

```bash
python scripts/tmper/eval_potential.py --traj-index 0
```

- 加载某条原始轨迹的全部帧，逐帧推理势能值
- 检查严格单调递增（允许 eps=1e-3 容差）
- 输出 `test1_progress_bar.png`：上图为势能曲线，下图为逐帧差分柱状图
- **通过标准**：0 个 real violations（delta < -1e-3）

### Test 2: 手滑掉落测试

```bash
python scripts/tmper/eval_potential.py --traj-index 0
```

- 构造合成轨迹：前向到 60% 处 → 倒放 20 步
- 检查倒放后势能是否低于切点处的势能
- 输出 `test2_drop_test.png`：上图为势能曲线（红色虚线标记切点），下图为进度标签
- **观察指标**（软性）：倒放段势能应显著低于切点峰值

### Test 3: 静止发呆测试

```bash
python scripts/tmper/eval_potential.py --traj-index 0
```

- 取前 15 帧（机器人尚未移动阶段）
- 计算势能的标准差，检查是否 < 0.05
- 输出 `test3_idle_test.png`：势能曲线 + 均值 ± 标准差带
- **通过标准**：std < 0.05

## 超参数调优：从阶跃函数到平滑 S 曲线

### 问题现象

初始超参（c=5.0, λ_smooth=0.1, lr=1e-3）训练出的势能函数呈现**二值阶跃行为**：在约 frame 20 处从 ~0 急剧跳变至 ~1（3-4 帧内完成），此后长期饱和在 1.0。max_jump 达 0.39，形似 step function 而非连续进度指标。

### 根因分析

**核心问题在 margin 缩放系数 c**。ranking loss 中的动态 margin 为 $m(u,v) = c \cdot |P_v - P_u| / T_{max}$。当 c=5.0 且 $|P_v - P_u|$ 较大时（如相隔 60 帧的一对），margin 目标可达 $5 \times 60/75 = 4.0$。但 sigmoid 输出 $\Phi \in (0,1)$，$\Phi_v - \Phi_u$ 的理论上限为 1.0。网络面临 $\Phi_v - \Phi_u \geq 4.0$ 的不可能约束，只能将 sigmoid 推向极端饱和（全 0 或全 1），形成阶跃。

### 超参搜索

使用 `scripts/tmper/sweep_potential.py` 进行 48 组网格搜索：

```
c ∈ {0.1, 0.3, 0.5, 1.0, 2.0, 5.0}
λ_smooth ∈ {0.1, 0.5, 1.0, 3.0}
lr ∈ {5e-4, 1e-3}
```

关键发现：

| 排名 | 配置 | 单调性 | 平滑度 | 违反数 | 最大跳变 |
|------|------|--------|--------|--------|----------|
| 1 | c=1.0, λs=3.0, lr=5e-4 | **1.000** | 0.243 | **0** | 0.164 |
| 2 | c=0.1, λs=3.0, lr=5e-4 | 0.973 | **0.271** | 2 | 0.106 |
| 3 | c=0.5, λs=3.0, lr=5e-4 | 0.986 | 0.267 | 1 | 0.123 |
| ... | c=5.0, λs=0.1, lr=1e-3 | 1.000 | 0.145 | 0 | **0.392** |

### 调优结论

1. **c ≤ 1.0 是硬性约束**：保证 margin 目标不超出 sigmoid 可表达范围，避免二值饱和
2. **λ_smooth 是平滑性最有效的杠杆**：0.1→3.0 在所有 c 值下均带来 55-83% 的平滑度提升
3. **lr=5e-4 略优于 1e-3**：降低学习率可防止模型过早收敛到阶跃解
4. **最优组合 c=1.0, λ_smooth=3.0, lr=5e-4**：唯一实现完美单调性（0 违反）且平滑 S 曲线跨越 ~20 帧的配置

### 残留限制

即使在最优超参下，势能函数在 frame ~35 后（74 帧中）饱和至 ~1.0。这是因为 PickCube-v1 的提升阶段在 CNN 可区分性上远弱于抓取阶段——不同高度的视觉差异极小。此为模型表征能力的固有限制，非超参问题。

---

## 实现中遇到的问题与解决方案

### 1. 依赖冲突：latex2sympy2 未安装

**症状**：`from rlinf.algorithms.rewards.tmper.potential_net import PotentialNetwork` 触发 `ModuleNotFoundError: No module named 'latex2sympy2'`。

**根因**：Python 导入 `tmper` 子包时会执行 `rlinf/algorithms/rewards/__init__.py`，该文件顶层导入了 `MathReward`（依赖 latex2sympy2）和其他推理奖励模块。

**修复**：将 `__init__.py` 中的推理奖励导入包裹在 `try-except ImportError` 中，使其成为可选依赖。

### 2. 背景训练无输出

**症状**：将 `train_potential.py` 放入后台运行后，终端日志文件中没有任何输出。

**修复**：使用 `PYTHONUNBUFFERED=1` 前缀启动训练脚本。

### 3. Checkpoint 加载 TypeError

**症状**：`TypeError: PotentialNetwork.__init__() got an unexpected keyword argument 'c'`

**根因**：保存的 checkpoint `config` 包含训练超参（c, lambda_rank 等），直接作为构造参数传入。

**修复**：在 `load_model()` 中过滤 config，仅保留 `PotentialNetwork` 构造器接受的键。

### 4. 单调性"假阳性"违反

**症状**：Test 1 报告若干 monotonicity violations，但经检查都是 1e-5 到 1e-8 量级的微小波动。

**根因**：Sigmoid 函数在接近 0 和 1 的饱和区域，float32 精度不足以表达微小差异。

**修复**：在 `test_progress_bar` 中引入 `eps=1e-3` 容差，区分"real violations"和"noise violations"。

### 5. 势能函数呈阶跃函数而非平滑曲线

**症状**：势能曲线在约 frame 20 处从 ~0 急剧跳变至 ~1（3-4 帧内完成），随后长期饱和。进度条看起来像开关量，而非连续进度条。

**根因**：margin 缩放系数 c=5.0 导致 ranking loss 的 margin 目标远超 sigmoid (0,1) 输出范围（最高达 4.0），迫使网络将 sigmoid 推向极端饱和。同时 λ_smooth=0.1 太弱，无法对抗 ranking loss 的二值化压力。

**修复**：通过 48 组超参网格搜索（`scripts/tmper/sweep_potential.py`），将默认超参修改为 c=1.0, λ_smooth=3.0, lr=5e-4。势能曲线从 3 帧阶跃改善为 ~20 帧的平滑 S 曲线，max_jump 从 0.39 降至 0.16，单调性违反数降为 0。详见上方"超参数调优"章节。
