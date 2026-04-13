这份文档是为你团队中的工程师量身定制的**实验执行指导书（Engineering Spec for Decisive Experiments）**。它褪去了论文里的学术叙事，直接将我们在 Section 5 中规划的三个核心实验转化为代码层面的“任务清单（What）”、“执行路径（How）”和“验证指标（Expectation）”。

你可以直接将这份文档发给负责代码实现的工程师。

---

# 实验执行指导书：Bridging the Monotonicity Trap

## 0. 全局实验设定 (Global Setup)

- **测试环境 (Environments)**:
    
    - _Meta-World_: Button Press, Drawer Open, Sweep Into, Hammer (验证细粒度操作)。
        
    - _DeepMind Control (DMC)_: Walker Walk, Quadruped Run (验证高频连续运动)。
        
- **数据集构建 (Data Generation)**:
    
    - 不需要人工标注。在每个环境中使用中等水平的 SAC 策略或添加噪声的专家策略收集轨迹库。
        
    - 利用环境底层的 Ground-Truth 进度（如距离目标的欧氏距离）或启发式规则（成功轨迹 $\succ$ 失败轨迹，时间倒放轨迹 $\prec$ 原轨迹）自动生成成对偏好数据集 $\mathcal{D} = \{(\tau_w, \tau_l)\}$。
        
- **下游 RL 算法**: PPO 和 SAC（必须测两个，以证明势函数对 On-policy 和 Off-policy 算法都稳定）。
    

---

## 1. 核心模型定义 (The 3 Conditions)

工程师需要编写并离线训练以下三个基础势函数模型 $\Phi(s)$：

- **Condition 1: Pure Ranking (Baseline)**
    
    - **Loss**: 仅使用 Bradley-Terry (BT) Loss。
        
    - **代码对应**: 传统的 T-REX / Rewarding DINO 机制。
        
- **Condition 2: Structure-Regularized (Ours)**
    
    - **Loss**: $\mathcal{L}_{total} = \mathcal{L}_{BT} + \lambda \mathcal{L}_{struct}$ (注意代码里 $\mathcal{L}_{struct}$ 的实现：求相邻帧差分 -> Softplus -> 轨迹内归一化 -> 算信息熵)。
        
    - **代码对应**: 我们的核心贡献。
        
- **Condition 3: Progress Oracle (Upper Bound)**
    
    - **Loss**: MSE Loss。直接回归环境给定的绝对进度值（如 $1.0 - d/d_{init}$）。
        
    - **代码对应**: ROBOMETER 的特权绝对进度机制。
        

> ⚠️ **部署前必须执行 (Mandatory Step)**: 所有三个模型在接入下游 RL 的 PBRS 接口 $F = \gamma\Phi(s') - \Phi(s)$ 前，**必须**使用验证集专家数据进行 Zero-Shot Canonicalization，统一映射到 $[0, 1]$ 区间，以保证 RL 接收到的 Reward Scale 是公平的。

---

## 2. 实验拆解与预期 (Experiment Breakdown)

### Phase I: 诊断测试 - 方差坍缩 (The Diagnostic Test)

**做什么 (What):** 证明单调性陷阱存在，且我们的方法能解决它。这是整篇论文最核心的决定性实验。

**怎么做 (How):**

1. 固定相同的训练集 $\mathcal{D}$。
    
2. 使用 5 个不同的随机种子（Random Seeds），训练 5 个 Cond 1 模型和 5 个 Cond 2 模型。
    
3. **离线评估**: 计算这 10 个模型在测试集上的 Kendall-$\tau$ 排序准确率。
    
4. **在线部署**: 将这 10 个冻结的模型作为 Reward 喂给 PPO，每个环境跑 200万步（或直到收敛）。记录每个种子的最终成功率。
    

**预期结果 (Expectation & Checkpoint):**

- **离线端**: Cond 1 和 Cond 2 的 Kendall-$\tau$ 都应该极高（例如 $> 90\%$），且两者没有统计学差异。
    
- **在线端**: Cond 1 的 5 个种子，其下游 RL 成功率会出现**极端方差**（比如 Seed A $95\%$，Seed B $10\%$ 梯度崩溃）。
    
- **核心胜利**: Cond 2 的 5 个种子，RL 成功率方差**急剧坍缩**，全部稳定在高位。
    
- _工程师复盘点_: 如果 Cond 1 的方差不大，说明现有的网络结构（如 Weight Decay）已经隐式锁死了曲率，此时需调大网络容量或去除正则化，必须把“坑”复现出来。
    

### Phase II: 样本效率与终极性能 (Sample Efficiency Benchmark)

**做什么 (What):** 证明我们只用纯偏好数据，就能打平依赖特权标签的 Oracle。

**怎么做 (How):**

1. 画出 Cond 1, Cond 2, Cond 3 在所有 6 个任务中的 RL 学习曲线（环境交互步数 vs. Success Rate）。
    
2. 计算学习曲线的 AUC (Area Under Curve) 来量化样本效率。
    

**预期结果 (Expectation & Checkpoint):**

- Cond 1 学习极慢，或者在中途崩溃。
    
- Cond 2 的收敛速度显著快于 Cond 1。
    
- **核心胜利**: Cond 2 的 AUC 达到 Cond 3 (Oracle) 的 $95\%$ 以上。向审稿人证明：加了 $\mathcal{L}_{struct}$，就再也不需要做人工进度打标了。
    

### Phase III: 结构约束消融实验 (Ablation on Curvature Penalty)

**做什么 (What):** 证明为什么要用“最大熵”，普通的“$L_2$ 平滑”为什么不行。

**怎么做 (How):**

1. 增加 **Condition 4 (L2 Smooth)**: $\mathcal{L}_{total} = \mathcal{L}_{BT} + \lambda_{smooth} \mathbb{E}[(\Delta\Phi_t)^2]$。
    
2. 比较 Cond 2 (MaxEnt) 和 Cond 4 (L2) 在有明显“语义瓶颈”的任务（如 Drawer Open，抓取瞬间需要 Reward Spike）上的表现。
    
3. 提取它们学到的势函数曲线，画出沿着一条专家轨迹的 $\Phi_t$ 变化图。
    

**预期结果 (Expectation & Checkpoint):**

- **曲线可视化**: Cond 4 (L2) 的势函数曲线会被拉得像一把笔直的尺子（完全线性），忽略了抓取物体的关键进度；而 Cond 2 (MaxEnt) 虽然整体平滑，但允许在关键状态保留合理的非线性爬升。
    
- **RL 性能**: Cond 2 的收敛速度和最终成功率跑赢 Cond 4。
    
- _工程师复盘点_: 这一步需要输出一张直观的 2D Line Chart，这很可能成为论文里的核心配图。
    

---

## 3. 工程师 Watch-outs (避坑指南)

1. **数值不稳定性 (Softplus 溢出)**：在算 $\mathcal{L}_{struct}$ 里的概率分布 $p_t$ 时，由于用到了 Softplus 分母求和，注意加上微小的 $\epsilon=1e-8$ 以防止除零错误。
    
2. **$\lambda$ 的调参 (Hyperparam Tuning)**：$\lambda$ 太大会破坏排序准确率（强行拉平），太小则约束不住方差。工程师需要在一个简单环境（如 Sweep Into）上跑一个 Grid Search，找到让 Kendall-$\tau$ 刚开始轻微下降之前的那个临界 $\lambda$ 值，并在所有任务中固定它。
    
3. **PBRS 伽马值对齐 (Gamma Matching)**：下游 PPO/SAC 的折扣因子 $\gamma$ 必须与 PBRS 公式 $F = \gamma \tilde{\Phi}(s') - \tilde{\Phi}(s)$ 中的 $\gamma$ 完全一致（通常设为 0.99），否则理论失效。