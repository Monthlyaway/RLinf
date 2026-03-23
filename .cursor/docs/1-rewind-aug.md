# 需求
在强化学习的长程稀疏任务中，仅利用专家演示（Expert Demonstrations）训练奖励模型会导致严重的“分布偏移（Distribution Shift）”与“奖励投机（Reward Hacking）”。因为专家的离线数据集具有结构性的“失败缺失”——模型只见过不断通向成功的理想轨迹，而从未见过失败边界。

- 分布偏移：假设你平时只教小狗在晴天去绿色的草地上捡球，结果比赛那天下了大雪，草地全白了。小狗懵了，完全不知道该怎么做。这就是“分布偏移”——AI在实际应用中遇到的真实情况，和它平时学习时见过的数据长得完全不一样，导致它原本学到的经验彻底失效了。研究人员通常会给AI看很多“人类完美完成任务”的视频（即成功的专家演示数据），用来教它识别什么是好动作。但是，当机器人自己去尝试探索时，它难免会犯错，比如把水杯拿起来一半又掉回了桌上。因为AI的“裁判打分系统”以前从来没见过这种失败的动作，这种没见过的新场景就是一种“分布偏移”。面对这种超纲题，打分系统常常会发生误判，瞎给分数，甚至把错误动作认成好动作。
- 奖励投机： 想象一下，你跟小孩说：“只要房间看起来干净，我就给你买冰淇淋作为奖励。” 结果小孩为了图省事，直接把满地乱七八糟的玩具全踢进了床底下。从表面上看，房间确实变干净了（小孩成功骗到了冰淇淋奖励），但实际上房间根本没收拾好。AI也是个极度聪明的“偷懒大王”，只要你的奖励规则有漏洞，它就不会去老老实实干活，而是想尽办法非法刷分。“奖励投机”是指机器人发现了一种能获得高分的方法，但其实根本没有推进真正的任务目标。比如，研究人员用一张“机械臂抓住方块”的图片作为满分标准，只要机器人的当前动作越像这张图，得分就越高。结果机器人根本不去学怎么抓方块，而是学会了**“伪装目标（Goal Mimicry）”**：它直接把机械臂摆出一个假装在抓东西的姿势，或者故意把机械臂挡在摄像机镜头前面，让画面“看起来”很像成功的样子。它通过这种“钻空子”的静态摆拍骗取了最高奖励，但实际上末端夹爪完全错过了物体，方块依然原封不动地躺在桌面上。

要一个包含失败轨迹的数据集，能够在训练的初期就明白什么是向成功推进，什么是向失败推进。

# 理解

## 倒放增广的实现原理

**底层数学表达式：**
为构建完整的时序流形，系统对单向正样本轨迹进行截断与翻转。定义原始专家成功轨迹的底层物理状态序列为 $\tau = (s_1, s_2, ..., s_T)$。

1. **切断与倒放采样**：在区间 $(1, T)$ 内均匀采样切断点索引 $i$，并在区间 $(1, i)$ 内随机采样倒放步数 $k$。
2. **序列拼接**：提取前向物理执行子序列 $S_{forward} = (s_1, ..., s_i)$，并提取 $s_i$ 之前的动作序列进行逆序处理，生成倒放子序列 $S_{rewind} = (s_{i-1}, ..., s_{i-k})$。将两者拼接得到增强状态序列 $\tilde{\tau} = [S_{forward}, S_{rewind}]$。
3. **分配进度标签**：为 $\tilde{\tau}$ 中的每一个物理状态分配其在原始绝对时间轴上的进度值，构建同步标签序列 $P$：
   $$P = (1, 2, ..., i, i-1, ..., i-k)$$


**外行人可以理解的通俗解释：**
想象你在看一段机器人完美“把水杯放到桌子上”的录像。为了教会AI“什么是不好的动作”，我们把录像播到一半（比如机器人刚好把水杯举到半空中），突然按下“倒退键”，让录像往回倒播几秒（机器人把水杯放回原处并松开手）。我们把这段“正常播放+倒退播放”的录像拼接在一起，并给每一帧贴上一个“进度百分比”。
在正常播放时，进度百分比不断增加；而一旦进入倒退播放，进度百分比就开始下降。这样一来，AI在学习时就会发现：“原来动作按照这个相反的顺序执行，进度是会倒退的”。通过这种方式，AI不仅学会了怎么走向成功，更深刻地记住了“把快做好的事情搞砸了”是需要被扣分的。


## ManiSkill 中的具体实施案例解释

以 ManiSkill 环境中的经典操作任务 `PickCube-v0`（抓取并抬起方块）为例。

假设一条真实的专家轨迹共有 $T = 100$ 步：
*   **步骤 1-30**：机械臂靠近桌面上的方块。
*   **步骤 31-40**：机械臂闭合夹爪，成功抓稳方块。
*   **步骤 41-100**：机械臂向上抬起方块到达目标高度。

**执行倒放增广：**
假设我们在 $i = 60$ 的地方切断，并设定倒放步数 $k = 25$。
*   **前向序列 ($S_{forward}$)**：包含步骤 $1 \to 60$。在第 60 步时，机械臂已经抓着方块抬到了一半的高度。这 60 步的进度标签 $P$ 严格递增为 $1 \to 60$。
*   **倒放序列 ($S_{rewind}$)**：从第 59 步逆向截取到第 35 步。在物理表现上，这段序列看起来像是：机械臂将抬到一半的方块**重新放回桌面，并开始松开夹爪**。对应的进度标签 $P$ 从 $59$ 严格递减至 $35$。

# 验收

提供一个高度工程化的数据管道（Data Pipeline），将纯成功的专家演示转化为可直接供下游reward网络进行自监督对比学习的 PyTorch Dataset。借鉴 ReWiND 等框架的设定，该模块应当能够在无需人类重新采集失败数据的前提下，自动合成满足物理时序惩罚的负样本
。

1. demo数据生成：在 ManiSkill 环境中，用pick cube v0任务，使用已收敛的模型生成10条成功轨迹
   1. 每条原始轨迹需包含state（RGB图，本体位姿），action以及termination, truncation, done
2. 增广数据集必须包含的字段
   1. demo 有的都要有
   2. progress_step  整数型物理进度标签。正向序列中单调递增，倒放序列中单调递减（例如 1,2,...,60,59,...,35）。用于计算 L_rank, 和 L_smooth 中的 P_u 和 P_v
   3. max_progress ：原轨迹的最大长度。用于在计算归一化动态边距 作为分母
   4. is_start_anchor：布尔值（Boolean）。用于标记该帧是否为物理任务的绝对起点，也就是P=1的帧。用于下游过滤数据以计算起始边界损失 pi(start)-0
   5. is_success_anchor：布尔值（Boolean）。用于标记该帧是否为真正触发环境 Success 的目标状态帧（即P=T_max的帧）。用于计算终止边界损失 pi(success)-1
   6. traj_id：轨迹唯一标识符（整数或字符串）。因为下游的L_rank要求状态对(s_u,s_v)必须来自同一条轨迹。跨轨迹采样会导致进度标签P失去相对约束标准。
3. 格式要求：输出的增广数据集可以直接被封装进 torch.utils.data.Dataset。
4. 提供一个调试脚本，将某一条生成好的带有进度标签 P的增广轨迹重新渲染为 .mp4 视频，并在画面左上角打印当前的 progress_step。人类肉眼观察视频时，当机械臂发生物理回退（例如松开夹爪导致方块掉落），画面上的进度数值必须同步下降。

---

# 开发文档：倒放增广模块使用教程

本节介绍倒放增广数据管道的具体实现与使用方法。完整的管道分为三个阶段：

1. **生成专家演示** — 使用 ManiSkill3 内置运动规划生成成功轨迹
2. **格式转换** — 将 `.h5` 轨迹文件转换为逐条 `.pkl` 文件
3. **倒放增广** — 对每条成功轨迹执行截断与翻转，生成带进度标签的增广数据集

## 快速开始

以 ManiSkill3 的 `PickCube-v1`（抓取方块）任务为例，完整执行全部阶段：

```bash
# 0. 激活虚拟环境
source .venv/bin/activate

# 1. 使用运动规划生成 10 条成功轨迹
python -m mani_skill.examples.motionplanning.panda.run \
  -e "PickCube-v1" -n 10 --only-count-success \
  --obs-mode none -b cpu --record-dir data/demos

# 2. 回放轨迹，提取 state 观测
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path data/demos/PickCube-v1/motionplanning/*.h5 \
  -o state --save-traj

# 3. 将 h5 转换为 pkl
python scripts/tmper/convert_h5_to_pkl.py \
  --h5-path data/demos/PickCube-v1/motionplanning/*.state.*.h5 \
  --output-dir data/demos/PickCube-v1/raw_pkl

# 4. 构建增广数据集（在 Python 中使用）
python -c "
from rlinf.data.rewind_augmentation import RewindAugmentedDataset
ds = RewindAugmentedDataset('data/demos/PickCube-v1/raw_pkl', num_augmentations=5, seed=42)
print(f'增广后轨迹总数: {len(ds)}')
ds.save('data/demos/PickCube-v1/augmented')
"

# 5. 可视化某条增广轨迹
python scripts/tmper/visualize_rewind.py \
  --raw-pkl data/demos/PickCube-v1/raw_pkl/traj_0.pkl \
  --aug-index 1 \
  --output data/demos/PickCube-v1/debug_rewind.mp4
```

---

## 阶段 1：生成专家演示数据

### 运动规划生成

ManiSkill3 内置了多个任务的运动规划求解器，无需训练 RL 模型即可生成高质量专家轨迹。支持的任务包括 `PickCube-v1`、`StackCube-v1`、`PegInsertionSide-v1` 等。

```bash
python -m mani_skill.examples.motionplanning.panda.run \
  -e "PickCube-v1" \
  -n 10 \
  --only-count-success \
  --obs-mode none \
  -b cpu \
  --record-dir data/demos
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `-e` | ManiSkill3 环境 ID |
| `-n` | 生成的轨迹数量 |
| `--only-count-success` | 仅保存成功轨迹，失败的自动丢弃重试 |
| `--obs-mode none` | 录制时不存储观测（后续回放时再提取） |
| `-b cpu` | 使用 CPU 后端（运动规划为单环境，无需 GPU） |
| `--record-dir` | 输出目录 |

**输出目录结构：**

```
data/demos/PickCube-v1/motionplanning/
├── 20260323_132703.h5      # 轨迹数据（HDF5 格式）
└── 20260323_132703.json    # 元数据（episode 信息）
```

**控制台输出示例：**

```
Motion Planning Running on PickCube-v1
proc_id: 0: 100%|██████████| 10/10 [00:03, success_rate=1, avg_episode_length=72.6, max_episode_length=88]
```

### 回放并提取 state 观测

运动规划阶段使用 `--obs-mode none` 录制，不包含观测数据。需要通过回放提取所需的 `state` 观测：

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path data/demos/PickCube-v1/motionplanning/20260323_132703.h5 \
  -o state \
  --save-traj
```

回放后会在同目录下生成新的 `.h5` 文件，文件名包含观测模式和控制模式：

```
data/demos/PickCube-v1/motionplanning/
├── 20260323_132703.h5
├── 20260323_132703.json
├── 20260323_132703.state.pd_joint_pos.physx_cpu.h5    # ← 带 state 观测的新文件
└── 20260323_132703.state.pd_joint_pos.physx_cpu.json
```

---

## H5 轨迹文件数据结构

回放后的 `.h5` 文件采用 HDF5 格式，按轨迹分组存储。以 PickCube-v1（T=74 步）为例：

```
├── traj_0/
│   ├── obs                (75, 42)    float32    # 状态观测，T+1 帧
│   ├── actions            (74, 8)     float32    # 关节位置动作，T 帧
│   ├── terminated         (74,)       bool       # 终止标志
│   ├── truncated          (74,)       bool       # 截断标志
│   ├── success            (74,)       bool       # 成功标志
│   └── env_states/                               # 完整物理状态（用于渲染）
│       ├── actors/
│       │   ├── table-workspace  (75, 13) float32
│       │   ├── cube             (75, 13) float32
│       │   └── goal_site        (75, 13) float32
│       └── articulations/
│           └── panda            (75, 31) float32
├── traj_1/
│   └── ...
└── traj_9/
    └── ...
```

**关键维度说明：**

| 字段 | 形状 | 说明 |
|------|------|------|
| `obs` | `(T+1, 42)` | 本体感知状态（关节角度、物体位姿等）。比 actions 多一帧，因为包含 reset 后的初始观测 |
| `actions` | `(T, 8)` | `pd_joint_pos` 控制模式下的 8 维关节位置指令 |
| `terminated` | `(T,)` | 是否因成功/失败而终止 |
| `truncated` | `(T,)` | 是否因达到最大步数而截断 |
| `success` | `(T,)` | 每步的成功标志；成功后所有后续步均为 True |
| `env_states` | 嵌套 group | 完整的物理仿真状态，可用于 `set_state_dict` 恢复场景并渲染 |

配套的 `.json` 文件记录每条轨迹的元信息：

```json
{
  "episodes": [
    {
      "episode_id": 0,
      "episode_seed": 0,
      "control_mode": "pd_joint_pos",
      "elapsed_steps": 74,
      "success": true
    }
  ]
}
```

---

## 阶段 2：格式转换（H5 → PKL）

### convert_h5_to_pkl.py 的功能

脚本位置：`scripts/tmper/convert_h5_to_pkl.py`

该脚本读取回放后的 `.h5` 文件，将每条轨迹拆分为独立的 `.pkl` 文件。格式与 RLinf 的 `CollectEpisode`（`rlinf/envs/wrappers/collect_episode.py`）pickle 导出对齐，方便后续模块复用。

```bash
python scripts/tmper/convert_h5_to_pkl.py \
  --h5-path data/demos/PickCube-v1/motionplanning/20260323_132703.state.pd_joint_pos.physx_cpu.h5 \
  --output-dir data/demos/PickCube-v1/raw_pkl
```

**控制台输出示例：**

```
  traj_0: T=74, state_dim=42, action_dim=8, success=True
  traj_1: T=74, state_dim=42, action_dim=8, success=True
  traj_2: T=50, state_dim=42, action_dim=8, success=True
  ...
  traj_9: T=84, state_dim=42, action_dim=8, success=True

Converted 10 trajectories to data/demos/PickCube-v1/raw_pkl
```

**输出目录结构：**

```
data/demos/PickCube-v1/raw_pkl/
├── traj_0.pkl
├── traj_1.pkl
├── ...
└── traj_9.pkl
```

### PKL 文件数据结构

每个 `.pkl` 文件是一个 Python 字典：

```python
{
    "observations": [obs_0, obs_1, ..., obs_T],     # list, 长度 T+1, 每个元素为 ndarray (42,)
    "actions":      [act_0, act_1, ..., act_{T-1}], # list, 长度 T, 每个元素为 ndarray (8,)
    "rewards":      [r_0, r_1, ..., r_{T-1}],       # list, 长度 T, 稀疏奖励 (成功=1.0, 否则=0.0)
    "terminated":   [t_0, t_1, ..., t_{T-1}],       # list, 长度 T, bool
    "truncated":    [tr_0, tr_1, ..., tr_{T-1}],     # list, 长度 T, bool
    "success":      True,                            # bool, 该轨迹是否成功
    "episode_id":   0,                               # int, 原始 episode 编号
    "elapsed_steps": 74,                             # int, 原始轨迹长度 T
    "env_states": {                                  # dict, 完整物理状态（用于可视化渲染）
        "actors/table-workspace":   ndarray (T+1, 13),
        "actors/cube":              ndarray (T+1, 13),
        "actors/goal_site":         ndarray (T+1, 13),
        "articulations/panda":      ndarray (T+1, 31),
    }
}
```

`observations` 的长度始终比 `actions` 多 1，因为第 0 帧是 `env.reset()` 返回的初始观测。

---

## 阶段 3：倒放增广

### 核心函数：rewind_augment_trajectory

位置：`rlinf/data/rewind_augmentation.py`

```python
from rlinf.data.rewind_augmentation import rewind_augment_trajectory
import numpy as np

augmented_list = rewind_augment_trajectory(
    traj=raw_traj,          # 原始轨迹字典（从 pkl 加载）
    traj_id=0,              # 基础轨迹编号
    num_augmentations=5,    # 每条原始轨迹生成几条增广变体
    rng=np.random.default_rng(42),  # 随机种子
)
# 返回 list[dict]，第 0 个是原始轨迹，第 1~5 个是增广变体
```

**增广算法（以 T=74 的 PickCube 轨迹为例）：**

1. 在 `[2, T-1]` 范围内均匀采样切断点 `i`，例如 `i=34`
2. 在 `[1, i-1]` 范围内均匀采样倒放步数 `k`，例如 `k=28`
3. 构建前向子序列：观测 `obs[0..34]`（35 帧），动作 `act[0..33]`（34 步）
4. 构建倒放子序列：观测 `obs[33], obs[32], ..., obs[6]`（28 帧），对应动作逆序
5. 拼接得到增广序列，分配进度标签：

```
前向段: P = [1, 2, 3, ..., 34, 35]        ← 严格递增
倒放段: P = [34, 33, 32, ..., 8, 7]       ← 严格递减
```

**输入/输出对照：**

| | 输入 | 输出（原始） | 输出（增广） |
|---|---|---|---|
| 数量 | 1 条成功轨迹 | 1 条（完整前向） | N 条（前向 + 倒放） |
| progress_step | 无 | `[1, 2, ..., T+1]` 单调递增 | `[1, ..., i+1, i, ..., i-k+1]` 先增后减 |
| is_success_anchor | 无 | 最后一帧为 True | 全部为 False |
| is_start_anchor | 无 | 第一帧为 True | 第一帧为 True |

### PyTorch Dataset：RewindAugmentedDataset

位置：`rlinf/data/rewind_augmentation.py`

```python
from rlinf.data.rewind_augmentation import RewindAugmentedDataset

ds = RewindAugmentedDataset(
    data_dir="data/demos/PickCube-v1/raw_pkl",  # 包含 pkl 文件的目录
    num_augmentations=5,   # 每条轨迹生成 5 条增广变体
    seed=42,               # 随机种子，确保可复现
)

print(len(ds))  # 60 = 10 条原始 × (1 原始 + 5 增广)
```

**初始化流程：**

1. 扫描 `data_dir` 下所有 `.pkl` 文件
2. 过滤掉 `success=False` 的轨迹
3. 对每条成功轨迹调用 `rewind_augment_trajectory`，生成 `1 + num_augmentations` 条输出
4. 所有结果展平存入 `self.trajectories` 列表

**`__getitem__` 返回值结构：**

每条轨迹是一个字典，包含以下字段（L 为观测帧数，PickCube-v1 的 state_dim=42, action_dim=8）：

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `states` | `torch.float32` | `[L, 42]` | 本体感知状态序列 |
| `actions` | `torch.float32` | `[L-1, 8]` | 动作序列 |
| `rewards` | `torch.float32` | `[L-1]` | 稀疏奖励 |
| `terminated` | `torch.bool` | `[L-1]` | 终止标志 |
| `truncated` | `torch.bool` | `[L-1]` | 截断标志 |
| `progress_step` | `torch.int64` | `[L]` | 进度标签，与 `states` 对齐 |
| `max_progress` | `int` | 标量 | 原始轨迹的最大进度值（= T+1） |
| `is_start_anchor` | `torch.bool` | `[L]` | 仅第一帧为 True |
| `is_success_anchor` | `torch.bool` | `[L]` | 仅原始轨迹的最后一帧为 True |
| `traj_id` | `int` | 标量 | 全局唯一轨迹标识符 |

**`states` 与 `actions` 的对齐关系：** `states` 有 L 帧，`actions` 有 L-1 帧。`states[t]` 是执行 `actions[t]` 前的状态，`states[t+1]` 是执行后的状态。`progress_step` 与 `states` 长度相同（L 帧）。

**持久化到磁盘：**

```python
ds.save("data/demos/PickCube-v1/augmented")
# 输出: aug_traj_0000.pkl, aug_traj_0001.pkl, ..., aug_traj_0059.pkl
```

**下游使用示例（采样状态对用于 L_rank）：**

```python
traj = ds[1]  # 取第 1 条增广轨迹
progress = traj["progress_step"]  # 例如 [1, 2, ..., 9, 8, 7, 6, 5, 4, 3]

# 采样同轨迹内的状态对 (s_u, s_v)，要求 P_u > P_v
u, v = 5, 12   # progress[5]=6, progress[12]=4 → P_u > P_v，可用于 L_rank
s_u = traj["states"][u]
s_v = traj["states"][v]
margin = (progress[u] - progress[v]).float() / traj["max_progress"]
```

---

## 可视化调试工具

### visualize_rewind.py

脚本位置：`scripts/tmper/visualize_rewind.py`

该脚本将增广轨迹渲染为 `.mp4` 视频，在画面左上角叠加当前 `progress_step` 数值和前进/倒放状态标识（绿色 `FORWARD` / 红色 `REWIND`）。

### 用法 1：从原始 pkl 动态生成增广并可视化

```bash
python scripts/tmper/visualize_rewind.py \
  --raw-pkl data/demos/PickCube-v1/raw_pkl/traj_0.pkl \
  --aug-index 1 \
  --output data/demos/PickCube-v1/debug_rewind.mp4
```

| 参数 | 说明 |
|------|------|
| `--raw-pkl` | 原始轨迹 pkl 路径（必须包含 `env_states`，用于物理状态恢复和渲染） |
| `--aug-index` | 增广变体索引：`0` = 原始轨迹（全程 FORWARD），`1~N` = 增广变体（含 REWIND 段） |
| `--num-augmentations` | 增广数量（默认 5），与 `--aug-index` 配合使用 |
| `--seed` | 随机种子（默认 42） |
| `--output` | 输出 `.mp4` 路径 |
| `--fps` | 视频帧率（默认 10） |
| `--resolution` | 渲染分辨率（默认 512） |

### 用法 2：可视化已保存的增广 pkl

```bash
python scripts/tmper/visualize_rewind.py \
  --raw-pkl data/demos/PickCube-v1/raw_pkl/traj_0.pkl \
  --aug-pkl data/demos/PickCube-v1/augmented/aug_traj_0001.pkl \
  --output data/demos/PickCube-v1/debug_rewind.mp4
```

### 输出示例

**原始轨迹（`--aug-index 0`）：** 75 帧，进度标签 `[1, 2, ..., 75]` 全程递增，左上角始终显示绿色 `FORWARD`。

**增广轨迹（`--aug-index 3`）：** 62 帧，进度标签 `[1, 2, ..., 34, 33, 32, ..., 6]`。前 34 帧显示绿色 `FORWARD`，从第 35 帧起切换为红色 `REWIND`，数值从 33 递减至 6。

```
控制台输出:
Saved 62 frames to data/demos/PickCube-v1/debug_rewind_aug3.mp4 (10 fps)
Progress sequence: [1, 2, ..., 34, 33, 32, ..., 6]
```

---

## 文件布局

```
scripts/tmper/
├── convert_h5_to_pkl.py          # H5 → PKL 格式转换
└── visualize_rewind.py           # 增广轨迹可视化（渲染 mp4）

rlinf/data/
└── rewind_augmentation.py        # 核心增广逻辑 + RewindAugmentedDataset

data/demos/PickCube-v1/           # 生成的数据（已 gitignore）
├── motionplanning/               #   运动规划原始 h5 + 回放后 h5
├── raw_pkl/                      #   逐条轨迹 pkl（含 env_states）
└── augmented/                    #   增广后的 pkl（不含 env_states）
```

---

## 注意事项

- **obs_mode 选择**：运动规划阶段使用 `--obs-mode none` 以减小文件体积，然后通过 `replay_trajectory` 按需提取所需的观测模式（`state`、`rgb` 等）。
- **T+1 与 T 的对齐**：observations/states 始终比 actions 多一帧。`states[0]` 是初始观测，`actions[0]` 是第一步动作，`states[1]` 是执行后的观测。
- **env_states 不进入增广数据集**：`env_states` 体积较大且与增广索引存在冗余。可视化工具通过 `progress_step - 1` 索引原始 pkl 中的 `env_states` 来恢复物理状态。
- **可逆性假设**：倒放增广假设状态转移在运动学上可逆。对于 `PickCube`、`StackCube` 等抓取放置类任务，倒放后的轨迹（放下→松手→远离）在物理上是合理的。对于切割、倾倒等不可逆操作，不适用本方法。
