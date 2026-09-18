# Isaac Lab：创建基于 Manager 的强化学习环境

> **英文原教程：** Creating a Manager-Based RL Environment
> **官方地址：** https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html
>
> 本文对应 Isaac Lab `main` 分支教程，最后更新时间：**2026-09-17**。
>
> 本教程是在上一篇 **Creating a Manager-Based Base Environment** 的基础上，进一步将普通的 Manager-Based 环境扩展为用于强化学习（Reinforcement Learning，RL）的任务环境。

---

## 1. 本教程要解决什么问题？

上一篇教程介绍了如何创建一个基础的 Manager-Based 环境。

基础环境可以完成：

```text
Agent
  │
  │ Action
  ▼
Environment
  │
  │ Observation
  ▼
Agent
```

也就是说，环境提供了一个基本的：

> **感知 → 动作 → 仿真 → 再感知**

接口。

这种接口对于以下任务已经足够：

* 传统运动规划
* 机器人控制
* 手工控制器
* 仿真测试

但是，对于强化学习来说，仅仅有：

```text
Action
Observation
```

是不够的。

强化学习还需要告诉智能体：

> **当前行为到底好不好？**

因此，需要增加：

```text
Reward
```

同时还需要知道：

> **一个 episode 什么时候结束？**

因此需要：

```text
Termination
```

对于某些任务，还需要：

```text
Command
Curriculum
```

于是，Isaac Lab 提供：

```python
ManagerBasedRLEnv
```

来扩展基础的：

```python
ManagerBasedEnv
```

使环境具备强化学习任务所需的任务定义能力。官方教程明确将 `ManagerBasedRLEnv` 定义为在基础环境之上加入 task specification 的环境。

---

# 2. ManagerBasedEnv 与 ManagerBasedRLEnv

可以把两者理解为：

```text
ManagerBasedEnv
│
├── Scene
├── Action
├── Observation
└── Event
```

而：

```text
ManagerBasedRLEnv
│
├── Scene
├── Action
├── Observation
├── Event
│
├── Reward
├── Termination
├── Command
└── Curriculum
```

因此：

```text
ManagerBasedEnv
       │
       │ 扩展
       ▼
ManagerBasedRLEnv
```

官方教程推荐不要直接修改 `ManagerBasedRLEnv` 基类，而是创建：

```python
ManagerBasedRLEnvCfg
```

配置类。

这样可以将：

```text
环境实现
```

与：

```text
任务定义
```

分离。

同一个机器人环境可以通过不同的配置，定义不同的强化学习任务。

---

# 3. 本教程的任务：Cartpole

本教程继续使用：

```text
Cartpole
```

也就是经典的倒立摆。

任务目标：

> **让杆尽可能保持竖直。**

可以把这个任务抽象成：

```text
             Pole
              │
              │
              │
             /│
            / │
       ┌───────────┐
       │    Cart   │
       └───────────┘
             │
        ←────────→
```

智能体控制：

```text
Cart 的水平运动
```

环境观察：

```text
Cart / Pole 的状态
```

奖励鼓励：

```text
Pole 保持竖直
Cart 速度较小
Pole 角速度较小
环境持续运行
```

失败条件包括：

```text
Episode 超时
Cart 超出 [-3, 3]
```

这些内容都直接对应官方教程中的配置。

---

# 4. Manager-Based RL 环境的整体结构

本教程最重要的结构可以画成：

```text
                  ManagerBasedRLEnv
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
    Scene             Actions          Observations
       │                 │                 │
       ▼                 ▼                 ▼
  仿真场景          ActionManager     ObservationManager
       │
       │
       └────────────────────────────────────┐
                                            │
       ┌────────────────────────────────────┘
       │
       ▼
     Events
       │
       ▼
     Rewards
       │
       ▼
  Terminations
       │
       ▼
 Commands / Curriculum
```

其中真正体现 RL 特征的是：

```text
Reward
Termination
Command
Curriculum
```

本教程的 Cartpole 示例实际定义了：

```text
Scene
Actions
Observations
Events
Rewards
Terminations
```

而：

```text
Commands = None
Curriculum = None
```

即本示例没有使用 Command Manager 和 Curriculum Manager。

---

# 5. 教程使用的文件

教程主要涉及两个文件。

## 5.1 环境配置

```text
isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py
```

它负责定义：

```text
Scene
Actions
Observations
Events
Rewards
Terminations
```

---

## 5.2 环境运行脚本

```text
scripts/tutorials/03_envs/run_cartpole_rl_env.py
```

它负责：

```text
启动 Isaac Sim
        ↓
创建 CartpoleEnvCfg
        ↓
创建 ManagerBasedRLEnv
        ↓
循环执行 env.step()
```

官方教程明确说明，该运行脚本与上一篇基础环境教程中的运行脚本非常相似，主要区别是使用：

```python
ManagerBasedRLEnv
```

而不是：

```python
ManagerBasedEnv
```

因此本教程重点放在 RL 组件。

---

# 6. 导入 RL 环境相关配置

当前官方教程的核心导入包括：

```python
import math

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp
```

其中新增的 RL 核心配置类型是：

```python
RewardTermCfg
```

和：

```python
TerminationTermCfg
```

相较上一篇基础环境教程，正是这两个部分将环境扩展到了 RL 任务层面。

---

# 7. Scene：定义 Cartpole 场景

教程首先定义场景：

```python
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0)
        ),
    )

    robot: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=500.0,
        ),
    )
```

这里包含三个主要对象：

```text
Scene
│
├── ground
│
├── robot
│    └── Cartpole
│
└── dome_light
```

官方配置中：

```python
GroundPlaneCfg(size=(100.0, 100.0))
```

定义 100m × 100m 的地面。

Cartpole 使用：

```python
CARTPOLE_CFG
```

并通过：

```python
replace(
    prim_path="{ENV_REGEX_NS}/Robot"
)
```

将机器人放置到每个并行环境的命名空间中。

---

# 8. Action：智能体如何控制 Cartpole？

Action 配置：

```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=100.0,
    )
```

这里定义：

```text
Action
  ↓
JointEffortActionCfg
  ↓
robot
  ↓
slider_to_cart
```

也就是说，智能体的 Action 最终作用于：

```text
slider_to_cart
```

这个关节。

Action 的 scale：

```python
scale=100.0
```

也是官方当前教程中的设置。

---

# 9. Observation：智能体能够看到什么？

Observation 配置：

```python
@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel
        )

        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

Policy Observation 包含：

```text
joint_pos_rel
joint_vel_rel
```

即：

```text
关节相对位置
+
关节相对速度
```

最终作为：

```python
obs["policy"]
```

提供给智能体。

---

# 10. Event：Episode Reset 时发生什么？

本教程使用 Event Manager 管理 reset。

## 10.1 Reset Cart

```python
reset_cart_position = EventTerm(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["slider_to_cart"]
        ),
        "position_range": (-1.0, 1.0),
        "velocity_range": (-0.5, 0.5),
    },
)
```

意思是：

每次 reset 时：

```text
Cart position
    ↓
随机范围 [-1, 1]

Cart velocity
    ↓
随机范围 [-0.5, 0.5]
```

---

## 10.2 Reset Pole

```python
reset_pole_position = EventTerm(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["cart_to_pole"]
        ),
        "position_range": (
            -0.25 * math.pi,
            0.25 * math.pi,
        ),
        "velocity_range": (
            -0.25 * math.pi,
            0.25 * math.pi,
        ),
    },
)
```

因此 Pole 每次 episode 开始时，也会从一个随机初始状态开始。

---

# 11. RL 的核心：Reward

这是本教程相较上一篇教程最重要的新增内容。

Isaac Lab 使用：

```python
RewardManager
```

计算奖励。

每一个奖励项使用：

```python
RewardTermCfg
```

配置。

可以抽象成：

```text
RewardManager
│
├── Reward Term 1
├── Reward Term 2
├── Reward Term 3
├── Reward Term 4
└── Reward Term 5
```

每一个 Term 通常包含：

```text
func
weight
params
```

其中：

```python
func
```

负责计算奖励。

```python
weight
```

负责奖励项的权重。

```python
params
```

提供传给奖励函数的额外参数。

---

# 12. Cartpole 的 Reward 设计

教程定义了 5 个 Reward Term：

```text
1. Alive Reward
2. Terminating Reward
3. Pole Position Reward
4. Cart Velocity Reward
5. Pole Velocity Reward
```

整体目标：

```text
保持 Pole 竖直
+
减少 Cart 速度
+
减少 Pole 角速度
+
尽量不要提前结束
```

---

# 13. Alive Reward

```python
alive = RewTerm(
    func=mdp.is_alive,
    weight=1.0
)
```

含义：

> 只要环境处于存活状态，就提供正奖励。

权重：

```text
+1.0
```

因此，持续保持任务状态本身就有奖励。

---

# 14. Terminating Reward

```python
terminating = RewTerm(
    func=mdp.is_terminated,
    weight=-2.0
)
```

作用：

> 对终止状态施加惩罚。

权重：

```text
-2.0
```

因此：

```text
提前失败
  ↓
termination
  ↓
negative reward
```

---

# 15. Pole Position Reward

这是任务最核心的奖励。

```python
pole_pos = RewTerm(
    func=mdp.joint_pos_target_l2,
    weight=-1.0,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["cart_to_pole"]
        ),
        "target": 0.0,
    },
)
```

这里：

```python
target=0.0
```

表示希望：

```text
Pole joint position → 0
```

也就是让 Pole 保持在目标竖直状态。

其奖励函数：

```python
mdp.joint_pos_target_l2
```

根据目标位置误差计算代价，然后乘以：

```text
-1.0
```

因此误差越大，奖励越低。

可以抽象为：

```text
Pole angle
     │
     ▼
与 target=0 比较
     │
     ▼
L2 error
     │
     ▼
× -1
     │
     ▼
Reward
```

这是本任务最主要的任务奖励。

---

# 16. Cart Velocity Reward

```python
cart_vel = RewTerm(
    func=mdp.joint_vel_l1,
    weight=-0.01,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["slider_to_cart"]
        )
    },
)
```

目的：

> 降低 Cart 的速度。

因为：

```python
weight=-0.01
```

所以：

```text
Cart 速度越大
    ↓
L1 velocity cost 越大
    ↓
负奖励越大
```

因此策略会倾向于减少 Cart 的运动速度。

---

# 17. Pole Velocity Reward

```python
pole_vel = RewTerm(
    func=mdp.joint_vel_l1,
    weight=-0.005,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["cart_to_pole"]
        )
    },
)
```

作用：

> 降低 Pole 的角速度。

同样：

```text
Pole angular velocity
        ↓
L1 cost
        ↓
× -0.005
        ↓
Reward
```

这样可以让 Pole 不仅保持竖直，而且减少快速摆动。

---

# 18. Reward 总体结构

因此，Cartpole 的 Reward 可以理解为：

```text
Total Reward
│
├── + Alive
│
├── - Termination penalty
│
├── - Pole position error
│
├── - Cart velocity
│
└── - Pole velocity
```

从任务角度来看：

```text
                    Cartpole Task
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
         Pole angle   Cart speed  Pole speed
             │           │           │
             ▼           ▼           ▼
          越接近 0      越小        越小
             │           │           │
             └───────────┼───────────┘
                         ▼
                       Reward
```

---

# 19. Reward Term 中的 weight 是什么？

例如：

```python
pole_pos = RewTerm(
    func=mdp.joint_pos_target_l2,
    weight=-1.0,
    ...
)
```

可以理解为：

```text
最终奖励贡献
=
reward function 的结果
×
weight
```

所以：

```text
weight > 0
```

通常表示鼓励某个量。

而：

```text
weight < 0
```

通常表示惩罚某个量。

但是实际使用时，不能简单地认为“负权重一定是坏的”，因为 reward function 本身可能计算的是 cost。

例如：

```python
joint_pos_target_l2
```

计算的是相对于目标的误差量，因此：

```text
误差越大 → cost 越大
```

再乘：

```text
-1
```

才变成：

```text
误差越大 → reward 越低
```

---

# 20. Termination：什么时候结束一个 Episode？

强化学习通常不是无限运行。

一次完整的任务运行称为：

```text
Episode
```

例如：

```text
Reset
  ↓
Step
  ↓
Step
  ↓
Step
  ↓
...
  ↓
Termination
  ↓
Reset
```

Cartpole 有两个终止条件：

```text
1. Time Out
2. Cart Out of Bounds
```

官方教程明确指出，这两个条件分别对应 episode 时间限制和 Cart 超出 `[-3, 3]`。

---

# 21. Time Out

```python
time_out = DoneTerm(
    func=mdp.time_out,
    time_out=True
)
```

这里有一个非常重要的配置：

```python
time_out=True
```

它表示：

> 这是一个 time-out / truncation 类型的结束，而不是任务意义上的 failure termination。

因此要区分：

```text
terminated
```

和：

```text
truncated
```

---

# 22. Terminated 与 Truncated

可以理解为：

### Terminated

任务真正进入了终止状态。

例如：

```text
Cart 出界
```

### Truncated

任务因为外部限制而结束。

例如：

```text
Episode 时间到了
```

本教程的：

```python
time_out=True
```

就是为了标记这个区别。

官方教程明确指出，这与 Gymnasium 对 `terminated` 和 `truncated` 的区分一致。

---

# 23. Cart Out of Bounds

第二个终止条件：

```python
cart_out_of_bounds = DoneTerm(
    func=mdp.joint_pos_out_of_manual_limit,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["slider_to_cart"]
        ),
        "bounds": (-3.0, 3.0),
    },
)
```

意思：

```text
Cart position < -3
        OR
Cart position > 3
        ↓
Episode Termination
```

因此：

```text
Cart ∈ [-3, 3]
```

是允许的范围。

---

# 24. Command Manager

Isaac Lab 还提供：

```python
CommandManager
```

用于 goal-conditioned task。

例如某些任务不是：

```text
“尽量保持不动”
```

而是：

```text
“移动到目标位置”
```

那么环境可以产生：

```text
Command
```

例如：

```text
目标位置
目标速度
目标姿态
目标方向
```

Command Manager 负责：

```text
生成 Command
      ↓
重新采样 Command
      ↓
更新 Command
      ↓
提供给 Observation
```

官方教程指出，Command Manager 也可以将 command 作为 observation 提供给智能体。

---

# 25. 为什么 Cartpole 没有 Command？

本教程中的 Cartpole 是一个简单任务。

目标固定：

```text
Pole → 保持竖直
```

不需要每个 episode 随机生成目标。

因此：

```text
Command Manager
=
None
```

官方教程明确说明，该示例没有使用 commands，并建议参考其他 locomotion 或 manipulation 任务了解 Command Manager。

---

# 26. Curriculum Manager

另一个 RL 组件是：

```python
CurriculumManager
```

它用于：

> 随着训练过程逐渐增加任务难度。

例如概念上可以：

```text
训练初期
简单任务
    ↓
训练中期
中等任务
    ↓
训练后期
困难任务
```

这就是：

```text
Curriculum Learning
```

官方教程指出 Isaac Lab 提供 `CurriculumManager`，但 **本 Cartpole 教程为了保持简单，没有配置 curriculum**。

因此这里不能把 Curriculum 理解成当前 Cartpole 示例已经启用了它。

---

# 27. 将所有组件组合起来

现在已经有：

```text
Scene
Actions
Observations
Events
Rewards
Terminations
```

因此可以创建：

```python
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    scene: CartpoleSceneCfg = CartpoleSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        clone_in_fabric=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
```

这里：

```text
ManagerBasedRLEnvCfg
```

就是整个 RL Task 的配置入口。

---

# 28. Cartpole 的环境配置参数

教程中的：

```python
self.decimation = 2
```

表示环境 Action 与底层 physics step 之间的降采样关系。

同时：

```python
self.episode_length_s = 5
```

表示：

```text
Episode 最大长度 = 5 秒
```

Viewer：

```python
self.viewer.eye = (8.0, 0.0, 5.0)
```

用于设置观察位置。

仿真时间步：

```python
self.sim.dt = 1 / 120
```

因此 physics timestep 是：

```text
1 / 120 s
≈ 0.00833 s
```

也就是：

```text
120 Hz
```

教程还设置：

```python
self.sim.render_interval = self.decimation
```

这些都是当前官方配置中的原始设置。

---

# 29. Episode 长度是多少？

配置：

```python
self.episode_length_s = 5
```

意味着：

```text
最大 Episode 时间 = 5 秒
```

结合：

```python
self.sim.dt = 1 / 120
```

和：

```python
self.decimation = 2
```

可以理解为：

```text
Physics step
    ↓
dt = 1/120 s

每 2 个 physics step
    ↓
环境进行一次 step

Episode
    ↓
最多 5 秒
```

因此一个 episode 在环境 step 层面对应的最大步数约为：

```text
5 / (2 × 1/120)
= 300 steps
```

这也与教程运行脚本中的：

```python
if count % 300 == 0:
```

形成对应关系。

> 注意：这里是根据教程明确给出的 `episode_length_s=5`、`dt=1/120` 和 `decimation=2` 计算得到的解释，而不是额外修改官方配置。

---

# 30. 创建 ManagerBasedRLEnv

运行脚本首先：

```python
env_cfg = CartpoleEnvCfg()
```

然后：

```python
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.sim.device = args_cli.device
```

最后：

```python
env = ManagerBasedRLEnv(
    cfg=env_cfg
)
```

也就是说：

```text
CartpoleEnvCfg
       │
       ▼
ManagerBasedRLEnv
```

环境根据配置自动创建对应的 Manager。

---

# 31. 与 ManagerBasedEnv 最大的区别

上一篇教程可能是：

```python
obs, info = env.step(action)
```

而现在：

```python
obs, rew, terminated, truncated, info = env.step(action)
```

多出了：

```text
rew
terminated
truncated
```

因此：

```text
ManagerBasedEnv.step()
        ↓
Observation

ManagerBasedRLEnv.step()
        ↓
Observation
Reward
Termination
Truncation
Info
```

这是从普通仿真环境进入 RL 环境最重要的变化之一。

---

# 32. RL Step 的数据流

可以把：

```python
obs, rew, terminated, truncated, info = env.step(action)
```

理解为：

```text
                    Action
                       │
                       ▼
                ActionManager
                       │
                       ▼
                   Physics
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
    Observation      Reward     Termination
          │            │            │
          └────────────┼────────────┘
                       ▼
                    Env.step
                       │
                       ▼
          obs, rew, terminated,
             truncated, info
```

---

# 33. Info 中有什么？

官方教程特别指出，`info` 字典还会维护用于日志记录的信息，例如：

```text
Reward 各 Term 的贡献
Termination 各 Term 的状态
Episode length
```

因此：

```python
info
```

不仅仅是一个“随便放数据的字典”，它是训练和调试 RL 环境的重要信息出口。

---

# 34. 运行脚本

官方当前教程给出的运行命令：

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32
```

这里：

```text
--num_envs 32
```

表示创建：

```text
32 个并行环境
```

官方脚本默认值为：

```python
default=16
```

但运行命令明确传入了：

```text
32
```

所以实际运行时为 32 个环境。

---

# 35. AppLauncher

运行脚本首先：

```python
from isaaclab.app import AppLauncher
```

然后：

```python
parser = argparse.ArgumentParser(
    description="Tutorial on running the cartpole RL environment."
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=16,
    help="Number of environments to spawn.",
)
```

接下来：

```python
AppLauncher.add_app_launcher_args(parser)
```

将 Isaac Lab / Isaac Sim 的启动参数加入 argparse。

然后：

```python
args_cli = parser.parse_args()
```

创建：

```python
app_launcher = AppLauncher(args_cli)
```

最后：

```python
simulation_app = app_launcher.app
```

启动 Isaac Sim。

---

# 36. 创建环境

进入 `main()`：

```python
env_cfg = CartpoleEnvCfg()

env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.sim.device = args_cli.device

env = ManagerBasedRLEnv(
    cfg=env_cfg
)
```

这里值得注意：

```python
env_cfg.scene.num_envs
```

可以在实例化配置后修改。

因此：

```text
配置默认值
    ↓
命令行覆盖
    ↓
创建 Environment
```

---

# 37. 使用随机 Action

教程没有在这里训练神经网络。

为了单纯演示环境：

```python
joint_efforts = torch.randn_like(
    env.action_manager.action
)
```

产生随机 Action。

这意味着：

```text
Random Policy
      ↓
Random Action
      ↓
Cartpole
```

所以：

> **运行本教程并不会让 Cartpole 学会保持平衡。**

这里的随机 Action 只是用于验证 RL Environment 的：

```text
Observation
Reward
Termination
Truncation
Info
```

是否正常工作。

---

# 38. 为什么使用 torch.inference_mode()？

运行循环使用：

```python
with torch.inference_mode():
```

原因是这个教程只是运行仿真，并没有进行反向传播。

因此不需要：

```text
Gradient
Autograd
Backward
```

使用 inference mode 可以避免不必要的梯度跟踪。

---

# 39. Reset

运行脚本：

```python
if count % 300 == 0:
    count = 0
    env.reset()
```

这里有一个容易混淆的地方。

脚本每 300 次循环手动调用：

```python
env.reset()
```

但是 RL 环境本身也会根据：

```python
TerminationsCfg
```

处理已经结束的环境。

官方教程明确说明：

> 与上一篇教程不同，这次各个环境会根据配置中的 termination criteria，在自己终止后进行 reset。

所以：

```text
手动 env.reset()
```

和：

```text
termination 后的自动处理
```

不是一回事。

---

# 40. env.step()

核心代码：

```python
obs, rew, terminated, truncated, info = env.step(
    joint_efforts
)
```

返回：

### `obs`

Observation。

例如：

```python
obs["policy"]
```

---

### `rew`

当前环境的 Reward。

通常是每个并行环境对应一个 reward。

---

### `terminated`

真正任务终止的状态。

---

### `truncated`

因为 timeout 等条件导致的截断状态。

---

### `info`

额外信息和日志数据。

这些返回值是 `ManagerBasedRLEnv` 相较于基础环境的核心扩展。

---

# 41. 查看 Pole 状态

教程中：

```python
print(
    "[Env 0]: Pole joint: ",
    obs["policy"][0][1].item()
)
```

这里：

```python
obs["policy"]
```

表示 Policy Observation Group。

```python
[0]
```

表示：

```text
第 0 个并行环境
```

而：

```python
[1]
```

表示 Observation 中对应 Pole 位置的元素。

因此代码只是为了简单地把第一个 Cartpole 的 Pole 状态打印出来。

---

# 42. 完整运行逻辑

整个程序可以总结成：

```text
启动 Isaac Sim
      │
      ▼
创建 CartpoleEnvCfg
      │
      ▼
设置 num_envs / device
      │
      ▼
创建 ManagerBasedRLEnv
      │
      ▼
循环
      │
      ├── 必要时 reset
      │
      ├── 生成随机 Action
      │
      ├── env.step(action)
      │
      ├── 得到 obs
      │
      ├── 得到 reward
      │
      ├── 得到 terminated
      │
      ├── 得到 truncated
      │
      └── 得到 info
      │
      ▼
继续仿真
```

---

# 43. 当前教程中的核心配置完整结构

把所有配置组合起来，就是：

```python
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):

    # Scene
    scene = CartpoleSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        clone_in_fabric=True,
    )

    # Basic environment components
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    # RL components
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):

        self.decimation = 2

        self.episode_length_s = 5

        self.viewer.eye = (
            8.0,
            0.0,
            5.0,
        )

        self.sim.dt = 1 / 120

        self.sim.render_interval = (
            self.decimation
        )
```

这就是整个 Manager-Based RL 环境的核心配置结构。

---

# 44. RL 环境的完整架构

现在可以把 Cartpole 的完整架构画出来：

```text
                         Cartpole RL Environment
                                  │
                    ManagerBasedRLEnvCfg
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
       ▼                          ▼                          ▼
     Scene                     Actions                 Observations
       │                          │                          │
       ▼                          ▼                          ▼
    Cartpole                Joint Effort              Joint Position
    Ground                  Action                    Joint Velocity
    Light
       │
       │
       ├───────────────────────────────────────────────┐
       │                                               │
       ▼                                               ▼
    Events                                           Reward
       │                                               │
       ▼                                               ▼
     Reset                                    ┌───────────────┐
       │                                      │ Alive         │
       │                                      │ Terminating   │
       │                                      │ Pole Position │
       │                                      │ Cart Velocity │
       │                                      │ Pole Velocity │
       │                                      └───────────────┘
       │
       ▼
 Terminations
       │
       ├── Time Out
       │
       └── Cart Out of Bounds
```

---

# 45. MDP 角度理解

从强化学习的 MDP 角度，可以理解为：

```text
Observation
     │
     ▼
  Policy
     │
     ▼
  Action
     │
     ▼
Environment
     │
     ├───────────────┐
     │               │
     ▼               ▼
 Observation       Reward
     │               │
     └───────┬───────┘
             │
             ▼
       Next State
```

而 Isaac Lab Manager-Based RL 环境进一步把这些组件拆成 Manager：

```text
ObservationManager
RewardManager
TerminationManager
CommandManager
CurriculumManager
```

因此每一个 RL 概念都可以独立配置。

---

# 46. Reward Manager 的设计意义

如果没有 Manager-Based 设计，可能需要在环境类中手动写：

```python
reward = 0

reward += ...
reward += ...
reward += ...
```

而 Manager-Based 写成：

```python
class RewardsCfg:

    alive = RewTerm(...)

    terminating = RewTerm(...)

    pole_pos = RewTerm(...)

    cart_vel = RewTerm(...)

    pole_vel = RewTerm(...)
```

这样做的好处是：

```text
Reward 结构清晰
       ↓
每个奖励项独立
       ↓
可以单独修改 weight
       ↓
可以复用 reward function
```

这也是 Isaac Lab Manager-Based Workflow 的重要设计思想。

---

# 47. Termination Manager 的设计意义

同样，终止条件也不需要全部写进：

```python
env.step()
```

而是配置：

```python
class TerminationsCfg:

    time_out = ...

    cart_out_of_bounds = ...
```

这样：

```text
Termination
├── Time Out
└── Cart Out of Bounds
```

可以独立修改。

例如以后增加新的终止条件，只需要增加新的 Term，而不是重写整个 Environment。

---

# 48. Command Manager 适合什么任务？

虽然本教程没有使用 Command，但理解它非常重要。

例如：

```text
机器人行走
```

可能需要：

```text
目标速度 = 1.0 m/s
```

或者：

```text
目标速度 = 0.5 m/s
目标转向 = 0.2 rad/s
```

那么：

```text
CommandManager
       ↓
产生 command
       ↓
Observation
       ↓
Policy
       ↓
Action
```

这样一个 Policy 可以学习：

```text
不同 command
       ↓
不同动作
```

而不是只能执行一个固定目标。

本教程只指出 Command Manager 的用途，并没有在 Cartpole 中定义 command。

---

# 49. Curriculum Manager 适合什么任务？

例如一个机器人一开始：

```text
简单地面
```

训练一段时间后：

```text
复杂地面
```

再之后：

```text
更高难度
```

可以通过 Curriculum：

```text
Training Progress
       │
       ▼
CurriculumManager
       │
       ├── Difficulty 1
       ├── Difficulty 2
       └── Difficulty 3
```

但需要特别强调：

> **本教程的 Cartpole 没有启用 Curriculum。**

官方只是介绍了这个 Manager 的用途，并说明可以在其他 locomotion 或 manipulation 任务中查看示例。

---

# 50. 与上一篇 Base Environment 教程的对比

| 功能                      | `ManagerBasedEnv` | `ManagerBasedRLEnv` |
| ----------------------- | ----------------: | ------------------: |
| Scene                   |                 ✓ |                   ✓ |
| Actions                 |                 ✓ |                   ✓ |
| Observations            |                 ✓ |                   ✓ |
| Events                  |                 ✓ |                   ✓ |
| Rewards                 |                 — |                   ✓ |
| Terminations            |                 — |                   ✓ |
| Commands                |                 — |                  可选 |
| Curriculum              |                 — |                  可选 |
| RL Task                 |              基础环境 |                   ✓ |
| `step()` 返回 Reward      |                 — |                   ✓ |
| `step()` 返回 Termination |                 — |                   ✓ |

因此：

```text
ManagerBasedEnv
```

更接近：

> **仿真环境**

而：

```text
ManagerBasedRLEnv
```

更接近：

> **强化学习任务环境**

---

# 51. 运行教程

官方给出的运行命令：

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32
```

运行后会打开类似上一篇教程的 Cartpole 仿真。

但这一次环境除了 Observation 外，还会返回：

```text
Reward
Termination
Truncation
```

并且环境会根据配置中的 termination 条件进行结束和 reset。

停止方式：

```text
关闭仿真窗口
```

或者：

```text
Ctrl + C
```

官方教程明确给出了这两种停止方式。

---

# 52. 一个重要认识：运行 ≠ 训练

运行：

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32
```

**并不是训练 RL Policy。**

当前教程只是：

```text
创建 RL Environment
       ↓
随机 Action
       ↓
观察 Reward
       ↓
观察 Termination
```

它的目的主要是：

> 验证 RL 环境接口是否正确。

真正的训练需要：

```text
RL Algorithm
       +
Policy
       +
Optimizer
       +
Rollout
       +
Environment
```

Isaac Lab 后续教程会进入：

```text
Training with an RL Agent
```

等内容。

---

# 53. 为什么教程先学习 ManagerBasedRLEnv？

因为在真正训练之前，需要先搞清楚：

```text
Observation 是什么？
Action 是什么？
Reward 是什么？
Termination 是什么？
Episode 是什么？
```

如果这些定义不合理，即使 RL 算法运行正常，也可能无法得到想要的行为。

所以推荐学习顺序：

```text
Base Environment
        ↓
Manager-Based RL Environment
        ↓
Register Environment
        ↓
RL Agent
        ↓
Training
        ↓
Policy Evaluation
```

这也是官方教程体系的组织方式。当前教程结束后，官方明确指出下一步将学习使用 `gymnasium.make()` 创建环境。

---

# 54. 本教程最重要的几个 API

建议重点记住以下 API。

## 环境

```python
ManagerBasedRLEnv
```

---

## 配置

```python
ManagerBasedRLEnvCfg
```

---

## Reward

```python
RewardManager
RewardTermCfg
```

---

## Termination

```python
TerminationTermCfg
```

---

## Command

```python
CommandManager
```

---

## Curriculum

```python
CurriculumManager
```

---

## Scene Entity

```python
SceneEntityCfg
```

---

## Observation

```python
ObservationGroupCfg
ObservationTermCfg
```

---

# 55. 从零创建自己的 Manager-Based RL 环境时的模板

根据本教程的结构，一个自定义任务通常可以按下面的思路组织：

```text
my_task/
│
├── mdp/
│   ├── rewards.py
│   ├── terminations.py
│   ├── observations.py
│   └── ...
│
├── my_env_cfg.py
│
└── run_my_env.py
```

然后：

```python
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):

    scene = ...

    observations = ...

    actions = ...

    events = ...

    rewards = ...

    terminations = ...
```

最后：

```python
env = ManagerBasedRLEnv(
    cfg=env_cfg
)
```

这就是从 Cartpole 向自己的机器人任务迁移时需要掌握的核心结构。

---

# 56. 如何设计 Reward？

本教程的 Cartpole Reward 实际体现了一个很典型的结构：

```text
Reward
│
├── 生存奖励
│
├── 失败惩罚
│
├── 核心任务奖励
│
└── Shaping Reward
       ├── Cart velocity
       └── Pole velocity
```

其中：

```text
核心任务
=
Pole 保持竖直
```

而：

```text
Shaping
=
让运动更加稳定
```

因此 Reward 设计可以理解为：

```text
“我要什么”
+
“我希望行为怎样更加稳定”
```

而不是简单地只定义一个最终成功条件。

---

# 57. 如何理解本教程的 Reward 权重？

官方配置：

```text
alive          +1.0
terminating    -2.0
pole_pos       -1.0
cart_vel       -0.01
pole_vel       -0.005
```

它们并不是简单的：

```text
数字越大越重要
```

而是必须结合对应 reward function 的输出范围来理解。

例如：

```python
joint_pos_target_l2
```

与：

```python
joint_vel_l1
```

的量纲和数值范围并不相同。

所以设计 Reward 时应该同时考虑：

```text
Reward Function
+
Weight
+
状态范围
+
Episode 累积效果
```

这比单纯调整一个 `weight` 更重要。

---

# 58. 本教程的数据流总结

最终，一个完整的 RL step 可以表示成：

```text
                    Policy
                      │
                      │ Action
                      ▼
              ┌───────────────┐
              │ ActionManager │
              └───────┬───────┘
                      │
                      ▼
                  Simulation
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
   Observation     Reward     Termination
     Manager       Manager       Manager
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
                   env.step()
                      │
          ┌───────────┼─────────────┐
          ▼           ▼             ▼
        obs          rew       terminated
                                   +
                                truncated
```

这就是 Manager-Based RL Environment 的核心。

---

# 59. 最终理解

如果只记住本教程的一个结构，可以记住：

```text
ManagerBasedRLEnv
│
├── Scene
│
├── Action
│
├── Observation
│
├── Event
│
├── Reward
│
├── Termination
│
├── Command       ← 本教程未使用
│
└── Curriculum    ← 本教程未使用
```

其中：

```text
Scene
```

回答：

> 仿真里有什么？

```text
Action
```

回答：

> Agent 能做什么？

```text
Observation
```

回答：

> Agent 能看到什么？

```text
Reward
```

回答：

> 什么行为更好？

```text
Termination
```

回答：

> 什么时候结束？

```text
Command
```

回答：

> Agent 当前要完成什么目标？

```text
Curriculum
```

回答：

> 如何随着训练进度调整任务难度？

---

# 60. 从 Base Environment 到 RL Environment

上一篇教程：

```text
ManagerBasedEnv
```

解决：

```text
怎么创建一个可交互的仿真环境？
```

本教程：

```text
ManagerBasedRLEnv
```

进一步解决：

```text
怎么把这个环境定义成一个 RL Task？
```

核心增加：

```text
Reward
Termination
```

同时提供：

```text
Command
Curriculum
```

作为可选机制。

最终形成：

```text
                    Isaac Lab
                       │
                       ▼
              Manager-Based Workflow
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 ManagerBasedEnv            ManagerBasedRLEnv
          │                         │
     基础仿真环境               RL 任务环境
                                    │
                      ┌─────────────┼─────────────┐
                      ▼             ▼             ▼
                   Reward      Termination    Command
                                                  │
                                             Curriculum
```

---

# 61. 下一步

本教程最后指出，虽然可以直接：

```python
ManagerBasedRLEnv(cfg=env_cfg)
```

创建环境，但如果每个任务都需要编写专门的运行脚本，这种方式并不容易扩展。

因此后续教程会使用：

```python
gymnasium.make()
```

通过 Gymnasium 接口创建环境。

即从：

```text
手动实例化 Environment
```

进一步走向：

```text
注册 Task
    ↓
gymnasium.make()
    ↓
RL Framework
    ↓
Training
```

这是从“理解环境”进入“真正训练 RL”的重要一步。官方教程明确将 `gymnasium.make()` 作为下一阶段内容。

---

# 62. 本教程知识点清单

完成本教程后，应该能够解释：

* [x] `ManagerBasedEnv` 与 `ManagerBasedRLEnv` 的区别
* [x] `ManagerBasedRLEnvCfg` 的作用
* [x] 如何定义 Reward
* [x] `RewardTermCfg` 的作用
* [x] `weight` 的含义
* [x] 如何定义 Termination
* [x] `terminated` 与 `truncated` 的区别
* [x] `episode_length_s` 的作用
* [x] Command Manager 的用途
* [x] Curriculum Manager 的用途
* [x] 如何创建 `ManagerBasedRLEnv`
* [x] `env.step()` 的 RL 返回值
* [x] `info` 在 RL 环境中的作用
* [x] 如何运行 Cartpole RL Environment
* [x] 为什么教程中的 Action 是随机的
* [x] 为什么运行教程不等于训练 Policy
* [x] Manager-Based RL 环境的整体数据流

---

# 参考

**官方 Isaac Lab 教程：**

https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html

**相关示例：**

```text
isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py

scripts/tutorials/03_envs/run_cartpole_rl_env.py
```

本文中的 API 名称、配置项、Cartpole Reward、Termination、运行命令等均以你提供的 Isaac Lab `main` 分支官方教程为依据。官方页面当前显示最后更新时间为 **2026-09-17**。
