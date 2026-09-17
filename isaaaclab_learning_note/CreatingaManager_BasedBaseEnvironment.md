# Isaac Lab：创建基于 Manager 的基础环境

> 本文根据 Isaac Lab 官方教程 **Creating a Manager-Based Base Environment** 整理翻译。
>
> 官方教程：
> [Isaac Lab 官方教程：Creating a Manager-Based Base Environment](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html?utm_source=chatgpt.com)
>
> 本教程使用 Cartpole（倒立摆）作为示例，介绍如何通过 Isaac Lab 的 Manager-Based 工作流创建一个基础仿真环境。

---

## 1. 教程目标

在 Isaac Lab 中，一个环境（Environment）需要把仿真中的多个部分组织起来，例如：

* 场景（Scene）
* 动作（Actions）
* 观测（Observations）
* 重置事件（Reset Events）
* 物理仿真参数
* 环境执行循环

Isaac Lab 提供了两类重要的 Manager-Based 环境：

```text
ManagerBasedEnv
        │
        └── 基础环境
             ├── Scene
             ├── Action Manager
             ├── Observation Manager
             └── Event Manager


ManagerBasedRLEnv
        │
        └── 强化学习环境
             ├── ManagerBasedEnv 的功能
             ├── Reward
             ├── Termination
             ├── Curriculum
             └── Command
```

其中：

### `ManagerBasedEnv`

主要用于传统机器人控制、运动规划等任务。

它负责：

* 接收 Action
* 执行仿真
* 计算 Observation
* 管理场景
* 执行事件

但它本身**不负责 Reward 和 Termination**。

### `ManagerBasedRLEnv`

主要用于强化学习。

除了基础环境功能之外，还提供：

* Reward
* Termination
* Curriculum
* Command generation
* MDP 相关信息

因此，本教程重点学习：

```python
ManagerBasedEnv
ManagerBasedEnvCfg
```

官方教程也是使用 Cartpole 来展示如何组合这些组件。

---

# 2. Manager-Based 环境的整体结构

`ManagerBasedEnv` 可以理解为一个环境协调器。

它把不同功能交给不同 Manager：

```text
                  ManagerBasedEnv
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
     Scene           ActionManager   ObservationManager
        │                │                │
        │                │                │
     机器人/物体       控制机器人        获取状态
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                   EventManager
                         │
                  重置 / 随机化
```

官方教程中主要涉及四个核心组件：

| 组件                   | 作用            |
| -------------------- | ------------- |
| `InteractiveScene`   | 管理仿真场景        |
| `ActionManager`      | 管理环境动作        |
| `ObservationManager` | 管理环境观测        |
| `EventManager`       | 管理启动、重置、周期性事件 |

这种设计的优势是：**通过修改配置类，就可以比较容易地创建不同环境，而不需要大量修改环境核心逻辑。**

---

# 3. Cartpole 示例

本教程对应 Isaac Lab 中的：

```text
scripts/tutorials/03_envs/create_cartpole_base_env.py
```

运行示例：

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32
```

如果使用 `uv`：

```bash
uv run python scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32 --viz kit
```

这里：

```text
--num_envs 32
```

表示创建 32 个并行环境。

Isaac Lab 的一个重要特点就是能够同时运行大量环境，从而充分利用 GPU 并行仿真能力。

---

# 4. 导入相关模块

教程中的环境配置主要依赖以下模块：

```python
import math
import torch

import isaaclab.envs.mdp as mdp

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass
```

另外，Cartpole 的场景配置来自：

```python
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg
```

这里最重要的几个类是：

```text
ManagerBasedEnv
        ↓
ManagerBasedEnvCfg
        ↓
ActionsCfg
ObservationsCfg
EventCfg
```

---

# 5. 定义 Action

首先定义环境需要接收什么动作。

教程中使用：

```python
@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=5.0,
    )
```

这里定义了一个名为：

```text
joint_efforts
```

的 Action Term。

它表示：

> 对 Cartpole 的 `slider_to_cart` 关节施加力矩/关节力控制。

其中：

```python
asset_name="robot"
```

表示控制的资产是：

```text
robot
```

而：

```python
joint_names=["slider_to_cart"]
```

表示控制的关节是：

```text
slider_to_cart
```

最后：

```python
scale=5.0
```

表示输入 Action 会经过对应的缩放后用于控制。

---

# 6. ActionManager 的设计思想

这是 Manager-Based 工作流非常重要的一点。

传统写法可能直接调用：

```python
robot.set_joint_effort_target(...)
```

而 Manager-Based 工作流则把 Action 定义为配置：

```text
Action
   ↓
ActionTerm
   ↓
ActionManager
   ↓
Robot
```

一个机器人可以拥有多个 Action Term。

例如机械臂：

```text
ActionManager
    │
    ├── ArmJointAction
    │       └── 控制机械臂关节
    │
    └── GripperAction
            └── 控制夹爪
```

这样就可以把不同控制对象拆开。

官方教程也指出，一个 `ActionManager` 可以包含多个 `ActionTerm`，每个 Action Term 负责环境中的某一部分控制。

---

# 7. 定义 Observation

Observation 是智能体能够观察到的环境状态。

在 Cartpole 中，教程定义：

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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy = PolicyCfg()
```

这里定义了一个 Observation Group：

```text
policy
```

这个 group 包含两个 observation term：

```text
joint_pos_rel
joint_vel_rel
```

分别表示：

```text
关节相对位置
关节相对速度
```

---

# 8. ObservationGroup

Observation 在 Isaac Lab 中可以分成多个 Group。

例如：

```text
Observations
│
├── policy
│    ├── joint_pos
│    └── joint_vel
│
├── critic
│    ├── joint_pos
│    ├── joint_vel
│    └── additional_state
│
└── camera
     └── image
```

这种设计特别适合复杂机器人和强化学习任务。

例如可以让：

```text
Policy
    ↓
只看到低维状态
```

而：

```text
Critic
    ↓
看到更多 privileged information
```

本教程只定义：

```text
policy
```

这一组观察。

---

# 9. ObservationTerm

单独的 Observation Term 通过：

```python
ObsTerm(...)
```

定义。

例如：

```python
joint_pos_rel = ObsTerm(
    func=mdp.joint_pos_rel
)
```

其中：

```python
func=mdp.joint_pos_rel
```

表示 Observation 的具体计算函数。

另一个：

```python
joint_vel_rel = ObsTerm(
    func=mdp.joint_vel_rel
)
```

用于计算关节相对速度。

Observation Term 还可以进一步配置：

* Noise
* Clipping
* Scaling
* Observation corruption

本教程为了保持简单，使用默认配置。

---

# 10. concatenate_terms

这里：

```python
self.concatenate_terms = True
```

表示将 Observation Group 中的多个 term 进行拼接。

例如：

```text
joint_pos_rel
    ↓
[ q1, q2 ]

joint_vel_rel
    ↓
[ dq1, dq2 ]
```

最终可能形成：

```text
policy observation

[ q1, q2, dq1, dq2 ]
```

而不是分别保存成多个独立 Tensor。

---

# 11. 定义 Event

接下来是 Event Manager。

Event Manager 用于管理环境中的各种事件，例如：

* 环境初始化
* 环境重置
* 物理参数随机化
* 质量随机化
* 摩擦系数随机化
* 视觉属性随机化

教程中的基本结构：

```python
@configclass
class EventCfg:
    ...
```

每一个 Event 都使用：

```python
EventTerm(...)
```

定义。

---

# 12. Event 的三种常见模式

Isaac Lab 提供了三个非常重要的 Event mode：

| Mode       | 含义         |
| ---------- | ---------- |
| `startup`  | 环境启动时执行    |
| `reset`    | 环境重置时执行    |
| `interval` | 按一定间隔周期性执行 |

也就是：

```text
startup
   ↓
环境创建
   ↓
reset
   ↓
运行
   ↓
interval
   ↓
周期性执行
```

官方教程说明，这三种模式是 `ManagerBasedEnv` 开箱即用的常用事件模式。

---

# 13. Startup：随机化 Pole 质量

教程中的一个 Event 是：

```python
add_pole_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            body_names=["pole"]
        ),
        "mass_distribution_params": (0.1, 0.5),
        "operation": "add",
    },
)
```

它的作用可以理解为：

```text
环境启动
    ↓
找到 robot
    ↓
找到 pole
    ↓
随机化 pole 的质量
```

这里：

```python
mode="startup"
```

意味着：

> 只在环境启动时执行一次。

为什么不每次 reset 都执行？

因为质量随机化可能比较昂贵，而且教程希望质量在一次环境运行期间保持固定。

这种思想也与强化学习中的 Domain Randomization 密切相关。

---

# 14. Reset：随机化小车状态

另一个 Event：

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
        "velocity_range": (-0.1, 0.1),
    },
)
```

意思是：

每次环境 reset 时：

```text
slider_to_cart
        ↓
随机设置位置
        ↓
[-1.0, 1.0]

随机设置速度
        ↓
[-0.1, 0.1]
```

---

# 15. Reset：随机化 Pole 状态

同样可以对 Pole 进行初始化随机化：

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
            -0.125 * math.pi,
            0.125 * math.pi
        ),
        "velocity_range": (
            -0.01 * math.pi,
            0.01 * math.pi
        ),
    },
)
```

这样每次 reset 后，Pole 都不会完全处于相同的初始状态。

---

# 16. 定义整个 Environment Configuration

前面已经定义：

```text
Scene
Action
Observation
Event
```

现在把它们组合到：

```python
ManagerBasedEnvCfg
```

中。

教程中的结构是：

```python
@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):

    scene = CartpoleSceneCfg(
        num_envs=1024,
        env_spacing=2.5,
    )

    observations = ObservationsCfg()

    actions = ActionsCfg()

    events = EventCfg()
```

可以把它理解成：

```text
CartpoleEnvCfg
│
├── scene
│
├── observations
│
├── actions
│
└── events
```

这就是 Manager-Based 环境配置的核心。

---

# 17. 配置 Viewer

在 `__post_init__()` 中可以修改 Viewer：

```python
def __post_init__(self):

    self.viewer.eye = [
        4.5,
        0.0,
        6.0,
    ]

    self.viewer.lookat = [
        0.0,
        0.0,
        2.0,
    ]
```

其中：

```text
viewer.eye
```

表示摄像机位置。

而：

```text
viewer.lookat
```

表示摄像机观察的目标位置。

---

# 18. 配置仿真频率

教程设置：

```python
self.decimation = 4
```

以及：

```python
self.sim.dt = 0.005
```

其中：

```text
dt = 0.005 s
```

即：

```text
5 ms
```

因此仿真频率：

```text
1 / 0.005 = 200 Hz
```

而：

```text
decimation = 4
```

意味着环境 Action 每 4 个 physics step 更新一次。

所以环境频率为：

```text
200 Hz / 4
= 50 Hz
```

也就是：

```text
Physics：
200 Hz

Environment：
50 Hz
```

可以理解为：

```text
Physics Step

0 ── 1 ── 2 ── 3 ── 4 ── 5 ── 6 ── 7 ── 8
│                    │                    │
└──── Environment ───┘                    │
                     └──── Environment ───┘
```

---

# 19. 创建 ManagerBasedEnv

所有配置完成后：

```python
env_cfg = CartpoleEnvCfg()

env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.sim.device = args_cli.device

env = ManagerBasedEnv(
    cfg=env_cfg
)
```

这里非常重要。

真正创建环境的是：

```python
ManagerBasedEnv(
    cfg=env_cfg
)
```

而 `env_cfg` 负责告诉环境：

```text
应该创建什么 Scene
应该有什么 Action
应该有什么 Observation
应该执行什么 Event
使用什么 Simulation 参数
```

---

# 20. Environment 的执行循环

Manager-Based 环境最大的优势之一是：

> 大量底层仿真逻辑已经被 Manager 封装起来。

因此主循环非常简单。

核心代码可以理解为：

```python
while simulation_app.is_running():

    env.reset()

    action = ...

    obs, info = env.step(action)
```

也就是说：

```text
Action
  ↓
env.step(action)
  ↓
ActionManager
  ↓
Physics
  ↓
ObservationManager
  ↓
Observation
```

---

# 21. Reset 环境

教程中每 300 个环境 step 执行一次 reset：

```python
if count % 300 == 0:
    count = 0
    env.reset()
```

需要注意：

`ManagerBasedEnv` 本身没有强化学习意义上的 Termination。

也就是说：

```text
ManagerBasedEnv
    │
    ├── reset()
    └── step()
```

但：

```text
ManagerBasedRLEnv
    │
    ├── reset()
    ├── step()
    ├── reward
    └── termination
```

因此在 `ManagerBasedEnv` 中，什么时候 reset 由用户自己决定。

本教程只是简单地：

```text
每 300 steps reset 一次
```

---

# 22. 生成随机 Action

教程为了演示环境运行，没有使用真正的控制策略，而是随机生成 Action：

```python
joint_efforts = torch.randn_like(
    env.action_manager.action
)
```

也就是根据 Action Manager 当前 Action 的形状创建随机 Tensor。

例如：

```text
Action Manager
      ↓
当前 Action shape
      ↓
torch.randn_like()
      ↓
随机 Action
```

---

# 23. 执行环境 Step

然后：

```python
obs, _ = env.step(joint_efforts)
```

执行一步环境。

返回：

```python
obs
```

和：

```python
info
```

其中：

```text
obs
```

是 Observation。

而：

```text
info
```

可以包含额外的环境信息。

---

# 24. 获取 Observation

因为我们之前定义了：

```python
policy = PolicyCfg()
```

因此可以通过：

```python
obs["policy"]
```

获取 Policy Observation。

例如教程中读取：

```python
obs["policy"][0][1]
```

用于打印某个 Cartpole 环境中的 Pole 状态。

整体数据结构可以理解为：

```text
obs
│
└── policy
      │
      ├── joint position
      └── joint velocity
```

---

# 25. 为什么使用 torch.inference_mode()

教程把仿真循环放在：

```python
with torch.inference_mode():
```

里面。

原因是 Isaac Lab 大量使用 PyTorch Tensor 进行计算。

如果只是运行仿真，而不是进行神经网络训练，则通常不需要：

```text
Autograd
Gradient
Backward
```

因此：

```python
torch.inference_mode()
```

可以避免不必要的梯度相关开销。

整体结构：

```python
while simulation_app.is_running():

    with torch.inference_mode():

        ...

        obs, _ = env.step(action)
```

官方教程特别强调了这一点，因为这样可以避免 PyTorch 自动求导机制给仿真循环带来的额外开销。

---

# 26. 完整的环境逻辑

整个教程的核心逻辑可以概括为：

```text
                    CartpoleEnvCfg
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
        Scene           Actions       Observations
          │               │                │
          │               ▼                ▼
          │        ActionManager    ObservationManager
          │               │                │
          └───────────────┼────────────────┘
                          │
                          ▼
                    EventManager
                          │
                          ▼
                  ManagerBasedEnv
                          │
                          ▼
                    env.step()
                          │
                          ▼
                     Simulation
                          │
                          ▼
                     Observation
```

这就是 Isaac Lab Manager-Based Workflow 的核心思想。

---

# 27. 一个简化后的完整示例

如果把整个教程浓缩成一个更容易理解的结构，可以写成：

```python
import torch

import isaaclab.envs.mdp as mdp

from isaaclab.envs import (
    ManagerBasedEnv,
    ManagerBasedEnvCfg,
)

from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
)

from isaaclab.utils import configclass


@configclass
class ActionsCfg:

    joint_efforts = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=5.0,
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObservationGroupCfg):

        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel
        )

        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel
        )

        def __post_init__(self):

            self.enable_corruption = False
            self.concatenate_terms = True

    policy = PolicyCfg()


@configclass
class EnvCfg(ManagerBasedEnvCfg):

    actions = ActionsCfg()
    observations = ObservationsCfg()


env = ManagerBasedEnv(
    cfg=EnvCfg()
)


while True:

    action = torch.randn_like(
        env.action_manager.action
    )

    obs, info = env.step(action)
```

> 注意：上面的代码是为了帮助理解 Manager-Based 架构而进行的简化示例，并不是官方 Cartpole 教程的完整可运行代码。实际运行时仍需要正确配置 Scene、Simulation 和 Isaac Lab AppLauncher。

---

# 28. 官方 Cartpole 教程运行

按照官方教程，可以运行：

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32
```

或者使用 `uv`：

```bash
uv run python \
    scripts/tutorials/03_envs/create_cartpole_base_env.py \
    --num_envs 32 \
    --viz kit
```

运行后应该能够看到：

```text
Ground Plane
     +
Light
     +
多个 Cartpole
```

并且 Cartpole 会接收到随机 Action。

同时 Isaac Lab UI 会显示在窗口中，用于调试和可视化。

停止仿真可以：

```text
关闭窗口
```

或者在终端按：

```text
Ctrl + C
```

官方教程对此运行流程有明确说明。

---

# 29. 其他 Manager-Based 示例

Isaac Lab 的教程目录中还有其他类似环境。

例如 Floating Cube：

```bash
./isaaclab.sh -p \
    scripts/tutorials/03_envs/create_cube_base_env.py \
    --num_envs 32
```

以及 Quadruped：

```bash
./isaaclab.sh -p \
    scripts/tutorials/03_envs/create_quadruped_base_env.py \
    --num_envs 32
```

这些示例可以帮助理解如何把 Manager-Based 设计应用到：

```text
Cube
Robot Arm
Quadruped
Cartpole
```

等不同类型的机器人环境中。

---

# 30. Manager-Based Workflow 的核心思想

学习完这个教程，最重要的不是记住某一行代码，而是理解下面的设计模式：

```text
                    Environment
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
           Scene       Actions   Observations
             │           │           │
             │           │           │
             ▼           ▼           ▼
          场景管理     动作管理     观测管理
             │           │           │
             └───────────┼───────────┘
                         │
                         ▼
                    Event Manager
                         │
                         ▼
                   仿真环境运行
```

也就是说，Isaac Lab 不鼓励把所有逻辑都写在一个巨大的 Environment 类里面，而是把不同职责拆分成不同 Manager。

---

# 31. Manager-Based 与传统写法的区别

传统方式可能是：

```text
Environment
│
├── 创建机器人
├── 设置关节
├── 读取关节
├── 计算 Observation
├── 设置 Action
├── Reset
├── 随机化
└── Physics Step
```

所有逻辑都可能集中在一个类中。

Manager-Based：

```text
Environment
│
├── Scene
│
├── ActionManager
│     ├── ActionTerm
│     └── ActionTerm
│
├── ObservationManager
│     ├── ObservationTerm
│     └── ObservationTerm
│
└── EventManager
      ├── Startup Event
      ├── Reset Event
      └── Interval Event
```

这样可以实现更好的模块化。

---

# 32. 本教程需要掌握的几个核心 API

建议重点记住以下类：

```python
ManagerBasedEnv
```

基础 Manager-Based 环境。

```python
ManagerBasedEnvCfg
```

环境配置。

```python
ActionManager
```

Action 管理。

```python
ObservationManager
```

Observation 管理。

```python
EventManager
```

事件管理。

```python
ActionTerm
```

具体 Action。

```python
ObservationTermCfg
```

具体 Observation。

```python
ObservationGroupCfg
```

Observation Group。

```python
EventTermCfg
```

具体 Event。

```python
SceneEntityCfg
```

用于指定 Scene 中的实体。

---

# 33. 三个最重要的配置类

在实际开发自己的 Isaac Lab 环境时，可以首先考虑：

```python
class ActionsCfg:
    ...
```

负责：

```text
机器人怎么被控制？
```

然后：

```python
class ObservationsCfg:
    ...
```

负责：

```text
智能体能够看到什么？
```

最后：

```python
class EventCfg:
    ...
```

负责：

```text
环境什么时候重置？
哪些参数需要随机化？
什么时候执行随机化？
```

最后把它们组合：

```python
class MyEnvCfg(ManagerBasedEnvCfg):

    scene = ...
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventCfg()
```

这就是创建 Manager-Based 环境的基本模板。

---

# 34. ManagerBasedEnv 与 ManagerBasedRLEnv

本教程最后一个重要知识点是：

```text
ManagerBasedEnv
```

并不是完整的强化学习 Task。

它主要提供：

```text
Action
    ↓
Environment
    ↓
Observation
```

可以理解为：

```text
Agent → Action → Environment → Observation → Agent
```

而：

```text
ManagerBasedRLEnv
```

进一步加入：

```text
Reward
Termination
Curriculum
Command
```

形成完整的强化学习 MDP：

```text
                ManagerBasedRLEnv
                       │
       ┌───────────────┼────────────────┐
       │               │                │
    Observation      Action           Reward
       │               │                │
       └───────────────┼────────────────┘
                       │
                  Environment
                       │
                       ▼
                  Termination
```

因此，如果目标是：

* 传统机器人控制
* 运动规划
* 手工控制
* 基础仿真

可以从：

```python
ManagerBasedEnv
```

开始。

如果目标是：

* PPO
* RL
* Policy Training
* Reward
* MDP

则通常继续学习：

```python
ManagerBasedRLEnv
```

官方教程的下一篇正是 **Creating a Manager-Based RL Environment**。

---

# 35. 总结

本教程的核心可以浓缩成一句话：

> **使用 `ManagerBasedEnv`，通过 Scene、Action、Observation 和 Event 等 Manager，把一个复杂的机器人仿真环境拆解成多个可配置的模块。**

核心结构：

```text
ManagerBasedEnv
│
├── Scene
│
├── ActionManager
│      └── ActionTerm
│
├── ObservationManager
│      └── ObservationGroup
│             └── ObservationTerm
│
└── EventManager
       └── EventTerm
```

实际创建环境时：

```text
① 配置 Scene
        ↓
② 定义 Actions
        ↓
③ 定义 Observations
        ↓
④ 定义 Events
        ↓
⑤ 创建 ManagerBasedEnvCfg
        ↓
⑥ 创建 ManagerBasedEnv
        ↓
⑦ env.reset()
        ↓
⑧ env.step(action)
```

理解这套结构之后，就可以进一步学习 Isaac Lab 中更完整的：

```text
ManagerBasedRLEnv
        ↓
Reward
        ↓
Termination
        ↓
Command
        ↓
Curriculum
        ↓
强化学习训练
```

---

## 参考

* Isaac Lab 官方文档：Creating a Manager-Based Base Environment
* 示例脚本：

```text
scripts/tutorials/03_envs/create_cartpole_base_env.py
```

官方当前版本页面显示该教程最后更新时间为 **2026-09-17**。
