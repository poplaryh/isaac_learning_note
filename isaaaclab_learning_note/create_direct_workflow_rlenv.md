# Isaac Lab：创建 Direct Workflow 强化学习环境

> **官方教程：** Creating a Direct Workflow RL Environment
> **官方文档：** https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html
>
> 本教程介绍 Isaac Lab 中的 **Direct Workflow（直接工作流）强化学习环境**。
>
> 示例任务使用经典的 **Cartpole（倒立摆）**，目标是通过强化学习让 Pole 保持直立。
>
> 与前面的 **Manager-Based RL Environment** 不同，Direct Workflow 不依赖 Manager 来分别管理 Action、Observation、Reward 等，而是把这些逻辑直接写在任务环境类中。

---

# 1. Direct Workflow 是什么？

Isaac Lab 提供两种主要的 RL 环境设计方式：

```text
Manager-Based Workflow
        │
        └── ManagerBasedRLEnv
                │
                ├── Action Manager
                ├── Observation Manager
                ├── Reward Manager
                ├── Termination Manager
                └── ...
```

以及：

```text
Direct Workflow
        │
        └── DirectRLEnv
                │
                ├── _setup_scene()
                ├── _pre_physics_step()
                ├── _apply_action()
                ├── _get_observations()
                ├── _get_rewards()
                ├── _get_dones()
                └── _reset_idx()
```

Manager-Based 工作流强调模块化和配置化。

Direct Workflow 则允许开发者直接编写环境逻辑，因此：

* 抽象层更少；
* 环境逻辑集中在任务代码中；
* 对环境执行过程拥有更直接的控制；
* 可以更容易地使用 PyTorch JIT 等机制；
* 阅读一个任务时，可以更直接地找到 Action、Observation、Reward、Reset 等代码。

---

# 2. Direct Workflow 与 Manager-Based Workflow 的区别

上一篇 Manager-Based RL 教程中，Reward 可能写成：

```python
class RewardsCfg:
    alive = RewTerm(...)
    pole_pos = RewTerm(...)
    cart_vel = RewTerm(...)
```

然后由：

```text
RewardManager
```

负责调用。

而 Direct Workflow 中：

```python
def _get_rewards(self):
    ...
```

直接负责计算 Reward。

同样：

```text
Manager-Based
    ObservationManager
          ↓
    Observation Terms
```

而 Direct Workflow：

```python
def _get_observations(self):
    ...
```

直接构造 Observation。

因此可以简单理解为：

| 项目          | Manager-Based          | Direct Workflow   |
| ----------- | ---------------------- | ----------------- |
| 环境基类        | `ManagerBasedRLEnv`    | `DirectRLEnv`     |
| 配置基类        | `ManagerBasedRLEnvCfg` | `DirectRLEnvCfg`  |
| Action      | Action Manager         | 直接实现              |
| Observation | Observation Manager    | 直接实现              |
| Reward      | Reward Manager         | 直接实现              |
| Termination | Termination Manager    | 直接实现              |
| Scene       | 框架/配置管理                | `_setup_scene()`  |
| 抽象程度        | 更高                     | 更低                |
| 代码集中程度      | 分散到多个 Manager          | 集中在 Environment 类 |
| 灵活性         | 配置驱动                   | 代码驱动              |

官方教程明确指出，Direct Workflow 通过任务脚本直接实现完整的 Reward 和 Observation 函数，从而提供更直接的环境控制。

---

# 3. 本教程的任务：Cartpole

本教程使用：

```text
isaaclab_tasks.direct.cartpole
```

中的 Cartpole 环境。

任务是：

> **让倒立摆的 Pole 保持直立。**

系统可以抽象成：

```text
                    Pole
                     │
                     │
                     │
                  ┌──┴──┐
                  │Cart  │
                  └──────┘
                     │
               ←──────────→
```

Agent 控制：

```text
Cart 的水平运动
```

环境提供：

```text
Pole 位置
Pole 速度
Cart 位置
Cart 速度
```

Reward 则鼓励：

```text
Pole 保持接近直立
Cart 速度较小
Pole 速度较小
环境保持运行
```

当：

```text
Cart 超出范围
```

或者：

```text
Episode 超时
```

则需要结束当前 episode。

---

# 4. 核心代码结构

Direct Workflow 的核心代码结构可以概括为：

```python
@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    ...


class CartpoleEnv(DirectRLEnv):
    ...

    def _setup_scene(self):
        ...

    def _pre_physics_step(self, actions):
        ...

    def _apply_action(self):
        ...

    def _get_observations(self):
        ...

    def _get_rewards(self):
        ...

    def _get_dones(self):
        ...

    def _reset_idx(self, env_ids):
        ...
```

这些函数分别负责：

```text
_setup_scene()
        ↓
创建仿真场景

_pre_physics_step()
        ↓
处理 RL Action

_apply_action()
        ↓
将 Action 应用到物理仿真

_get_observations()
        ↓
生成 Observation

_get_rewards()
        ↓
计算 Reward

_get_dones()
        ↓
计算 Termination / Timeout

_reset_idx()
        ↓
重置指定环境
```

这些接口是本教程理解 Direct Workflow 的核心。

---

# 5. 导入模块

官方教程中的核心导入：

```python
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import (
    GroundPlaneCfg,
    spawn_ground_plane,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
```

其中最重要的是：

```python
DirectRLEnv
```

和：

```python
DirectRLEnvCfg
```

以及：

```python
Articulation
```

用于直接操作 Cartpole。

---

# 6. 创建环境配置类

Direct Workflow 使用：

```python
DirectRLEnvCfg
```

作为配置类基类。

```python
@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    ...
```

这与 Manager-Based 环境中的：

```python
ManagerBasedRLEnvCfg
```

不同。

---

# 7. 环境基本参数

官方 Cartpole 配置：

```python
@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):

    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]

    action_space = 1
    observation_space = 4
    state_space = 0
```

这些参数分别表示：

| 参数                  |       值 | 含义                             |
| ------------------- | ------: | ------------------------------ |
| `decimation`        |     `2` | 每个 RL step 对应的 physics step 数量 |
| `episode_length_s`  |   `5.0` | Episode 最大时长                   |
| `action_scale`      | `100.0` | Action 缩放                      |
| `action_space`      |     `1` | Action 维度                      |
| `observation_space` |     `4` | Observation 维度                 |
| `state_space`       |     `0` | State 维度                       |

由于 Direct Workflow 没有 Action Manager 和 Observation Manager，所以这些空间大小需要直接在配置中指定。

---

# 8. Action Space

Cartpole 只有一个控制量：

```text
Cart 水平驱动力 / 力矩
```

所以：

```python
action_space = 1
```

意味着：

```text
Action:

[a]
```

只有一个元素。

---

# 9. Observation Space

Observation 一共有 4 个元素：

```text
1. Pole position
2. Pole velocity
3. Cart position
4. Cart velocity
```

因此：

```python
observation_space = 4
```

对应：

```text
Observation
[
    pole_pos,
    pole_vel,
    cart_pos,
    cart_vel
]
```

后面 `_get_observations()` 会具体构造这个 Tensor。

---

# 10. State Space

配置：

```python
state_space = 0
```

表示该任务没有额外的 State Buffer。

对于使用 asymmetric actor-critic 的任务，如果存在额外的 critic state，可以使用相应的 state 信息；官方教程在 Observation 部分也说明，对于 asymmetric policies，可以返回：

```python
{
    "policy": obs,
    "critic": states,
}
```

本 Cartpole 示例只返回：

```python
{
    "policy": obs
}
```

---

# 11. Simulation 配置

官方代码：

```python
sim: SimulationCfg = SimulationCfg(
    dt=1 / 120,
    render_interval=decimation,
)
```

因此：

```text
Physics timestep:

dt = 1 / 120 s
```

即：

```text
Physics frequency = 120 Hz
```

同时：

```python
render_interval = decimation
```

当前：

```python
decimation = 2
```

因此渲染间隔与 decimation 保持一致。

---

# 12. Cartpole Robot 配置

机器人配置：

```python
robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(
    prim_path="/World/envs/env_.*/Robot"
)
```

这里使用官方提供的：

```python
CARTPOLE_CFG
```

然后替换：

```python
prim_path
```

为：

```text
/World/envs/env_.*/Robot
```

这样可以对应并行环境中的 Cartpole。

---

# 13. 定义两个 DOF 名称

代码：

```python
cart_dof_name = "slider_to_cart"
pole_dof_name = "cart_to_pole"
```

分别表示：

```text
slider_to_cart
    ↓
Cart 的运动关节
```

以及：

```text
cart_to_pole
    ↓
Pole 的旋转关节
```

之后环境会通过这两个名称查找对应的 joint index。

---

# 14. Scene 配置

官方代码：

```python
scene: InteractiveSceneCfg = InteractiveSceneCfg(
    num_envs=4096,
    env_spacing=4.0,
    replicate_physics=True,
    clone_in_fabric=True,
)
```

这里创建：

```text
4096
```

个并行环境。

环境之间间隔：

```text
4.0
```

并启用：

```python
replicate_physics=True
clone_in_fabric=True
```

这些参数属于官方当前教程中的 Scene 配置。

---

# 15. Reset 参数

Direct Workflow 将 reset 所需的参数直接放在配置类里。

```python
max_cart_pos = 3.0
```

表示：

```text
Cart 超过 ±3.0 m
    ↓
触发 reset / termination
```

另一个参数：

```python
initial_pole_angle_range = [-0.25, 0.25]
```

表示 Pole 初始角度采样范围。

需要注意：

代码后面会乘：

```python
math.pi
```

因此实际采样范围是：

```text
[-0.25π, 0.25π]
```

也就是：

```text
约 [-45°, 45°]
```

官方代码正是通过：

```python
self.cfg.initial_pole_angle_range[0] * math.pi
```

和：

```python
self.cfg.initial_pole_angle_range[1] * math.pi
```

进行采样。

---

# 16. Reward Scale

官方配置：

```python
rew_scale_alive = 1.0
rew_scale_terminated = -2.0
rew_scale_pole_pos = -1.0
rew_scale_cart_vel = -0.01
rew_scale_pole_vel = -0.005
```

分别用于：

```text
Alive Reward
Termination Reward
Pole Position Reward
Cart Velocity Reward
Pole Velocity Reward
```

这些参数不会通过 `RewardManager` 注册，而是直接传入：

```python
compute_rewards(...)
```

中。

---

# 17. 创建 DirectRLEnv

接下来创建真正的环境类：

```python
class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg
```

这就是 Direct Workflow 最核心的区别：

```text
Manager-Based
    ManagerBasedRLEnv
         +
    多个 Manager

Direct
    DirectRLEnv
         +
    一个任务类
```

官方要求新 Direct Workflow 环境继承自：

```python
DirectRLEnv
```

---

# 18. `__init__()`

官方代码：

```python
def __init__(
    self,
    cfg: CartpoleEnvCfg,
    render_mode: str | None = None,
    **kwargs,
):
    super().__init__(cfg, render_mode, **kwargs)

    self._cart_dof_idx, _ = self.cartpole.find_joints(
        self.cfg.cart_dof_name
    )

    self._pole_dof_idx, _ = self.cartpole.find_joints(
        self.cfg.pole_dof_name
    )

    self.action_scale = self.cfg.action_scale

    self.joint_pos = self.cartpole.data.joint_pos
    self.joint_vel = self.cartpole.data.joint_vel
```

这里主要完成四件事情：

### ① 调用父类初始化

```python
super().__init__(...)
```

### ② 查找 Cart joint

```python
self.cartpole.find_joints(...)
```

### ③ 查找 Pole joint

```python
self.cartpole.find_joints(...)
```

### ④ 保存常用数据引用

```python
self.joint_pos
self.joint_vel
```

---

# 19. 为什么要查找 DOF Index？

配置里只有：

```python
cart_dof_name = "slider_to_cart"
```

但实际 Tensor 通常按照 joint index 排列。

例如：

```text
joint_pos
[
    joint_0,
    joint_1,
    ...
]
```

因此需要：

```python
self.cartpole.find_joints(
    "slider_to_cart"
)
```

找到对应 index。

然后可以：

```python
self.joint_pos[:, self._cart_dof_idx]
```

访问 Cart 的状态。

同样：

```python
self._pole_dof_idx
```

用于访问 Pole。

---

# 20. `_setup_scene()`：创建场景

Direct Workflow 中，场景创建由开发者自己实现。

核心函数：

```python
def _setup_scene(self):
    ...
```

官方指出，这里负责：

* 添加 Actor；
* Clone environments；
* 设置碰撞过滤；
* 将 Actor 加入 Scene；
* 添加 Ground Plane；
* 添加灯光等额外对象。

---

# 21. 创建 Cartpole

```python
self.cartpole = Articulation(
    self.cfg.robot_cfg
)
```

这里直接创建：

```python
Articulation
```

而不是依赖 Manager 自动创建。

这正体现了 Direct Workflow 的“直接控制”。

---

# 22. 创建 Ground Plane

```python
spawn_ground_plane(
    prim_path="/World/ground",
    cfg=GroundPlaneCfg(),
)
```

创建地面：

```text
/World/ground
```

---

# 23. Clone 环境

```python
self.scene.clone_environments(
    copy_from_source=False
)
```

用于创建并行环境。

因为配置中：

```python
num_envs=4096
```

所以最终需要大量 Cartpole 环境。

---

# 24. CPU 碰撞过滤

官方代码：

```python
if self.device == "cpu":
    self.scene.filter_collisions(
        global_prim_paths=[]
    )
```

这里是针对 CPU simulation 的碰撞过滤处理。

这段代码是官方当前教程的一部分，不应简单删除后认为所有设备上的行为完全一样。

---

# 25. 将 Cartpole 加入 Scene

```python
self.scene.articulations["cartpole"] = self.cartpole
```

这样：

```text
InteractiveScene
       │
       └── articulations
              │
              └── cartpole
```

就可以通过 Scene 管理这个 articulation。

---

# 26. 添加灯光

```python
light_cfg = sim_utils.DomeLightCfg(
    intensity=2000.0,
    color=(0.75, 0.75, 0.75),
)

light_cfg.func(
    "/World/Light",
    light_cfg,
)
```

添加 Dome Light：

```text
/World/Light
```

亮度：

```text
2000
```

颜色：

```text
(0.75, 0.75, 0.75)
```

这是用于仿真场景的可视化。

---

# 27. Action 执行流程

Direct Workflow 中 Action 有两个重要函数：

```python
_pre_physics_step()
```

以及：

```python
_apply_action()
```

两者的调用时机不同。

---

# 28. `_pre_physics_step()`

官方代码：

```python
def _pre_physics_step(
    self,
    actions: torch.Tensor,
) -> None:

    self.actions = (
        self.action_scale * actions.clone()
    )
```

这个函数：

> 每一个 RL step 调用一次。

它接收 Policy 输出的 Action：

```text
actions
```

然后进行缩放：

```text
actions × action_scale
```

保存到：

```python
self.actions
```

中。

---

# 29. Action Scaling

配置：

```python
action_scale = 100.0
```

因此：

```python
self.actions = 100.0 * actions
```

例如：

```text
Policy Action:
0.5
```

经过 scale：

```text
0.5 × 100
= 50
```

最终使用：

```text
50 N
```

作为目标关节力。

官方配置中的 `action_scale` 注释明确标记单位为 `[N]`。

---

# 30. `_apply_action()`

官方代码：

```python
def _apply_action(self) -> None:
    self.cartpole.set_joint_effort_target(
        self.actions,
        joint_ids=self._cart_dof_idx,
    )
```

这个函数负责：

> 真正把 Action 应用到物理仿真。

调用：

```python
set_joint_effort_target()
```

控制：

```text
slider_to_cart
```

对应的 joint。

---

# 31. `_pre_physics_step()` 与 `_apply_action()` 的区别

这是 Direct Workflow 非常重要的概念。

```text
RL Step
   │
   ▼
_pre_physics_step(actions)
   │
   │ 处理 / 缓存 Action
   ▼
Physics Step 1
   │
   ▼
_apply_action()
   │
   ▼
Physics Step 2
   │
   ▼
_apply_action()
   │
   ...
```

因为：

```python
decimation = 2
```

所以：

```text
每一个 RL step
    ↓
2 个 physics step
```

而：

```python
_apply_action()
```

会在每个 physics step 前调用。

官方文档明确说明：

* `_pre_physics_step()` 每个 RL step 调用一次；
* `_apply_action()` 每个 RL step 中按照 `decimation` 调用。

---

# 32. Observation

Observation 在：

```python
_get_observations()
```

中计算。

官方代码：

```python
def _get_observations(self) -> dict:

    obs = torch.cat(
        (
            self.joint_pos[
                :,
                self._pole_dof_idx[0]
            ].unsqueeze(dim=1),

            self.joint_vel[
                :,
                self._pole_dof_idx[0]
            ].unsqueeze(dim=1),

            self.joint_pos[
                :,
                self._cart_dof_idx[0]
            ].unsqueeze(dim=1),

            self.joint_vel[
                :,
                self._cart_dof_idx[0]
            ].unsqueeze(dim=1),
        ),
        dim=-1,
    )

    observations = {
        "policy": obs
    }

    return observations
```

---

# 33. Observation 的具体组成

最终：

```text
obs =
[
    pole_position,
    pole_velocity,
    cart_position,
    cart_velocity
]
```

因此：

```python
observation_space = 4
```

与代码完全对应。

---

# 34. 为什么使用 `torch.cat()`？

因为每个状态首先是一个独立的 Tensor。

最终希望形成：

```text
[4]
```

维 Observation。

所以：

```python
torch.cat(
    (...),
    dim=-1
)
```

将：

```text
Pole Position
+
Pole Velocity
+
Cart Position
+
Cart Velocity
```

拼接为：

```text
Observation
```

---

# 35. 返回 `"policy"`

最终：

```python
observations = {
    "policy": obs
}
```

这里：

```text
policy
```

是 Observation Dictionary 的 key。

RL Agent 从这里读取 Policy Observation。

对于需要 asymmetric actor-critic 的环境，官方说明还可以增加：

```python
{
    "policy": obs,
    "critic": states,
}
```

其中 `critic` 提供给价值函数使用。

---

# 36. Reward

Reward 的核心入口：

```python
_get_rewards()
```

官方代码：

```python
def _get_rewards(self) -> torch.Tensor:

    total_reward = compute_rewards(
        self.cfg.rew_scale_alive,
        self.cfg.rew_scale_terminated,
        self.cfg.rew_scale_pole_pos,
        self.cfg.rew_scale_cart_vel,
        self.cfg.rew_scale_pole_vel,
        self.joint_pos[:, self._pole_dof_idx[0]],
        self.joint_vel[:, self._pole_dof_idx[0]],
        self.joint_pos[:, self._cart_dof_idx[0]],
        self.joint_vel[:, self._cart_dof_idx[0]],
        self.reset_terminated,
    )

    return total_reward
```

这里真正的 Reward 计算放到了：

```python
compute_rewards()
```

函数中。

---

# 37. 为什么单独定义 `compute_rewards()`？

官方代码使用：

```python
@torch.jit.script
def compute_rewards(...):
    ...
```

这样可以使用 PyTorch JIT 对 Reward 计算函数进行处理。

这也是官方在介绍 Direct Workflow 时提到的优势之一：

> Direct Workflow 可以让开发者更直接地控制代码实现，并方便使用 PyTorch JIT 等机制。

---

# 38. `compute_rewards()` 完整逻辑

官方 Reward 计算：

```python
@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):

    rew_alive = (
        rew_scale_alive
        * (1.0 - reset_terminated.float())
    )

    rew_termination = (
        rew_scale_terminated
        * reset_terminated.float()
    )

    rew_pole_pos = (
        rew_scale_pole_pos
        * torch.sum(
            torch.square(pole_pos),
            dim=-1
        )
    )

    rew_cart_vel = (
        rew_scale_cart_vel
        * torch.sum(
            torch.abs(cart_vel),
            dim=-1
        )
    )

    rew_pole_vel = (
        rew_scale_pole_vel
        * torch.sum(
            torch.abs(pole_vel),
            dim=-1
        )
    )

    total_reward = (
        rew_alive
        + rew_termination
        + rew_pole_pos
        + rew_cart_vel
        + rew_pole_vel
    )

    return total_reward
```

---

# 39. Alive Reward

```python
rew_alive = (
    rew_scale_alive
    * (1.0 - reset_terminated.float())
)
```

配置：

```python
rew_scale_alive = 1.0
```

所以如果环境没有终止：

```text
reset_terminated = 0
```

那么：

```text
Alive Reward = 1
```

如果当前环境已经 terminated：

```text
reset_terminated = 1
```

那么：

```text
Alive Reward = 0
```

---

# 40. Termination Reward

```python
rew_termination = (
    rew_scale_terminated
    * reset_terminated.float()
)
```

配置：

```python
rew_scale_terminated = -2.0
```

所以终止时：

```text
Termination Reward = -2
```

---

# 41. Pole Position Reward

```python
rew_pole_pos = (
    rew_scale_pole_pos
    * torch.sum(
        torch.square(pole_pos),
        dim=-1
    )
)
```

配置：

```python
rew_scale_pole_pos = -1.0
```

因此：

```text
Pole position 越接近 0
    ↓
平方误差越小
    ↓
负奖励越小
    ↓
总 Reward 越高
```

也就是说，该项鼓励：

```text
pole_pos → 0
```

---

# 42. Cart Velocity Reward

```python
rew_cart_vel = (
    rew_scale_cart_vel
    * torch.sum(
        torch.abs(cart_vel),
        dim=-1
    )
)
```

配置：

```python
rew_scale_cart_vel = -0.01
```

因此：

```text
Cart velocity 越大
        ↓
|velocity| 越大
        ↓
负奖励越大
```

即鼓励 Cart 减少运动速度。

---

# 43. Pole Velocity Reward

```python
rew_pole_vel = (
    rew_scale_pole_vel
    * torch.sum(
        torch.abs(pole_vel),
        dim=-1
    )
)
```

配置：

```python
rew_scale_pole_vel = -0.005
```

因此鼓励：

```text
Pole velocity → 0
```

---

# 44. Total Reward

最终：

```python
total_reward = (
    rew_alive
    + rew_termination
    + rew_pole_pos
    + rew_cart_vel
    + rew_pole_vel
)
```

所以：

```text
                  Total Reward
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
   Alive Reward   Termination      State Penalties
                       │                │
                       │        ┌───────┼────────┐
                       │        │       │        │
                       │      Pole     Cart     Pole
                       │      Pos      Vel      Vel
```

---

# 45. 注意：`cart_pos` 没有用于 Reward

`compute_rewards()` 的参数中包含：

```python
cart_pos
```

但当前官方代码中的 Reward 计算没有使用它。

使用的是：

```python
pole_pos
pole_vel
cart_vel
reset_terminated
```

因此不要自行推断：

> Cart Position 一定参与 Reward。

当前教程的实际代码并没有这样做。

---

# 46. Dones

Direct Workflow 中：

```python
_get_dones()
```

负责计算：

```text
Termination
+
Timeout
```

官方代码：

```python
def _get_dones(
    self,
) -> tuple[torch.Tensor, torch.Tensor]:

    self.joint_pos = self.cartpole.data.joint_pos
    self.joint_vel = self.cartpole.data.joint_vel

    time_out = (
        self.episode_length_buf
        >= self.max_episode_length - 1
    )

    out_of_bounds = torch.any(
        torch.abs(
            self.joint_pos[:, self._cart_dof_idx]
        ) > self.cfg.max_cart_pos,
        dim=1,
    )

    out_of_bounds = (
        out_of_bounds
        | torch.any(
            torch.abs(
                self.joint_pos[:, self._pole_dof_idx]
            ) > math.pi / 2,
            dim=1,
        )
    )

    return out_of_bounds, time_out
```

---

# 47. `time_out`

代码：

```python
time_out = (
    self.episode_length_buf
    >= self.max_episode_length - 1
)
```

用于判断：

```text
Episode 是否达到最大长度
```

配置：

```python
episode_length_s = 5.0
```

所以 Episode 最长为：

```text
5 秒
```

---

# 48. Cart 超出范围

代码：

```python
torch.abs(
    self.joint_pos[:, self._cart_dof_idx]
) > self.cfg.max_cart_pos
```

而：

```python
max_cart_pos = 3.0
```

所以：

```text
|Cart Position| > 3
```

时：

```text
out_of_bounds = True
```

---

# 49. Pole 倒下

还有一个重要条件：

```python
torch.abs(
    self.joint_pos[:, self._pole_dof_idx]
) > math.pi / 2
```

即：

```text
|Pole Angle| > π/2
```

时认为 Pole 已经超出允许范围。

所以实际终止条件是：

```text
Cart 出界
        OR
Pole 角度超过 ±π/2
```

官方代码明确实现了这一逻辑。

---

# 50. `_get_dones()` 返回两个 Tensor

返回：

```python
return out_of_bounds, time_out
```

也就是说：

```text
第一个：
Termination

第二个：
Time Out
```

两者都是：

```python
torch.Tensor
```

并且通常是每个并行环境一个布尔值。

例如：

```text
4096 个环境
    ↓
4096 个 termination flags
+
4096 个 timeout flags
```

---

# 51. Reset

确定哪些环境需要 reset 后：

```python
_reset_idx(env_ids)
```

负责真正执行 reset。

官方说明：

> `_get_dones()` 计算需要 reset 的环境后，`_reset_idx()` 对这些环境执行实际状态重置，并把新的状态写入仿真。

---

# 52. `_reset_idx()` 的入口

```python
def _reset_idx(
    self,
    env_ids: Sequence[int] | None,
):
```

如果：

```python
env_ids is None
```

则：

```python
env_ids = self.cartpole._ALL_INDICES
```

也就是：

```text
所有环境
```

否则只 reset：

```text
指定 env_ids
```

这对于并行环境非常重要。

---

# 53. 调用父类 Reset

```python
super()._reset_idx(env_ids)
```

先让父类执行基础 reset 逻辑。

然后 Direct Workflow 任务自己设置 Cartpole 的具体状态。

---

# 54. 获取默认 Joint Position

```python
joint_pos = (
    self.cartpole
    .data
    .default_joint_pos[env_ids]
)
```

从默认状态开始。

然后对 Pole 角度增加随机偏移：

```python
joint_pos[:, self._pole_dof_idx] += sample_uniform(
    self.cfg.initial_pole_angle_range[0] * math.pi,
    self.cfg.initial_pole_angle_range[1] * math.pi,
    joint_pos[:, self._pole_dof_idx].shape,
    joint_pos.device,
)
```

因此每次 reset 时：

```text
Pole 初始角度
    ↓
随机采样
    ↓
[-0.25π, 0.25π]
```

---

# 55. Reset Joint Velocity

```python
joint_vel = (
    self.cartpole
    .data
    .default_joint_vel[env_ids]
)
```

使用 Cartpole 默认 joint velocity。

当前教程没有像 Pole angle 那样额外对 velocity 添加随机偏移。

---

# 56. Reset Root State

```python
default_root_state = (
    self.cartpole
    .data
    .default_root_state[env_ids]
)

default_root_state[:, :3] += (
    self.scene.env_origins[env_ids]
)
```

这里需要注意：

```text
default_root_state
```

是机器人根状态。

通过：

```python
self.scene.env_origins
```

将状态转换到对应并行环境的位置。

---

# 57. 更新本地状态缓存

```python
self.joint_pos[env_ids] = joint_pos
self.joint_vel[env_ids] = joint_vel
```

更新环境中保存的状态引用。

---

# 58. 将状态写入仿真

最后：

```python
self.cartpole.write_root_pose_to_sim(
    default_root_state[:, :7],
    env_ids,
)

self.cartpole.write_root_velocity_to_sim(
    default_root_state[:, 7:],
    env_ids,
)

self.cartpole.write_joint_state_to_sim(
    joint_pos,
    joint_vel,
    None,
    env_ids,
)
```

也就是说，Reset 并不是只修改 Python Tensor。

而是：

```text
新状态
  ↓
写入 Simulation
  ↓
下一次 Physics Step 使用新状态
```

这是自定义 Direct Environment 时非常重要的一点。

---

# 59. Direct Workflow 的完整生命周期

现在可以把一个 RL Step 完整表示为：

```text
                Policy
                   │
                   │ Action
                   ▼
       _pre_physics_step(actions)
                   │
                   ▼
            Action Scaling
                   │
                   ▼
          ┌────────────────┐
          │ Physics Step 1 │
          └───────┬────────┘
                  │
           _apply_action()
                  │
                  ▼
          ┌────────────────┐
          │ Physics Step 2 │
          └───────┬────────┘
                  │
           _apply_action()
                  │
                  ▼
          _get_observations()
                  │
          _get_rewards()
                  │
            _get_dones()
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
      Obs        Reward     Dones
```

这里：

```text
decimation = 2
```

所以每一个 RL step 有两个 physics steps。

---

# 60. Direct Workflow 的核心接口

可以把 DirectRLEnv 的关键接口记成：

```text
DirectRLEnv
│
├── _setup_scene()
│
├── _pre_physics_step()
│
├── _apply_action()
│
├── _get_observations()
│
├── _get_rewards()
│
├── _get_dones()
│
└── _reset_idx()
```

这七个部分基本覆盖了本教程的核心。

---

# 61. 与 Manager-Based RL 的对应关系

如果你已经学习了前面的 Manager-Based RL 教程，那么可以这样对应：

| Manager-Based          | Direct Workflow                           |
| ---------------------- | ----------------------------------------- |
| `SceneCfg`             | `_setup_scene()`                          |
| Action Manager         | `_pre_physics_step()` + `_apply_action()` |
| Observation Manager    | `_get_observations()`                     |
| Reward Manager         | `_get_rewards()`                          |
| Termination Manager    | `_get_dones()`                            |
| Event Manager / reset  | `_reset_idx()`                            |
| `ManagerBasedRLEnvCfg` | `DirectRLEnvCfg`                          |
| `ManagerBasedRLEnv`    | `DirectRLEnv`                             |

这种对应关系有助于从 Manager-Based 环境迁移到 Direct Workflow。

---

# 62. 为什么 Direct Workflow 更“直接”？

例如 Reward。

Manager-Based：

```text
RewardTermCfg
       ↓
RewardManager
       ↓
reward function
       ↓
total reward
```

Direct：

```text
_get_rewards()
       ↓
compute_rewards()
       ↓
total reward
```

Observation 同样如此。

Manager-Based：

```text
ObservationTerm
        ↓
ObservationManager
        ↓
Observation
```

Direct：

```text
_get_observations()
        ↓
torch.cat(...)
        ↓
Observation
```

因此 Direct Workflow 的代码路径更短。

---

# 63. Direct Workflow 的一个重要优势：PyTorch JIT

教程明确使用：

```python
@torch.jit.script
def compute_rewards(...):
    ...
```

这意味着 Reward 的主要计算逻辑被独立出来，可以使用 PyTorch JIT 的能力。

这也是 Direct Workflow 比 Manager-Based 方式更适合某些高度定制任务的原因之一。官方教程在开头明确将“更直接的控制”和“使用 PyTorch JIT 等功能”列为 Direct Workflow 的特点。

---

# 64. 运行 Direct Cartpole

官方教程给出的训练命令：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-Direct-v0
```

这里使用：

```text
rl_games
```

进行训练。

任务名称：

```text
Isaac-Cartpole-Direct-v0
```

注意：

```text
-Direct
```

后缀用于区分 Direct Workflow 实现。

---

# 65. `-Direct` 后缀

官方说明：

> 所有 Direct Workflow 任务都会在任务名称后增加 `-Direct` 后缀，用于区分不同的实现方式。

因此可以看到：

```text
Isaac-Cartpole-Direct-v0
```

而不是：

```text
Isaac-Cartpole-v0
```

这个命名约定在寻找任务时非常重要。

---

# 66. 训练流程

运行：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task=Isaac-Cartpole-Direct-v0
```

大体流程：

```text
train.py
   │
   ▼
Gymnasium Task
   │
   ▼
Isaac-Cartpole-Direct-v0
   │
   ▼
DirectRLEnv
   │
   ▼
CartpoleEnv
   │
   ├── Observation
   ├── Action
   ├── Reward
   └── Termination
   │
   ▼
RL Games
   │
   ▼
Policy Training
```

---

# 67. Domain Randomization

虽然 Cartpole 的核心实现没有使用复杂的 Domain Randomization，但教程专门介绍了 Direct Workflow 下的随机化配置。

Direct Workflow 中可以使用：

```python
configclass
```

定义：

```python
EventTermCfg
```

变量。

例如：

```python
@configclass
class EventCfg:

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            ...
        },
    )
```

官方教程使用这个方式说明 Direct Workflow 同样可以使用 Event 配置进行 Domain Randomization。

---

# 68. 随机化摩擦系数

官方示例：

```python
robot_physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            body_names=".*"
        ),
        "static_friction_range": (0.7, 1.3),
        "dynamic_friction_range": (1.0, 1.0),
        "restitution_range": (1.0, 1.0),
        "num_buckets": 250,
    },
)
```

这里随机化：

```text
Static friction
Dynamic friction
Restitution
```

并且：

```python
mode="reset"
```

表示在 reset 时执行。

---

# 69. 随机化 Actuator Gains

官方示例：

```python
robot_joint_stiffness_and_damping = EventTerm(
    func=mdp.randomize_actuator_gains,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=".*"
        ),
        "stiffness_distribution_params": (
            0.75,
            1.5
        ),
        "damping_distribution_params": (
            0.3,
            3.0
        ),
        "operation": "scale",
        "distribution": "log_uniform",
    },
)
```

作用是对：

```text
Joint stiffness
Joint damping
```

进行随机化。

---

# 70. 随机化 Gravity

官方示例：

```python
reset_gravity = EventTerm(
    func=mdp.randomize_physics_scene_gravity,
    mode="interval",
    is_global_time=True,
    interval_range_s=(36.0, 36.0),
    params={
        "gravity_distribution_params": (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.4],
        ),
        "operation": "add",
        "distribution": "gaussian",
    },
)
```

这里：

```text
mode="interval"
```

表示不是只在 reset 时改变，而是按照 interval 周期执行。

教程注释说明：

```text
time_s = num_steps * (decimation * dt)
```

用于理解 interval 时间与仿真 step 的关系。

---

# 71. EventTerm 的三个 mode

官方说明 `EventTermCfg` 的 `mode` 可以是：

```text
startup
reset
interval
```

分别表示：

### `startup`

环境启动时执行。

### `reset`

环境 reset 时执行。

### `interval`

按照指定时间间隔执行。

Event 的具体函数通过：

```python
func
```

指定，而参数通过：

```python
params
```

提供。

---

# 72. `SceneEntityCfg`

Domain Randomization 示例中经常看到：

```python
SceneEntityCfg(
    "robot",
    body_names=".*"
)
```

这里：

```text
robot
```

是 Actor 的名称。

```text
body_names=".*"
```

是匹配 body 名称的正则表达式。

也就是说，随机化作用到：

```text
robot
    ↓
匹配的 body
```

官方教程明确说明，Actor 名称以及 body/joint 名称可以通过正则表达式指定随机化对象。

---

# 73. 将 Event 配置加入任务

定义完：

```python
class EventCfg:
    ...
```

之后，还需要加入主任务配置：

```python
@configclass
class MyTaskConfig:

    events: EventCfg = EventCfg()
```

官方教程明确说明，只有将 Event 配置赋值给主任务配置中的 `events` 后，这些随机化项才被纳入任务配置。

---

# 74. Action Noise 和 Observation Noise

Direct Workflow 还支持：

```text
Action Noise
Observation Noise
```

通过：

```python
action_noise_model
```

和：

```python
observation_noise_model
```

添加。

例如：

```python
@configclass
class MyTaskConfig:

    action_noise_model = ...
    observation_noise_model = ...
```

官方教程使用：

```python
NoiseModelWithAdditiveBiasCfg
```

展示如何同时加入：

```text
Gaussian Noise
+
Bias
```

---

# 75. Noise Model with Additive Bias

官方示例：

```python
action_noise_model = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=GaussianNoiseCfg(
        mean=0.0,
        std=0.05,
        operation="add",
    ),
    bias_noise_cfg=GaussianNoiseCfg(
        mean=0.0,
        std=0.015,
        operation="abs",
    ),
)
```

Observation：

```python
observation_noise_model = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=GaussianNoiseCfg(
        mean=0.0,
        std=0.002,
        operation="add",
    ),
    bias_noise_cfg=GaussianNoiseCfg(
        mean=0.0,
        std=0.0001,
        operation="abs",
    ),
)
```

---

# 76. Noise 的两种时间尺度

官方对：

```python
noise_cfg
```

和：

```python
bias_noise_cfg
```

进行了明确区分。

### `noise_cfg`

每一个 step 重新采样：

```text
Step 1 → Noise A
Step 2 → Noise B
Step 3 → Noise C
```

所以它是不相关的逐步噪声。

### `bias_noise_cfg`

在 reset 时采样：

```text
Reset
 ↓
Bias A
 ↓
Step 1 → A
Step 2 → A
Step 3 → A
...
Next Reset
 ↓
Bias B
```

因此它在一个 episode 中保持一致，在下一次 reset 时重新采样。

官方明确将这种 bias 描述为 correlated noise。

---

# 77. 如果只需要每 Step 的 Gaussian Noise

官方提供：

```python
GaussianNoiseCfg
```

例如：

```python
@configclass
class MyTaskConfig:

    action_noise_model = GaussianNoiseCfg(
        mean=0.0,
        std=0.05,
        operation="add",
    )
```

这样可以只添加：

```text
每 Step Gaussian Noise
```

而不使用 reset 时重新采样并持续整个 episode 的 bias。

---

# 78. Direct Workflow 的完整结构

最终可以将整个 Direct Workflow 理解成：

```text
                    DirectRLEnv
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
        Scene Setup    Actions    Observations
             │           │           │
             │           │           │
             ▼           ▼           ▼
      _setup_scene()   _pre_...   _get_observations()
                         │
                         ▼
                   _apply_action()
                         │
                         ▼
                     Physics
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          Rewards      Dones      Reset
              │          │          │
              ▼          ▼          ▼
       _get_rewards  _get_dones  _reset_idx
```

---

# 79. Direct Workflow 的核心思想

如果用一句话总结：

> **把一个 RL 环境需要的核心逻辑直接写进 `DirectRLEnv` 子类中。**

而不是：

```text
ActionManager
ObservationManager
RewardManager
TerminationManager
...
```

分别管理。

因此 Direct Workflow 更像：

```text
一个任务
   ↓
一个主要 Environment 类
   ↓
自己负责：
    Scene
    Action
    Observation
    Reward
    Done
    Reset
```

---

# 80. Direct Workflow 什么时候比较合适？

根据官方对该工作流的描述，它适合需要：

```text
更直接的环境控制
```

以及：

```text
更少抽象层
```

的场景。

尤其是：

* Reward 逻辑高度定制；
* Observation 逻辑高度定制；
* 需要直接操作 Tensor；
* 希望使用 PyTorch JIT；
* 希望在一个任务类中快速定位环境逻辑。

这些都是 Direct Workflow 的设计目标。

---

# 81. Manager-Based 与 Direct：如何选择？

可以从代码组织方式理解：

### Manager-Based

```text
MyTask
│
├── Scene Config
├── Action Config
├── Observation Config
├── Reward Config
├── Termination Config
└── Event Config
```

更强调：

```text
模块化
复用
配置化
```

### Direct

```text
MyTask
│
└── MyEnv(DirectRLEnv)
      ├── scene
      ├── action
      ├── observation
      ├── reward
      ├── done
      └── reset
```

更强调：

```text
直接控制
集中实现
灵活性
```

官方 Reference Architecture 也将 Direct Workflow 描述为由单个类负责计算 Observation、应用 Action 和计算 Reward 的工作流。

---

# 82. 本教程中的关键数据流

最终一个 RL step：

```text
Policy
  │
  │ Action
  ▼
_pre_physics_step()
  │
  │ scale
  ▼
self.actions
  │
  ▼
_apply_action()
  │
  ▼
Physics Simulation
  │
  ├───────────────┐
  │               │
  ▼               ▼
Observation      State
  │
  ▼
_get_observations()

State
  │
  ├───────────────┐
  ▼               ▼
Reward          Dones
  │               │
  ▼               ▼
_get_rewards()  _get_dones()
```

如果需要 Reset：

```text
_get_dones()
     │
     ▼
env_ids
     │
     ▼
_reset_idx(env_ids)
     │
     ▼
写入新的 Simulation State
```

---

# 83. 本教程的完整核心代码骨架

为了理解 Direct Workflow，可以将官方 Cartpole 实现抽象成下面的结构：

```python
@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):

    decimation = 2
    episode_length_s = 5.0

    action_scale = 100.0

    action_space = 1
    observation_space = 4
    state_space = 0

    sim = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )

    robot_cfg = CARTPOLE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    scene = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    max_cart_pos = 3.0
    initial_pole_angle_range = [-0.25, 0.25]

    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


class CartpoleEnv(DirectRLEnv):

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(
            cfg,
            render_mode,
            **kwargs,
        )

        self._cart_dof_idx, _ = (
            self.cartpole.find_joints(
                self.cfg.cart_dof_name
            )
        )

        self._pole_dof_idx, _ = (
            self.cartpole.find_joints(
                self.cfg.pole_dof_name
            )
        )

        self.action_scale = (
            self.cfg.action_scale
        )

        self.joint_pos = (
            self.cartpole.data.joint_pos
        )

        self.joint_vel = (
            self.cartpole.data.joint_vel
        )

    def _setup_scene(self):
        ...

    def _pre_physics_step(self, actions):
        ...

    def _apply_action(self):
        ...

    def _get_observations(self):
        ...

    def _get_rewards(self):
        ...

    def _get_dones(self):
        ...

    def _reset_idx(self, env_ids):
        ...
```

这段骨架反映的是官方教程的实际结构，而不是另造一个不同的环境架构。

---

# 84. 与前两个教程的学习关系

如果你已经学习了前面的：

```text
Creating a Manager-Based Base Environment
```

和：

```text
Creating a Manager-Based RL Environment
```

那么当前教程可以理解成：

```text
                    Isaac Lab
                       │
              Environment Workflows
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      Manager-Based          Direct
             │                   │
             ▼                   ▼
    ManagerBasedRLEnv       DirectRLEnv
             │                   │
       配置多个 Manager      直接实现任务逻辑
```

Manager-Based：

```text
Reward
   ↓
RewardManager
```

Direct：

```text
Reward
   ↓
_get_rewards()
```

Manager-Based：

```text
Observation
   ↓
ObservationManager
```

Direct：

```text
Observation
   ↓
_get_observations()
```

---

# 85. 最重要的 API 总结

本教程建议重点掌握：

```python
DirectRLEnv
```

Direct RL 环境基类。

```python
DirectRLEnvCfg
```

Direct RL 环境配置基类。

```python
_setup_scene()
```

创建场景。

```python
_pre_physics_step()
```

每个 RL step 预处理 Action。

```python
_apply_action()
```

在 physics step 前应用 Action。

```python
_get_observations()
```

构造 Observation。

```python
_get_rewards()
```

计算 Reward。

```python
_get_dones()
```

计算 Termination 和 Timeout。

```python
_reset_idx()
```

重置指定环境。

---

# 86. 最重要的调用关系

建议记住下面这张图：

```text
                 DirectRLEnv
                     │
                     ▼
              _setup_scene()
                     │
                     ▼
              Simulation Ready
                     │
                     ▼
              RL Step Begins
                     │
                     ▼
          _pre_physics_step()
                     │
                     ▼
               Physics Step
                     │
                     ▼
             _apply_action()
                     │
                     ▼
               Physics Step
                     │
                     ▼
          _get_observations()
                     │
                     ▼
             _get_rewards()
                     │
                     ▼
              _get_dones()
                     │
          ┌──────────┴──────────┐
          │                     │
        Done                  Not Done
          │                     │
          ▼                     ▼
     _reset_idx()            Next Step
```

---

# 87. 为什么不是所有东西都写在 `step()`？

Direct Workflow 虽然叫“Direct”，但并不是让用户完全重写所有底层 RL 循环。

Isaac Lab 的 `DirectRLEnv` 已经定义了环境生命周期。

开发者主要通过重载：

```text
_setup_scene
_pre_physics_step
_apply_action
_get_observations
_get_rewards
_get_dones
_reset_idx
```

提供任务-specific 的逻辑。

因此：

```text
Direct
```

并不意味着：

```text
完全脱离 Isaac Lab Framework
```

而是：

> **减少 Manager 抽象，把任务逻辑直接放到 Environment 子类中。**

---

# 88. 一个非常重要的理解：并行环境

配置：

```python
num_envs = 4096
```

意味着：

```text
4096 个 Cartpole
```

同时运行。

因此所有主要状态都是 Tensor：

```text
joint_pos
    ↓
[4096, num_joints]

joint_vel
    ↓
[4096, num_joints]

reward
    ↓
[4096]

terminated
    ↓
[4096]

timeout
    ↓
[4096]
```

这也是为什么教程中的代码大量使用：

```python
torch.Tensor
```

以及：

```python
torch.abs()
torch.square()
torch.cat()
torch.any()
```

而不是 Python `for` 循环逐个环境处理。

---

# 89. Direct Workflow 的核心设计思想

最终可以把 Direct Workflow 总结成：

```text
配置：
DirectRLEnvCfg
       │
       ▼
环境：
DirectRLEnv
       │
       ├── Scene
       ├── Action
       ├── Observation
       ├── Reward
       ├── Done
       └── Reset
```

其中：

```text
Scene
```

决定：

> 仿真中有什么。

```text
Action
```

决定：

> Agent 能做什么。

```text
Observation
```

决定：

> Agent 能看到什么。

```text
Reward
```

决定：

> 什么行为被鼓励。

```text
Done
```

决定：

> Episode 什么时候结束。

```text
Reset
```

决定：

> Episode 如何重新开始。

---

# 90. 最终总结

本教程的核心目标不是训练一个 Cartpole，而是学习如何**直接实现一个 RL Task**。

整个任务结构：

```text
CartpoleEnvCfg
       │
       ▼
   DirectRLEnv
       │
       ├── _setup_scene()
       │
       ├── _pre_physics_step()
       │
       ├── _apply_action()
       │
       ├── _get_observations()
       │
       ├── _get_rewards()
       │
       ├── _get_dones()
       │
       └── _reset_idx()
```

与 Manager-Based Workflow 相比：

```text
Manager-Based
    ↓
多个 Manager + Config
```

Direct Workflow：

```text
Direct
    ↓
一个 Environment 类
    ↓
直接实现任务逻辑
```

同时，Direct Workflow 并没有放弃 Isaac Lab 的配置体系。环境仍然使用：

```python
@configclass
```

以及：

```python
DirectRLEnvCfg
```

并且仍然可以使用：

```text
Domain Randomization
Action Noise
Observation Noise
```

等机制。

---

# 91. 下一步

本教程最后指出，虽然可以直接手动创建：

```python
DirectRLEnv
```

的实例，但如果每一个任务都需要一个专门的脚本，这种方式并不适合大规模管理多个任务。

因此下一步是使用：

```python
gymnasium.make()
```

通过 Gymnasium 注册表创建环境。

也就是从：

```text
手动创建 DirectRLEnv
        ↓
Gymnasium 注册
        ↓
gymnasium.make()
        ↓
RL Framework
```

进入更加标准化的 RL 任务使用方式。

官方教程明确将 **Registering an Environment** 作为下一步学习内容。

---

# 92. 本教程知识点检查清单

完成本教程后，应该能够解释：

* [x] 什么是 Direct Workflow
* [x] `DirectRLEnv` 的作用
* [x] `DirectRLEnvCfg` 的作用
* [x] Direct Workflow 与 Manager-Based Workflow 的区别
* [x] 如何创建 Scene
* [x] 如何处理 Action
* [x] `_pre_physics_step()` 的作用
* [x] `_apply_action()` 的作用
* [x] `decimation` 的作用
* [x] 如何创建 Observation
* [x] 如何计算 Reward
* [x] 为什么可以使用 `torch.jit.script`
* [x] 如何计算 Termination
* [x] 如何处理 Timeout
* [x] 如何执行 Reset
* [x] 如何对并行环境进行局部 Reset
* [x] Domain Randomization 的配置方式
* [x] Action / Observation Noise 的配置方式
* [x] `NoiseModelWithAdditiveBiasCfg` 的基本作用
* [x] `GaussianNoiseCfg` 的基本作用
* [x] `Isaac-Cartpole-Direct-v0` 的任务命名方式
* [x] 如何使用 RL Games 训练 Direct Cartpole
* [x] 为什么最终还需要 Gymnasium 注册机制

---

# 参考

**Isaac Lab 官方教程：**

https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html

**相关模块：**

```text
isaaclab_tasks.direct.cartpole
```

**主要环境类：**

```python
DirectRLEnv
DirectRLEnvCfg
```

**Cartpole 任务：**

```text
Isaac-Cartpole-Direct-v0
```

**官方训练命令：**

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task=Isaac-Cartpole-Direct-v0
```

本文按照你提供的 Isaac Lab `main` 分支页面整理；官方页面当前显示最后更新时间为 **2026-09-19**。
