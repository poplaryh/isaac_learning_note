# 与表面夹爪进行交互
## Interacting with a Surface Gripper

本教程演示如何在仿真中与一个**末端执行器安装了表面夹爪的关节机器人（articulated robot）**进行交互。

它是上一篇 **Interacting with an articulation（与 Articulation 进行交互）** 教程的延续。在上一篇教程中，我们学习了如何与一个关节机器人进行交互；这一篇进一步加入 Surface Gripper。

需要特别注意：

> **从 Isaac Sim 5.0 开始，Surface Gripper 目前只支持 CPU backend（CPU 后端）。**

因此运行本教程时必须使用：

```bash
./isaaclab.sh -p scripts/tutorials/01_assets/run_surface_gripper.py --device=cpu
```

---

# 一、代码

本教程对应：

```text
scripts/tutorials/01_assets/run_surface_gripper.py
```

运行方式：

```bash
./isaaclab.sh -p scripts/tutorials/01_assets/run_surface_gripper.py --device=cpu
```

---

# 二、启动 Isaac Sim

首先：

```python
import argparse

from isaaclab.app import AppLauncher
```

创建命令行参数：

```python
parser = argparse.ArgumentParser(
    description="Tutorial on spawning and interacting with a Surface Gripper."
)
```

添加 `AppLauncher` 参数：

```python
AppLauncher.add_app_launcher_args(parser)
```

解析：

```python
args_cli = parser.parse_args()
```

启动 Omniverse：

```python
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

这一部分与前面几个教程完全相同。

---

# 三、导入相关模块

接下来：

```python
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, SurfaceGripper, SurfaceGripperCfg
from isaaclab.sim import SimulationContext
```

这里新增了两个关键类：

```python
SurfaceGripper
SurfaceGripperCfg
```

它们分别用于：

```text
SurfaceGripperCfg
        ↓
描述夹爪配置

SurfaceGripper
        ↓
实际管理夹爪
```

此外仍然使用：

```python
Articulation
```

来管理机器人本体。

---

# 四、使用预定义的 Pick-and-Place 机器人

教程使用一个预定义的：

```python
PICK_AND_PLACE_CFG
```

配置：

```python
from isaaclab_assets import PICK_AND_PLACE_CFG
```

这是一个已经配置好的 Pick-and-Place 机器人。

它包含：

- 机器人几何模型；
- 关节；
- 物理属性；
- 表面夹爪。

这个机器人是一个非常简单的三轴机器人：

- 可以沿 X 轴移动；
- 可以沿 Y 轴移动；
- 可以沿 Z 轴上下移动。

其末端执行器安装了 Surface Gripper。

---

# 五、设计场景

定义：

```python
def design_scene():
    """Designs the scene."""
```

## 1. 创建地面

```python
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/defaultGroundPlane", cfg)
```

创建：

```text
/World/defaultGroundPlane
```

## 2. 创建灯光

```python
cfg = sim_utils.DomeLightCfg(
    intensity=3000.0,
    color=(0.75, 0.75, 0.75)
)

cfg.func("/World/Light", cfg)
```

这与之前几个教程相同。

---

# 六、创建两个机器人 Origin

教程创建：

```python
origins = [
    [2.75, 0.0, 0.0],
    [-2.75, 0.0, 0.0]
]
```

也就是说两个机器人分别放在：

```text
Origin1 = [ 2.75, 0, 0]
Origin2 = [-2.75, 0, 0]
```

然后：

```python
sim_utils.create_prim(
    "/World/Origin1",
    "Xform",
    translation=origins[0]
)

sim_utils.create_prim(
    "/World/Origin2",
    "Xform",
    translation=origins[1]
)
```

于是：

```text
/World/Origin1
/World/Origin2
```

下面分别放置一个机器人。

---

# 七、配置机器人 Articulation

首先：

```python
pick_and_place_robot_cfg = PICK_AND_PLACE_CFG.copy()
```

复制预定义的机器人配置。

然后：

```python
pick_and_place_robot_cfg.prim_path = "/World/Origin.*/Robot"
```

利用：

```text
.*
```

匹配：

```text
/World/Origin1/Robot
/World/Origin2/Robot
```

最后：

```python
pick_and_place_robot = Articulation(
    cfg=pick_and_place_robot_cfg
)
```

创建机器人。

---

# 八、配置 Surface Gripper

这是本教程的核心。

首先：

```python
surface_gripper_cfg = SurfaceGripperCfg()
```

创建表面夹爪配置对象。

然后必须告诉 Isaac Lab：

> **哪一个 Prim 是 Surface Gripper。**

于是：

```python
surface_gripper_cfg.prim_path = \
    "/World/Origin.*/Robot/picker_head/SurfaceGripper"
```

这里对应机器人 USD 中的：

```text
picker_head
    └── SurfaceGripper
```

因此 Isaac Lab 就可以找到实际的夹爪。

---

# 九、Surface Gripper 的主要参数

教程设置了四个关键参数。

## 1. 最大抓取距离

```python
surface_gripper_cfg.max_grip_distance = 0.1
```

单位：

```text
m
```

含义：

> 夹爪能够抓住物体的最大距离。

也就是目标物体必须在距离夹爪不超过 0.1 m 的范围内，夹爪才可能抓住它。

## 2. 横向剪切力限制

```python
surface_gripper_cfg.shear_force_limit = 500.0
```

单位：

```text
N
```

表示：

> 夹爪在垂直于夹爪轴线的方向能够施加的最大力。

## 3. 轴向力限制

```python
surface_gripper_cfg.coaxial_force_limit = 500.0
```

单位：

```text
N
```

表示：

> 沿着夹爪自身轴线方向能够施加的最大力。

## 4. 重试时间

```python
surface_gripper_cfg.retry_interval = 0.1
```

单位：

```text
seconds
```

含义：

> 夹爪保持抓取状态的时间。

也就是夹爪处于 grasping state 时，在指定时间内进行相关抓取过程。

---

# 十、创建 SurfaceGripper

有了配置以后：

```python
surface_gripper = SurfaceGripper(
    cfg=surface_gripper_cfg
)
```

这样就创建出了真正用于控制的：

```python
SurfaceGripper
```

对象。

这里和 `Articulation` 的关系可以理解成：

```text
PICK_AND_PLACE_CFG
        ↓
    Articulation
        ↓
      机器人

SurfaceGripperCfg
        ↓
   SurfaceGripper
        ↓
      夹爪
```

---

# 十一、保存场景实体

教程最后把两个对象保存到字典：

```python
scene_entities = {
    "pick_and_place_robot": pick_and_place_robot,
    "surface_gripper": surface_gripper
}
```

最终返回：

```python
return scene_entities, origins
```

所以场景中有：

```text
pick_and_place_robot
surface_gripper
```

两个主要实体。

---

# 十二、运行仿真循环

定义：

```python
def run_simulator(
    sim,
    entities,
    origins
):
```

首先从字典中取出：

```python
robot = entities["pick_and_place_robot"]
```

以及：

```python
surface_gripper = entities["surface_gripper"]
```

然后得到 physics timestep：

```python
sim_dt = sim.get_physics_dt()
```

并设置：

```python
count = 0
```

进入：

```python
while simulation_app.is_running():
```

循环。

---

# 十三、重置机器人

每隔：

```text
500
```

个 step：

```python
if count % 500 == 0:
```

执行一次重置。

首先获取默认根状态：

```python
root_state = robot.data.default_root_state.clone()
```

再加上环境 Origin：

```python
root_state[:, :3] += origins
```

然后：

```python
robot.write_root_pose_to_sim(
    root_state[:, :7]
)
```

以及：

```python
robot.write_root_velocity_to_sim(
    root_state[:, 7:]
)
```

将根状态重新写入模拟器。

---

# 十四、重置机器人关节状态

读取：

```python
joint_pos = robot.data.default_joint_pos.clone()
joint_vel = robot.data.default_joint_vel.clone()
```

然后：

```python
joint_pos += torch.rand_like(joint_pos) * 0.1
```

给关节位置增加少量随机噪声。

之后：

```python
robot.write_joint_state_to_sim(
    joint_pos,
    joint_vel
)
```

最后：

```python
robot.reset()
```

这样机器人状态就恢复了。

---

# 十五、重置 Surface Gripper

与机器人不同，夹爪只需要：

```python
surface_gripper.reset()
```

教程明确说明，这会：

> 重置夹爪的内部 buffers 和 caches，并确保夹爪处于打开状态。

然后打印：

```text
[INFO]: Resetting gripper state...
```

---

# 十六、给夹爪生成随机命令

现在是本教程最核心的控制部分。

首先：

```python
gripper_commands = (
    torch.rand(surface_gripper.num_instances) * 2.0 - 1.0
)
```

随机生成：

```text
-1 ~ 1
```

之间的数。

然后根据数值区间决定夹爪行为。

## 命令范围与行为

### `-1 < command < -0.3`

```text
Opening
```

即：

> 打开夹爪。

### `-0.3 < command < 0.3`

```text
Idle
```

即：

> 保持空闲状态。

### `0.3 < command < 1`

```text
Closing
```

即：

> 关闭夹爪。

可以把它理解成：

```text
command
   │
   ├── [-1, -0.3) → Open
   │
   ├── [-0.3, 0.3] → Idle
   │
   └── (0.3, 1] → Close
```

---

# 十七、打印命令

教程先打印随机命令：

```python
print(f"[INFO]: Gripper commands: {gripper_commands}")
```

然后将数值转换成人类可读的状态：

```python
mapped_commands = [
    "Opening"
    if command < -0.3
    else "Closing"
    if command > 0.3
    else "Idle"
    for command in gripper_commands
]
```

然后：

```python
print(
    f"[INFO]: Mapped commands: {mapped_commands}"
)
```

这样终端就能直接看到类似：

```text
Gripper commands: ...
Mapped commands: ['Opening', 'Idle']
```

---

# 十八、设置夹爪命令

真正向夹爪发送命令：

```python
surface_gripper.set_grippers_command(
    gripper_commands
)
```

这一步表示：

> 设置夹爪的目标命令。

需要注意：

这时候只是**设置 Command**，并没有完全写入 PhysX。

---

# 十九、将命令写入仿真

然后：

```python
surface_gripper.write_data_to_sim()
```

这一阶段会：

> 根据 Surface Gripper 的配置，把相应的数据转换并写入 PhysX buffer。

因此夹爪控制也是两个阶段：

```text
set_grippers_command()
        ↓
设置目标
        ↓
write_data_to_sim()
        ↓
写入 PhysX
```

---

# 二十、执行 simulation step

之后：

```python
sim.step()
```

推进一次仿真。

再：

```python
count += 1
```

更新计数器。

---

# 二十一、更新夹爪状态

现在我们需要知道：

> 夹爪到底执行得怎么样了？

调用：

```python
surface_gripper.update(sim_dt)
```

从模拟器读取最新状态，并更新内部 buffer。

然后：

```python
surface_gripper_state = surface_gripper.state
```

读取当前夹爪状态。

---

# 二十二、Surface Gripper 的状态值

`surface_gripper.state` 返回：

```text
[num_envs]
```

大小的 Tensor。

其中每个元素只有三个可能值：

```text
-1
 0
 1
```

对应：

| 数值 | 状态 |
|---:|---|
| `-1` | Open，打开 |
| `0` | Closing，正在关闭 |
| `1` | Closed，已关闭 |

也就是说：

```text
-1 → Open
 0 → Closing
 1 → Closed
```

官方说明 `state` 属性会在每次调用：

```python
surface_gripper.update()
```

时更新。

---

# 二十三、打印夹爪状态

代码：

```python
print(
    f"[INFO]: Gripper state: {surface_gripper_state}"
)
```

然后也会把数值映射成人类容易理解的文本：

```python
mapped_commands = [
    "Open"
    if state == -1
    else "Closing"
    if state == 0
    else "Closed"
    for state in surface_gripper_state.tolist()
]
```

最后：

```python
print(
    f"[INFO]: Mapped commands: {mapped_commands}"
)
```

因此终端中可以看到类似：

```text
Gripper state: tensor([-1., 0.])
Mapped commands: ['Open', 'Closing']
```

---

# 二十四、主函数

主函数创建仿真：

```python
sim_cfg = sim_utils.SimulationCfg(
    device=args_cli.device
)

sim = SimulationContext(sim_cfg)
```

注意，这里由于 Surface Gripper 只支持 CPU，所以最终应该通过：

```bash
--device cpu
```

启动。

---

# 二十五、设置摄像机

教程设置：

```python
sim.set_camera_view(
    [2.75, 7.5, 10.0],
    [2.75, 0.0, 0.0]
)
```

这样可以从合适的角度观察两个 Pick-and-Place 机器人。

---

# 二十六、创建场景

调用：

```python
scene_entities, scene_origins = design_scene()
```

然后：

```python
scene_origins = torch.tensor(
    scene_origins,
    device=sim.device
)
```

把 Origin 转换成 Tensor。

接下来：

```python
sim.reset()
```

启动仿真。

然后：

```python
print("[INFO]: Setup complete...")
```

最后：

```python
run_simulator(
    sim,
    scene_entities,
    scene_origins
)
```

进入循环。

---

# 二十七、运行结果

官方教程运行：

```bash
./isaaclab.sh -p scripts/tutorials/01_assets/run_surface_gripper.py --device cpu
```

应该打开一个场景，其中包括：

- 一个地面；
- 灯光；
- 两个 Pick-and-Place 机器人。

同时终端会持续打印：

```text
Gripper commands
Mapped commands
Gripper state
Mapped commands
```

因此你可以同时：

> 在 Isaac Sim 窗口观察机器人和夹爪，在终端观察夹爪的命令与实际状态。

---

# 二十八、本教程真正教会你的三个 API

这一篇最重要的其实就是这三个操作。

## 1. 设置命令

```python
surface_gripper.set_grippers_command(command)
```

表示：

> 我要夹爪执行什么动作。

## 2. 写入仿真

```python
surface_gripper.write_data_to_sim()
```

表示：

> 将命令真正写入物理仿真。

## 3. 更新状态

```python
surface_gripper.update(sim_dt)
```

表示：

> 从仿真器读取最新状态。

于是形成：

```text
Command
   ↓
set_grippers_command()
   ↓
write_data_to_sim()
   ↓
sim.step()
   ↓
update()
   ↓
state
```

这是整个教程最核心的数据流。

---

# 二十九、Surface Gripper 与 Articulation 的区别

上一篇教程中：

```text
Articulation
```

主要负责：

```text
机器人根状态
关节位置
关节速度
关节 effort
```

而这里增加：

```text
SurfaceGripper
```

因此完整结构变成：

```text
Pick-and-Place Robot
│
├── Articulation
│    ├── Root State
│    ├── Joint Position
│    ├── Joint Velocity
│    └── Joint Command
│
└── Surface Gripper
     ├── Command
     └── State
```

这就是一个带末端执行器的机器人系统。

---

# 三十、实际工程中应该怎么使用？

教程最后指出：

实际使用时，建议把：

```python
assets.SurfaceGripper
```

实例注册到：

```python
isaaclab.InteractiveScene
```

中。

这样 `InteractiveScene` 可以统一处理：

```python
surface_gripper.write_data_to_sim()
```

以及：

```python
surface_gripper.update()
```

的调用。

这意味着后面的教程中，你不需要每次手动调用：

```python
surface_gripper.write_data_to_sim()
surface_gripper.update()
```

而是可以让：

```text
InteractiveScene
```

统一管理。

---

# 三十一、官方还提供了 Pick-and-Place Demo

除了这个基础教程，官方还提供了：

```text
scripts/demos/pick_and_place.py
```

用于：

> 生成多个 Pick-and-Place 机器人并执行抓取与放置任务。

运行方式：

```bash
./isaaclab.sh -p scripts/demos/pick_and_place.py --viz kit
```

官方说明该 demo 就是把这里学习到的机器人和 Surface Gripper 机制进一步用于 Pick-and-Place。

---

# 三十二、本教程的整体数据流

把这一篇压缩成一张图，就是：

```text
                Pick-and-Place Robot
                         │
             ┌───────────┴───────────┐
             │                       │
        Articulation          SurfaceGripper
             │                       │
        Joint Control            Gripper Command
             │                       │
             │             set_grippers_command()
             │                       │
             │              write_data_to_sim()
             │                       │
             └───────────┬───────────┘
                         ↓
                     sim.step()
                         ↓
                    Physics Engine
                         ↓
                  update(sim_dt)
                         ↓
             ┌───────────┴───────────┐
             │                       │
       Robot State             Gripper State
```

---

# 三十三、对强化学习最重要的理解

如果把这篇教程放进强化学习框架，它就变成：

```text
                RL Policy
                    │
                    ↓
              Action Tensor
              /           \
             /             \
      Robot Action       Gripper Action
           │                   │
           ↓                   ↓
   Articulation          SurfaceGripper
           │                   │
           └─────────┬─────────┘
                     ↓
                  Physics
                     ↓
                 Observation
                     ↓
                  RL Policy
```

例如以后机械臂抓取任务可以设计成：

```text
Observation:
    机械臂关节位置
    机械臂关节速度
    末端位置
    物体位置
    Gripper state

Action:
    joint1 effort
    joint2 effort
    ...
    gripper command
```

其中 Surface Gripper 的命令在这个教程里被设计成：

```text
[-1, -0.3) → Opening
[-0.3, 0.3] → Idle
(0.3, 1] → Closing
```

而夹爪反馈状态则是：

```text
-1 → Open
 0 → Closing
 1 → Closed
```

这正好可以作为以后强化学习环境中的 **Action / Observation 接口**。

---

# 总结

本教程的核心是：

1. 使用 `SurfaceGripperCfg` 描述夹爪；
2. 使用 `SurfaceGripper` 管理夹爪；
3. 使用 `set_grippers_command()` 设置夹爪目标；
4. 使用 `write_data_to_sim()` 将命令写入物理仿真；
5. 使用 `sim.step()` 推进物理；
6. 使用 `update()` 和 `state` 读取夹爪状态；
7. 通过 `Articulation + SurfaceGripper` 构成完整的抓取机器人系统。

特别注意：本教程中的 Surface Gripper 需要 CPU backend，因此运行时使用 `--device cpu`。
