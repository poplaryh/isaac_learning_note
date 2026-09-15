# 使用 Interactive Scene

> Isaac Lab 官方教程：**Using the Interactive Scene**
>
> 原文：https://isaac-sim.github.io/IsaacLab/main/source/tutorials/02_scene/create_scene.html
>
> 本文按照官方教程结构翻译，代码中的类名、函数名、参数名和 API 名称保留英文。

## 教程简介

到目前为止，前面的教程都是手动向仿真中生成各种资产（assets），并创建对象实例来与它们进行交互。但是，随着场景复杂度不断增加，手动执行这些操作会越来越繁琐。

本教程介绍 `scene.InteractiveScene` 类。它提供了一个方便的接口，用于在仿真中生成 Prim，并对它们进行统一管理。

从较高层次来看，Interactive Scene 是一组场景实体（scene entities）。每个实体可以是：

- 非交互式 Prim，例如地面、光源；
- 交互式 Prim，例如 articulation、rigid object；
- 传感器，例如 camera、lidar。

Interactive Scene 为这些实体的生成和仿真过程中的管理提供了统一、方便的接口。

### 相比手动方式的优势

与手动创建场景相比，Interactive Scene 主要有以下优势：

1. 用户不再需要分别生成每个资产，因为这些操作会由 Interactive Scene 隐式完成。
2. 可以方便地把场景中的 Prim 克隆到多个环境中。
3. 可以把所有场景实体收集到一个对象中，因此更容易统一管理。

在本教程中，我们使用上一教程的 Cartpole 示例，并将原来的 `design_scene` 函数替换为 `scene.InteractiveScene` 对象。对于这么简单的示例来说，使用 Interactive Scene 似乎有些“杀鸡用牛刀”，但是随着后续场景中增加越来越多的资产和传感器，它会变得非常有用。

## 代码

本教程对应：

```text
scripts/tutorials/02_scene/create_scene.py
```

### `create_scene.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import CARTPOLE_CFG  # isort:skip


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["cartpole"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
```

## 代码解析

虽然代码与前一个教程类似，但有几个关键变化。

### 1. Scene configuration：场景配置

场景由一组实体组成，每个实体都有自己的配置。这些配置定义在一个继承自 `scene.InteractiveSceneCfg` 的配置类中。然后把这个配置类实例传递给 `scene.InteractiveScene` 的构造函数，用它创建场景。

对于 Cartpole 示例，我们使用与上一教程相同的场景，但现在不再手动生成场景，而是把场景实体列在 `CartpoleSceneCfg` 配置类中：

```python
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75)
        )
    )

    # articulation
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
```

配置类中的变量名，会成为从 `scene.InteractiveScene` 对象中访问对应实体时使用的 key。例如：

```python
scene["cartpole"]
```

可以访问 Cartpole。具体的访问方法将在后面介绍。

首先来看每个场景实体是如何配置的。

与前面刚体对象和 articulation 的配置方式类似，场景实体也是通过配置类进行描述的。但地面、光源和 Cartpole 之间存在一个重要区别：

- 地面和光源属于**非交互式 Prim**；
- Cartpole 属于**交互式 Prim**。

这个区别会体现在它们使用的配置类上。

地面和光源使用：

```python
assets.AssetBaseCfg
```

而 Cartpole 使用：

```python
assets.ArticulationCfg
```

任何不是交互式 Prim 的实体，也就是既不是 asset 又不是 sensor 的对象，都不会由场景在每一个仿真 step 中进行处理。

---

### 2. 不同 Prim 的路径

本教程中不同 Prim 的路径分别为：

```text
Ground plane: /World/defaultGroundPlane
Light source: /World/Light
Cartpole:     {ENV_REGEX_NS}/Robot
```

正如前面的教程所介绍，Omniverse 会在 USD Stage 中创建一棵 Prim 图。

Prim path 用于指定 Prim 在这棵图中的位置。

地面和光源使用**绝对路径（absolute paths）**，而 Cartpole 使用**相对路径（relative path）**。

相对路径中的：

```text
{ENV_REGEX_NS}
```

是一个特殊变量。

在创建场景时，它会被替换为具体的环境名称。

任何 Prim path 中包含：

```text
{ENV_REGEX_NS}
```

的实体，都会针对每一个环境进行克隆。

这个变量最终会被场景对象替换成：

```text
/World/envs/env_{i}
```

其中 `i` 是环境索引。

例如，当环境索引为 `0` 时：

```text
{ENV_REGEX_NS}/Robot
```

会变成：

```text
/World/envs/env_0/Robot
```

而环境索引为 `1` 时则变成：

```text
/World/envs/env_1/Robot
```

等等。

这就是 Isaac Lab 可以高效创建多个并行环境的关键机制之一。

---

### 3. Scene instantiation：创建 InteractiveScene

以前的教程中，我们通过调用：

```python
design_scene()
```

来手动创建场景。

现在则改为创建：

```python
scene.InteractiveScene
```

对象，并把配置对象传给它的构造函数。

创建 `CartpoleSceneCfg` 时，通过：

```python
num_envs
```

参数指定想创建多少个环境副本。

然后这些配置会被用于把场景克隆到每一个环境中。

对应代码：

```python
# Design scene
scene_cfg = CartpoleSceneCfg(
    num_envs=args_cli.num_envs,
    env_spacing=2.0
)

scene = InteractiveScene(scene_cfg)
```

这意味着：

```text
CartpoleSceneCfg
      ↓
指定 num_envs
      ↓
InteractiveScene
      ↓
克隆场景
      ↓
env_0, env_1, ..., env_N
```

---

### 4. Accessing scene elements：访问场景元素

与前面教程中通过字典访问实体类似，现在可以直接通过 `InteractiveScene` 对象的 `[]` 运算符来访问场景元素。

这个运算符接收一个字符串 key，并返回对应的实体。

这个 key 是在配置类中通过变量名指定的。

例如配置中：

```python
cartpole: ArticulationCfg = CARTPOLE_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot"
)
```

这里变量名是：

```text
cartpole
```

因此可以这样访问：

```python
robot = scene["cartpole"]
```

官方代码：

```python
# Extract scene entities
# note: we only do this here for readability.
robot = scene["cartpole"]
```

这里把实体提取到 `robot` 变量中只是为了提高后续代码的可读性；并不是必须这样做。

---

### 5. Running the simulation loop：运行仿真循环

后面的仿真代码与前面的 `assets.Articulation` 教程非常相似，但有一些调用方式发生了变化。

主要变化如下：

| 原来的调用 | 现在的调用 |
|---|---|
| `assets.Articulation.reset()` | `scene.InteractiveScene.reset()` |
| `assets.Articulation.write_data_to_sim()` | `scene.InteractiveScene.write_data_to_sim()` |
| `assets.Articulation.update()` | `scene.InteractiveScene.update()` |

也就是说，不再需要对每一个实体单独调用这些方法，而是可以直接让 `InteractiveScene` 统一处理。

在内部，`InteractiveScene` 的这些方法会进一步调用场景中各个实体对应的方法。

因此可以理解为：

```text
InteractiveScene
      │
      ├── Articulation
      ├── RigidObject
      ├── Sensor
      └── 其他实体
```

当执行：

```python
scene.write_data_to_sim()
```

InteractiveScene 会把需要写入仿真的数据统一分发给相应实体。

同样：

```python
scene.update(sim_dt)
```

会统一更新场景中各个实体的内部状态。

---

## 仿真循环中的具体流程

教程中的 reset 过程为：

```python
if count % 500 == 0:
    count = 0

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos, joint_vel = (
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )

    joint_pos += torch.rand_like(joint_pos) * 0.1
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    scene.reset()
```

其中：

```python
root_state[:, :3] += scene.env_origins
```

非常重要，因为 root state 是以 simulation world frame 写入的。如果不加上环境原点，不同环境中的机器人就会被写到整个仿真世界的 `(0, 0, 0)` 位置。

之后设置关节位置的随机噪声，并使用：

```python
scene.reset()
```

清除场景中实体的内部 buffer。

---

### 随机动作

示例使用随机 effort 控制 Cartpole：

```python
efforts = torch.randn_like(robot.data.joint_pos) * 5.0
robot.set_joint_effort_target(efforts)
```

然后不再调用机器人自己的 `write_data_to_sim()`，而是调用：

```python
scene.write_data_to_sim()
```

接着：

```python
sim.step()
```

推进物理仿真。

最后使用：

```python
scene.update(sim_dt)
```

更新场景中实体的状态 buffer。

因此整个循环可以概括为：

```text
读取/修改实体状态
      ↓
设置 Action
      ↓
scene.write_data_to_sim()
      ↓
sim.step()
      ↓
scene.update(sim_dt)
      ↓
获得最新状态
```

---

# 代码执行

现在运行脚本，让场景中模拟 32 个 Cartpole。

通过向脚本传递：

```text
--num_envs
```

参数来指定环境数量。

运行命令：

```bash
./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32
```

这会打开一个包含 **32 个 Cartpole** 的场景，并且这些 Cartpole 会随机摆动。

可以使用：

- 鼠标旋转摄像机；
- 方向键在场景中移动。

教程中的示例结果就是一个同时显示多个 Cartpole 的场景。citeturn689867view0

---

# 教程总结

本教程介绍了如何使用：

```python
scene.InteractiveScene
```

创建一个包含多个资产的场景。

同时也介绍了如何使用：

```text
num_envs
```

参数，把场景克隆到多个环境中。

与手动创建场景相比，Interactive Scene 可以让场景管理更加统一和方便，尤其适用于包含大量资产、传感器和并行环境的复杂场景。

官方还指出，在 `isaaclab_tasks` 扩展中的各种任务里，有大量 `scene.InteractiveSceneCfg` 的实际使用示例。对于更复杂的场景，可以进一步查看这些任务的源代码，了解 Interactive Scene 在实际项目中的使用方式。

---

# 本教程最重要的知识点

## 1. InteractiveSceneCfg

用于描述整个场景的配置：

```python
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ...
```

## 2. InteractiveScene

根据配置创建和管理场景：

```python
scene = InteractiveScene(scene_cfg)
```

## 3. 场景实体通过变量名访问

例如：

```python
cartpole: ArticulationCfg = ...
```

那么访问方式是：

```python
scene["cartpole"]
```

## 4. `{ENV_REGEX_NS}` 用于多环境克隆

例如：

```python
prim_path="{ENV_REGEX_NS}/Robot"
```

会在每个环境下生成对应机器人：

```text
/World/envs/env_0/Robot
/World/envs/env_1/Robot
/World/envs/env_2/Robot
...
```

## 5. InteractiveScene 统一管理实体

以前：

```python
robot.reset()
robot.write_data_to_sim()
robot.update(sim_dt)
```

现在：

```python
scene.reset()
scene.write_data_to_sim()
scene.update(sim_dt)
```

这正是本教程从前面的 Articulation 教程进一步抽象到 Scene 管理层的核心。citeturn155013view0turn689867view0

---

# 与你当前 StudyProject02 的关系

这篇教程和你前面学习的 Cartpole RL 环境已经非常接近了。

你现在的理解可以逐步形成下面的层次：

```text
USD Prim
   ↓
Asset / Articulation
   ↓
ArticulationCfg
   ↓
InteractiveSceneCfg
   ↓
InteractiveScene
   ↓
Manager-Based Environment
   ↓
Observation / Action / Reward
   ↓
PPO / RSL-RL
```

其中 `InteractiveScene` 解决的是：

> **“一个环境中到底有哪些对象，以及如何统一创建、克隆、访问和更新这些对象。”**

而下一层的 Manager-Based Environment 才会进一步解决：

> **“哪些数据作为 Observation、Action、Reward、Termination，以及这些数据怎样组织成强化学习环境。”**

这正是从场景构建过渡到你正在学习的强化学习任务设计的关键一步。

