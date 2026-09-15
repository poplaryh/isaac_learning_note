# 向 Isaac Lab 添加一个新的机器人
## Adding a New Robot to Isaac Lab

模拟和训练一个新的机器人是一个**多步骤的过程**，首先需要将机器人导入 Isaac Sim。

机器人导入 Isaac Sim 后，还需要针对仿真进行调试和参数调整。关于机器人如何导入 Isaac Sim，官方 Isaac Sim 文档有更加详细的说明。

完成机器人导入并调试好之后，无论选择哪一种工作流（workflow）或训练框架（training framework），我们还需要定义一些必要的接口，用于：

- 将机器人复制（clone）到多个环境中；
- 驱动机器人的关节；
- 正确地重置机器人。

本教程将介绍如何向 Isaac Lab 添加一个新的机器人。

其中最关键的一步，是创建一个：

```python
AssetBaseCfg
```

它定义了机器人 USD articulation 与 Isaac Lab 中可用学习算法之间的接口。



---

# 代码

本教程对应于：

```text
scripts/tutorials/01_assets/add_new_robot.py
```

中的 `add_new_robot` 脚本。

---

## 1. 启动 Isaac Sim

首先导入：

```python
import argparse

from isaaclab.app import AppLauncher
```

创建命令行参数解析器：

```python
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
```

添加环境数量参数：

```python
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to spawn."
)
```

这里：

```text
--num_envs
```

用于指定要创建多少个仿真环境。

默认：

```text
1
```

然后加入 AppLauncher 的命令行参数：

```python
AppLauncher.add_app_launcher_args(parser)
```

解析参数：

```python
args_cli = parser.parse_args()
```

最后启动 Omniverse：

```python
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

这部分流程与前面的 AppLauncher 教程一致。

---

# 2. 导入 Isaac Lab 所需要的模块

```python
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
```

其中比较重要的是：

```python
ArticulationCfg
```

它用于定义机器人这种具有多个关节驱动的 articulation。

---

# 3. 配置 Jetbot

教程首先添加一个：

**Jetbot**

Jetbot 是一个非常简单的：

- 两轮差速移动机器人；
- 顶部安装有摄像头。

Isaac Sim 中已经有很多 Jetbot 的演示和教程，所以这个机器人已经可以直接使用。

因为机器人本质上是一个带关节驱动器的 articulation，所以这里使用：

```python
ArticulationCfg
```

来描述 Jetbot。

代码：

```python
JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            damping=None,
            stiffness=None
        )
    },
)
```



---

# 4. 最小机器人配置

这是 Isaac Lab 中一个机器人的**最小配置**。

只有两个必需参数：

```text
spawn
actuators
```

即：

```python
ArticulationCfg(
    spawn=...,
    actuators=...,
)
```

---

## `spawn`

```python
spawn=sim_utils.UsdFileCfg(...)
```

`spawn` 接收一个：

```text
SpawnerCfg
```

它用于指定仿真中的机器人 USD 资产。

Isaac Lab 的：

```python
isaaclab.sim
```

提供了：

```python
UsdFileCfg
```

这个配置类。

它接收 USD 文件路径，并生成所需要的 `SpawnerCfg`。

本教程中的 Jetbot USD 文件位于 Isaac Assets 中：

```text
Robots/Jetbot/jetbot.usd
```



---

# 5. `actuators`

`actuators` 是一个：

```python
dictionary
```

字典中保存的是各种：

```text
ActuatorCfg
```

其作用是定义：

> 我们希望智能体（agent）控制机器人哪些部分。

机器人关节从当前状态向目标状态变化，可以采用很多不同的方法。

Isaac Lab 提供了一系列 actuator 类，用于：

- 匹配常见的执行器模型；
- 或实现自定义执行器模型。

Jetbot 的轮子比较简单，因此这里使用：

```python
ImplicitActuatorCfg
```

并采用默认设置。



---

# 6. 使用正则表达式指定关节

这里：

```python
joint_names_expr=[".*"]
```

使用的是一个简单的正则表达式：

```text
.*
```

意思是：

> 匹配所有关节。

Jetbot 的关节比较少，而且我们希望直接使用 USD 资产中已经定义好的默认设置，因此使用这个表达式即可。

当然，也可以使用其他正则表达式，把不同关节分成不同组，然后为不同的关节组设置不同的 actuator 参数。

---

## 注意

对于：

```python
ImplicitActuatorCfg
```

`stiffness` 和 `damping` 都必须指定。

但是，如果指定：

```python
None
```

表示：

> 使用 USD 资产中定义好的默认值。

例如：

```python
damping=None
stiffness=None
```

就是让 Jetbot 使用 USD 文件中的默认刚度和阻尼。



---

# 7. 更复杂的机器人配置

虽然 Jetbot 的配置已经是最小形式，但实际上我们还可以设置更多参数。

教程第二个机器人使用：

**Dofbot**

Dofbot 是一个具有多个关节的机械臂，因此需要更加详细的配置。

代码如下：

```python
DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
        },
        pos=(0.25, -0.25, 0.0),
    ),

    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),

        "joint3_act": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),

        "joint4_act": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)
```



---

# 8. Dofbot 与 Jetbot 的主要区别

Dofbot 比 Jetbot 多配置了两个非常重要的部分：

```text
physics properties
init_state
```

也就是：

**物理属性**

以及：

**初始状态**

教程指出，Dofbot 最明显的两个区别就是：

1. 增加了物理属性相关配置；
2. 增加了机器人的初始状态 `init_state`。



---

# 9. `rigid_props`

在 `UsdFileCfg` 中：

```python
rigid_props=sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
)
```

`rigid_props` 接收：

```python
RigidBodyPropertiesCfg
```

用于定义机器人作为一个“物理对象”时的相关属性。

这里：

```python
disable_gravity=False
```

意味着：

> 不关闭重力。

因此机器人会受到重力作用。

---

```python
max_depenetration_velocity=5.0
```

设置最大去穿透速度。

---

# 10. `articulation_props`

另外：

```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=0
)
```

这个配置主要控制：

> 机器人 articulation 在仿真过程中使用的求解器相关属性。

例如：

```python
enabled_self_collisions=True
```

表示启用机器人自身部件之间的碰撞。

同时设置：

```python
solver_position_iteration_count=8
solver_velocity_iteration_count=0
```

控制求解器的迭代次数。

Isaac Lab 还提供了很多其他物理属性，可以通过：

```python
isaaclab.sim.schemas
```

中的配置进行指定。



---

# 11. `init_state`：定义机器人初始状态

`ArticulationCfg` 可以选择性地包含：

```python
init_state
```

它用于定义 articulation 的：

> 初始状态。

这个初始状态是用户自己定义的状态，会在以下情况下使用：

- 机器人刚刚生成时；
- Isaac Lab 重置机器人时。

例如：

```python
init_state=ArticulationCfg.InitialStateCfg(
    joint_pos={
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": 0.0,
    },
    pos=(0.25, -0.25, 0.0),
)
```

这里：

```python
joint_pos
```

用于定义关节初始位置。

---

## 一个非常重要的细节

`joint_pos` 中的字典 key：

```text
joint1
joint2
joint3
joint4
```

必须使用：

> **USD 中定义的关节名称**

而不是 actuator 的名称。

也就是说：

```python
"joint3"
```

是 USD joint name。

而：

```python
"joint3_act"
```

是 actuator name。

两者不是一回事。



---

# 12. 初始位置 `pos` 的坐标系

另一个值得注意的地方是：

```python
pos=(0.25, -0.25, 0.0)
```

这里的坐标不是相对于整个世界（world）的坐标，而是：

> **相对于当前环境（environment）的坐标系。**

因此：

```python
pos=(0.25, -0.25, 0.0)
```

表示机器人相对于当前环境原点偏移：

```text
x = 0.25
y = -0.25
z = 0
```

而不是相对于整个世界坐标原点进行偏移。



---

# 13. 给 Dofbot 配置 actuator

Dofbot 的 actuator 被划分为三个组。

第一组：

```python
"front_joints"
```

控制：

```python
joint[1-2]
```

即：

```text
joint1
joint2
```

使用：

```python
effort_limit_sim=100.0
velocity_limit_sim=100.0
stiffness=10000.0
damping=100.0
```

---

第二组：

```python
"joint3_act"
```

只控制：

```python
joint3
```

---

第三组：

```python
"joint4_act"
```

只控制：

```python
joint4
```

这样就可以针对不同的关节组设置不同的 actuator 参数。



---

# 14. 将机器人加入 Interactive Scene

有了前面的机器人配置之后，现在就可以把它们加入场景。

首先定义：

```python
class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
```

---

## 地面

```python
ground = AssetBaseCfg(
    prim_path="/World/defaultGroundPlane",
    spawn=sim_utils.GroundPlaneCfg()
)
```

这里：

```python
AssetBaseCfg
```

表示一个场景资产配置。

地面的 Prim 路径为：

```text
/World/defaultGroundPlane
```

---

## 灯光

```python
dome_light = AssetBaseCfg(
    prim_path="/World/Light",
    spawn=sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75)
    )
)
```

创建一个 Dome Light。

---

## Jetbot

```python
Jetbot = JETBOT_CONFIG.replace(
    prim_path="{ENV_REGEX_NS}/Jetbot"
)
```

将前面定义好的：

```python
JETBOT_CONFIG
```

复制为一个新的配置，并把：

```text
prim_path
```

替换成：

```text
{ENV_REGEX_NS}/Jetbot
```

---

## Dofbot

同样：

```python
Dofbot = DOFBOT_CONFIG.replace(
    prim_path="{ENV_REGEX_NS}/Dofbot"
)
```

这样，Jetbot 和 Dofbot 都可以随着环境一起复制到不同的环境实例中。



---

# 15. 运行仿真

定义：

```python
def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene
):
```

先获取物理时间步长：

```python
sim_dt = sim.get_physics_dt()
```

然后初始化：

```python
sim_time = 0.0
count = 0
```

进入仿真循环：

```python
while simulation_app.is_running():
```

---

# 16. 每 500 步重置机器人

代码：

```python
if count % 500 == 0:
```

表示：

> 每运行 500 个仿真步，就执行一次重置。

首先：

```python
count = 0
```

重新计算计数器。

然后读取 Jetbot 的默认根状态：

```python
root_jetbot_state = (
    scene["Jetbot"].data.default_root_state.clone()
)
```

再加上环境原点：

```python
root_jetbot_state[:, :3] += scene.env_origins
```

这样，每个环境中的机器人都能得到对应的世界位置。

Dofbot 也执行同样的操作。



---

# 17. 将根状态写回仿真

对于 Jetbot：

```python
scene["Jetbot"].write_root_pose_to_sim(
    root_jetbot_state[:, :7]
)

scene["Jetbot"].write_root_velocity_to_sim(
    root_jetbot_state[:, 7:]
)
```

这里：

```python
root_jetbot_state[:, :7]
```

表示根部位姿。

而：

```python
root_jetbot_state[:, 7:]
```

表示根部速度。

Dofbot 也进行同样操作。



---

# 18. 重置关节状态

首先读取 Jetbot 默认关节位置和速度：

```python
joint_pos, joint_vel = (
    scene["Jetbot"].data.default_joint_pos.clone(),
    scene["Jetbot"].data.default_joint_vel.clone(),
)
```

然后写回仿真：

```python
scene["Jetbot"].write_joint_state_to_sim(
    joint_pos,
    joint_vel
)
```

Dofbot 同理：

```python
joint_pos, joint_vel = (
    scene["Dofbot"].data.default_joint_pos.clone(),
    scene["Dofbot"].data.default_joint_vel.clone(),
)

scene["Dofbot"].write_joint_state_to_sim(
    joint_pos,
    joint_vel
)
```

最后清理内部缓存：

```python
scene.reset()
```

并输出：

```text
[INFO]: Resetting Jetbot and Dofbot state...
```

---

# 19. 控制 Jetbot 移动

接下来让 Jetbot 运动。

代码：

```python
if count % 100 < 75:
    action = torch.Tensor([[10.0, 10.0]])
else:
    action = torch.Tensor([[5.0, -5.0]])
```

前 75 个步：

```text
[10.0, 10.0]
```

两个轮子的速度相同，因此：

> Jetbot 直线运动。

后 25 个步：

```text
[5.0, -5.0]
```

左右轮速度不同，因此：

> Jetbot 转弯。

然后：

```python
scene["Jetbot"].set_joint_velocity_target(action)
```

把轮子目标速度写入 Jetbot。



---

# 20. 控制 Dofbot 做摆动动作

Dofbot 使用正弦函数产生关节运动：

```python
wave_action = scene["Dofbot"].data.default_joint_pos

wave_action[:, 0:4] = (
    0.25
    * np.sin(2 * np.pi * 0.5 * sim_time)
)
```

也就是让前四个关节进行周期性摆动。

然后：

```python
scene["Dofbot"].set_joint_position_target(
    wave_action
)
```

把目标位置发送给 Dofbot。



---

# 21. 写入仿真并执行一步

在每一个仿真循环中：

```python
scene.write_data_to_sim()
```

首先把场景数据写入模拟器。

然后：

```python
sim.step()
```

执行一步物理仿真。

接着：

```python
sim_time += sim_dt
```

更新时间。

再：

```python
count += 1
```

增加仿真步计数。

最后：

```python
scene.update(sim_dt)
```

更新场景中的各个实体状态。

完整流程可以理解为：

```text
设置机器人目标
       ↓
scene.write_data_to_sim()
       ↓
sim.step()
       ↓
scene.update()
       ↓
读取新的机器人状态
```



---

# 22. 主函数

最后在 `main()` 中初始化仿真。

```python
sim_cfg = sim_utils.SimulationCfg(
    device=args_cli.device
)

sim = sim_utils.SimulationContext(sim_cfg)
```

然后设置摄像机：

```python
sim.set_camera_view(
    [3.5, 0.0, 3.2],
    [0.0, 0.0, 0.5]
)
```

创建场景配置：

```python
scene_cfg = NewRobotsSceneCfg(
    args_cli.num_envs,
    env_spacing=2.0
)
```

创建 Interactive Scene：

```python
scene = InteractiveScene(scene_cfg)
```

重置模拟器：

```python
sim.reset()
```

输出：

```text
[INFO]: Setup complete...
```

然后运行：

```python
run_simulator(sim, scene)
```



---

# 23. 程序入口

最后：

```python
if __name__ == "__main__":
    main()
    simulation_app.close()
```

也就是说：

1. 启动程序；
2. 创建场景；
3. 创建机器人；
4. 运行模拟；
5. 关闭 Isaac Sim。

---

# 教程核心思想

这个教程真正想让你掌握的，并不仅仅是“把 Jetbot 和 Dofbot 加进场景”。

核心是理解：

> **Isaac Lab 如何通过各种 Configuration 类，把一个 USD 机器人资产转换成一个可以被 Isaac Lab 管理、复制、驱动和重置的机器人资产。**

最关键的结构是：

```text
USD Robot
   │
   ↓
ArticulationCfg
   │
   ├── spawn
   │      └── UsdFileCfg
   │
   ├── init_state
   │      ├── joint_pos
   │      └── pos
   │
   └── actuators
          └── ImplicitActuatorCfg
```

然后：

```text
ArticulationCfg
       ↓
InteractiveSceneCfg
       ↓
InteractiveScene
       ↓
多个环境中的机器人
```

---

# 教程最后的注意事项

运行这个教程时，你可能会看到：

```text
warning that not all actuators are configured
```

这是**正常现象**。

原因是：

> 本教程没有处理 Dofbot 的夹爪（gripper）。

因此并不是所有 actuator 都进行了配置。

