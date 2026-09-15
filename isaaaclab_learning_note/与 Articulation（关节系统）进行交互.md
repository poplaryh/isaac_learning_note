# 与 Articulation（关节系统）进行交互

本教程介绍如何在仿真环境中与一个**关节式机器人（articulated robot）**进行交互。

它是上一篇：

> **Interacting with a rigid object（与刚体对象进行交互）**

教程的延续。

上一篇教程主要学习了如何与一个刚体对象进行交互，而本教程将在此基础上进一步学习：

- 如何设置 articulation 的根状态（root state）；
- 如何设置关节状态（joint state）；
- 如何向 articulation 发送控制命令。



---

# 一、代码

本教程对应的脚本为：

```text
scripts/tutorials/01_assets/run_articulation.py
```

运行方式：

```bash
./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py
```

该脚本用于演示：

> 如何生成一个 cart-pole，并与这个 articulation 进行交互。



---

# 二、启动 Isaac Sim

首先导入 Python 的参数解析模块：

```python
import argparse
```

然后导入：

```python
from isaaclab.app import AppLauncher
```

创建参数解析器：

```python
parser = argparse.ArgumentParser(
    description="Tutorial on spawning and interacting with an articulation."
)
```

将 AppLauncher 的命令行参数加入解析器：

```python
AppLauncher.add_app_launcher_args(parser)
```

解析命令行参数：

```python
args_cli = parser.parse_args()
```

然后启动 Omniverse 应用：

```python
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

这一部分和前面的 AppLauncher 教程以及刚体教程相同。



---

# 三、导入相关模块

完成 Isaac Sim 启动之后，再导入：

```python
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
```

这里最重要的是：

```python
from isaaclab.assets import Articulation
```

与上一篇教程中的：

```python
RigidObject
```

相比，这里使用的是：

```python
Articulation
```

因为本教程操作的不是单个独立刚体，而是一个：

> **由多个刚体通过关节连接起来的系统。**

例如本教程的 cart-pole，就是一个 articulation。



---

# 四、预定义的 Cartpole 配置

教程直接使用 Isaac Lab 已经定义好的 cart-pole 配置：

```python
from isaaclab_assets import CARTPOLE_CFG
```

这里的：

```python
CARTPOLE_CFG
```

是一个预先定义好的：

```python
ArticulationCfg
```

对象。

这个配置中已经包含了 cart-pole 所需要的相关信息。

---

# 五、设计场景

定义：

```python
def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
```

这个函数负责设计整个仿真场景。

---

## 1. 创建地面

和上一篇教程一样：

```python
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/defaultGroundPlane", cfg)
```

创建一个地面：

```text
/World/defaultGroundPlane
```

---

## 2. 创建灯光

```python
cfg = sim_utils.DomeLightCfg(
    intensity=3000.0,
    color=(0.75, 0.75, 0.75)
)

cfg.func(
    "/World/Light",
    cfg
)
```

创建一个 Dome Light。

---

# 六、创建两个机器人 Origin

接下来创建两个不同的位置：

```python
origins = [
    [0.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0]
]
```

也就是说：

```text
Origin1 = [0.0, 0.0, 0.0]
Origin2 = [-1.0, 0.0, 0.0]
```

然后分别创建两个 Xform：

```python
sim_utils.create_prim(
    "/World/Origin1",
    "Xform",
    translation=origins[0]
)
```

以及：

```python
sim_utils.create_prim(
    "/World/Origin2",
    "Xform",
    translation=origins[1]
)
```

所以最终场景中有：

```text
/World/Origin1
/World/Origin2
```

两个独立的环境原点。

每个 Origin 下都会放一个 cart-pole。



---

# 七、生成 Articulation

这是本教程最关键的地方。

首先：

```python
cartpole_cfg = CARTPOLE_CFG.copy()
```

复制一份预定义的 Cartpole 配置。

然后修改它的 Prim 路径：

```python
cartpole_cfg.prim_path = "/World/Origin.*/Robot"
```

这里：

```text
.*
```

表示匹配：

```text
/World/Origin1/Robot
/World/Origin2/Robot
```

因此这一份配置可以对应两个 cart-pole。

最后：

```python
cartpole = Articulation(
    cfg=cartpole_cfg
)
```

创建 `Articulation` 对象。

这里可以理解为：

```text
CARTPOLE_CFG
      ↓
ArticulationCfg
      ↓
Articulation
      ↓
真正由 Isaac Lab 管理的 Cartpole
```



---

# 八、返回场景实体

把生成的 cart-pole 放入字典：

```python
scene_entities = {
    "cartpole": cartpole
}
```

然后返回：

```python
return scene_entities, origins
```

所以最终得到：

```text
scene_entities
    └── cartpole

origins
    ├── Origin1
    └── Origin2
```

---

# 九、运行仿真循环

定义：

```python
def run_simulator(
    sim: sim_utils.SimulationContext,
    entities: dict[str, Articulation],
    origins: torch.Tensor
):
```

首先取得 cart-pole：

```python
robot = entities["cartpole"]
```

教程特别说明，这里只是为了代码可读性，把它从字典中取出来。

实际上也可以直接：

```python
entities["cartpole"]
```

访问。

而后面的教程会使用：

```python
InteractiveScene
```

来更加方便地管理场景实体。

---

# 十、获得仿真时间步长

```python
sim_dt = sim.get_physics_dt()
```

获得 physics timestep。

然后：

```python
count = 0
```

初始化计数器。

进入循环：

```python
while simulation_app.is_running():
```

只要 Isaac Sim 还在运行，就不断推进仿真。

---

# 十一、重置 Cartpole

教程每：

```python
500
```

个仿真步执行一次重置：

```python
if count % 500 == 0:
```

---

## 1. 重置计数器

```python
count = 0
```

---

# 十二、设置根状态

和上一篇刚体教程一样，articulation 也具有：

> **root state（根状态）**

它对应 articulation 树中的：

> **根刚体（root body）**

首先获取默认根状态：

```python
root_state = robot.data.default_root_state.clone()
```

---

## 为什么需要加 `origins`？

教程中特别强调：

> 根状态是按照 simulation world frame（仿真世界坐标系）写入的。

因此：

```python
root_state[:, :3] += origins
```

需要将每个 robot 的 origin 添加到根状态的位置中。

否则，两个机器人都会被生成到：

```text
(0, 0, 0)
```

即世界坐标原点。

所以这里的作用是：

```text
机器人默认位置
       +
对应 Origin
       ↓
机器人在世界中的实际位置
```



---

# 十三、写入根部姿态

```python
robot.write_root_pose_to_sim(
    root_state[:, :7]
)
```

将：

```python
root_state[:, :7]
```

写入模拟器。

这里表示：

> 根部位姿（root pose）

包括位置和姿态。

---

# 十四、写入根部速度

```python
robot.write_root_velocity_to_sim(
    root_state[:, 7:]
)
```

将根部速度写回仿真。

因此：

```text
root_state
├── [:7]
│    └── root pose
│
└── [7:]
     └── root velocity
```

---

# 十五、设置关节状态

这是 Articulation 和简单 RigidObject 的一个重要区别。

Articulation 除了 root state 之外，还有：

> **joint states（关节状态）**

这些状态对应：

- 关节位置；
- 关节速度。

教程首先获取默认关节状态：

```python
joint_pos, joint_vel = (
    robot.data.default_joint_pos.clone(),
    robot.data.default_joint_vel.clone()
)
```

然后给关节位置添加少量随机噪声：

```python
joint_pos += torch.rand_like(joint_pos) * 0.1
```

也就是说：

> 每次重置时，关节位置都会在默认值附近随机偏移一点。



---

# 十六、将关节状态写入仿真

```python
robot.write_joint_state_to_sim(
    joint_pos,
    joint_vel
)
```

将：

```text
joint position
joint velocity
```

写入模拟器。

---

# 十七、清除内部缓存

最后：

```python
robot.reset()
```

重置 Articulation 自己内部的：

- buffers；
- caches。

然后输出：

```python
print("[INFO]: Resetting robot state...")
```

因此完整的重置过程就是：

```text
获取默认 root state
       ↓
加上 Origin
       ↓
写入 root pose
       ↓
写入 root velocity
       ↓
获取默认 joint state
       ↓
添加随机噪声
       ↓
写入 joint state
       ↓
robot.reset()
```



---

# 十八、给机器人施加随机动作

每一个仿真 step，都生成随机关节力矩：

```python
efforts = torch.randn_like(
    robot.data.joint_pos
) * 5.0
```

这里：

```python
torch.randn_like(...)
```

生成与关节位置 shape 相同的随机张量。

乘：

```text
5.0
```

表示随机力矩的幅度大约在这个尺度上。

---

# 十九、给 Articulation 设置关节力

调用：

```python
robot.set_joint_effort_target(
    efforts
)
```

把随机生成的关节 effort 作为目标发送给 articulation。

这里的“effort”可以理解成：

> 关节执行器施加的力或力矩命令。

对于这个 cart-pole，教程通过随机 effort，让小车和杆子随机运动。



---

# 二十、将控制数据写入仿真

设置 target 后：

```python
robot.write_data_to_sim()
```

将数据写入 PhysX buffer。

这一过程不是简单地“把数值直接塞给物理引擎”。

Isaac Lab 会根据 articulation 的配置进行：

> **actuation conversion（执行器转换）**

然后把转换后的值写入 PhysX。



---

# 二十一、执行仿真 Step

之后：

```python
sim.step()
```

推进一个 physics step。

然后：

```python
count += 1
```

增加计数器。

---

# 二十二、更新 Articulation 状态

执行：

```python
robot.update(sim_dt)
```

这一步非常重要。

每一个 `Articulation` 对象内部都有一个：

```python
ArticulationData
```

对象。

这个对象存储 articulation 当前的各种状态。

执行：

```python
robot.update(sim_dt)
```

之后：

> 内部 buffers 中的数据会更新为当前仿真状态。



---

# 二十三、Articulation 的状态结构

与 `RigidObject` 相比：

```text
RigidObject
    ↓
root state
```

而：

```text
Articulation
    ↓
root state
+
joint state
```

也就是说：

```text
Articulation
├── root state
│   ├── position
│   ├── orientation
│   ├── linear velocity
│   └── angular velocity
│
└── joint state
    ├── joint position
    └── joint velocity
```

这是理解机器人资产的一个非常重要的结构。

---

# 二十四、设置仿真环境

在 `main()` 中：

```python
sim_cfg = sim_utils.SimulationCfg(
    device=args_cli.device
)
```

创建 Simulation Configuration。

然后：

```python
sim = SimulationContext(sim_cfg)
```

创建仿真上下文。

---

# 二十五、设置摄像机

教程设置：

```python
sim.set_camera_view(
    [2.5, 0.0, 4.0],
    [0.0, 0.0, 2.0]
)
```

第一组坐标：

```text
[2.5, 0.0, 4.0]
```

表示摄像机位置。

第二组：

```text
[0.0, 0.0, 2.0]
```

表示摄像机观察目标。

---

# 二十六、创建场景

调用：

```python
scene_entities, scene_origins = design_scene()
```

创建：

```text
地面
灯光
Origin1
Origin2
Cartpole1
Cartpole2
```

然后：

```python
scene_origins = torch.tensor(
    scene_origins,
    device=sim.device
)
```

将 Origin 转换成 PyTorch Tensor，并放到与仿真相同的设备上。

例如：

```text
cuda:0
```

---

# 二十七、启动模拟

执行：

```python
sim.reset()
```

重置模拟器。

然后：

```python
print("[INFO]: Setup complete...")
```

输出：

```text
[INFO]: Setup complete...
```

最后进入仿真循环：

```python
run_simulator(
    sim,
    scene_entities,
    scene_origins
)
```

---

# 二十八、程序结束

最后：

```python
if __name__ == "__main__":
    main()
    simulation_app.close()
```

也就是：

```text
启动 Isaac Sim
      ↓
创建场景
      ↓
创建两个 Cartpole
      ↓
重置机器人
      ↓
随机施加关节力
      ↓
推进仿真
      ↓
更新 Articulation 状态
      ↓
循环运行
      ↓
关闭 Isaac Sim
```

---

# 二十九、教程核心：Reset、Action、Update 三个阶段

这个教程最值得掌握的是整个 Articulation 的交互循环。

可以把它总结成：

```text
                ┌──────────────┐
                │   Reset      │
                └──────┬───────┘
                       ↓
            设置 root state
                       ↓
            设置 joint state
                       ↓
                robot.reset()
                       ↓
                ┌──────────────┐
                │    Action    │
                └──────┬───────┘
                       ↓
          set_joint_effort_target()
                       ↓
            write_data_to_sim()
                       ↓
                ┌──────────────┐
                │  Simulation  │
                └──────┬───────┘
                       ↓
                   sim.step()
                       ↓
                ┌──────────────┐
                │    Update    │
                └──────┬───────┘
                       ↓
                  robot.update()
                       ↓
              获取新的状态数据
                       │
                       └──────→ 下一步
```

---

# 三十、如何向 Articulation 发送命令？

教程特别指出，对 articulation 施加命令需要两个步骤。

## 第一步：设置 Joint Target

首先设置你希望达到的：

- 关节位置；
- 关节速度；
- 关节 effort。

例如：

```python
robot.set_joint_effort_target(efforts)
```

---

## 第二步：写入 Simulation

然后：

```python
robot.write_data_to_sim()
```

这一步根据 articulation 的配置执行必要的 actuator conversion，并把最终的数据写入 PhysX buffer。

因此不能简单理解成：

```text
set_joint_effort_target()
```

就已经执行了物理控制。

完整流程是：

```text
set_joint_effort_target()
        ↓
设置目标
        ↓
write_data_to_sim()
        ↓
转换 actuator command
        ↓
写入 PhysX
        ↓
sim.step()
        ↓
机器人运动
```



---

# 三十一、为什么这个教程使用 Effort Control？

本教程通过：

```python
robot.set_joint_effort_target(efforts)
```

控制 articulation。

要使这种 effort 控制方式正常工作，cart-pole 预定义配置中的：

```text
stiffness
damping
```

已经提前设置为：

```text
0
```

也就是说，教程采用的是比较直接的：

> **Joint Effort Control（关节力/力矩控制）**

而不是利用位置/速度 PD 控制器产生 effort。

这一点非常重要。

如果 actuator 中配置了明显的：

```text
stiffness
damping
```

那么控制效果会与单纯 effort 控制有所不同。



---

# 三十二、Articulation 的状态在哪里？

每一个 `Articulation` 都包含：

```python
robot.data
```

其中核心数据对象是：

```python
ArticulationData
```

它保存 articulation 的状态。

例如：

```python
robot.data.default_root_state
```

表示默认根状态。

以及：

```python
robot.data.default_joint_pos
```

表示默认关节位置。

```python
robot.data.default_joint_vel
```

表示默认关节速度。

因此可以理解成：

```text
robot
  │
  └── data
       │
       ├── default_root_state
       ├── default_joint_pos
       ├── default_joint_vel
       ├── 当前 root state
       ├── 当前 joint position
       └── 当前 joint velocity
```

---

# 三十三、运行教程

执行：

```bash
./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py
```

程序会打开一个场景。

场景中包括：

- 一个地面；
- 灯光；
- 两个 cart-pole。

两个 cart-pole 会受到随机的关节力作用，因此会随机运动。

教程运行过程中，可以直接观察两个 Cartpole 的运动情况。

停止仿真可以：

- 直接关闭 Isaac Sim 窗口；
- 或在终端按：

```text
Ctrl+C
```

