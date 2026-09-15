# 深入理解 AppLauncher

## 教程简介

在本教程中，我们将深入学习 `app.AppLauncher` 类，了解如何通过**命令行参数（CLI arguments）**和**环境变量（environment variables，简称 envars）**来配置模拟器。

本教程主要介绍如何使用 `AppLauncher`：

- 启用流式传输（livestreaming）
- 配置它所封装的 `isaacsim.simulation_app.SimulationApp` 实例
- 同时保留用户自己提供的其他配置选项

---

## 什么是 AppLauncher？

`AppLauncher` 是 `SimulationApp` 的一个**封装器（wrapper）**，用于简化 `SimulationApp` 的配置工作。

`SimulationApp` 内部包含大量扩展（extensions），不同功能依赖不同的扩展，而且这些扩展之间可能存在：

- 加载顺序依赖
- 相互依赖关系

此外，某些启动选项，例如：

```text
headless
```

必须在 `SimulationApp` 创建时就确定。

而且，这些启动参数与某些扩展之间存在隐含关系，例如：

```text
headless
        ↓
livestreaming extensions
```

`AppLauncher` 提供了一个统一的接口，可以处理这些扩展和启动选项，从而让代码能够在不同的使用场景下更加方便地运行。

为实现这一点，Isaac Lab 提供了：

- 命令行参数（CLI flags）
- 环境变量（environment variables）

这些参数可以和用户自定义的命令行参数合并，同时把属于 `SimulationApp` 的参数继续向下传递给 `SimulationApp`。



---

# 代码

本教程对应于：

```text
scripts/tutorials/00_sim/launch_app.py
```

脚本。

运行方式：

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py
```

---

## 启动 Isaac Sim

首先导入 Python 标准库：

```python
import argparse
```

然后导入 Isaac Lab 的：

```python
from isaaclab.app import AppLauncher
```

---

## 创建参数解析器

创建一个标准的 `argparse.ArgumentParser`：

```python
parser = argparse.ArgumentParser(
    description="Tutorial on running IsaacSim via the AppLauncher."
)
```

然后添加一个脚本自定义参数：

```python
parser.add_argument(
    "--size",
    type=float,
    default=1.0,
    help="Side-length of cuboid"
)
```

这个参数用于设置立方体边长。

默认值为：

```text
1.0
```

---

## 添加 SimulationApp 参数

然后加入 `SimulationApp` 使用的：

```text
--width
--height
```

参数。

```python
parser.add_argument(
    "--width",
    type=int,
    default=1280,
    help="Width of the viewport and generated images. Defaults to 1280"
)

parser.add_argument(
    "--height",
    type=int,
    default=720,
    help="Height of the viewport and generated images. Defaults to 720"
)
```

也就是说，默认视口大小是：

```text
宽度：1280
高度：720
```

---

## 添加 AppLauncher 参数

接下来是本教程最关键的一步：

```python
AppLauncher.add_app_launcher_args(parser)
```

这个函数会向参数解析器中加入 `AppLauncher` 自身提供的命令行参数。

然后解析全部参数：

```python
args_cli = parser.parse_args()
```

最后启动 Omniverse 应用：

```python
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

因此，最基本的启动流程就是：

```python
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Tutorial on running IsaacSim via the AppLauncher."
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```



---

# 场景设计

在这个例子中，启动模拟器后会创建一个简单场景。

首先创建地面：

```python
cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func(
    "/World/defaultGroundPlane",
    cfg_ground
)
```

然后创建一个远距离光源：

```python
cfg_light_distant = sim_utils.DistantLightCfg(
    intensity=3000.0,
    color=(0.75, 0.75, 0.75),
)

cfg_light_distant.func(
    "/World/lightDistant",
    cfg_light_distant,
    translation=(1, 0, 10)
)
```

之后创建一个立方体：

```python
cfg_cuboid = sim_utils.CuboidCfg(
    size=[args_cli.size] * 3,
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 1.0, 1.0)
    ),
)
```

最后将立方体生成到：

```text
/World/Object
```

并根据其尺寸调整 Z 轴位置：

```python
cfg_cuboid.func(
    "/World/Object",
    cfg_cuboid,
    translation=(0.0, 0.0, args_cli.size / 2)
)
```



---

# 主函数

首先创建模拟配置：

```python
sim_cfg = sim_utils.SimulationCfg(
    dt=0.01,
    device=args_cli.device
)
```

这里：

```text
dt = 0.01
```

表示仿真的时间步长。

然后创建：

```python
sim = sim_utils.SimulationContext(sim_cfg)
```

---

## 设置主摄像机

使用：

```python
sim.set_camera_view(
    [2.0, 0.0, 2.5],
    [-0.5, 0.0, 0.5]
)
```

设置摄像机位置和朝向。

---

## 创建场景

调用：

```python
design_scene()
```

完成地面、灯光和立方体的生成。

---

## 重置并运行模拟

```python
sim.reset()
```

然后进入仿真循环：

```python
while simulation_app.is_running():
    sim.step()
```

也就是说，只要 Isaac Sim 应用仍然运行，就不断执行模拟步。

最后关闭应用：

```python
simulation_app.close()
```



---

# 代码解释

## 1. 向 ArgumentParser 添加参数

`AppLauncher` 的设计目标之一，就是能够兼容用户为自己脚本定义的命令行参数，同时提供一个统一、可移植的启动接口。

在本教程中，首先创建标准的：

```python
argparse.ArgumentParser
```

然后添加：

```text
--size
--height
--width
```

这几个参数。

其中：

```text
--size
```

是脚本自己使用的参数；

而：

```text
--height
--width
```

会被传递给 `SimulationApp`。



---

## 2. `--size` 与 AppLauncher 的关系

`--size` 并不是 `AppLauncher` 自己使用的参数。

但是，它可以与 `AppLauncher` 的参数接口**无缝组合**。

实现方式是：

```python
AppLauncher.add_app_launcher_args(parser)
```

该函数会：

> 返回一个已经附加了 `AppLauncher` 参数的 `ArgumentParser`。

然后通过 Python 标准方法：

```python
parser.parse_args()
```

将这些参数解析成：

```python
argparse.Namespace
```

再把这个 Namespace 直接传递给：

```python
AppLauncher
```

进行实例化。



---

# 3. 理解 `--help` 的输出

运行：

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py --help
```

可以看到：

```text
usage: launch_app.py [-h]
                     [--size SIZE]
                     [--width WIDTH]
                     [--height HEIGHT]
                     [--headless]
                     [--livestream {0,1,2}]
                     [--enable_cameras]
                     [--verbose]
                     [--experience EXPERIENCE]
```

其中，脚本自己定义的参数有：

```text
--size
--width
--height
```

而 `AppLauncher` 增加的参数包括：

```text
--headless
--livestream
--enable_cameras
--verbose
--experience
```

---

## `--headless`

```text
--headless
```

含义是：

> 强制关闭显示。

也就是让 Isaac Sim 不显示图形界面。

---

## `--livestream`

```text
--livestream {0,1,2}
```

含义是：

> 强制启用流式传输。

三个值的具体含义与环境变量：

```text
LIVESTREAM
```

的映射方式相对应。

---

## `--enable_cameras`

```text
--enable_cameras
```

表示：

> 在没有 GUI 的情况下启用摄像机。

这个选项主要用于相机传感器和离屏渲染。

---

## `--verbose`

```text
--verbose
```

表示：

> 启用更详细的 `SimulationApp` 终端日志。

---

## `--experience`

```text
--experience EXPERIENCE
```

用于指定启动 `SimulationApp` 时加载的 experience 文件。

如果传入空字符串：

```text
""
```

则会根据 headless 标志决定使用哪个 experience 文件。

如果给出相对路径，则会按照以下顺序查找：

1. Isaac Sim 的 `apps` 文件夹
2. Isaac Lab 的 `apps` 文件夹



---

# 4. 哪些参数会传给 SimulationApp？

在教程示例中，运行 `--help` 时会看到：

```text
[INFO][AppLauncher]:
The argument 'width' will be used to configure the SimulationApp.

[INFO][AppLauncher]:
The argument 'height' will be used to configure the SimulationApp.
```

这意味着：

```text
--width
--height
```

会被识别为 `SimulationApp` 的参数。

判断标准是：

> 如果某个参数的名称和类型与 `SimulationApp` 支持的参数匹配，那么 `AppLauncher` 就会将其传递给 `SimulationApp`。



---

# 5. 使用环境变量

除了命令行参数之外，`AppLauncher` 还支持环境变量。

例如：

```text
--livestream
--headless
```

都有对应的环境变量。

具体设置方式在 `isaaclab.app` API 文档中定义。

命令行参数与环境变量具有等价的配置效果。

也就是说：

通过 CLI：

```bash
--headless
```

与在 shell 中设置对应环境变量，是等价的。

---

## 为什么要使用环境变量？

环境变量主要是为了方便实现：

**持久化的会话配置（session-persistent configurations）**

例如可以在：

```text
${HOME}/.bashrc
```

中设置环境变量。

这样以后启动新的 shell 会话时，这些设置可以继续保留。

---

## CLI 参数优先级更高

如果环境变量与命令行参数发生冲突：

> **命令行参数优先于环境变量。**

例如环境变量设置：

```text
LIVESTREAM=0
```

同时命令行指定：

```bash
--livestream 2
```

那么最终会采用：

```text
--livestream 2
```

而不是环境变量中的：

```text
LIVESTREAM=0
```



---

# 6. `--enable_cameras` 的特殊性

这些 `AppLauncher` 参数原则上可以用于任何通过 `AppLauncher` 启动模拟器的脚本。

但有一个例外：

```text
--enable_cameras
```

这个设置会让渲染管线使用：

**离屏渲染器（offscreen renderer）**

但是，它只兼容：

```python
isaaclab.sim.SimulationContext
```

而不兼容 Isaac Sim 自己的：

```python
isaacsim.core.api.simulation_context.SimulationContext
```

类。



---

# 7. 执行示例

现在运行：

```bash
LIVESTREAM=2 ./isaaclab.sh -p \
    scripts/tutorials/00_sim/launch_app.py \
    --size 0.5
```

这会创建一个：

```text
0.5 m³
```

的立方体。

但是：

> 此时不会出现 GUI。

其效果与使用：

```text
--headless
```

类似。

原因是：

```text
LIVESTREAM
```

环境变量已经隐含地让程序采用 headless 模式。



---

# 8. 如何查看流式画面？

如果需要可视化，可以通过 Isaac 的：

**WebRTC Livestreaming**

来查看仿真画面。

教程中特别说明：

> 在容器环境中，目前唯一受支持的可视化方式是流式传输。

也就是说，容器中不能依赖普通 GUI 窗口，而应该通过 streaming 方式查看仿真。

停止程序可以在启动终端中按：

```text
Ctrl+C
```

---

# 9. 当命令行参数与环境变量冲突时

执行：

```bash
LIVESTREAM=0 ./isaaclab.sh -p \
    scripts/tutorials/00_sim/launch_app.py \
    --size 0.5 \
    --livestream 2
```

虽然环境变量写的是：

```text
LIVESTREAM=0
```

但是命令行指定：

```text
--livestream 2
```

因此最终仍然使用：

```text
--livestream 2
```

也就是说：

```text
CLI 参数
   ↓
优先级更高
   ↓
覆盖环境变量
```



---

# 10. 向 SimulationApp 传递分辨率参数

最后执行：

```bash
LIVESTREAM=2 ./isaaclab.sh -p \
    scripts/tutorials/00_sim/launch_app.py \
    --size 0.5 \
    --width 1920 \
    --height 1080
```

这一次仍然会使用 livestream 模式。

但是视口渲染分辨率变成：

```text
1920 × 1080
```

这在以下场景中很有用：

- 获取高分辨率视频
- 录制训练过程
- 需要更清晰的仿真画面

反过来，如果更关注模拟速度，则可以使用较低的分辨率，因为更低的渲染分辨率通常可以降低渲染负担，从而提高仿真性能。



---

# 本教程核心总结

这个教程最重要的并不是那个立方体，而是理解：

```text
用户脚本
   │
   ├── 自定义参数
   │      └── --size
   │
   └── AppLauncher 参数
          ├── --headless
          ├── --livestream
          ├── --enable_cameras
          ├── --verbose
          └── --experience
                    │
                    ↓
              AppLauncher
                    │
                    ↓
              SimulationApp
```

也就是说，`AppLauncher` 的核心作用可以理解为：

> **在用户自己的 Python 脚本与 Isaac Sim 的 `SimulationApp` 之间建立一个统一的启动和配置接口。**

它负责处理：

```text
CLI 参数
环境变量
SimulationApp 参数
Isaac Sim Extensions
启动模式
```

然后统一启动 Isaac Sim。