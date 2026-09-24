# 注册环境（Registering an Environment）

> 本文根据 Isaac Lab 官方文档当前版本的 **Registering an Environment** 教程翻译整理。
>
> 原文：https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html
>
> **说明：** 翻译过程中保留了 Python 模块名、类名、函数名、环境 ID、命令行参数和命令等代码标识符；代码本身不做功能性修改，仅对代码中的英文注释/说明进行中文化。没有对官方页面未提供的数据、输出结果或示例内容进行补充或猜测。

## 目录

- [注册环境](#注册环境)
- [代码](#代码)
- [代码解析](#代码解析)
  - [使用 Gym 注册表](#使用-gym-注册表)
  - [Manager-Based 环境](#manager-based-环境)
  - [Direct 环境](#direct-环境)
  - [创建环境](#创建环境)
- [代码执行](#代码执行)

# 注册环境

在上一教程中，我们学习了如何创建一个自定义的 Cartpole 环境。我们通过导入环境类及其配置类，手动创建了环境实例。

## 上一教程中的环境创建方式

```python
# 创建环境配置
env_cfg = CartpoleEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.sim.device = args_cli.device

# 设置强化学习环境
env = ManagerBasedRLEnv(cfg=env_cfg)
```

虽然这种方式很直接，但当环境数量较多、需要维护一整套环境时，这种方式并不容易扩展。

本教程将介绍如何使用 `gymnasium.register()` 方法，将环境注册到 `gymnasium` 注册表（registry）中。这样，我们就可以通过 `gymnasium.make()` 函数创建环境。

## 本教程中的环境创建方式

```python
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """使用 Isaac Lab 环境运行随机动作智能体。"""
    # 创建环境配置
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)
```

> 注：官方示例代码中保留了 `# PLACEHOLDER: Extension template (do not remove this comment)` 占位注释；该注释属于代码模板的一部分，因此不应删除。

# 代码

本教程对应 `scripts/environments` 目录中的 `random_agent.py` 脚本。

## `random_agent.py`

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用随机动作智能体运行环境的脚本。"""

"""首先启动 Isaac Sim Simulator。"""

import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="禁用 Fabric，并使用 USD I/O 操作。",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="要模拟的环境数量。",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="任务名称。",
)

# 添加 AppLauncher 的命令行参数
AppLauncher.add_app_launcher_args(parser)

# 解析参数
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下为其余代码。"""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """使用 Isaac Lab 环境运行随机动作智能体。"""
    # 创建环境配置
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 打印信息（这是一个向量化环境）
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # 重置环境
    env.reset()

    # 仿真环境
    while simulation_app.is_running():
        # 在推理模式下执行所有操作
        with torch.inference_mode():
            # 在 -1 到 1 的范围内采样动作
            actions = 2 * torch.rand(
                env.action_space.shape,
                device=env.unwrapped.device,
            ) - 1

            # 应用动作
            env.step(actions)

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行 main 函数
    main()

    # 关闭仿真应用
    simulation_app.close()
```

# 代码解析

`envs.ManagerBasedRLEnv` 类继承自 `gymnasium.Env` 类，以遵循标准接口。

但是，与传统的 Gym 环境不同，`envs.ManagerBasedRLEnv` 实现的是**向量化环境（vectorized environment）**。这意味着多个环境实例会在同一个进程中同时运行，并且所有数据都会以批量（batched）的形式返回。

类似地，对于 Direct 工作流，`envs.DirectRLEnv` 类也继承自 `gymnasium.Env` 类。

对于 `envs.DirectMARLEnv`，虽然它并不继承自 Gymnasium，但仍然可以使用相同的方式进行注册和创建。

## 使用 Gym 注册表

要注册一个环境，可以使用 `gymnasium.register()` 方法。

该方法需要提供环境名称、环境类的入口点（entry point），以及环境配置类的入口点。

> **注意**
>
> `gymnasium` 注册表是一个**全局注册表**。因此，需要确保环境名称具有唯一性；否则，在注册环境时，注册表会抛出错误。

## Manager-Based 环境

对于 Manager-Based 环境，下面展示了 `isaaclab_tasks.manager_based.classic.cartpole` 子包中 Cartpole 环境的注册方式：

```python
import gymnasium as gym

from . import agents

##
# 注册 Gym 环境。
##
gym.register(
    id="Isaac-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleRGBCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleDepthCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-RGB-ResNet18-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleResNet18CameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-RGB-TheiaTiny-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleTheiaTinyCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)
```

### `id`：环境名称

`id` 参数表示环境名称。

Isaac Lab 采用了一种命名约定：所有环境通常都使用 `Isaac-` 作为前缀，以便更容易在注册表中搜索环境。

环境名称通常依次包含：

1. `Isaac-` 前缀；
2. 任务名称；
3. 机器人名称；
4. 可选的其他变体信息；
5. 版本号 `v<N>`。

例如，对于**平地上的 ANYmal C 腿式运动任务**，环境名称为：

```text
Isaac-Velocity-Flat-Anymal-C-v0
```

版本号 `v<N>` 通常用于表示同一环境的不同变体。

否则，环境名称可能会变得过长并且难以阅读。

### `entry_point`：环境类入口点

`entry_point` 参数指定环境类的入口点。

入口点是如下形式的字符串：

```text
<module>:<class>
```

以 Cartpole 环境为例：

```text
isaaclab.envs:ManagerBasedRLEnv
```

创建环境实例时，系统会使用这个入口点导入对应的环境类。

### `env_cfg_entry_point`：环境配置入口点

`env_cfg_entry_point` 参数指定环境的默认配置。

默认配置通过 `isaaclab_tasks.utils.parse_env_cfg()` 函数加载，然后传递给 `gymnasium.make()`，以创建环境实例。

配置入口点可以是：

- YAML 文件；
- Python 配置类。

### 其他 RL 配置入口点

Manager-Based Cartpole 示例中还注册了多个强化学习框架对应的配置入口点，例如：

```text
rl_games_cfg_entry_point
rsl_rl_cfg_entry_point
rsl_rl_with_symmetry_cfg_entry_point
skrl_cfg_entry_point
sb3_cfg_entry_point
```

这些参数用于关联相应的 RL 配置。

## Direct 环境

对于 Direct 环境，注册方式与 Manager-Based 环境类似。

区别在于：

- Manager-Based 环境将环境的 `entry_point` 注册为 `ManagerBasedRLEnv`；
- Direct 环境则将 `entry_point` 注册为具体环境的实现类；
- 为了将 Direct 环境与 Manager-Based 环境区分开，环境名称通常添加 `-Direct` 后缀。

例如，下面展示了 `isaaclab_tasks.direct.cartpole` 子包中 Cartpole 环境的注册方式：

```python
import gymnasium as gym

from . import agents

##
# 注册 Gym 环境。
##
gym.register(
    id="Isaac-Cartpole-Direct-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-RGB-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env:CartpoleRGBCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Depth-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_camera_env:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env:CartpoleDepthCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)
```

## 创建环境

为了让 `gym` 注册表获得 `isaaclab_tasks` 扩展所提供的所有环境，需要在脚本开始处导入该模块：

```python
import isaaclab_tasks  # noqa: F401
```

执行这个导入操作时，会执行 `__init__.py` 文件。该文件会遍历各个子包，并注册它们对应的环境。

在本教程中，任务名称从命令行参数中读取。

任务名称同时用于：

1. 解析默认环境配置；
2. 创建环境实例。

此外，其他已经解析的命令行参数，例如：

- 环境数量；
- 仿真设备；
- 是否进行渲染；

可以用于覆盖默认配置。

创建环境配置：

```python
# 创建环境配置
env_cfg = parse_env_cfg(
    args_cli.task,
    device=args_cli.device,
    num_envs=args_cli.num_envs,
    use_fabric=not args_cli.disable_fabric,
)

# 创建环境
env = gym.make(args_cli.task, cfg=env_cfg)
```

创建环境之后，其余执行流程遵循标准的环境重置（reset）和步进（step）流程。

# 代码执行

完成代码介绍之后，可以运行脚本查看结果：

```bash
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 32
```

运行后应该会打开一个 Stage，其内容与“创建 Manager-Based RL 环境”教程中的内容类似。

要停止仿真，可以：

- 直接关闭窗口；
- 或者在终端中按 `Ctrl+C`。

教程原页面还提供了一张 `random_agent.py` 的运行结果图片；本文不复制该图片。

## 使用 CPU 运行

此外，还可以通过显式设置 `--device` 参数，将仿真设备从 GPU 切换到 CPU：

```bash
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 32 --device cpu
```

使用：

```text
--device cpu
```

后，仿真将在 CPU 上运行。

这对于调试仿真非常有用。不过，与 GPU 相比，仿真运行速度会明显更慢。

---

## 关键流程总结

本教程的核心流程可以概括为：

```text
定义环境及其配置
        │
        ▼
使用 gymnasium.register() 注册环境
        │
        ├── id
        ├── entry_point
        └── env_cfg_entry_point
        │
        ▼
导入 isaaclab_tasks
        │
        ▼
环境注册信息进入 Gymnasium registry
        │
        ▼
parse_env_cfg() 解析环境配置
        │
        ▼
gym.make(args_cli.task, cfg=env_cfg)
        │
        ▼
创建向量化 Isaac Lab 环境
        │
        ▼
env.reset()
        │
        ▼
env.step(actions)
```

其中最关键的变化，是从上一教程中的：

```python
env = ManagerBasedRLEnv(cfg=env_cfg)
```

转变为通过 Gymnasium 注册表创建：

```python
env = gym.make(args_cli.task, cfg=env_cfg)
```

这种方式使 Isaac Lab 可以通过统一的环境 ID 管理大量环境，并让不同的 RL 工作流使用标准化的环境创建接口。

## 官方版本信息

- 文档标题：Registering an Environment
- 当前页面：Isaac Lab `main`
- 官方页面标注的最后更新时间：2026-09-20
- 原始教程：https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html
