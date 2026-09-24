# 从 FSM 控制程序迁移到 Isaac Lab 强化学习环境

> 本文档整理自一次关于「如何把现有 Franka 抓取搬运程序转换为 Isaac Lab 强化学习训练任务」的技术对话。
> 原始程序：`Pasted code(8).py`（约 2900 行，基于 Isaac Lab + cuMotion + ROS 2 的确定性 FSM 控制程序）。

---

## 0. 原始提问

> 上面是我的代码（`Pasted code(8).py`，Python），我现在想要用 Isaac Lab 来进行强化学习训练。我应该怎么把上面的代码进行转换，形成 Isaac Lab 可以强化学习训练的部分内容？Isaac Lab 要进行自定义场景、reward、action 以及任务的强化学习，应该怎么一步一步实现呢？

---

## 一、先理解：你现在的代码和 RL 到底差在哪里

你当前主循环实际上是：

```
读取当前状态
    ↓
判断 phase
    ↓
根据 phase 给出固定目标
    ↓
cuMotion
    ↓
得到 joint target
    ↓
robot.set_joint_position_target()
    ↓
physics step
    ↓
判断 EE 是否到目标
    ↓
进入下一个 phase
```

例如你的 `PREGRASP`：

```python
target = pregrasp[env_idx]

desired = controller.forward(
    estimated,
    setpoint,
    phase_time[env_idx],
)

apply_cumotion_joint_target(
    robot,
    desired,
    env_idx=env_idx,
)
```

然后通过：

```python
error = get_env_ee_error(
    env_idx,
    target,
)
```

判断是否到达目标。你的代码目前用的是 **1 mm 的位置误差阈值**，并且至少运行 **20 个 simulation steps** 才允许 phase 完成。

这是一套非常典型的传统机器人控制 FSM。

RL 则要把它改成：

```
                  ┌───────────────┐
                  │   Policy π    │
                  └───────┬───────┘
                          │
                       action
                          ↓
                  ┌───────────────┐
                  │ Action Manager│
                  └───────┬───────┘
                          ↓
                     Robot control
                          ↓
                  ┌───────────────┐
                  │ Isaac Physics │
                  └───────┬───────┘
                          ↓
             ┌────────────┴────────────┐
             ↓                         ↓
        observation                  reward
             │                         │
             └──────────→ Policy ←─────┘
```

所以真正需要增加的是：

- **Observation**（观测）
- **Action**（动作）
- **Reward**（奖励）
- **Termination**（终止）
- **Reset**（重置）
- **Episode**（回合）
- **RL algorithm / PPO 等**（强化学习算法）

---

## 二、你的代码应该拆成什么结构

我建议最终目录不要继续使用现在这种：

```
main()
 ├── SceneCfg
 ├── cuMotion
 ├── FSM
 ├── ROS
 ├── simulation loop
 ├── print
 └── final verification
```

而是改成标准 Isaac Lab RL task 结构：

```
my_franka_rl/
│
├── source/
│   └── my_franka/
│       │
│       ├── tasks/
│       │   └── franka_pick_place/
│       │       │
│       │       ├── __init__.py
│       │       │
│       │       ├── franka_pick_place_env.py
│       │       │
│       │       ├── franka_pick_place_env_cfg.py
│       │       │
│       │       ├── mdp/
│       │       │   ├── observations.py
│       │       │   ├── actions.py
│       │       │   ├── rewards.py
│       │       │   ├── terminations.py
│       │       │   └── events.py
│       │       │
│       │       └── agents/
│       │           └── rsl_rl_ppo_cfg.py
│       │
│       └── __init__.py
│
├── scripts/
│   └── reinforcement_learning/
│       └── train.py
│
└── config/
```

这是你后面最值得做的事情。

---

## 三、第一步：你的 SceneCfg 基本可以直接保留

这是目前最好的部分。你现在的：

```python
@configclass
class SceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(...)

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        ...
    )

    katao: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/katao",
        ...
    )

    stick_part: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stick_part",
        ...
    )

    joint_wrench = JointWrenchSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        ...
    )
```

这一层基本就是 Isaac Lab RL Scene 应该有的东西。尤其你已经使用 `{ENV_REGEX_NS}`，这意味着它天然适合：

```
env_0
env_1
env_2
...
env_N
```

并行训练。你的当前代码已经明确设计了每个 environment 独立的 `robot`、`katao`、`stick_part` 和 `JointWrenchSensor`。所以 **SceneCfg 不需要为了 RL 大改**。

---

## 四、第二步：把你的 TaskPhase 从"控制 FSM"变成"任务状态"

你现在：

```python
class TaskPhase(IntEnum):

    PREGRASP = 0
    GRASP_APPROACH = 1
    CLOSE_GRIPPER = 2
    LIFT = 3
    TRANSPORT = 4
    PLACE = 5
    OPEN_GRIPPER = 6
    FINAL_LIFT = 7
    JOINTMOVE = 8
    DONE = 9
    FAILED = 10
```

这个东西不要直接删除。但是它不能继续承担「机器人下一步应该怎么运动」，而应该变成「任务当前处于哪个阶段」。也就是说：

```python
# 以前
if phase == PREGRASP:
    controller.forward(...)

# RL 后
if phase == PREGRASP:
    reward = ...
```

甚至**第一版 RL 我建议暂时完全不使用 FSM**。

---

## 五、第三步：首先确定 RL 的 Action（最重要的一步）

你现在的动作实际上是：`cuMotion → joint position target`。

但是 policy 应该产生什么？

我建议你第一版**不要直接让 RL 输出 7 个 joint torque**。你的任务是 Franka + 抓取 + 搬运 + 放置，而且你已经有成熟的 cuMotion / RMPFlow。所以最适合你的第一版是：

**RL 输出 Cartesian action**

例如：

```
action_dim = 4

dx
dy
dz
gripper
```

或者：

```
action_dim = 7

dx
dy
dz
droll
dpitch
dyaw
gripper
```

---

## 六、推荐的第一版 Action

我非常建议你先做：

```
action = [Δx, Δy, Δz, gripper]
```

例如：

```python
action = torch.clamp(action, -1.0, 1.0)

delta_pos = action[:, 0:3] * 0.02
gripper = action[:, 3]
```

流程是：

```
policy
   │
   ├── dx
   ├── dy
   ├── dz
   └── gripper
        ↓
Cartesian target
        ↓
cuMotion
        ↓
joint target
        ↓
Franka
```

这实际上是把你现在的 cuMotion 当成一个 **low-level controller**，而 RL 是 **high-level Cartesian policy**，这是非常合理的 hierarchical control。

---

## 七、你现在的 cuMotion 不需要马上删除

你代码中已经有：

```python
controller = RmpFlowController(...)

desired = controller.forward(
    estimated,
    setpoint,
    phase_time[env_idx],
)
```

然后 `apply_cumotion_joint_target(...)`。所以你可以形成：

```
RL Policy → Cartesian Δpose → Target EE pose → cuMotion RMPFlow → Joint Position Target → Franka
```

这样 RL 的学习难度会比直接学 torque 小很多。

---

## 八、第四步：定义 Observation

你现在已经能够获取：

| 数据 | 来源 |
|------|------|
| Franka joint position | `robot.data.joint_pos` |
| Franka joint velocity | `robot.data.joint_vel` |
| EE pose | `robot.data.body_pose_w`（`get_ee_pose()`） |
| katao pose | `katao.data.root_link_pose_w` |
| stick_part pose | `stick_part.data.root_link_pose_w` |
| Joint wrench | `wrench_sensor.data.force` / `wrench_sensor.data.torque` |

你当前代码已经读取 `panda_link7` 的 force [N] 和 torque [Nm]，并用于 ROS 发布。

---

## 九、我建议你的 Observation 第一版这样设计

不要一上来放太多东西。先：

```
Observation =
[
    joint_pos,        # 9 (7 arm + 2 gripper)
    joint_vel,        # 9
    ee_pos,           # 3
    ee_quat,          # 4
    katao_pos,        # 3
    katao_quat,       # 4
    stick_pos,        # 3
]
```

总计：`9 + 9 + 3 + 4 + 3 + 4 + 3 = 35`，所以 **obs_dim = 35**。第一版可以先这么做。

---

## 十、但是更推荐：Observation 使用相对坐标

不要直接给 `katao_pos_world`，而使用：

```python
object_rel = katao_pos - ee_pos
```

这样 observation 为：

```
joint_pos
joint_vel
object_relative_position
target_relative_position
gripper_state
```

会比直接给 world coordinate 更容易学。

---

## 十一、JointWrenchSensor 可以作为 Observation

你这个传感器非常有价值。你现在已经有：

```python
force = wrench_sensor.data.force[env_idx, sensor_body_id]
torque = wrench_sensor.data.torque[env_idx, sensor_body_id]
```

那么可以把 `Fx Fy Fz Tx Ty Tz` 放进 Observation：

```python
wrench = torch.cat([force, torque], dim=-1)
```

于是 `robot state + object state + wrench`，这对于你后面研究 **force-aware grasping**（力感知抓取）会很有意义。

---

## 十二、第五步：Reward 怎么从你现在的程序转换

你现在没有 reward。但其实你已经写出了很多可以直接转换成 reward 的东西。比如：

```python
error = np.linalg.norm(ee_pos - target)
```

这就是非常好的 reward 基础。

---

## 十三、最简单的 Reward

第一版：

```python
reward = -torch.linalg.norm(ee_pos - target_pos, dim=-1)
```

例如：距离 0.5m → -0.5，距离 0.1m → -0.1，距离 0.01m → -0.01。机器人自然会学习靠近目标。

---

## 十四、然后增加阶段 reward

你现在的任务阶段：`PREGRASP → GRASP → CLOSE → LIFT → TRANSPORT → PLACE → OPEN`。可以分阶段给奖励：

```
到达 grasp      +1
成功抓住物体     +2
成功 lift        +2
transport 成功   +2
place 成功       +5
成功释放         +5
```

但是我建议：**第一版不要直接把 FSM reward 全加进去**。先让机器人学 `reach object`，再 `grasp`，再 `pick`，再 `place`。

---

## 十五、Reward 可以分成四层

```python
# 1. Reach reward
r_reach = -distance(ee, grasp_target)

# 2. Grasp reward
# 物体在夹爪之间 + 夹爪闭合 + 物体跟随机械臂

# 3. Transport reward
r_transport = -distance(object_pos, target_pos)

# 4. Place reward
if object_in_target:
    reward += 10
```

---

## 十六、你的 katao displacement 已经是一个很好的成功指标

你现在最终计算：

```python
katao_displacements.append(
    np.linalg.norm(
        final_katao_pos_env
        - katao_positions[env_idx]
    )
)
```

这实际上可以直接转化成 grasp success：

```python
object_lifted = (
    katao_pos[:, 2]
    - initial_katao_pos[:, 2]
    > threshold
)
```

或者更严格：物体 position ≈ 夹爪 position，这可以作为 grasp reward。

---

## 十七、Force reward 也可以利用你现在的传感器

例如你希望避免撞击：

```python
force_penalty = torch.linalg.norm(wrench_force, dim=-1)
reward -= 0.001 * force_penalty
```

但要特别注意：**不要简单地让力越小越好**，因为抓取本身需要一定夹持力。更合理的是：

```python
force_excess = torch.relu(force_norm - force_limit)
reward -= k_force * force_excess
```

正常 grasp force 给 reward，过大的 force 给 penalty。

---

## 十八、你的 timeout 直接变成 termination

现在：

```python
if phase_step[env_idx] >= timeout_steps:
    phase[env_idx] = TaskPhase.FAILED
```

RL 中就变成：

```python
timeout = episode_time > max_episode_length
terminated = task_failed       # 任务失败
truncated = timeout             # 超时截断
```

这是标准 RL episode 概念。

---

## 十九、最关键的一点：不要把现在的 FSM 原封不动放进 RL

这是很多人第一次把机器人程序改 RL 时最容易犯的错误。如果你保留 `PREGRASP → GRASP → CLOSE → LIFT → TRANSPORT → PLACE` 然后代码自动推进，那么 policy 实际上没有学会「什么时候抓」，因为你的程序已经决定了。真正的 RL 应该是：

```
Policy → 决定 dx dy dz gripper → 环境 → 观察 → reward
```

而不是 `FSM → 决定下一步 → cuMotion`。

---

## 二十、所以我建议你做三个版本

这是最稳妥的路线。

### Version 0：你现在的程序（FSM + cuMotion + Isaac Lab）
作用：验证物理环境和任务。**你现在实际上已经完成了这个版本。**

### Version 1：RL Reach
只做 Franka + katao，任务 `EE → katao`：
- Action：`dx, dy, dz`
- Observation：`joint_pos, joint_vel, EE, object relative position`
- Reward：`-distance(EE, object)`
- 成功：`distance < 0.01m`
- 先把 num_envs 调小（如 16/32），把这个训练起来。

### Version 2：RL Grasp
- Action：`dx, dy, dz, gripper`
- Reward：`reach + grasp + lift`
- Success：物体高度 > 初始高度 + 阈值

### Version 3：完整 Pick & Place
- 流程：reach → grasp → lift → transport → place → release
- Reward：reach + grasp + lift + transport + place + collision penalty + excessive force penalty

---

## 二十一、你的 RL Environment 应该长什么样

核心大概是：

```python
class FrankaPickPlaceEnv(DirectRLEnv):

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.katao = self.scene["katao"]
        self.stick_part = self.scene["stick_part"]
        self.wrench_sensor = self.scene["joint_wrench"]

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

这就是你整个程序要变成的核心。

---

## 二十二、_pre_physics_step() 放什么？

这里处理 policy action：

```python
def _pre_physics_step(self, actions):
    self.actions = actions
    delta_pos = actions[:, :3] * 0.02
    self.ee_target_pos += delta_pos
    self.gripper_target = actions[:, 3]
```

流程：`policy action → ee_target → cuMotion`。

---

## 二十三、_apply_action() 放你的 cuMotion

你现在的 `desired = controller.forward(...)` 应该迁移到这里：

```python
def _apply_action(self):
    for env_id in range(self.num_envs):
        target = self.ee_target_pos[env_id]
        estimated = make_estimated_state(...)
        setpoint = make_setpoint_state(tool_frame, robot_site_space, target, self.ee_target_quat)
        desired = self.controllers[env_id].forward(estimated, setpoint, self.sim_time)
        apply_cumotion_joint_target(self.robot, desired, env_id)
```

不过这里我特别建议你后面再优化成完全 batch 化。

---

## 二十四、你现在的 cuMotion controller 是逐环境 Python loop

你目前 `for env_idx in range(num_envs): controller = controllers[env_idx]`。而 RL 最终可能是 4096 / 8192 envs，那么 `for env_idx in range(4096):` 会成为严重的性能瓶颈。所以：

- **第一阶段**：`num_envs = 16` 甚至 `32`，先把 RL 跑通。
- **第二阶段**：再考虑 cuMotion batch，或者 RL 直接输出 joint target，减少 cuMotion Python 调用。

---

## 二十五、_get_observations() 应该这样组织

```python
def _get_observations(self):
    joint_pos = self.robot.data.joint_pos
    joint_vel = self.robot.data.joint_vel
    ee_pos = self.robot.data.body_pos_w[:, self.ee_body_id]
    object_pos = self.katao.data.root_pos_w
    object_rel = object_pos - ee_pos

    obs = torch.cat([joint_pos, joint_vel, object_rel], dim=-1)
    return {"policy": obs}
```

这就是：`Isaac Lab → observation tensor → RL policy`。

---

## 二十六、Reward 也应该 tensor 化

不要再 `for env_idx in range(num_envs): numpy()`，而应该：

```python
distance = torch.linalg.norm(ee_pos - target_pos, dim=-1)
reward = -distance
```

这样 4096 envs 可以一次计算。

---

## 二十七、Reset 是你现在程序缺少的核心

你的现在程序是「启动 → 跑一次 → DONE」。RL 则是「episode 0 失败 → reset → episode 1 成功 → reset → ...」：

```python
def _reset_idx(self, env_ids):
    super()._reset_idx(env_ids)
    # reset robot
    # reset katao
    # reset stick
    # reset target
    self.robot.write_root_state_to_sim(...)
    self.robot.write_joint_state_to_sim(...)
    self.katao.write_root_state_to_sim(...)
    self.stick_part.write_root_state_to_sim(...)
```

---

## 二十八、你现在固定的目标必须随机化

你现在 katao init `(0.35, 0.6, 0.037057)`、stick `(0.75, 0.0, 0.131)`。对于 RL，最好变成每 episode 随机：

```
episode 1: katao = (0.35, 0.60)
episode 2: katao = (0.42, 0.52)
episode 3: katao = (0.30, 0.68)
...
```

否则 policy 很容易学成「固定位置动作」，而不是「理解如何抓取」。

---

## 二十九、目标也随机化

```python
katao_xy = uniform([-0.2, -0.3], [0.2, 0.3])
place_xy = uniform(...)
```

让 Object 和 Target 都是随机位置，policy 才能学到 general manipulation。

---

## 三十、katao 和 stick_part 有一个地方要特别注意

你现在 `katao: kinematic_enabled=False`，而 `stick_part: kinematic_enabled=True`，意味着它们在物理上承担的角色不同。对于真正的 pick-and-place RL：

- `katao` 应该是可以被夹爪真正拿起来的**动态刚体**。
- `stick_part` 如果是目标位置/容器/插入结构，要根据实际任务确定它是否应该动态。

建议 RL 第一版先把任务简化：**Franka + katao + 固定 place target**，不要一开始同时训练抓取 + 搬运 + 插入 + 力控制 + ROS。

---

## 三十一、ROS 2 不应该放进 RL training loop

你现在有 `WrenchPublisher`（`/franka/env_0/joint_wrench` ...）。这个对于验证和真实机器人部署很有价值，但是训练阶段建议关闭 ROS，因为 RL training 需要高频 + 大量 env + GPU + batch，而 ROS（`rclpy` / `publisher` / `spin_once` / CPU）会拖慢训练。

```
Training:    Isaac Lab → tensor → PPO
Deployment:  Isaac Lab → JointWrenchSensor → ROS2
```

分开。

---

## 三十二、你的 JointWrenchSensor 可以保留

它在 SceneCfg：

```python
joint_wrench = JointWrenchSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    update_period=0.0,
    debug_vis=True,
)
```

这个非常适合直接成为 observation source（`force` / `torque`），以后甚至可以研究 force-aware grasping。

---

## 三十三、我建议你的 Observation 最终变成

如果使用 7 DOF arm：

```
┌─────────────────────────────┐
│ Robot                       │
│ q[7]                        │
│ dq[7]                       │
│ gripper[1]                  │
├─────────────────────────────┤
│ EE                          │
│ position[3]                 │
│ orientation[4]              │
├─────────────────────────────┤
│ Object                      │
│ relative position[3]        │
│ relative orientation[3/4]   │
├─────────────────────────────┤
│ Place target                │
│ relative position[3]        │
├─────────────────────────────┤
│ Force/Torque                │
│ F[3]                        │
│ T[3]                        │
└─────────────────────────────┘
```

---

## 三十四、Action 最终建议

我会先用 `Action = 4`：

```
[dx, dy, dz, gripper]
```

而不是 `[τ1 ... τ7]`，因为你的 cuMotion 已经承担了 Cartesian → joint motion planning/control，RL 只负责 task-level manipulation decision，这和你当前软件架构最匹配。

---

## 三十五、整个系统最终会变成

```
                 PPO Policy
                     │
                [dx dy dz grip]
                     ▼
              Cartesian Target
                     ▼
                cuMotion
                     ▼
             Joint Position Target
                     ▼
                  Franka
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
           Physics       Sensors
              │       ┌─────┴─────┐
              │       │           │
              │       ▼           ▼
              │      EE       JointWrench
              │       │           │
              └───────┴───────────┘
                      ▼
                  Observation
                      ▼
                  PPO Policy
                      +
                   Reward
```

---

## 三十六、具体迁移关系

| 现在的代码 | RL 中变成 |
|-----------|-----------|
| `SceneCfg` | `EnvCfg.scene` |
| `robot` | `scene["robot"]` |
| `katao` | `scene["katao"]` |
| `stick_part` | `scene["stick_part"]` |
| `JointWrenchSensor` | observation |
| `get_ee_pose()` | observation / reward |
| `get_prim_pose()` | observation / reward |
| `TaskPhase` | task state / 后期可删除 |
| `phase_targets` | curriculum / target |
| `cuMotion` | low-level controller |
| `apply_cumotion_joint_target()` | action application |
| `get_env_ee_error()` | reward |
| `phase timeout` | termination |
| `FAILED` | terminated |
| `DONE` | success termination |
| `warmup` | 通常删除 |
| ROS publisher | training 时删除 |
| `print_wrench` | debug only |
| final verification | episode metrics |
| `main while loop` | Isaac Lab RL runner |

---

## 三十七、你现在最适合的实现顺序

不要一次改完，严格按照下面顺序：

1. **Step 1**：保留 `SceneCfg`，创建 `Franka + katao + stick + sensor`，确认 `num_envs = 64` 能正常启动。
2. **Step 2**：创建 `FrankaPickPlaceEnv`，让它能 `reset()` / `step(action)` 正常运行。
3. **Step 3**：实现 Observation（`q, dq, EE, object, target`），打印 `obs.shape` 确认 `[num_envs, obs_dim]`。
4. **Step 4**：实现最简单 Action（`dx, dy, dz`），让机器人能 `action → EE target → cuMotion → robot`。
5. **Step 5**：暂只训练 `EE → object`，Reward `-distance`，成功 `distance < 0.02`。
6. **Step 6**：加入 `gripper action` → `dx, dy, dz, gripper`。
7. **Step 7**：加入 grasp reward（object lifted）。
8. **Step 8**：加入 place target（object → target）。
9. **Step 9**：加入 force / torque observation。
10. **Step 10**：加入 force penalty、collision penalty、smoothness penalty。
11. **Step 11**：加入 domain randomization（object position / orientation / friction / mass / target position / robot initial q）。
12. **Step 12**：最后才考虑 ROS2、real robot、sim-to-real。

---

## 三十八、最重要的架构建议

对于你的任务，我实际上更推荐：

```
RL → High-level policy → Cartesian action → cuMotion → Joint position target → Robot
```

而不是：

```
RL → 7 joint torque → Franka
```

因为你已经有成熟的 cuMotion。这样你不是把已有系统全部推翻，而是在已有系统上增加一个 RL decision layer。

---

## 三十九、第一版我甚至建议把任务缩成这样

先完全不要 `PREGRASP / GRASP_APPROACH / CLOSE / LIFT / TRANSPORT / PLACE / OPEN / FINAL_LIFT / JOINTMOVE`，先：

- **Environment**：Franka + Katao
- **Action**：`dx dy dz gripper`
- **Observation**：`q, dq, EE, Katao, wrench`
- **Goal**：grasp + lift
- **Reward**：reach + grasp + lift
- **Termination**：success / timeout / failure

等这个能稳定训练（success rate ↑），再加 place。

---

## 四十、你这份代码中有一个很重要的现成优势

你的多环境结构已经是正确方向。你现在明确维护：

```python
phase[env_idx]
phase_step[env_idx]
phase_time[env_idx]
phase_initialized[env_idx]
phase_error[env_idx]
completed_phases[env_idx]
```

也就是说 env0 / env1 / env2 / env3 彼此独立，这和 RL 的 N parallel environments 概念完全一致。区别只是：

- 现在：每个 env 有一个 FSM
- RL：每个 env 有一个 episode

所以你的代码不是「不能改 RL」，而是**已经有很好的 simulator/task 基础，只需要把控制逻辑从 FSM 驱动改成 policy 驱动**。

---

## 下一步建议

基于你这份约 2906 行的 `Pasted code(8).py`，可以直接设计并写出一个可放进 Isaac Lab 的完整 RL task 第一版，而不是只讲概念。目标结构：

```
Franka + katao + stick_part
        ↓
Isaac Lab DirectRLEnv
        ↓
Observation: q/dq, EE pose, katao pose, target pose, wrench
        ↓
Action: dx, dy, dz, gripper
        ↓
cuMotion RMPFlow
        ↓
Franka
        ↓
Reward: reach / grasp / lift / transport / place / force penalty
        ↓
Termination: success / failure / timeout
        ↓
PPO / rsl_rl
```

进一步可整理成：**具体文件结构 + 每个 .py 文件完整代码 + `train.py` + `play.py` + config**，这样你可以直接在 **Isaac Lab 3.0 / Isaac Sim 6.0.1** 环境里逐步运行，而不是自己从 2900 行代码里一点点拆。
