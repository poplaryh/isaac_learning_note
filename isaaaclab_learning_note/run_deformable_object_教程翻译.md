# 与可变形物体交互（Interacting with a deformable object）

> 原文：Isaac Lab 官方教程 — `run_deformable_object.py`（位于 `scripts/tutorials/01_assets` 目录）
> 说明：本翻译保留原教程的标题层级、代码块与注意事项框（Note / Attention）。技术术语首次出现时附英文原文，方便对照官方 API 文档。代码为原样保留，中文解释放在各代码块下方的说明中。

尽管"**可变形物体（deformable objects）**"有时泛指一大类物体（如布料、流体和软体），但在 **PhysX** 中，可变形物体在语法上对应**软体（soft bodies）**。与刚体不同，软体在受到外力或碰撞时会**发生形变**。

PhysX 中使用**有限元法（Finite Element Method, FEM）**来模拟软体。软体由**两套四面体网格（tetrahedral meshes）**组成——一套是**仿真网格（simulation mesh）**，一套是**碰撞网格（collision mesh）**。仿真网格用于模拟软体的形变，碰撞网格则用于检测它与场景中其它物体的碰撞。更多细节请参阅 PhysX 官方文档。

本教程展示如何在仿真中与可变形物体交互。我们将生成一组软立方体，并学习如何设置它们的**节点位置（nodal positions）**与**节点速度（nodal velocities）**，同时向网格节点施加**运动学指令（kinematic commands）**来移动软体。

---

## 代码（The Code）

本教程对应的脚本是 `scripts/tutorials/01_assets` 目录下的 `run_deformable_object.py`。

---

## 代码详解（The Code Explained）

### 设计场景（Designing the scene）

与"与刚体交互"教程类似，我们先往场景中加入地面和光源。此外，我们使用 `assets.DeformableObject` 类向场景添加一个可变形物体。该类负责在给定路径下生成 prim，并初始化其对应的**可变形体物理句柄（deformable body physics handles）**。

本教程中，我们用法与"生成物体"教程中可变形立方体相似的生成配置，创建一个立方体软体。唯一区别是：现在我们把生成配置**封装**进了 `assets.DeformableObjectCfg` 类。这个类包含资产的**生成策略（spawning strategy）**与**默认初始状态（default initial state）**信息。当把这个配置对象传给 `assets.DeformableObject` 类时，它会在仿真播放（play）时生成物体并初始化对应的物理句柄。

> **Note（注意）**
> 可变形物体**仅在 GPU 仿真**中受支持，且生成时必须给网格物体带上**可变形体物理属性（deformable body physics properties）**。

和刚体教程一样，我们通过在构造函数中传入配置对象、创建 `assets.DeformableObject` 类的实例，以类似方式把可变形物体生成进场景。

```python
# Create separate groups called "Origin1", "Origin2", "Origin3"
# Each group will have a robot in it
# 创建名为 "Origin0"~"Origin3" 的若干分组，每个分组里放置一个物体
origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
for i, origin in enumerate(origins):
    sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

# Deformable Object
# 可变形物体配置
cfg = DeformableObjectCfg(
    prim_path="/World/Origin.*/Cube",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.2, 0.2, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    debug_vis=True,
)
cube_object = DeformableObject(cfg=cfg)
```

**代码说明**
- `origins` 定义了 4 个分组的平移位置（Xform），用于摆放 4 个立方体。
- `DeformableObjectCfg` 是"配置类"：
  - `prim_path="/World/Origin.*/Cube"` 用通配符一次性生成多个 Cube。
  - `spawn` 里用 `MeshCuboidCfg` 描述立方体，关键是带上了 `deformable_props`（可变形体属性）和 `physics_material`（软体材质：泊松比 `poissons_ratio` 与杨氏模量 `youngs_modulus`——模量越大越"硬"）。
  - `init_state` 指定生成时的初始位姿。
  - `debug_vis=True` 打开调试可视化（会显示运动学目标标记）。
- 最后把 `cfg` 传给 `DeformableObject` 实例化，完成生成与句柄初始化。

### 运行仿真循环（Running the simulation loop）

延续刚体教程的思路，我们定期重置仿真、向可变形体施加运动学指令、执行仿真步进，并更新可变形物体的内部缓冲区。

### 重置仿真状态（Resetting the simulation state）

与刚体和关节体不同，可变形物体有**不同的状态表示**。可变形物体的状态由网格的**节点位置**与**节点速度**定义。这些节点位置和速度定义在**仿真世界坐标系（world frame）**下，并存储在 `assets.DeformableObject.data` 属性中。

我们使用 `assets.DeformableObject.data.default_nodal_state_w` 属性来获取生成物体 prim 的**默认节点状态**。这个默认状态可通过 `assets.DeformableObjectCfg.init_state` 属性配置；本教程中我们保持其为**单位变换（identity，即不变）**。

> **Attention（注意）**
> 配置 `assets.DeformableObjectCfg` 中的初始状态，指定的是可变形物体在**生成时刻的位姿（pose）**。基于这个初始状态，在仿真**首次播放（play）**时即得到默认节点状态。

我们对节点位置施加变换，以随机化可变形物体的初始状态。

```python
# reset the nodal state of the object
# 获取默认节点状态
nodal_state = cube_object.data.default_nodal_state_w.clone()
# apply random pose to the object
# 随机生成平移与旋转，用于随机化初始状态
pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + origins
quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)
```

**代码说明**
- `default_nodal_state_w` 的形状是 `[num_instances, num_nodes, 7]`，其中每节点的坐标占前 3 位（位置）、后 4 位（速度/姿态）。
- `transform_nodal_pos` 按给定的平移 `pos_w` 和旋转 `quat_w`，把节点位置变换到世界坐标系下，实现随机摆放。

要重置可变形物体，我们首先调用 `assets.DeformableObject.write_nodal_state_to_sim()` 方法设置节点状态。该方法把可变形物体 prim 的节点状态写入**仿真缓冲区（simulation buffer）**。此外，我们调用 `assets.DeformableObject.write_nodal_kinematic_target_to_sim()` 方法，**释放（free）**上一步中为节点设置的所有运动学目标。运动学目标我们会在下一节解释。

最后，调用 `assets.DeformableObject.reset()` 方法重置所有内部缓冲区和缓存。

```python
# write nodal state to simulation
# 把节点状态写入仿真
cube_object.write_nodal_state_to_sim(nodal_state)

# Write the nodal state to the kinematic target and free all vertices
# 把节点状态写入运动学目标，并"释放"所有顶点
nodal_kinematic_target[..., :3] = nodal_state[..., :3]
nodal_kinematic_target[..., 3] = 1.0
cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

# reset buffers
# 重置缓冲区
cube_object.reset()
```

**代码说明**
- `nodal_kinematic_target` 的第 4 个分量（索引 3）是**约束标志**：`1.0` 表示释放（自由），`0.0` 表示约束（kinematic）。这里设为 `1.0` 即释放所有顶点，回到由 FEM 自由模拟的状态。
- `write_nodal_kinematic_target_to_sim` 把运动学目标写入仿真缓冲区。

### 执行仿真步进（Stepping the simulation）

软体支持**用户驱动的运动学控制（kinematic control）**：用户可以指定部分网格节点的位置目标，而其余节点仍由 **FEM 求解器**模拟。这种**局部运动学控制（partial kinematic control）**在需要以可控方式与可变形物体交互的应用中非常有用。

本教程中，我们给场景中 4 个立方体里的**两个**施加运动学指令。我们把**索引为 0（左下角）**的节点设为位置目标，使立方体沿 z 轴移动。

每一步，我们都给该节点的运动学位置目标增加一个**小增量**。同时，我们设置标志位，表明该目标在仿真缓冲区中是该节点的运动学目标。这些内容通过调用 `assets.DeformableObject.write_nodal_kinematic_target_to_sim()` 方法写入仿真缓冲区。

```python
# update the kinematic target for cubes at index 0 and 3
# 更新索引为 0 和 3 的两个立方体的运动学目标
# we slightly move the cube in the z-direction by picking the vertex at index 0
# 取索引 0 的节点（顶点），让其沿 z 方向小幅移动
nodal_kinematic_target[[0, 3], 0, 2] += 0.001
# set vertex at index 0 to be kinematically constrained
# 将索引 0 的节点设为运动学约束：0 = 约束(constrained)，1 = 自由(free)
nodal_kinematic_target[[0, 3], 0, 3] = 0.0
# write kinematic target to simulation
# 写入仿真
cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
```

**代码说明**
- `nodal_kinematic_target[[0, 3], 0, 2] += 0.001`：对第 0、3 号立方体，取它们各自的节点 0，把 z 分量（索引 2）每步增加 `0.001`，实现缓慢抬升。
- `nodal_kinematic_target[[0, 3], 0, 3] = 0.0`：把该节点的约束标志设为 `0.0`（约束），即该节点被"钉"在运动学目标位置上，由我们控制；其余节点仍由 FEM 自由模拟。

与刚体和关节体类似，我们在执行仿真步进前调用 `assets.DeformableObject.write_data_to_sim()` 方法。对可变形物体而言，该方法**不会**施加任何外力。但我们保留它以保证完整性，也为将来扩展之用。

```python
# write internal data to simulation
# 把内部数据写入仿真
cube_object.write_data_to_sim()
```

### 更新状态（Updating the state）

仿真步进之后，我们更新可变形物体 prim 的内部缓冲区，使 `assets.DeformableObject.data` 属性反映其新状态。这通过 `assets.DeformableObject.update()` 方法完成。

我们以固定间隔把可变形物体的**根位置（root position）**打印到终端。如前所述，可变形物体**没有"根状态（root state）"的概念**。不过我们把根位置**近似计算为网格所有节点的平均位置**。

```python
# update buffers
# 更新缓冲区
cube_object.update(sim_dt)
# print the root position
# 打印根位置
if count % 50 == 0:
    print(f"Root position (in world): {cube_object.data.root_pos_w[:, :3]}")
```

**代码说明**
- `cube_object.data.root_pos_w` 即为"所有节点平均位置"近似出的根坐标，形状为 `[num_instances, 3]`。
- 每 50 步打印一次，便于观察物体是否按预期运动。

---

## 代码执行（The Code Execution）

讲解完代码后，我们来运行脚本看看效果：

```console
./isaaclab.sh -p scripts/tutorials/01_assets/run_deformable_object.py
```

这会打开一个舞台，里面有地面、灯光和若干绿色立方体。4 个立方体中有 2 个会从高处**下落并落在地面上静止**；与此同时，另外 2 个会沿 **z 轴移动**。你应该能看到一个**标记（marker）**，表示立方体左下角节点的运动学目标位置。要停止仿真，可以关闭窗口，或在终端按 `Ctrl+C`。

![run_deformable_object.py 的运行结果](result_of_run_deformable_object.png)

---

## 小结

本教程展示了如何生成可变形物体，并将其封装进 `DeformableObject` 类以初始化其物理句柄，从而能够设置和获取它们的状态。我们还看到了如何向可变形物体施加运动学指令，以可控方式移动网格节点。下一个教程中，我们将学习如何使用 `InteractiveScene` 类来创建场景。

---

## 关键要点速记（Cheat Sheet）

| 概念 | 说明 |
| --- | --- |
| 可变形物体 = PhysX 软体 | 用 **FEM（有限元）** 模拟，由仿真网格 + 碰撞网格两套四面体网格组成 |
| `DeformableObjectCfg` | 配置类：描述生成策略、默认初始状态、软体材质（杨氏模量/泊松比） |
| `DeformableObject` | 运行类：生成 prim 并初始化**物理句柄**，提供读写节点状态的方法 |
| 状态表示 | 由**节点位置 + 节点速度**定义（世界坐标系），存于 `.data` |
| `default_nodal_state_w` | 默认节点状态；基于 `init_state` 在首次 play 时生成 |
| 运动学约束标志 | 第 4 分量：`0.0`=约束（你控制），`1.0`=释放（FEM 自由模拟） |
| 局部运动学控制 | 只"钉住"部分节点（如节点 0），其余由 FEM 模拟 |
| 仅支持 GPU 仿真 | 必须在 GPU 上跑，且网格需带可变形体物理属性 |
| 没有根状态 | 根位置近似为所有节点的**平均位置**（`.data.root_pos_w`） |
| 核心方法 | `write_nodal_state_to_sim` / `write_nodal_kinematic_target_to_sim` / `reset` / `update` / `write_data_to_sim` |
