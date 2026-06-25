from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.prims import is_prim_path_valid
from isaacsim.core.prims import SingleArticulation
from omni.isaac.motion_generation import ArticulationKinematicsSolver

import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial


def move_to_joint_safe(
    robot,
    world,
    target_joint: np.ndarray,
    steps: int = 150,
    max_timeout_steps: int = 300,
    pos_tol: float = 0.005
    ) -> tuple[bool, str]:
    """
    平滑运动到目标关节位置，带超时+收敛判定防卡死
    :param robot: Franka 铰接对象
    :param world: 仿真World
    :param target_joint: 目标关节数组
    :param steps: 正常插值总步数（运动时长 = steps * dt）
    :param max_timeout_steps: 最大允许步数，超过直接判定超时退出
    :param pos_tol: 关节位置收敛误差阈值(rad)
    :return: (是否成功, 状态描述)
    """
    current_joint = robot.get_joint_positions()
    if current_joint is None:
        return False, "获取当前关节位置失败，铰接未初始化"

    # 异常值过滤
    if np.any(np.isnan(target_joint)) or np.any(np.isinf(target_joint)):
        return False, "目标关节存在NaN/Inf非法数值"

    step_cnt = 0
    success = False

    while step_cnt < max_timeout_steps:
        alpha = min(step_cnt / steps, 1.0)
        interp_joint = (1 - alpha) * current_joint + alpha * target_joint

        # 下发动作
        action = ArticulationAction(joint_positions=interp_joint)
        robot.apply_action(action)
        world.step(render=True)
        step_cnt += 1

        # 获取最新关节位置，判断是否收敛
        curr_now = robot.get_joint_positions()
        if curr_now is None:
            continue

        # joint_error = np.max(np.abs(curr_now - target_joint))
        joint_error = np.max(np.abs(curr_now[:-2] - target_joint[:-2]))
        if joint_error < pos_tol:
            success = True
            break

    print(f"当前关节位置: {curr_now}")
    print(f'误差具体: {curr_now[:-2] - target_joint[:-2]}')
    if success:
        return True, f"运动完成，总步数:{step_cnt}, 最大关节误差:{joint_error:.4f}"
    else:
        return False, f"运动超时退出，已执行{step_cnt}步，最终最大误差:{joint_error:.4f}"

def move_gripper_smooth(
    robot,
    world,
    target_gripper_joints: np.ndarray,
    steps: int = 80
    ) -> tuple[bool, str]:
    """
    夹爪专用平滑开合：每帧实时读取当前关节插值，持续保持目标位置夹紧
    不做位置收敛判断，避免接触物体超时，依靠PD持续施加夹紧力
    """
    if np.any(np.isnan(target_gripper_joints)) or np.any(np.isinf(target_gripper_joints)):
        return False, "目标关节存在NaN/Inf非法数值"

    for i in range(1, steps + 1):
        # 核心修复：每一帧实时获取当前实际关节位置
        current_joint = robot.get_joint_positions()
        if current_joint is None:
            world.step(render=True)
            continue

        alpha = i / steps
        interp_joint = (1 - alpha) * current_joint + alpha * target_gripper_joints
        action = ArticulationAction(joint_positions=interp_joint)
        robot.apply_action(action)
        world.step(render=True)

    # 运动结束后，持续下发最终目标位置，保持夹紧力
    final_action = ArticulationAction(joint_positions=target_gripper_joints)
    robot.apply_action(final_action)

    return True, f"夹爪平滑运动完成，步数：{steps}，已锁定目标位置保持夹紧"

usd_path = "/home/yh/abc/isaac_learn/woche2/Collected_World0/World0.usd"
prim_path = "/World/Franka"

open_stage(usd_path)

world = World()

world.reset()

for _ in range(60):
    world.step(render=False)

while not is_prim_path_valid(prim_path):
    print("Waiting for prim to be valid...")
    world.step(render=False)

# # Create Articulation view for the robot
# robot = Articulation(prim_paths_expr=prim_path, name="franka_panda_view")

# # Create and initialize the articulation controller with the articulation view
# articulation_controller = ArticulationController()
# articulation_controller.initialize(robot)

# # Get current joint positions
# current_positions = robot.get_joint_positions()
# print('------------------------------------------------')
# print(f"Current joint positions: {current_positions}")
# print(type(robot))
# print('------------------------------------------------')

# Create SingleArticulation wrapper (automatically creates articulation controller)
robot = SingleArticulation(prim_path=prim_path, name="franka_panda")

# Initialize the robot (initializes articulation controller internally)
robot.initialize()

# Get current joint positions
current_positions = robot.get_joint_positions()
print('------------------------------------------------')
print(f"Current joint positions from the very beginning: {current_positions}")
print('------------------------------------------------')

cube_pos = np.array([0.475, 0.0, 0.05])

cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=cube_pos,
        scale=np.array([0.05, 0.05, 0.05]),
        mass=0.05,
        color=np.array([1.0, 0.2, 0.2]),
    )
)
gripper_open = np.array([0.5, 0.5])
gripper_close = np.array([0.002, 0.002])

home = np.array([
    0.0,
    -0.7,
    0.0,
    -2.1,
    0.0,
    1.6,
    0.8,
    0.0,
    0.0,
])

pre_grasp = np.array([
    3.5,
    27.2,
    -3.1,
    -131.8,
    -11.4,
    145.8,
    48.1,
    0.04,
    0.04,
])
pre_grasp = np.deg2rad(pre_grasp)
pre_grasp[-2:] = gripper_open[:]


grasp = np.array([
    3.5,
    27.2,
    -3.1,
    -131.8,
    -11.4,
    145.8,
    48.1,
    0.04,
    0.04,
])
grasp = np.deg2rad(grasp)
grasp[-2:] = gripper_close[:]

# robot.apply_action(
#     ArticulationAction(
#         joint_positions=pre_grasp
#     )
# )

# for _ in range(120):
#     world.step(render=True)

success, msg = move_to_joint_safe(robot, world, pre_grasp)
print(msg)
if success:
    current_positions = robot.get_joint_positions()
    cube_position = cube.get_world_pose()
    print('------------------------------------------------')
    print(f"Current joint positions after pre_grasp: {current_positions}")
    print(f'Postion of Cube: {cube_position}')
    print('------------------------------------------------')

for _ in range(60):
    world.step(render=True)

# robot.apply_action(
#     ArticulationAction(
#         joint_positions=grasp
#     )
# )

# for _ in range(120):
#     world.step(render=True)

# success, msg = move_to_joint_safe(robot, world, grasp)
success, msg = move_gripper_smooth(robot, world, grasp)
print(msg)
if success:
    current_positions = robot.get_joint_positions()
    cube_position = cube.get_world_pose()
    print('------------------------------------------------')
    print(f"Current joint positions after grasp: {current_positions}")
    print(f'Postion of Cube: {cube_position}')
    print('------------------------------------------------')

# robot.apply_action(
#     ArticulationAction(
#         joint_positions=home
#     )
# )

for _ in range(60):
    world.step(render=True)

success, msg = move_to_joint_safe(robot, world, home)
print(msg)
if success:
    current_positions = robot.get_joint_positions()
    print('------------------------------------------------')
    print(f"Current joint positions after home position: {current_positions}")
    print('------------------------------------------------')

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()

# for _ in range(120):
#     world.step(render=True)

# current_positions = robot.get_joint_positions()
# print('------------------------------------------------')
# print(f"Current joint positions: {current_positions}")
# print('------------------------------------------------')

# from isaacsim.core.utils.prims import get_prim_children
# from pxr import UsdGeom

# stage = world.stage

# def print_tree(path, depth=0):
#     prim = stage.GetPrimAtPath(path)
#     print("  " * depth + prim.GetName())
#     for child in prim.GetChildren():
#         print_tree(str(child.GetPath()), depth + 1)

# print_tree("/World/Franka")


# lula = LulaKinematicsSolver()

# ik = ArticulationKinematicsSolver(
#     robot,
#     lula,
#     end_effector_frame_name="panda_hand",  # 如果不动改 panda_hand_tcp
# )

# ik = ArticulationKinematicsSolver(robot)

# def open_gripper():
#     q = robot.get_joint_positions()
#     q[-2:] = 0.04
#     robot.apply_action(ArticulationAction(joint_positions=q))


# def close_gripper():
#     q = robot.get_joint_positions()
#     q[-2:] = 0.0
#     robot.apply_action(ArticulationAction(joint_positions=q))

# def goto(pos, quat=np.array([0, 1, 0, 0]), tol=0.01):

#     while simulation_app.is_running():

#         action, success = ik.compute_inverse_kinematics(
#             target_position=pos,
#             target_orientation=quat,
#         )

#         robot.apply_action(action)

#         world.step(render=True)

#         ee_pos, _ = ik.compute_end_effector_pose()

#         if np.linalg.norm(ee_pos - pos) < tol:
#             break

# pre_grasp = cube_pos + np.array([0, 0, 0.20])
# grasp     = cube_pos + np.array([0, 0, 0.08])
# lift      = cube_pos + np.array([0, 0, 0.25])

# place     = np.array([0.6, -0.3, 0.05])

# quat = np.array([0, 1, 0, 0])

# open_gripper()

# goto(pre_grasp, quat)

# goto(grasp, quat)

# close_gripper()

# for _ in range(120):
#     world.step(render=True)

# goto(lift, quat)

# goto(place, quat)

# open_gripper()
