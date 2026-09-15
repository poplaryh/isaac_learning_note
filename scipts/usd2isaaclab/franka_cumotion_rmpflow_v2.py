#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isaac Lab 3.0 / Isaac Sim 6.0.1 Franka + cuMotion RMPFlow pick/move verification. (v2)

USD:
    /Franka
    /World/katao/katao
    /World/stick_part/stick_part

v2 修正记录（相对上一版，针对"误差稳定 0.059 m 不收敛"问题）：

[修正 1] 误差测量点对齐 cuMotion 的 tool frame。
    上一版测量的是 panda_hand（法兰）位置，但 Isaac Sim 6.0.1 内置 cuMotion Franka 配置
    （isaacsim.robot_motion.cumotion/robot_configurations/franka/robot.xrdf）里：
        tool_frames: ["panda_leftfingertip"]
    URDF 运动链：
        panda_hand --(z +0.0584, prismatic)--> panda_leftfinger --(z +0.045, fixed)--> panda_leftfingertip
    即控制点(左指尖)与测量点(法兰)恒差 sqrt(0.1034^2 + q_finger^2) ≈ 0.103~0.111 m。
    即使 cuMotion 完美收敛，法兰误差也永远 > 0.10 m，tolerance=0.03 数学上不可达，
    每个 phase 都会跑满 10 s 超时 —— 这就是"误差总是稳定不下来"的主因。
    v2 用 panda_leftfinger body 的实际位姿 + URDF 偏移 [0,0,0.045] 计算指尖位置来度量误差。

[修正 2] 每步同步机器人 base 位姿到 CumotionWorldInterface。
    cuMotion 全部在"机器人 base 坐标系"内运算，而 base 位姿【不会】被自动同步
    （官方文档原话："The robot base position is not automatically synchronized"）。
    若 USD 中 /Franka 不在世界原点，上一版会产生一个恒定的额外偏差。
    v2 每步调用 update_world_to_robot_root_transforms()。base 在原点时此调用无副作用。

[修正 3] --ee-quat 参数改为 (QW, QX, QY, QZ) 的 wxyz 约定。
    cuMotion motion_generation API 的四元数是 wxyz（官方 Graph Planner 教程明确注释
    "quaternion wxyz"，且 RMPflow 教程直接传入 Isaac Sim get_world_poses() 的 wxyz 输出）。
    上一版按 xyzw 传 [0,1,0,0]，实际被解释为"绕 X 轴 180°"（碰巧也是竖直朝下，但爪口
    方向与预期差 90°）。v2 默认 (0,0,1,0) = 绕 Y 轴 180°，工具竖直向下。
    如需调整爪口朝向（yaw），改水平轴分量即可，例如 (0, 0.7071, 0, 0.7071)。

[修正 4] prim_path 由 " /Franka  "（前后带空格）修正为 "/Franka"。

[修正 5] 新增诊断打印：机器人 base 位姿、误差向量 xyz 分量、闭合后手指关节实际位置。

[语义变化] --grasp-height / --pregrasp-height / --lift-height 现在表示
    【指尖】相对 katao 原点的高度（上一版实际效果也是如此——cuMotion 控制的是指尖——
    只是没有被明确意识到）。若 katao 原点在物体几何中心，建议 grasp-height 从
    +0.02~+0.04 开始试（指尖平面略高于物体中心，指间接触区正对物体中部）。

Sequence:
    read katao pose
      -> open gripper
      -> pregrasp
      -> grasp
      -> close gripper
      -> lift
      -> transport to an arbitrary pose
      -> hold
"""

import argparse
import traceback

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Minimal Franka cuMotion RMPFlow pick/move verification (v2)"
)
parser.add_argument(
    "--usd",
    type=str,
    default="/home/yh/tianji/mission_docs/mission01_v2/World0.usd",
)
parser.add_argument(
    "--katao",
    type=str,
    default="/World/katao/katao",
)
parser.add_argument(
    "--pregrasp-height",
    type=float,
    default=0.12,
)
parser.add_argument(
    "--grasp-height",
    type=float,
    default=0.03,
)
parser.add_argument(
    "--lift-height",
    type=float,
    default=0.20,
)
parser.add_argument(
    "--move-offset",
    type=float,
    nargs=3,
    default=(0.25, -0.25, 0.20),
    metavar=("DX", "DY", "DZ"),
)
parser.add_argument(
    "--ee-quat",
    type=float,
    nargs=4,
    default=(0.0, 0.0, 1.0, 0.0),
    metavar=("QW", "QX", "QY", "QZ"),
    help=(
        "World-frame cuMotion tool orientation, wxyz convention. "
        "Default (0,0,1,0) = rotate 180 deg about Y -> tool pointing down. "
        "NOTE: cuMotion motion_generation API expects wxyz."
    ),
)
parser.add_argument(
    "--open",
    type=float,
    default=0.04,
)
parser.add_argument(
    "--close",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--grasp-wait",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--tolerance",
    type=float,
    default=0.02,
)
parser.add_argument(
    "--phase-timeout",
    type=float,
    default=10.0,
)
parser.add_argument(
    "--warmup",
    type=int,
    default=120,
)
parser.add_argument(
    "--print-every",
    type=int,
    default=30,
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Isaac Sim imports must happen after SimulationApp.
# -----------------------------------------------------------------------------

import numpy as np
import torch
import warp as wp
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# cuMotion extension is already known to work in this environment.
import omni.kit.app

_extension_manager = omni.kit.app.get_app().get_extension_manager()
if not _extension_manager.is_extension_enabled("isaacsim.robot_motion.cumotion"):
    result = _extension_manager.set_extension_enabled_immediate(
        "isaacsim.robot_motion.cumotion", True
    )
    if not result:
        raise RuntimeError(
            "Could not enable isaacsim.robot_motion.cumotion"
        )
simulation_app.update()

import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.robot_motion.cumotion import (
    CumotionWorldInterface,
    RmpFlowController,
    load_cumotion_supported_robot,
)

# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


@configclass
# class SceneCfg(InteractiveSceneCfg):
#     # Reuse the Franka that already exists in the USD.
#     # v2: removed stray spaces around the prim path.
#     robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
#         prim_path="/Franka",
#         spawn=None,
#     )
class SceneCfg(InteractiveSceneCfg):
    # Reuse the Franka that already exists in the USD.
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.0),
        ),
    )


# -----------------------------------------------------------------------------
# USD / robot helpers
# -----------------------------------------------------------------------------


def get_prim_world_pose(prim_path: str):
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(
            f"USD prim does not exist: {prim_path}"
        )
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    p = matrix.ExtractTranslation()
    q = matrix.ExtractRotationQuat()
    position = np.array(
        [float(p[0]), float(p[1]), float(p[2])],
        dtype=np.float32,
    )
    quat = np.array(
        [
            float(q.GetImaginary()[0]),
            float(q.GetImaginary()[1]),
            float(q.GetImaginary()[2]),
            float(q.GetReal()),
        ],
        dtype=np.float32,
    )
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise RuntimeError(
            f"Invalid quaternion at {prim_path}: {quat}"
        )
    quat /= norm
    return position, quat


def get_panda_indices(robot):
    arm_ids = []
    for i, name in enumerate(robot.joint_names):
        if name.startswith("panda_joint"):
            suffix = name[len("panda_joint"):]
            if suffix.isdigit():
                arm_ids.append(i)
    arm_ids.sort(
        key=lambda i: int(
            robot.joint_names[i][len("panda_joint"):]
        )
    )
    if len(arm_ids) != 7:
        raise RuntimeError(
            "Expected 7 Panda arm joints, got "
            f"{len(arm_ids)}: "
            f"{[robot.joint_names[i] for i in arm_ids]}"
        )
    finger_ids = [
        i
        for i, name in enumerate(robot.joint_names)
        if name in (
            "panda_finger_joint1",
            "panda_finger_joint2",
        )
    ]
    hand_ids = [
        i
        for i, name in enumerate(robot.body_names)
        if name == "panda_hand"
    ]
    if len(hand_ids) != 1:
        raise RuntimeError(
            "Could not uniquely find panda_hand: "
            f"{hand_ids}"
        )
    return arm_ids, finger_ids, hand_ids[0]


def quat_wxyz_to_rotmat(q):
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = (
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    )
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def get_body_pos_quat_wxyz(robot, body_id):
    """Return (position, quat_wxyz) of a body in the world frame.

    IMPORTANT: Isaac Lab's body_pose_w packs the quaternion in
    (w, x, y, z) order, NOT (x, y, z, w).
    """
    pose = robot.data.body_pose_w[0, body_id]
    position = (
        pose[:3]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    quat = (
        pose[3:7]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    quat /= np.linalg.norm(quat)
    return position, quat


def get_fingertip_pose(robot, lf_body_id):
    """Position of cuMotion's tool frame: panda_leftfingertip.

    cuMotion Franka URDF:
        panda_leftfinger -> panda_leftfingertip : fixed joint, xyz = [0, 0, 0.045]
    Isaac Lab's USD has no fingertip body, so we reconstruct it from the
    panda_leftfinger body pose (finger prismatic offset is included
    automatically because the body pose already reflects it).
    """
    pos, quat_wxyz = get_body_pos_quat_wxyz(robot, lf_body_id)
    tip_pos = pos + quat_wxyz_to_rotmat(quat_wxyz) @ np.array(
        [0.0, 0.0, 0.045], dtype=np.float32
    )
    return tip_pos, quat_wxyz


def get_ee_pose(robot, ee_body_id):
    # Kept for reference printing. Note: quaternion is wxyz.
    return get_body_pos_quat_wxyz(robot, ee_body_id)


# -----------------------------------------------------------------------------
# cuMotion RobotState helpers
# -----------------------------------------------------------------------------


def make_estimated_state(robot, joint_space):
    # The joint-space definition is the full Isaac Lab joint_names list,
    # so the measured state must contain the corresponding full 9-DOF vector.
    q = (
        robot.data.joint_pos[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    dq = (
        robot.data.joint_vel[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return mg.RobotState(
        joints=mg.JointState.from_name(
            robot_joint_space=joint_space,
            positions=(
                joint_space,
                wp.array(q, dtype=wp.float32),
            ),
            velocities=(
                joint_space,
                wp.array(dq, dtype=wp.float32),
            ),
        )
    )


def make_setpoint_state(
    tool_frame,
    site_space,
    position,
    orientation,
):
    position = np.asarray(
        position,
        dtype=np.float32,
    ).reshape(1, 3)
    orientation = np.asarray(
        orientation,
        dtype=np.float32,
    ).reshape(1, 4)
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=site_space,
            positions=(
                [tool_frame],
                wp.array(
                    position,
                    dtype=wp.float32,
                ),
            ),
            orientations=(
                [tool_frame],
                wp.array(
                    orientation,
                    dtype=wp.float32,
                ),
            ),
        )
    )


def sync_robot_base(world_interface, robot):
    """Tell cuMotion where the robot base actually is.

    cuMotion operates in the robot base frame and does NOT pick up the base
    pose automatically. If /Franka is not at the world origin in the USD,
    skipping this call produces a constant Cartesian offset on every phase.
    """
    world_interface.update_world_to_robot_root_transforms(
        poses=(
            robot.data.root_pos_w.detach().cpu(),
            robot.data.root_quat_w.detach().cpu(),  # wxyz
        )
    )


def apply_cumotion_joint_target(
    robot,
    desired_state,
):
    if desired_state is None:
        raise RuntimeError(
            "cuMotion returned None desired_state."
        )
    if desired_state.joints is None:
        raise RuntimeError(
            "cuMotion returned desired_state without joints."
        )
    if desired_state.joints.positions is None:
        raise RuntimeError(
            "cuMotion returned no joint positions."
        )
    positions = desired_state.joints.positions
    if hasattr(positions, "numpy"):
        positions_np = positions.numpy()
    else:
        positions_np = np.asarray(positions)
    positions_np = np.asarray(
        positions_np,
        dtype=np.float32,
    ).reshape(1, -1)
    indices = desired_state.joints.position_indices
    if hasattr(indices, "numpy"):
        indices = indices.numpy()
    indices = np.asarray(
        indices,
        dtype=np.int64,
    ).reshape(-1).tolist()
    target = torch.as_tensor(
        positions_np,
        dtype=torch.float32,
        device=robot.device,
    )
    # NOTE: kept as set_joint_position_target_index -- the API that exists
    # and works in your (github main) Isaac Lab installation.
    robot.set_joint_position_target_index(
        target=target,
        joint_ids=indices,
    )


def set_gripper(robot, finger_ids, value):
    if not finger_ids:
        return
    target = torch.full(
        (1, len(finger_ids)),
        float(value),
        dtype=torch.float32,
        device=robot.device,
    )
    robot.set_joint_position_target_index(
        target=target,
        joint_ids=finger_ids,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    print("\n==========================================================")
    print("  FRANKA + cuMotion RMPFlow MINIMAL TEST  (v2)")
    print("==========================================================")
    print("[INFO] Starting main()")

    # -------------------------------------------------------------------------
    # Open USD
    # -------------------------------------------------------------------------
    print("[INFO] Opening USD...")
    if sim_utils.open_stage(args_cli.usd) is False:
        raise RuntimeError(
            f"Failed to open USD: {args_cli.usd}"
        )
    print("[INFO] USD opened.")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.01,
            device=args_cli.device,
        )
    )
    sim.set_camera_view(
        eye=[1.8, 1.8, 1.4],
        target=[0.45, 0.0, 0.45],
    )

    print("[INFO] Creating InteractiveScene...")
    scene = InteractiveScene(
        SceneCfg(
            num_envs=1,
            env_spacing=2.0,
        )
    )

    print("[INFO] Resetting simulation...")
    sim.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    arm_ids, finger_ids, ee_body_id = get_panda_indices(robot)

    # v2: body id of panda_leftfinger, used to reconstruct the cuMotion
    # tool frame position (panda_leftfingertip).
    lf_body_id = robot.body_names.index("panda_leftfinger")

    print("\n================ FRANKA CHECK =============================")
    print(f"[INFO] DOFs       : {robot.num_joints}")
    print(f"[INFO] Bodies     : {robot.num_bodies}")
    print(
        "[INFO] Arm joints : "
        f"{[robot.joint_names[i] for i in arm_ids]}"
    )
    print(f"[INFO] Finger ids : {finger_ids}")
    print(
        "[INFO] EE body    : "
        f"{robot.body_names[ee_body_id]}"
    )
    print(
        "[INFO] Tool body  : "
        f"{robot.body_names[lf_body_id]} (panda_leftfinger, +0.045 z -> fingertip)"
    )

    # v2: diagnose robot base pose. If this is not [0,0,0]/[1,0,0,0],
    # the missing update_world_to_robot_root_transforms() call in the
    # previous version was contributing a constant Cartesian error.
    root_pos = (
        robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
    )
    root_quat = (
        robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
    )
    print("\n================ ROBOT BASE (DIAGNOSTIC) ==================")
    print(f"[INFO] base pos         : {np.round(root_pos, 5)}")
    print(f"[INFO] base quat (wxyz) : {np.round(root_quat, 5)}")
    if (
        np.linalg.norm(root_pos[:2]) > 1e-4
        or abs(root_pos[2]) > 1e-4
    ):
        print(
            "[WARN] Robot base is NOT at the world origin. "
            "The previous version (without base-transform sync) would see a "
            "constant Cartesian offset of about "
            f"{np.linalg.norm(root_pos):.4f} m on every phase."
        )

    # -------------------------------------------------------------------------
    # Read katao
    # -------------------------------------------------------------------------
    katao_pos, katao_quat = get_prim_world_pose(
        args_cli.katao
    )
    initial_ee_pos, initial_ee_quat = get_ee_pose(
        robot,
        ee_body_id,
    )
    initial_tip_pos, _ = get_fingertip_pose(
        robot,
        lf_body_id,
    )

    print("\n================ TARGET READ ===============================")
    print(
        f"[INFO] katao prim : {args_cli.katao}"
    )
    print(
        f"[INFO] katao pos  : {katao_pos}"
    )
    print(
        f"[INFO] katao quat : {katao_quat} (x,y,z,w)"
    )
    print(
        f"[INFO] initial hand (flange) : {initial_ee_pos}"
    )
    print(
        f"[INFO] initial fingertip     : {initial_tip_pos}"
    )

    # v2: wxyz convention, default (0,0,1,0) = 180 deg about Y (tool down).
    target_quat = np.asarray(
        args_cli.ee_quat,
        dtype=np.float32,
    )
    target_quat /= np.linalg.norm(target_quat)
    print(
        f"[INFO] commanded tool quat : "
        f"{target_quat} (w,x,y,z)"
    )

    # -------------------------------------------------------------------------
    # cuMotion model + world
    # -------------------------------------------------------------------------
    print(
        "\n[STEP] Loading built-in cuMotion Franka configuration..."
    )
    cumotion_robot = load_cumotion_supported_robot(
        "franka"
    )
    print(
        "[OK] load_cumotion_supported_robot('franka')"
    )

    robot_joint_space = list(robot.joint_names)
    # NOTE: for the built-in Franka config in Isaac Sim 6.0.1 this is
    # ['panda_leftfingertip'] -- the LEFT FINGER TIP, not panda_hand!
    # All setpoint positions are interpreted at this frame, and all error
    # measurements in this script are now taken at this frame too.
    robot_site_space = (
        cumotion_robot.robot_description.tool_frame_names()
    )
    if len(robot_joint_space) != robot.num_joints:
        raise RuntimeError(
            f"Joint-space mismatch: {len(robot_joint_space)} names for "
            f"{robot.num_joints} articulation DOFs."
        )
    if not robot_site_space:
        raise RuntimeError(
            "cuMotion Franka configuration has no tool frames."
        )
    tool_frame = robot_site_space[0]
    print(
        f"[INFO] Isaac Lab joint space : "
        f"{robot_joint_space}"
    )
    print(
        f"[INFO] cuMotion controlled joints : "
        f"{cumotion_robot.controlled_joint_names}"
    )
    print(
        f"[INFO] cuMotion tool frames : "
        f"{robot_site_space}"
    )
    print(
        f"[INFO] selected tool frame : "
        f"{tool_frame}"
    )

    print(
        "\n[STEP] Creating CumotionWorldInterface..."
    )
    # NO obstacle tracking in this smoke test, but the world interface is
    # still where the robot base transform must be registered.
    world_interface = CumotionWorldInterface()
    print(
        "[OK] CumotionWorldInterface created."
    )

    print(
        "\n[STEP] Creating RmpFlowController..."
    )
    controller = RmpFlowController(
        cumotion_robot=cumotion_robot,
        cumotion_world_interface=world_interface,
        robot_joint_space=robot_joint_space,
        robot_site_space=robot_site_space,
        tool_frame=tool_frame,
    )
    print(
        "[OK] RmpFlowController created."
    )

    cfg = controller.get_rmp_flow_config()
    cfg.set_param(
        "cspace_target_rmp/metric_scalar",
        1.0,
    )
    print(
        "[OK] cspace_target_rmp/metric_scalar = 1.0"
    )

    # -------------------------------------------------------------------------
    # Targets
    # NOTE (v2): heights are now interpreted at the FINGERTIP frame,
    # which is what cuMotion actually controls.
    # -------------------------------------------------------------------------
    pregrasp = katao_pos + np.array(
        [0.0, 0.0, args_cli.pregrasp_height],
        dtype=np.float32,
    )
    grasp = katao_pos + np.array(
        [0.0, 0.0, args_cli.grasp_height],
        dtype=np.float32,
    )
    print(
        "[INFO] Heights are fingertip-relative to the katao origin. "
        "If the grip is too high/low, tune --grasp-height."
    )
    lift = grasp + np.array(
        [0.0, 0.0, args_cli.lift_height],
        dtype=np.float32,
    )
    transport = katao_pos + np.asarray(
        args_cli.move_offset,
        dtype=np.float32,
    )

    print("\n================ MOTION TARGETS ============================")
    print(f"[INFO] PREGRASP  : {pregrasp}")
    print(f"[INFO] GRASP     : {grasp}")
    print(f"[INFO] LIFT      : {lift}")
    print(f"[INFO] TRANSPORT : {transport}")

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------
    print(
        f"\n[STEP] Warmup for {args_cli.warmup} simulation frames..."
    )
    for i in range(args_cli.warmup):
        set_gripper(
            robot,
            finger_ids,
            args_cli.open,
        )
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    print("[OK] Warmup complete.")

    # v2: register the true robot base pose with cuMotion once before
    # the first controller.reset().
    sync_robot_base(world_interface, robot)
    print("[OK] Robot base transform synced to cuMotion world interface.")

    # -------------------------------------------------------------------------
    # One phase of RMPFlow.
    # -------------------------------------------------------------------------
    dt = sim.get_physics_dt()
    timeout_steps = max(
        1,
        int(args_cli.phase_timeout / dt),
    )

    def run_phase(
        phase_name,
        target_position,
        gripper_position,
    ):
        print(
            "\n----------------------------------------------------------"
        )
        print(
            f"[PHASE] {phase_name}"
        )
        print(
            f"[PHASE] target = "
            f"{np.round(target_position, 5)} (fingertip frame)"
        )
        estimated = make_estimated_state(
            robot,
            robot_joint_space,
        )
        setpoint = make_setpoint_state(
            tool_frame,
            robot_site_space,
            target_position,
            target_quat,
        )
        print("[STEP] controller.reset() ...")
        reset_ok = controller.reset(
            estimated,
            setpoint,
            t=0.0,
        )
        print(
            f"[INFO] controller.reset() -> {reset_ok}"
        )
        if not reset_ok:
            raise RuntimeError(
                f"cuMotion reset failed in phase {phase_name}"
            )

        t = 0.0
        for step in range(timeout_steps):
            set_gripper(
                robot,
                finger_ids,
                gripper_position,
            )
            # v2: keep cuMotion's view of the robot base up to date.
            sync_robot_base(world_interface, robot)
            estimated = make_estimated_state(
                robot,
                robot_joint_space,
            )
            setpoint = make_setpoint_state(
                tool_frame,
                robot_site_space,
                target_position,
                target_quat,
            )
            desired = controller.forward(
                estimated,
                setpoint,
                t,
            )
            if desired is None:
                raise RuntimeError(
                    f"cuMotion returned None in {phase_name}"
                )
            apply_cumotion_joint_target(
                robot,
                desired,
            )
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            t += dt

            # v2: error measured at the fingertip == cuMotion tool frame.
            tip_pos, _ = get_fingertip_pose(
                robot,
                lf_body_id,
            )
            err_vec = tip_pos - target_position
            error = float(np.linalg.norm(err_vec))
            if step % args_cli.print_every == 0:
                print(
                    f"[STEP {step:4d}] "
                    f"tip error = {error:.4f} m | "
                    f"err xyz = {np.round(err_vec, 4)} | "
                    f"tip = {np.round(tip_pos, 4)}"
                )
            if (
                error < args_cli.tolerance
                and step >= 20
            ):
                print(
                    f"[OK] {phase_name} reached. "
                    f"tip error = {error:.4f} m"
                )
                return True

        tip_pos, _ = get_fingertip_pose(
            robot,
            lf_body_id,
        )
        error = float(
            np.linalg.norm(
                tip_pos - target_position
            )
        )
        print(
            f"[WARN] {phase_name} timeout. "
            f"Final tip error = {error:.4f} m"
        )
        return False

    # -------------------------------------------------------------------------
    # Execute
    # -------------------------------------------------------------------------
    print(
        "\n================ EXECUTION ================================"
    )
    run_phase(
        "PREGRASP",
        pregrasp,
        args_cli.open,
    )
    run_phase(
        "GRASP_APPROACH",
        grasp,
        args_cli.open,
    )

    print(
        "\n[PHASE] CLOSE_GRIPPER"
    )
    close_steps = max(
        1,
        int(args_cli.grasp_wait / dt),
    )
    for _ in range(close_steps):
        set_gripper(
            robot,
            finger_ids,
            args_cli.close,
        )
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
    print("[OK] Gripper close wait complete.")
    # v2: diagnostic -- if the fingers stop well above --close, they are
    # pressing on the object (grasp contact); if they reach --close fully,
    # the object was missed.
    finger_now = (
        robot.data.joint_pos[0, finger_ids]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    print(
        f"[INFO] finger joints after close: {np.round(finger_now, 4)} "
        f"(fully open = {args_cli.open}; ~{args_cli.close} = missed the object)"
    )

    run_phase(
        "LIFT",
        lift,
        args_cli.close,
    )
    run_phase(
        "TRANSPORT",
        transport,
        args_cli.close,
    )

    # -------------------------------------------------------------------------
    # Final verification
    # -------------------------------------------------------------------------
    final_tip_pos, _ = get_fingertip_pose(
        robot,
        lf_body_id,
    )
    final_hand_pos, _ = get_ee_pose(
        robot,
        ee_body_id,
    )
    final_katao_pos, _ = get_prim_world_pose(
        args_cli.katao
    )
    final_tip_error = float(
        np.linalg.norm(
            final_tip_pos - transport
        )
    )
    katao_displacement = float(
        np.linalg.norm(
            final_katao_pos - katao_pos
        )
    )

    print(
        "\n=========================================================="
    )
    print(
        "  TEST FINISHED  (v2)"
    )
    print(
        "=========================================================="
    )
    print(
        f"[RESULT] final fingertip pos : "
        f"{np.round(final_tip_pos, 5)}"
    )
    print(
        f"[RESULT] final hand (flange) : "
        f"{np.round(final_hand_pos, 5)}"
    )
    print(
        f"[RESULT] transport target    : "
        f"{np.round(transport, 5)}"
    )
    print(
        f"[RESULT] final tip error     : "
        f"{final_tip_error:.4f} m"
    )
    print(
        f"[RESULT] initial katao       : "
        f"{np.round(katao_pos, 5)}"
    )
    print(
        f"[RESULT] final katao         : "
        f"{np.round(final_katao_pos, 5)}"
    )
    print(
        f"[RESULT] katao displacement  : "
        f"{katao_displacement:.4f} m"
    )
    if final_tip_error < args_cli.tolerance:
        print(
            "\n[SUCCESS] cuMotion RMPFlow control chain "
            "reached the transport target (fingertip frame)."
        )
    else:
        print(
            "\n[FAIL] cuMotion chain did not reach "
            "the final transport target."
        )
    if katao_displacement > 0.03:
        print(
            "[INFO] katao moved with the gripper."
        )
    else:
        print(
            "[INFO] katao did not move significantly. "
            "Check: (a) --grasp-height vs object geometry, "
            "(b) katao has RigidBody+Collider with enough friction, "
            "(c) finger joints stopped above 0 => contact happened."
        )

    # Keep the application alive for inspection.
    print(
        "\n[INFO] Test is complete. "
        "Window will remain open until you close it."
    )
    while simulation_app.is_running():
        scene.update(dt)
        sim.step()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n==========================================================")
        print("  SCRIPT EXCEPTION")
        print("==========================================================")
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except Exception:
            pass
    finally:
        simulation_app.close()
