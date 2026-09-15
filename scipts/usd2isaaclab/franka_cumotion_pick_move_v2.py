#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Isaac Lab 3.0 / Isaac Sim 6.0.1
Franka + cuMotion RMPFlow pick-and-move verification.

Expected USD:
    /Franka
    /World/stick_part/stick_part
    /World/katao/katao

Sequence:
    1) Read /World/katao/katao world pose.
    2) Open Panda gripper.
    3) cuMotion RMPFlow -> pre-grasp pose above katao.
    4) cuMotion RMPFlow -> grasp pose.
    5) Close Panda gripper and wait.
    6) cuMotion RMPFlow -> lift.
    7) cuMotion RMPFlow -> arbitrary transport pose.
    8) Hold.

This is a first-link validation. It deliberately does NOT create a fake
fixed joint between the gripper and katao. Therefore, if katao does not
follow the gripper, that is a USD contact/rigid-body/gripper-friction issue,
not proof that cuMotion failed.
"""

import argparse
from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Franka cuMotion RMPFlow pick-and-move verification"
)
parser.add_argument(
    "--usd",
    type=str,
    default="/home/yh/tianji/mission_docs/mission01_v1/World0.usd",
)
parser.add_argument(
    "--katao",
    type=str,
    default="/World/katao/katao",
    help="Prim whose world pose is read as the grasp target.",
)
parser.add_argument(
    "--stick",
    type=str,
    default="/World/stick_part/stick_part",
    help="Second part; optional cuMotion collision obstacle.",
)
parser.add_argument(
    "--pregrasp-height",
    type=float,
    default=0.15,
    help="EE Z offset above katao for pre-grasp [m].",
)
parser.add_argument(
    "--grasp-height",
    type=float,
    default=0.055,
    help="EE Z offset above katao for grasp [m]. Tune for your katao geometry.",
)
parser.add_argument(
    "--lift-height",
    type=float,
    default=0.18,
    help="Extra Z offset for lift [m].",
)
parser.add_argument(
    "--move-offset",
    type=float,
    nargs=3,
    default=(0.30, -0.20, 0.25),
    metavar=("DX", "DY", "DZ"),
    help="Final transport offset from katao initial world position [m].",
)
parser.add_argument(
    "--ee-quat",
    type=float,
    nargs=4,
    default=None,
    metavar=("QX", "QY", "QZ", "QW"),
    help="Fixed EE orientation in world frame. Default: current panda_hand orientation.",
)
parser.add_argument("--open", type=float, default=0.04)
parser.add_argument("--close", type=float, default=0.0)
parser.add_argument(
    "--grasp-wait",
    type=float,
    default=1.0,
    help="Seconds to hold closed gripper before lift.",
)
parser.add_argument("--tolerance", type=float, default=0.025)
parser.add_argument("--phase-timeout", type=float, default=8.0)
parser.add_argument("--print-every", type=int, default=30)

parser.add_argument(
    "--with-obstacle",
    action="store_true",
    help="Add /World/stick_part/stick_part to cuMotion collision world.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Enable cuMotion before importing its Python modules.
# This also makes the script work without relying on --kit_args.
# -----------------------------------------------------------------------------
import omni.kit.app

_ext_mgr = omni.kit.app.get_app().get_extension_manager()
if not _ext_mgr.is_extension_enabled("isaacsim.robot_motion.cumotion"):
    if not _ext_mgr.set_extension_enabled_immediate(
        "isaacsim.robot_motion.cumotion", True
    ):
        raise RuntimeError(
            "Failed to enable isaacsim.robot_motion.cumotion"
        )

simulation_app.update()


# -----------------------------------------------------------------------------
# Isaac / cuMotion imports -- after SimulationApp and extension enable.
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
class SceneCfg(InteractiveSceneCfg):
    # Reuse the Franka already present in the USD.
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/Franka",
        spawn=None,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_prim_world_pose(prim_path: str):
    """Read USD prim world pose as position + quaternion (x,y,z,w)."""
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"USD prim does not exist: {prim_path}")

    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    p = matrix.ExtractTranslation()
    q = matrix.ExtractRotationQuat()

    pos = np.array(
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
    quat /= np.linalg.norm(quat)
    return pos, quat


def get_panda_indices(robot):
    """Resolve 7 arm joints, two fingers and panda_hand body."""
    arm_ids = []
    for i, name in enumerate(robot.joint_names):
        if name.startswith("panda_joint"):
            suffix = name[len("panda_joint") :]
            if suffix.isdigit():
                arm_ids.append(i)

    arm_ids.sort(
        key=lambda i: int(
            robot.joint_names[i][len("panda_joint") :]
        )
    )

    if len(arm_ids) != 7:
        raise RuntimeError(
            f"Expected 7 Panda arm joints, got {len(arm_ids)}: "
            f"{[robot.joint_names[i] for i in arm_ids]}"
        )

    finger_ids = [
        i
        for i, name in enumerate(robot.joint_names)
        if name in ("panda_finger_joint1", "panda_finger_joint2")
    ]

    hand_ids = [
        i
        for i, name in enumerate(robot.body_names)
        if name == "panda_hand"
    ]
    if len(hand_ids) != 1:
        raise RuntimeError(
            f"Expected one panda_hand body, got {hand_ids}. "
            f"Bodies={robot.body_names}"
        )

    return arm_ids, finger_ids, hand_ids[0]


def robot_root_world_pose_np(robot):
    """Isaac Lab root pose -> numpy arrays for cuMotion WorldBinding."""
    pos = (
        robot.data.root_pos_w[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    quat = (
        robot.data.root_quat_w[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return pos[None, :], quat[None, :]


def get_ee_pose(robot, ee_body_id):
    """Current panda_hand world pose."""
    pose = robot.data.body_pose_w[0, ee_body_id]
    pos = pose[:3].detach().cpu().numpy().astype(np.float32)
    quat = pose[3:7].detach().cpu().numpy().astype(np.float32)
    quat /= np.linalg.norm(quat)
    return pos, quat


def make_estimated_state(robot, robot_joint_space):
    """Build current RobotState for RMPFlow."""
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
            robot_joint_space=robot_joint_space,
            positions=(robot_joint_space, wp.array(q, dtype=wp.float32)),
            velocities=(robot_joint_space, wp.array(dq, dtype=wp.float32)),
        )
    )


def make_setpoint_state(tool_frame, site_space, position, orientation):
    """Build task-space target for the selected cuMotion tool frame."""
    p = np.asarray(position, dtype=np.float32).reshape(1, 3)
    q = np.asarray(orientation, dtype=np.float32).reshape(1, 4)

    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=site_space,
            positions=(
                [tool_frame],
                wp.array(p, dtype=wp.float32),
            ),
            orientations=(
                [tool_frame],
                wp.array(q, dtype=wp.float32),
            ),
        )
    )


def set_gripper(robot, finger_ids, target):
    """Set both Panda finger joint position targets."""
    if not finger_ids:
        return

    target_tensor = torch.full(
        (1, len(finger_ids)),
        float(target),
        dtype=torch.float32,
        device=robot.device,
    )
    robot.set_joint_position_target_index(
        target=target_tensor,
        joint_ids=finger_ids,
    )


def apply_cumotion_target(robot, desired):
    """Apply cuMotion's own joint ordering/indices to Isaac Lab."""
    if desired is None or desired.joints is None:
        raise RuntimeError("cuMotion returned no joint state.")

    if desired.joints.positions is None:
        raise RuntimeError("cuMotion returned no joint position target.")

    # Isaac Lab 3.0 accepts Warp arrays and Warp joint indices here.
    robot.set_joint_position_target_index(
        target=desired.joints.positions,
        joint_ids=desired.joints.position_indices,
    )


def phase_reached(robot, ee_body_id, target):
    ee_pos, _ = get_ee_pose(robot, ee_body_id)
    error = float(np.linalg.norm(ee_pos - target))
    return error, error < args_cli.tolerance


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("\n============================================================")
    print(" Franka + cuMotion RMPFlow PICK / MOVE verification")
    print("============================================================")
    print(f"[INFO] USD    : {args_cli.usd}")
    print("[INFO] Franka : /Franka")
    print(f"[INFO] Katao  : {args_cli.katao}")
    print(f"[INFO] Stick  : {args_cli.stick}")

    if sim_utils.open_stage(args_cli.usd) is False:
        raise RuntimeError(f"Failed to open USD: {args_cli.usd}")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    )
    sim.set_camera_view(
        eye=[1.8, 1.8, 1.4],
        target=[0.45, 0.0, 0.45],
    )

    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    arm_ids, finger_ids, ee_body_id = get_panda_indices(robot)

    print("\n================ FRANKA CHECK =============================")
    print(f"[INFO] DOFs       : {robot.num_joints}")
    print(f"[INFO] Bodies     : {robot.num_bodies}")
    print(
        f"[INFO] Arm joints : "
        f"{[robot.joint_names[i] for i in arm_ids]}"
    )
    print(f"[INFO] Finger ids : {finger_ids}")
    print(f"[INFO] EE body    : {robot.body_names[ee_body_id]}")

    # -------------------------------------------------------------------------
    # Read katao pose from the USD.
    # -------------------------------------------------------------------------
    katao_pos, katao_quat = get_prim_world_pose(args_cli.katao)
    ee_initial_pos, ee_initial_quat = get_ee_pose(robot, ee_body_id)

    if args_cli.ee_quat is None:
        ee_quat = ee_initial_quat.copy()
    else:
        ee_quat = np.asarray(args_cli.ee_quat, dtype=np.float32)
        ee_quat /= np.linalg.norm(ee_quat)

    print("\n================ TARGET READ ===============================")
    print(f"[INFO] katao position : {katao_pos}")
    print(f"[INFO] katao quat     : {katao_quat} (x,y,z,w)")
    print(f"[INFO] initial EE     : {ee_initial_pos}")
    print(f"[INFO] EE orientation : {ee_quat} (x,y,z,w)")

    # -------------------------------------------------------------------------
    # cuMotion collision world.
    #
    # katao is excluded because the gripper must approach it.
    # The optional stick part is treated as an OBB obstacle.
    # -------------------------------------------------------------------------
    robot_pos_w, robot_quat_w = robot_root_world_pose_np(robot)

    if args_cli.with_obstacle:
        stage = sim_utils.get_current_stage()
        stick_prim = stage.GetPrimAtPath(args_cli.stick)
        if not stick_prim.IsValid():
            raise RuntimeError(
                f"--with-obstacle requested, but stick prim does not exist: "
                f"{args_cli.stick}"
            )

        tracked_objects = [args_cli.stick]
        obstacle_strategy = mg.ObstacleStrategy()
        obstacle_strategy.set_configuration_overrides(
            {
                args_cli.stick: mg.ObstacleConfiguration(
                    "obb", 0.01
                )
            }
        )

        print(
            f"[INFO] cuMotion obstacle enabled: {args_cli.stick}"
        )
    else:
        tracked_objects = []
        obstacle_strategy = mg.ObstacleStrategy()
        print("[INFO] cuMotion obstacle world disabled for smoke test.")

    world_binding = mg.WorldBinding(
        world_interface=CumotionWorldInterface(),
        obstacle_strategy=obstacle_strategy,
        tracked_prims=tracked_objects,
        tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
    )
    world_binding.initialize()
    world_binding.get_world_interface().update_world_to_robot_root_transforms(
        poses=(robot_pos_w, robot_quat_w)
    )
    world_binding.synchronize_transforms()

    # -------------------------------------------------------------------------
    # Official built-in Franka cuMotion configuration.
    # -------------------------------------------------------------------------
    cumotion_robot = load_cumotion_supported_robot("franka")
    site_space = cumotion_robot.robot_description.tool_frame_names()
    if not site_space:
        raise RuntimeError(
            "Built-in cuMotion Franka configuration has no tool frame."
        )

    tool_frame = site_space[0]
    robot_joint_space = robot.dof_names

    print("\n================ CUMOTION MODEL ============================")
    print(f"[INFO] cuMotion robot      : franka")
    print(
        f"[INFO] controlled joints   : "
        f"{cumotion_robot.controlled_joint_names}"
    )
    print(f"[INFO] tool frames         : {site_space}")
    print(f"[INFO] selected tool frame : {tool_frame}")

    controller = RmpFlowController(
        cumotion_robot=cumotion_robot,
        cumotion_world_interface=world_binding.get_world_interface(),
        robot_joint_space=robot_joint_space,
        robot_site_space=site_space,
        tool_frame=tool_frame,
    )

    # Reduce the posture/c-space pull toward the initial joint state.
    controller.get_rmp_flow_config().set_param(
        "cspace_target_rmp/metric_scalar", 1.0
    )

    # -------------------------------------------------------------------------
    # Targets. All are derived from katao's READ world position.
    # -------------------------------------------------------------------------
    pregrasp = katao_pos + np.array(
        [0.0, 0.0, args_cli.pregrasp_height],
        dtype=np.float32,
    )
    grasp = katao_pos + np.array(
        [0.0, 0.0, args_cli.grasp_height],
        dtype=np.float32,
    )
    lift = grasp + np.array(
        [0.0, 0.0, args_cli.lift_height],
        dtype=np.float32,
    )
    transport = katao_pos + np.asarray(
        args_cli.move_offset,
        dtype=np.float32,
    )

    print("\n================ TASK TARGETS ==============================")
    print(f"[INFO] pregrasp : {pregrasp}")
    print(f"[INFO] grasp    : {grasp}")
    print(f"[INFO] lift     : {lift}")
    print(f"[INFO] transport: {transport}")

    # -------------------------------------------------------------------------
    # First let the gripper reach the requested OPEN target.
    # -------------------------------------------------------------------------
    dt = sim.get_physics_dt()
    for _ in range(60):
        set_gripper(robot, finger_ids, args_cli.open)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

    # Each arm phase gets a fresh RMPFlow reset from the actual current robot
    # state. This avoids carrying a stale internal setpoint across a large
    # phase transition.
    motion_phases = [
        ("PREGRASP", pregrasp, args_cli.open),
        ("GRASP_APPROACH", grasp, args_cli.open),
        ("LIFT", lift, args_cli.close),
        ("TRANSPORT", transport, args_cli.close),
    ]

    timeout_steps = max(
        1,
        int(args_cli.phase_timeout / dt),
    )

    print("\n================ EXECUTION ================================")

    for phase_name, target, gripper_target in motion_phases:
        print(f"\n[PHASE] {phase_name}")

        set_gripper(robot, finger_ids, gripper_target)

        estimated = make_estimated_state(robot, robot_joint_space)
        setpoint = make_setpoint_state(
            tool_frame,
            site_space,
            target,
            ee_quat,
        )

        if not controller.reset(
            estimated,
            setpoint,
            t=0.0,
        ):
            raise RuntimeError(
                f"RmpFlowController.reset() failed in {phase_name}"
            )

        print(f"[INFO] RMPFlow reset OK -> {target}")

        controller_time = 0.0
        reached = False

        for step in range(timeout_steps):
            set_gripper(robot, finger_ids, gripper_target)

            robot_pos_w, robot_quat_w = robot_root_world_pose_np(robot)
            world_binding.get_world_interface().update_world_to_robot_root_transforms(
                poses=(robot_pos_w, robot_quat_w)
            )
            world_binding.synchronize_transforms()

            estimated = make_estimated_state(
                robot,
                robot_joint_space,
            )
            setpoint = make_setpoint_state(
                tool_frame,
                site_space,
                target,
                ee_quat,
            )

            desired = controller.forward(
                estimated,
                setpoint,
                controller_time,
            )
            apply_cumotion_target(robot, desired)

            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

            controller_time += dt

            error, ok = phase_reached(
                robot,
                ee_body_id,
                target,
            )

            if step % args_cli.print_every == 0:
                ee_pos, _ = get_ee_pose(
                    robot,
                    ee_body_id,
                )
                print(
                    f"[STEP {step:4d}] "
                    f"{phase_name:16s} "
                    f"EEerr={error:.4f} m "
                    f"EE={np.round(ee_pos, 4)} "
                    f"T={np.round(target, 4)}"
                )

            if ok and step >= 20:
                print(
                    f"[OK] {phase_name} reached, "
                    f"EE error={error:.4f} m"
                )
                reached = True
                break

        if not reached:
            ee_pos, _ = get_ee_pose(robot, ee_body_id)
            error = float(np.linalg.norm(ee_pos - target))
            print(
                f"[WARN] {phase_name} timed out after "
                f"{args_cli.phase_timeout:.2f}s; "
                f"EE error={error:.4f} m"
            )

        # Grasp phase: close and hold before lifting.
        if phase_name == "GRASP_APPROACH":
            print(
                f"[PHASE] CLOSE_GRIPPER "
                f"(holding for {args_cli.grasp_wait:.2f}s)"
            )
            wait_steps = max(
                1,
                int(args_cli.grasp_wait / dt),
            )
            for _ in range(wait_steps):
                set_gripper(
                    robot,
                    finger_ids,
                    args_cli.close,
                )
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)

    # -------------------------------------------------------------------------
    # Final result.
    # -------------------------------------------------------------------------
    final_ee, _ = get_ee_pose(robot, ee_body_id)
    final_katao, _ = get_prim_world_pose(args_cli.katao)

    print("\n============================================================")
    print(" TEST SEQUENCE FINISHED")
    print("============================================================")
    print(f"[RESULT] final EE        : {final_ee}")
    print(f"[RESULT] transport goal  : {transport}")
    print(
        f"[RESULT] final EE error  : "
        f"{np.linalg.norm(final_ee - transport):.4f} m"
    )
    print(f"[RESULT] final katao     : {final_katao}")

    displacement = float(np.linalg.norm(final_katao - katao_pos))
    print(
        f"[RESULT] katao displacement: "
        f"{displacement:.4f} m"
    )

    if displacement > 0.03:
        print(
            "[RESULT] katao moved with the robot. "
            "Physical grasp/contact appears to be working."
        )
    else:
        print(
            "[RESULT] katao did not move significantly. "
            "cuMotion can still be considered verified if the EE reached "
            "the transport target; the remaining issue is the physical "
            "grasp/contact setup of katao."
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
