#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Isaac Lab 3.0 / Isaac Sim 6.0.1
Franka + cuMotion RMPFlow minimal pick/move verification.

USD:
    /Franka
    /World/katao/katao
    /World/stick_part/stick_part

IMPORTANT:
    This version intentionally does NOT use WorldBinding or obstacle tracking.
    The first goal is to isolate the cuMotion RMPFlow control chain itself.

Sequence:
    read katao pose
      -> open gripper
      -> pre-grasp
      -> grasp
      -> close gripper
      -> lift
      -> transport to an arbitrary pose
      -> hold

After the chain is verified, WorldBinding / collision avoidance / grasp
attachment can be added back one by one.
"""

import argparse
import traceback

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Minimal Franka cuMotion RMPFlow pick/move verification"
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
)
parser.add_argument(
    "--pregrasp-height",
    type=float,
    default=0.18,
)
parser.add_argument(
    "--grasp-height",
    type=float,
    default=0.06,
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
    default=0.03,
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
# Enable it explicitly in this process before importing the cuMotion Python API.
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
class SceneCfg(InteractiveSceneCfg):
    # Reuse the Franka that already exists in the USD.
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/Franka",
        spawn=None,
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


def get_ee_pose(robot, ee_body_id):
    pose = robot.data.body_pose_w[0, ee_body_id]

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


# -----------------------------------------------------------------------------
# cuMotion RobotState helpers
# -----------------------------------------------------------------------------
def make_estimated_state(robot, joint_space):
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

    # The cuMotion controller returns positions in its requested
    # robot_joint_space and explicitly returns the corresponding indices.
    #
    # Convert to torch here because Isaac Lab's Articulation command API
    # accepts torch tensors naturally.
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
    print("  FRANKA + cuMotion RMPFlow MINIMAL TEST")
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
        f"[INFO] initial EE : {initial_ee_pos}"
    )
    print(
        f"[INFO] initial EE quat : "
        f"{initial_ee_quat} (x,y,z,w)"
    )

    # Keep the current EE orientation for all phases.
    # This avoids introducing an additional orientation-control problem.
    target_quat = initial_ee_quat.copy()

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

    robot_joint_space = robot.dof_names
    robot_site_space = (
        cumotion_robot.robot_description.tool_frame_names()
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

    # NO WorldBinding in this smoke test.
    # The official RMPFlow controller requires a world interface, but the
    # world can be empty when we are only validating the controller chain.
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

    print("\n================ MOTION TARGETS ============================")
    print(f"[INFO] PREGRASP  : {pregrasp}")
    print(f"[INFO] GRASP     : {grasp}")
    print(f"[INFO] LIFT      : {lift}")
    print(f"[INFO] TRANSPORT : {transport}")

    # -------------------------------------------------------------------------
    # Warmup: leave the physics/articulation stable before calling cuMotion.
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
            f"{np.round(target_position, 5)}"
        )

        # Fresh controller reset at every major arm segment.
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

            ee_pos, _ = get_ee_pose(
                robot,
                ee_body_id,
            )

            error = float(
                np.linalg.norm(
                    ee_pos - target_position
                )
            )

            if step % args_cli.print_every == 0:
                print(
                    f"[STEP {step:4d}] "
                    f"EE error = {error:.4f} m | "
                    f"EE = {np.round(ee_pos, 4)}"
                )

            if (
                error < args_cli.tolerance
                and step >= 20
            ):
                print(
                    f"[OK] {phase_name} reached. "
                    f"EE error = {error:.4f} m"
                )
                return True

        ee_pos, _ = get_ee_pose(
            robot,
            ee_body_id,
        )
        error = float(
            np.linalg.norm(
                ee_pos - target_position
            )
        )

        print(
            f"[WARN] {phase_name} timeout. "
            f"Final EE error = {error:.4f} m"
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
    final_ee_pos, _ = get_ee_pose(
        robot,
        ee_body_id,
    )

    final_katao_pos, _ = get_prim_world_pose(
        args_cli.katao
    )

    final_ee_error = float(
        np.linalg.norm(
            final_ee_pos - transport
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
        "  TEST FINISHED"
    )
    print(
        "=========================================================="
    )
    print(
        f"[RESULT] final EE position : "
        f"{np.round(final_ee_pos, 5)}"
    )
    print(
        f"[RESULT] transport target  : "
        f"{np.round(transport, 5)}"
    )
    print(
        f"[RESULT] final EE error    : "
        f"{final_ee_error:.4f} m"
    )
    print(
        f"[RESULT] initial katao     : "
        f"{np.round(katao_pos, 5)}"
    )
    print(
        f"[RESULT] final katao       : "
        f"{np.round(final_katao_pos, 5)}"
    )
    print(
        f"[RESULT] katao displacement: "
        f"{katao_displacement:.4f} m"
    )

    if final_ee_error < args_cli.tolerance:
        print(
            "\n[SUCCESS] cuMotion RMPFlow control chain "
            "reached the transport target."
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
            "This is a grasp/contact issue unless the EE target "
            "itself also failed."
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

        # Do not immediately hide the exception behind a fast application
        # shutdown. Leave the Isaac Sim window open for inspection.
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except Exception:
            pass
    finally:
        simulation_app.close()
