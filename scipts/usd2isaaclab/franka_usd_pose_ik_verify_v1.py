#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Isaac Lab 3.0 / Isaac Sim 6.x
Quick verification:
1. Open an existing USD scene.
2. Reuse the Franka articulation already at /World/Franka.
3. Read the world pose of /World/tighter_v1 (card sleeve) or /World/part01.
4. Build an end-effector target pose from that object pose.
5. Use Isaac Lab DifferentialIKController (DLS) to compute Panda arm joint targets.
6. Execute the joint targets with the Franka position actuators.

IMPORTANT:
- This script assumes the USD already contains a valid Franka articulation at /World/Franka.
- The screenshot supplied by the user shows /World/Franka with panda_hand and
  panda_joint-style bodies, and a rootjoint PhysicsJoint.
- The script controls the 7 arm joints only; the gripper fingers are controlled
  separately with --gripper-open / --gripper-close.
- Isaac Lab's DifferentialIKController expects the target pose in the robot base frame.
"""

import argparse
import math

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Quick Franka pose -> IK verification")
parser.add_argument("--usd", type=str, default="/home/yh/tianji/mission_docs/simple_version2/World1.usd", help="Path to the USD scene.")
parser.add_argument(
    "--target-part",
    type=str,
    default="/World/tighter_v1",
    choices=["/World/tighter_v1", "/World/part01"],
    help="USD prim whose world pose is used as the target source.",
)
parser.add_argument(
    "--offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.10),
    metavar=("DX", "DY", "DZ"),
    help="Target EE position offset expressed in the target-part local frame, meters.",
)
parser.add_argument(
    "--target-quat",
    type=float,
    nargs=4,
    default=None,
    metavar=("QX", "QY", "QZ", "QW"),
    help="Optional fixed EE quaternion in WORLD frame. If omitted, use target-part rotation.",
)
parser.add_argument(
    "--gripper",
    type=str,
    default="open",
    choices=["open", "close"],
    help="Gripper command during the test.",
)
parser.add_argument(
    "--hold",
    type=int,
    default=300,
    help="Number of simulation steps to hold the target after reaching it.",
)
parser.add_argument(
    "--print-every",
    type=int,
    default=50,
    help="Print status every N simulation steps.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after AppLauncher
# -----------------------------------------------------------------------------
import torch
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# -----------------------------------------------------------------------------
# Scene configuration
# -----------------------------------------------------------------------------
@configclass
class SceneCfg(InteractiveSceneCfg):
    # IMPORTANT:
    # spawn=None means: do NOT create another Panda.
    # The Panda must already exist in the opened USD at /World/Franka.
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/Franka",
        spawn=None,
    )


# -----------------------------------------------------------------------------
# USD pose helper
# -----------------------------------------------------------------------------
def get_prim_world_pose(prim_path: str):
    """Return (position, quaternion_xyzw) from a USD prim."""
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise RuntimeError(f"USD prim does not exist: {prim_path}")

    xform_cache = UsdGeom.XformCache()
    matrix = xform_cache.GetLocalToWorldTransform(prim)
    transform = matrix

    # Gf.Matrix4d -> translation
    p = transform.ExtractTranslation()

    # Gf.Matrix4d -> quaternion
    q = transform.ExtractRotationQuat()

    # USD Gf quaternion is real + imaginary(x,y,z).
    # Isaac Lab uses (x,y,z,w).
    pos = torch.tensor(
        [float(p[0]), float(p[1]), float(p[2])],
        dtype=torch.float32,
    )
    quat = torch.tensor(
        [
            float(q.GetImaginary()[0]),
            float(q.GetImaginary()[1]),
            float(q.GetImaginary()[2]),
            float(q.GetReal()),
        ],
        dtype=torch.float32,
    )
    quat = quat / torch.linalg.norm(quat)

    return pos, quat


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Open the user's existing USD scene.
    print(f"[INFO] Opening USD: {args_cli.usd}")
    opened = sim_utils.open_stage(args_cli.usd)
    if opened is False:
        raise RuntimeError(f"Failed to open USD: {args_cli.usd}")

    # Simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Camera: adjust if needed.
    sim.set_camera_view(
        eye=[1.8, 1.8, 1.5],
        target=[0.5, 0.0, 0.5],
    )

    # One environment because we are controlling the user's existing USD stage.
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Start physics once so PhysX articulation handles / Jacobians are available.
    sim.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]

    print("\n================ FRANKA / USD CHECK ================")
    print(f"[INFO] Robot prim       : /World/Franka")
    print(f"[INFO] Target prim      : {args_cli.target_part}")
    print(f"[INFO] Is fixed base    : {robot.is_fixed_base}")
    print(f"[INFO] Number of joints : {robot.num_joints}")
    print(f"[INFO] Number of bodies : {robot.num_bodies}")
    print(f"[INFO] Joint names      : {robot.joint_names}")
    print(f"[INFO] Body names       : {robot.body_names}")

    # Resolve the 7 Panda arm joints and panda_hand.
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["panda_joint.*"],
        body_names=["panda_hand"],
    )
    robot_entity_cfg.resolve(scene)

    arm_joint_ids = robot_entity_cfg.joint_ids
    ee_body_id = robot_entity_cfg.body_ids[0]

    if len(arm_joint_ids) != 7:
        raise RuntimeError(
            f"Expected 7 Panda arm joints, got {len(arm_joint_ids)}: "
            f"{[robot.joint_names[i] for i in arm_joint_ids]}"
        )

    print(f"[INFO] Arm joint ids   : {arm_joint_ids}")
    print(
        "[INFO] Arm joint names : "
        f"{[robot.joint_names[i] for i in arm_joint_ids]}"
    )
    print(f"[INFO] EE body         : {robot.body_names[ee_body_id]}")

    # Fixed-base Jacobian index convention.
    # PhysX Jacobian does not contain the fixed root's 6 base DoFs.
    if robot.is_fixed_base:
        ee_jacobi_idx = ee_body_id - 1
    else:
        ee_jacobi_idx = ee_body_id

    # Differential IK controller.
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    ik_controller = DifferentialIKController(
        ik_cfg,
        num_envs=1,
        device=robot.device,
    )

    # -------------------------------------------------------------------------
    # Read target object pose from USD.
    # -------------------------------------------------------------------------
    target_pos_w, target_quat_w = get_prim_world_pose(args_cli.target_part)

    # Optional fixed orientation supplied by the user.
    if args_cli.target_quat is not None:
        target_quat_w = torch.tensor(
            args_cli.target_quat,
            dtype=torch.float32,
            device=robot.device,
        )
        target_quat_w /= torch.linalg.norm(target_quat_w)

    target_pos_w = target_pos_w.to(robot.device)
    target_quat_w = target_quat_w.to(robot.device)

    # Offset is expressed in the target-part local frame:
    # target EE = target part * local offset.
    offset_pos = torch.tensor(
        args_cli.offset,
        dtype=torch.float32,
        device=robot.device,
    ).reshape(1, 3)

    target_part_pos_w = target_pos_w.reshape(1, 3)
    target_part_quat_w = target_quat_w.reshape(1, 4)

    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        target_part_pos_w,
        target_part_quat_w,
        offset_pos,
        torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=robot.device,
        ),
    )

    print("\n================ TARGET POSE =======================")
    print(
        "[INFO] Target part world position : "
        f"{target_part_pos_w[0].detach().cpu().numpy()}"
    )
    print(
        "[INFO] Target part world quat     : "
        f"{target_part_quat_w[0].detach().cpu().numpy()}  (x,y,z,w)"
    )
    print(
        "[INFO] EE target world position   : "
        f"{ee_target_pos_w[0].detach().cpu().numpy()}"
    )
    print(
        "[INFO] EE target world quat       : "
        f"{ee_target_quat_w[0].detach().cpu().numpy()}  (x,y,z,w)"
    )

    # -------------------------------------------------------------------------
    # Read current robot state.
    # -------------------------------------------------------------------------
    root_pose_w = robot.data.root_pose_w.torch
    joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids].clone()

    # Current EE pose in world.
    ee_pose_w = robot.data.body_pose_w.torch[:, ee_body_id]
    ee_pos_w = ee_pose_w[:, 0:3]
    ee_quat_w = ee_pose_w[:, 3:7]

    # Convert current EE pose and target pose from WORLD -> ROBOT BASE.
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        ee_pos_w,
        ee_quat_w,
    )

    target_pos_b, target_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        ee_target_pos_w,
        ee_target_quat_w,
    )

    # Set absolute pose command in robot base frame.
    ik_command = torch.cat((target_pos_b, target_quat_b), dim=-1)
    ik_controller.set_command(ik_command)

    # -------------------------------------------------------------------------
    # Initial gripper target.
    # -------------------------------------------------------------------------
    # Panda finger joints are normally panda_finger_joint1/2 in Isaac Lab assets.
    gripper_cfg = SceneEntityCfg(
        "robot",
        joint_names=["panda_finger_joint1", "panda_finger_joint2"],
    )
    try:
        gripper_cfg.resolve(scene)
        gripper_joint_ids = gripper_cfg.joint_ids
    except Exception:
        gripper_joint_ids = []

    # Typical Panda finger targets in meters.
    if args_cli.gripper == "open":
        gripper_target = torch.tensor([[0.04, 0.04]], device=robot.device)
    else:
        gripper_target = torch.tensor([[0.0, 0.0]], device=robot.device)

    # -------------------------------------------------------------------------
    # Execution loop.
    # -------------------------------------------------------------------------
    print("\n================ EXECUTION =========================")
    print("[INFO] Starting IK tracking.")
    print("[INFO] Close the simulator window or Ctrl+C to stop.\n")

    count = 0

    while simulation_app.is_running():
        # PhysX Jacobian, expressed in world frame.
        # In Isaac Lab 3.0 the Jacobian columns include base DoFs first.
        jacobi_joint_ids = [
            j + robot.num_base_dofs for j in arm_joint_ids
        ]

        jacobian = robot.data.body_link_jacobian_w.torch[
            :,
            ee_jacobi_idx,
            :,
            jacobi_joint_ids,
        ]

        # Current EE pose.
        ee_pose_w = robot.data.body_pose_w.torch[:, ee_body_id]
        ee_pos_w = ee_pose_w[:, 0:3]
        ee_quat_w = ee_pose_w[:, 3:7]

        # Robot base pose.
        root_pose_w = robot.data.root_pose_w.torch

        # Current arm joint positions.
        joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]

        # Current EE pose in robot base frame.
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pos_w,
            ee_quat_w,
        )

        # Differential IK:
        # target EE pose -> desired 7 joint positions.
        joint_pos_des = ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            jacobian,
            joint_pos,
        )

        # Apply Panda arm position target.
        robot.set_joint_position_target_index(
            target=joint_pos_des,
            joint_ids=arm_joint_ids,
        )

        # Apply gripper target if finger joints were found.
        if gripper_joint_ids:
            robot.set_joint_position_target_index(
                target=gripper_target,
                joint_ids=gripper_joint_ids,
            )

        # Send commands to PhysX.
        scene.write_data_to_sim()

        # Step simulation.
        sim.step()

        # Update Isaac Lab buffers.
        scene.update(sim.get_physics_dt())

        if count % args_cli.print_every == 0:
            pos_error = torch.linalg.norm(
                ee_target_pos_w - ee_pos_w,
                dim=-1,
            )[0].item()

            print(
                f"[STEP {count:5d}] "
                f"EE position error = {pos_error:.4f} m | "
                f"target = {ee_target_pos_w[0].detach().cpu().numpy()} | "
                f"current = {ee_pos_w[0].detach().cpu().numpy()}"
            )

        count += 1

        # After hold steps, keep holding the pose.
        # This makes it easy to inspect the result manually in the viewport.
        if count >= args_cli.hold:
            # Continue holding the same target indefinitely.
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
