#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Isaac Lab 3.0 / Isaac Sim 6.0.1
Franka + cuMotion RMPFlow task-space verification.

Expected USD:
    /Franka                  <- existing Panda articulation
    /World/tighter_v1        <- sleeve / collar
    /World/part01            <- rod

This version only verifies stable task-space motion to a target pose.
It does not yet grasp or insert the sleeve.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka cuMotion RMPFlow verification")
parser.add_argument("--usd", type=str,
                    default="/home/yh/tianji/mission_docs/mission01_v1/World0.usd")
parser.add_argument("--target-part", type=str,
                    default="/World/katao/katao",
                    choices=["/World/tighter_v1", "/World/part01"])
parser.add_argument("--offset", type=float, nargs=3,
                    default=(0.0, 0.0, 0.10),
                    metavar=("DX", "DY", "DZ"),
                    help="EE offset in target-part local frame, meters.")
parser.add_argument("--target-quat", type=float, nargs=4, default=None,
                    metavar=("QX", "QY", "QZ", "QW"))
parser.add_argument("--gripper", type=str, default="open",
                    choices=["open", "close"])
parser.add_argument("--print-every", type=int, default=50)
parser.add_argument("--target-tolerance", type=float, default=0.015)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from isaacsim.robot_motion.cumotion import RmpFlowController


@configclass
class SceneCfg(InteractiveSceneCfg):
    # The existing Panda is /Franka, not /World/Franka.
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/Franka",
        spawn=None,
    )


def get_prim_world_pose(prim_path: str):
    """Return USD prim world pose as position + quaternion (x,y,z,w)."""
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"USD prim does not exist: {prim_path}")

    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    p = matrix.ExtractTranslation()
    q = matrix.ExtractRotationQuat()

    pos = torch.tensor([float(p[0]), float(p[1]), float(p[2])],
                       dtype=torch.float32)
    quat = torch.tensor([
        float(q.GetImaginary()[0]),
        float(q.GetImaginary()[1]),
        float(q.GetImaginary()[2]),
        float(q.GetReal()),
    ], dtype=torch.float32)
    quat /= torch.linalg.norm(quat)
    return pos, quat


def get_panda_indices(robot):
    arm_ids = []
    for i, name in enumerate(robot.joint_names):
        if name.startswith("panda_joint"):
            suffix = name[len("panda_joint"):]
            if suffix.isdigit():
                arm_ids.append(i)
    arm_ids.sort(key=lambda i: int(robot.joint_names[i][len("panda_joint"):]))
    if len(arm_ids) != 7:
        raise RuntimeError(
            f"Expected 7 Panda arm joints, got {len(arm_ids)}: "
            f"{[robot.joint_names[i] for i in arm_ids]}"
        )

    finger_ids = [
        i for i, name in enumerate(robot.joint_names)
        if name in ("panda_finger_joint1", "panda_finger_joint2")
    ]
    hand_ids = [
        i for i, name in enumerate(robot.body_names)
        if name == "panda_hand"
    ]
    if len(hand_ids) != 1:
        raise RuntimeError(
            f"Expected one panda_hand body, got {hand_ids}. "
            f"Bodies={robot.body_names}"
        )
    return arm_ids, finger_ids, hand_ids[0]


def create_controller():
    """
    Create the Isaac Sim 6.0.1 cuMotion RMPFlow controller.

    The exact constructor can vary between extension builds. We try the
    two common Franka forms and otherwise stop with an API probe command.
    """
    try:
        return RmpFlowController(
            robot_description="franka",
            robot_prim_path="/Franka",
        )
    except TypeError:
        try:
            return RmpFlowController(
                robot_name="franka",
                robot_prim_path="/Franka",
            )
        except TypeError as exc:
            raise RuntimeError(
                "RmpFlowController constructor differs in this installation.\n"
                "Run:\n"
                "  python /home/yh/IsaacLab/scripts/diytest/probe_rmpflow.py\n"
                "and send the output."
            ) from exc


def extract_joint_positions(action):
    if hasattr(action, "joint_positions"):
        return action.joint_positions
    if hasattr(action, "joint_position"):
        return action.joint_position
    if isinstance(action, dict):
        value = action.get("joint_positions",
                          action.get("joint_position"))
        if value is not None:
            return value
    return action


def main():
    print(f"[INFO] Opening USD: {args_cli.usd}")
    if sim_utils.open_stage(args_cli.usd) is False:
        raise RuntimeError(f"Failed to open USD: {args_cli.usd}")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    )
    sim.set_camera_view(
        eye=[1.8, 1.8, 1.5],
        target=[0.5, 0.0, 0.5],
    )

    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    arm_ids, finger_ids, ee_body_id = get_panda_indices(robot)

    print("\n================ FRANKA / USD CHECK ================")
    print("[INFO] Robot prim       : /Franka")
    print(f"[INFO] Target prim      : {args_cli.target_part}")
    print(f"[INFO] Fixed base       : {robot.is_fixed_base}")
    print(f"[INFO] Joints           : {robot.num_joints}")
    print(f"[INFO] Bodies           : {robot.num_bodies}")
    print(f"[INFO] Arm joints       : {[robot.joint_names[i] for i in arm_ids]}")
    print(f"[INFO] EE body          : {robot.body_names[ee_body_id]}")
    print(f"[INFO] Finger ids       : {finger_ids}")

    # -------------------------------------------------------------------------
    # Target = /World/tighter_v1 pose + local offset
    # -------------------------------------------------------------------------
    target_pos_w, target_quat_w = get_prim_world_pose(args_cli.target_part)

    if args_cli.target_quat is not None:
        target_quat_w = torch.tensor(
            args_cli.target_quat, dtype=torch.float32
        )
        target_quat_w /= torch.linalg.norm(target_quat_w)

    target_pos_w = target_pos_w.to(robot.device).reshape(1, 3)
    target_quat_w = target_quat_w.to(robot.device).reshape(1, 4)

    offset_pos = torch.tensor(
        args_cli.offset, dtype=torch.float32, device=robot.device
    ).reshape(1, 3)

    identity_quat = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=robot.device,
    )

    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        target_pos_w,
        target_quat_w,
        offset_pos,
        identity_quat,
    )

    print("\n================ TARGET POSE =======================")
    print("[INFO] Part position :",
          target_pos_w[0].detach().cpu().numpy())
    print("[INFO] Part quat     :",
          target_quat_w[0].detach().cpu().numpy(), "(x,y,z,w)")
    print("[INFO] EE target     :",
          ee_target_pos_w[0].detach().cpu().numpy())
    print("[INFO] EE target quat:",
          ee_target_quat_w[0].detach().cpu().numpy(), "(x,y,z,w)")

    # -------------------------------------------------------------------------
    # RMPFlow
    # -------------------------------------------------------------------------
    print("\n================ CUMOTION RMPFLOW ==================")
    controller = create_controller()
    if hasattr(controller, "reset"):
        controller.reset()
    print("[INFO] RmpFlowController created.")

    if args_cli.gripper == "open":
        gripper_target = torch.tensor(
            [[0.04, 0.04]], dtype=torch.float32, device=robot.device
        )
    else:
        gripper_target = torch.tensor(
            [[0.0, 0.0]], dtype=torch.float32, device=robot.device
        )

    print("\n================ EXECUTION =========================")
    print("[INFO] RMPFlow tracking only; no grasp/insertion yet.")

    count = 0
    while simulation_app.is_running():
        scene.update(sim.get_physics_dt())

        joint_pos = robot.data.joint_pos.torch[:, arm_ids]
        joint_vel = robot.data.joint_vel.torch[:, arm_ids]

        target_position = ee_target_pos_w[0].detach().cpu().numpy()
        target_orientation = ee_target_quat_w[0].detach().cpu().numpy()
        q = joint_pos[0].detach().cpu().numpy()
        dq = joint_vel[0].detach().cpu().numpy()

        action = None

        # First try the common cuMotion controller.forward API.
        if hasattr(controller, "forward"):
            for kwargs in (
                dict(
                    target_position=target_position,
                    target_orientation=target_orientation,
                    joint_positions=q,
                    joint_velocities=dq,
                ),
                dict(
                    target_position=target_position,
                    target_orientation=target_orientation,
                ),
            ):
                try:
                    action = controller.forward(**kwargs)
                    break
                except TypeError:
                    continue

        # Alternate API names used by some extension builds.
        if action is None and hasattr(controller, "set_target"):
            controller.set_target(
                target_position=target_position,
                target_orientation=target_orientation,
            )
            if hasattr(controller, "compute_action"):
                for kwargs in (
                    dict(joint_positions=q, joint_velocities=dq),
                    dict(),
                ):
                    try:
                        action = controller.compute_action(**kwargs)
                        break
                    except TypeError:
                        continue

        if action is None:
            raise RuntimeError(
                "Unable to call the installed RmpFlowController API.\n"
                "Run the API probe command shown below and send its output."
            )

        q_target_np = extract_joint_positions(action)
        if q_target_np is None:
            raise RuntimeError("RMPFlow action contains no joint positions.")

        q_target = torch.as_tensor(
            q_target_np, dtype=torch.float32, device=robot.device
        ).reshape(1, -1)

        if q_target.shape[1] != 7:
            raise RuntimeError(
                f"RMPFlow returned {q_target.shape[1]} joint values; expected 7."
            )

        robot.set_joint_position_target_index(
            target=q_target, joint_ids=arm_ids
        )

        if finger_ids:
            robot.set_joint_position_target_index(
                target=gripper_target, joint_ids=finger_ids
            )

        scene.write_data_to_sim()
        sim.step()

        if count % args_cli.print_every == 0:
            ee_pose_w = robot.data.body_pose_w.torch[:, ee_body_id]
            ee_pos_w = ee_pose_w[:, :3]
            error = torch.linalg.norm(
                ee_target_pos_w - ee_pos_w, dim=-1
            )[0].item()
            print(
                f"[STEP {count:5d}] EE error={error:.4f} m | "
                f"target={ee_target_pos_w[0].detach().cpu().numpy()} | "
                f"current={ee_pos_w[0].detach().cpu().numpy()}"
            )
            if error < args_cli.target_tolerance:
                print(
                    f"[INFO] Position target reached "
                    f"(<{args_cli.target_tolerance:.3f} m)."
                )

        count += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
