#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Isaac Lab 3.0 / Isaac Sim 6.0.1
Franka + cuMotion RMPFlow minimal pick/move verification (multi-env).

USD:
    /Franka
    /World/katao/katao
    /World/stick_part/stick_part

IMPORTANT:
    This version intentionally does NOT use WorldBinding or obstacle tracking.
    num_envs is supported by solving one independent RMPFlow controller per env.
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
    default="/home/yh/tianji/mission_docs/mission01_v2/World0.usd",
)
parser.add_argument(
    "--katao",
    type=str,
    default="/home/yh/tianji/mission_docs/mission01_v2/ihihi/katao02.usd",
)
parser.add_argument(
    "--stick_part",
    type=str,
    default="/home/yh/tianji/mission_docs/mission01_v2/ihihi/stick_part02.usd",
)
parser.add_argument(
    "--pregrasp-height",
    type=float,
    default=0.22,
)
parser.add_argument(
    "--grasp-height",
    type=float,
    default=0.086,
)
parser.add_argument(
    "--lift-height",
    type=float,
    default=0.20,
)
parser.add_argument(
    "--place-height",
    type=float,
    default=-0.085,
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
    default=(0.0, 1.0, 0.0, 0.0),
    # default=(0.0, 0.0, 0.7071, 0.7071),
    metavar=("QX", "QY", "QZ", "QW"),
    help="World-frame cuMotion tool orientation. Default is top-down [0,1,0,0].",
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
    "--dt",
    type=float,
    default=0.01,
)
parser.add_argument(
    "--tolerance",
    type=float,
    default=0.001,
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
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
    help="Number of replicated environments. Each env gets an independent cuMotion controller.",
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
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Usd

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg
# from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg, UsdPhysicsCollisionCfg, MassCfg
from isaaclab_physx.sim.schemas import PhysxSDFMeshPropertiesCfg


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
from isaacsim.robot_motion.cumotion import load_cumotion_robot


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------
@configclass
class SceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Reuse the Franka that already exists in the USD.
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    katao: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/katao",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.katao,
            # Rigid body
            rigid_props=RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),

            # Mass
            # mass_props=MassPropertiesCfg(
            #     mass=0.5,
            # ),

            # Collision
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.6, 0.037057),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    stick_part: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stick_part",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.stick_part,
            rigid_props=RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=False,
            ),

            # mass_props=MassPropertiesCfg(
            #     mass=0.1,
            # ),

            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.131),
            rot=(0.0, -0.707, 0.707, 0.0),
        ),
    )



# -----------------------------------------------------------------------------
# USD / robot helpers
# -----------------------------------------------------------------------------
def get_prim_world_pose(object01, env_idx=0):
    # katao_state = katao.data.root_state_w[0]
    object01_state = object01.data.root_link_pose_w[env_idx]

    position = (
        object01_state[:3]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    quat = (
        object01_state[3:7]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    ) # wxyz

    quat = np.array([
        quat[1],
        quat[2],
        quat[3],
        quat[0],
    ], dtype=np.float32) # xyzw

    quat /= np.linalg.norm(quat)
    print(f"[INFO] object01 pose: {position}, quat: {quat}")

    #####################测试区域#####################

    # print("root_lin_vel:",
    #     object01.data.root_link_vel_w[env_idx])

    # print("root_ang_vel:",
    #     object01.data.root_link_ang_vel_w[env_idx])

    # print("root_link_pose_w:",
    #     object01.data.root_link_pose_w[env_idx])

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


def get_ee_pose(robot, ee_body_id, env_idx=0):
    pose = robot.data.body_pose_w[env_idx, ee_body_id]
    # print(f"pose: {robot.data.body_names}")

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
def make_estimated_state(robot, joint_space, env_idx=0):
    # The joint-space definition is the full Isaac Lab joint_names list,
    # so the measured state must contain the corresponding full 9-DOF vector.
    q = (
        robot.data.joint_pos[env_idx]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    dq = (
        robot.data.joint_vel[env_idx]
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
    env_idx,
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

    # cuMotion is solved independently for each environment. Restrict this
    # command to env_idx instead of broadcasting env-0's command to all envs.
    env_ids = torch.tensor([env_idx], dtype=torch.long, device=robot.device)
    robot.set_joint_position_target_index(
        target=target,
        joint_ids=indices,
        env_ids=env_ids,
    )


def set_gripper(robot, finger_ids, value):
    if not finger_ids:
        return

    target = torch.full(
        (args_cli.num_envs, len(finger_ids)),
        float(value),
        dtype=torch.float32,
        device=robot.device,
    )

    robot.set_joint_position_target_index(
        target=target,
        joint_ids=finger_ids,
    )

def make_gripper_state(robot, finger_ids, move, sim, scene):
    open_steps = max(
        1,
        int(args_cli.grasp_wait / args_cli.dt),
    )

    for _ in range(open_steps):
        set_gripper(
            robot,
            finger_ids,
            move,
        )

        scene.write_data_to_sim()
        sim.step()
        scene.update(args_cli.dt)

    print("[OK] Gripper movement wait complete.")
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
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=args_cli.dt,
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
            num_envs=args_cli.num_envs,
            env_spacing=2.0,
        )
    )

    print("[INFO] Resetting simulation...")

    sim.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    katao = scene["katao"]
    stick_part = scene["stick_part"]

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
    katao_pos, katao_quat = get_prim_world_pose(katao, env_idx=0)
    stick_part_pos, stick_part_quat = get_prim_world_pose(stick_part, env_idx=0)

    initial_ee_pos, initial_ee_quat = get_ee_pose(
        robot,
        ee_body_id,
        env_idx=0,
    )

    print("\n================ TARGET READ ===============================")
    print(f"[INFO] num_envs   : {args_cli.num_envs}")
    print(f"[INFO] katao prim : {args_cli.katao}")
    print(f"[INFO] env0 katao : {katao_pos}")
    print(f"[INFO] env0 stick : {stick_part_pos}")
    print(f"[INFO] env0 EE    : {initial_ee_pos}")
    print(f"[INFO] env0 EE quat: {initial_ee_quat} (x,y,z,w)")

    # IMPORTANT:
    # Do NOT reuse the current panda_hand orientation here.
    # The previous version did that, which caused the gripper to approach
    # katao with the jaws/tool pointing in the wrong direction.
    #
    # For a top-down parallel-jaw grasp, command the cuMotion tool frame to
    # the downward orientation corresponding to Euler XYZ = [0, pi, 0].
    # Quaternion convention is (x, y, z, w):
    #       [0, 1, 0, 0]
    #
    # This is intentionally independent of katao's own orientation.
    target_quat = np.asarray(
        args_cli.ee_quat,
        dtype=np.float32,
    )
    target_quat /= np.linalg.norm(target_quat)

    print(
        f"[INFO] commanded EE quat : "
        f"{target_quat} (x,y,z,w)"
    )

    # -------------------------------------------------------------------------
    # cuMotion model + world
    # -------------------------------------------------------------------------
    print(
        "\n[STEP] Loading built-in cuMotion Franka configuration..."
    )

    # cumotion_robot = load_cumotion_supported_robot(
    #     "franka"
    # )

    cumotion_robot = load_cumotion_robot(
        directory="/home/yh/cumotion_robots/franka",
        urdf_filename="robot.urdf",
        xrdf_filename="robot.xrdf",
    )

    print(
        "[OK] load_cumotion_supported_robot('franka')"
    )

    # Isaac Lab 3.0 Articulation exposes joint_names rather than dof_names.
    # cuMotion expects the FULL ordered joint space, including the two
    # gripper joints. Its Franka XRDF marks the arm joints as active and
    # the gripper joints as fixed/non-controlled.
    robot_joint_space = list(robot.joint_names)
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

    # -------------------------------------------------------------------------
    # One cuMotion controller per environment.
    #
    # RmpFlowController keeps internal state, so controllers must not be
    # shared between replicated environments. Each env is solved independently.
    # -------------------------------------------------------------------------
    print(
        f"\n[STEP] Creating {args_cli.num_envs} CumotionWorldInterface/RmpFlowController pairs..."
    )

    controllers = []
    for env_idx in range(args_cli.num_envs):
        # NO WorldBinding in this smoke test.
        # The world can be empty while validating the controller chain.
        world_interface = CumotionWorldInterface()

        controller = RmpFlowController(
            cumotion_robot=cumotion_robot,
            cumotion_world_interface=world_interface,
            robot_joint_space=robot_joint_space,
            robot_site_space=robot_site_space,
            tool_frame=tool_frame,
        )

        cfg = controller.get_rmp_flow_config()
        cfg.set_param("cspace_target_rmp/metric_scalar", 1.0)
        cfg.set_param("collision_rmp/metric_scalar", 0.0)
        cfg.set_param("target_rmp/max_metric_scalar", 1000.0)

        controllers.append(controller)
        print(f"[OK] controller[{env_idx}] created.")

    print(f"[OK] Created {len(controllers)} independent cuMotion controllers.")

    # -------------------------------------------------------------------------
    # Targets
    # -------------------------------------------------------------------------
    # Every replicated environment has its own world-space object pose.
    # Never reuse env-0 coordinates for env-1..N.
    katao_positions = np.stack(
        [get_prim_world_pose(katao, env_idx=e)[0] for e in range(args_cli.num_envs)],
        axis=0,
    )
    stick_part_positions = np.stack(
        [get_prim_world_pose(stick_part, env_idx=e)[0] for e in range(args_cli.num_envs)],
        axis=0,
    )

    pregrasp = katao_positions + np.asarray(
        [0.0, 0.0, args_cli.pregrasp_height], dtype=np.float32
    )
    grasp = katao_positions + np.asarray(
        [0.0, 0.0, args_cli.grasp_height], dtype=np.float32
    )
    lift = grasp + np.asarray(
        [0.0, 0.0, args_cli.lift_height], dtype=np.float32
    )
    transport = stick_part_positions + np.asarray(
        [0.0, 0.0, args_cli.lift_height], dtype=np.float32
    )
    place = transport + np.asarray(
        [0.0, 0.0, args_cli.place_height], dtype=np.float32
    )

    print("\n================ MOTION TARGETS ============================")
    for e in range(args_cli.num_envs):
        print(
            f"[ENV {e}] PREGRASP={np.round(pregrasp[e], 4)} | "
            f"GRASP={np.round(grasp[e], 4)} | "
            f"LIFT={np.round(lift[e], 4)} | "
            f"TRANSPORT={np.round(transport[e], 4)} | "
            f"PLACE={np.round(place[e], 4)}"
        )

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
        target_positions,
        gripper_position,
    ):
        """Run one Cartesian phase independently for every environment."""
        print(
            "\n----------------------------------------------------------"
        )
        print(f"[PHASE] {phase_name}")

        target_positions = np.asarray(target_positions, dtype=np.float32)
        expected_shape = (args_cli.num_envs, 3)
        if target_positions.shape != expected_shape:
            raise ValueError(
                f"{phase_name}: expected target shape "
                f"{expected_shape}, got {target_positions.shape}"
            )

        # Reset each controller with the state/target belonging to that env.
        for env_idx, controller in enumerate(controllers):
            estimated = make_estimated_state(
                robot,
                robot_joint_space,
                env_idx=env_idx,
            )
            setpoint = make_setpoint_state(
                tool_frame,
                robot_site_space,
                target_positions[env_idx],
                target_quat,
            )

            print(
                f"[ENV {env_idx}] controller.reset() -> target="
                f"{np.round(target_positions[env_idx], 5)}"
            )
            reset_ok = controller.reset(
                estimated,
                setpoint,
                t=0.0,
            )
            if not reset_ok:
                raise RuntimeError(
                    f"cuMotion reset failed in {phase_name}, env={env_idx}"
                )

        t = 0.0
        max_error = float("inf")

        for step in range(timeout_steps):
            set_gripper(
                robot,
                finger_ids,
                gripper_position,
            )

            # Solve all environments independently, then advance physics once.
            for env_idx, controller in enumerate(controllers):
                estimated = make_estimated_state(
                    robot,
                    robot_joint_space,
                    env_idx=env_idx,
                )
                setpoint = make_setpoint_state(
                    tool_frame,
                    robot_site_space,
                    target_positions[env_idx],
                    target_quat,
                )

                desired = controller.forward(
                    estimated,
                    setpoint,
                    t,
                )
                if desired is None:
                    raise RuntimeError(
                        f"cuMotion returned None in {phase_name}, env={env_idx}"
                    )

                apply_cumotion_joint_target(
                    robot,
                    desired,
                    env_idx=env_idx,
                )

            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            t += dt

            errors = []
            for env_idx in range(args_cli.num_envs):
                ee_pos, _ = get_ee_pose(
                    robot,
                    ee_body_id,
                    env_idx=env_idx,
                )
                errors.append(
                    float(np.linalg.norm(
                        ee_pos - target_positions[env_idx]
                    ))
                )

            max_error = max(errors)

            if step % args_cli.print_every == 0:
                error_str = ", ".join(
                    f"e{e}={errors[e]:.4f}m"
                    for e in range(args_cli.num_envs)
                )
                print(
                    f"[STEP {step:4d}] max EE error = {max_error:.4f} m | "
                    f"{error_str}"
                )

            if max_error < args_cli.tolerance and step >= 20:
                print(
                    f"[OK] {phase_name} reached in all {args_cli.num_envs} envs. "
                    f"max EE error = {max_error:.4f} m"
                )
                return True

        print(
            f"[WARN] {phase_name} timeout. "
            f"max final EE error = {max_error:.4f} m"
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

    make_gripper_state(
        robot,
        finger_ids,
        args_cli.close,
        sim,
        scene,
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

    run_phase(
        "PLACE",
        place,
        args_cli.close,
    )

    make_gripper_state(
        robot,
        finger_ids,
        args_cli.open,
        sim,
        scene,
    )

    # -------------------------------------------------------------------------
    # Final verification
    # -------------------------------------------------------------------------
    final_ee_positions = []
    final_ee_errors = []
    final_katao_positions = []
    katao_displacements = []

    for env_idx in range(args_cli.num_envs):
        final_ee_pos, _ = get_ee_pose(
            robot,
            ee_body_id,
            env_idx=env_idx,
        )
        final_katao_pos, _ = get_prim_world_pose(
            katao,
            env_idx=env_idx,
        )

        final_ee_positions.append(final_ee_pos)
        final_ee_errors.append(
            float(np.linalg.norm(
                final_ee_pos - transport[env_idx]
            ))
        )
        final_katao_positions.append(final_katao_pos)
        katao_displacements.append(
            float(np.linalg.norm(
                final_katao_pos - katao_positions[env_idx]
            ))
        )

    print(
        "\n=========================================================="
    )
    print("  TEST FINISHED")
    print("==========================================================")

    for env_idx in range(args_cli.num_envs):
        print(
            f"[ENV {env_idx}] final EE={np.round(final_ee_positions[env_idx], 5)} | "
            f"target={np.round(transport[env_idx], 5)} | "
            f"error={final_ee_errors[env_idx]:.4f} m | "
            f"katao displacement={katao_displacements[env_idx]:.4f} m"
        )

    max_final_ee_error = max(final_ee_errors)
    all_reached = max_final_ee_error < args_cli.tolerance

    if all_reached:
        print(
            f"\n[SUCCESS] All {args_cli.num_envs} environments reached "
            "the transport target."
        )
    else:
        print(
            f"\n[FAIL] At least one environment missed the transport target. "
            f"max EE error={max_final_ee_error:.4f} m"
        )

    for env_idx, displacement in enumerate(katao_displacements):
        if displacement > 0.03:
            print(f"[ENV {env_idx}] katao moved with the gripper.")
        else:
            print(
                f"[ENV {env_idx}] katao did not move significantly. "
                "This is a grasp/contact issue unless the EE target itself also failed."
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
