#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Isaac Lab 3.0 / Isaac Sim 6.0.1
Franka + cuMotion RMPFlow multi-env independent pick/place verification
+ JointWrenchSensor -> ROS 2

IMPORTANT:
    Each replicated environment owns an independent task state machine.

    Physics is still advanced synchronously for all environments with one
    sim.step() per simulation frame, but task phases are asynchronous:

        ENV 0: PREGRASP -> GRASP -> CLOSE -> LIFT -> ...
        ENV 1: PREGRASP -> GRASP -> CLOSE -> ...
        ENV 2: PREGRASP -> ...
        ENV 3: ...

    One environment never waits for another environment to finish its
    current phase.

    Each environment also has its own JointWrenchSensor measurement.

ROS 2:
    /franka/env_0/joint_wrench
    /franka/env_1/joint_wrench
    /franka/env_2/joint_wrench
    /franka/env_3/joint_wrench

    Each topic publishes the wrench measured at panda_link7 for the
    corresponding environment.

USD:
    /Franka
    /World/katao/katao
    /World/stick_part/stick_part

Sequence per environment:
    open gripper
      -> pre-grasp
      -> grasp
      -> close gripper
      -> lift
      -> transport
      -> place
      -> open gripper
      -> done

cuMotion:
    One independent RmpFlowController is created for each environment.

Coordinate system:
    Isaac Lab reports root_link_pose_w in global /World coordinates.
    cuMotion operates on the local robot coordinate frame.
    Therefore every Cartesian target is converted using:

        p_env = p_world - scene.env_origins[env_idx]

JointWrenchSensor:
    The sensor is attached to each replicated Franka robot and panda_link7
    is selected as the sensor body.

    Isaac Lab JointWrenchSensor convention used here:
        force  : incoming joint reaction force [N]
        torque : incoming joint reaction torque [N*m]
        frame  : child-side incoming joint frame
        point  : child-side joint anchor
"""


import argparse
import traceback
from enum import IntEnum

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description=(
        "Franka cuMotion RMPFlow independent multi-env pick/place "
        "with JointWrenchSensor ROS 2 publishing"
    )
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
    default=0.088,
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
    default=(0.0, 0.8660254, 0.5, 0.0),
    # default=(0.0, 1.0, 0.0, 0.0),
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
    help=(
        "Number of replicated environments. "
        "Each environment gets an independent cuMotion controller, "
        "task state, and JointWrenchSensor measurement."
    ),
)

# -----------------------------------------------------------------------------
# ROS 2 options
# -----------------------------------------------------------------------------

parser.add_argument(
    "--ros_rate",
    type=float,
    default=100.0,
    help="ROS 2 wrench publishing rate [Hz].",
)

parser.add_argument(
    "--wrench_topic_prefix",
    type=str,
    default="/franka",
    help=(
        "ROS 2 topic prefix. For example, with the default value, "
        "env0 publishes to /franka/env_0/joint_wrench."
    ),
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

import rclpy

from geometry_msgs.msg import WrenchStamped

from pxr import UsdGeom, UsdPhysics, PhysxSchema, Usd

import isaaclab.sim as sim_utils

from isaaclab.assets import (
    ArticulationCfg,
    RigidObjectCfg,
    AssetBaseCfg,
)

from isaaclab.scene import (
    InteractiveScene,
    InteractiveSceneCfg,
)

from isaaclab.sensors import (
    JointWrenchSensorCfg,
)

from isaaclab.utils.configclass import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from isaaclab.sim.schemas import (
    RigidBodyPropertiesCfg,
    MassPropertiesCfg,
    CollisionPropertiesCfg,
)

from isaaclab_physx.sim.schemas import (
    PhysxSDFMeshPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
    PhysxCollisionPropertiesCfg,
)

from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg


# -----------------------------------------------------------------------------
# Enable cuMotion extension.
# -----------------------------------------------------------------------------

import omni.kit.app


_extension_manager = (
    omni.kit.app.get_app().get_extension_manager()
)

if not _extension_manager.is_extension_enabled(
    "isaacsim.robot_motion.cumotion"
):

    result = _extension_manager.set_extension_enabled_immediate(
        "isaacsim.robot_motion.cumotion",
        True,
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

from isaacsim.robot_motion.cumotion import (
    load_cumotion_robot,
)


# -----------------------------------------------------------------------------
# ROS 2 publisher
# -----------------------------------------------------------------------------

class WrenchPublisher:
    """
    ROS 2 publisher for one JointWrenchSensor environment.

    One WrenchPublisher instance owns one ROS topic.

    Example:
        /franka/env_0/joint_wrench
        /franka/env_1/joint_wrench
    """

    def __init__(
        self,
        topic_prefix,
        env_idx,
    ):

        if not rclpy.ok():
            rclpy.init(args=None)

        self.env_idx = env_idx

        self.topic = (
            f"{topic_prefix.rstrip('/')}"
            f"/env_{env_idx}"
            f"/joint_wrench"
        )

        # All environments can share one ROS node.
        # The node is created lazily on the first publisher.
        global _ros_node

        if _ros_node is None:
            _ros_node = rclpy.create_node(
                "isaaclab_joint_wrench_publisher"
            )

        self.node = _ros_node

        self.publisher = self.node.create_publisher(
            WrenchStamped,
            self.topic,
            10,
        )

        print(
            f"[ROS 2] Publisher created: {self.topic}"
        )

    def publish(
        self,
        force_xyz,
        torque_xyz,
        frame_id,
    ):

        msg = WrenchStamped()

        msg.header.stamp = (
            self.node
            .get_clock()
            .now()
            .to_msg()
        )

        msg.header.frame_id = frame_id

        force = [
            float(x)
            for x in force_xyz.detach().cpu()
        ]

        torque = [
            float(x)
            for x in torque_xyz.detach().cpu()
        ]

        msg.wrench.force.x = force[0]
        msg.wrench.force.y = force[1]
        msg.wrench.force.z = force[2]

        msg.wrench.torque.x = torque[0]
        msg.wrench.torque.y = torque[1]
        msg.wrench.torque.z = torque[2]

        self.publisher.publish(
            msg
        )


    def destroy(self):

        if getattr(
            self,
            "publisher",
            None,
        ) is not None:

            self.node.destroy_publisher(
                self.publisher
            )

            self.publisher = None


_ros_node = None


def ros_spin_some():

    global _ros_node

    if (
        _ros_node is not None
        and rclpy.ok()
    ):

        rclpy.spin_once(
            _ros_node,
            timeout_sec=0.0,
        )


def ros_shutdown():

    global _ros_node

    if _ros_node is not None:

        try:
            _ros_node.destroy_node()
        except Exception:
            pass

        _ros_node = None

    if rclpy.ok():
        rclpy.shutdown()


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

@configclass
class SceneCfg(InteractiveSceneCfg):

    # Ground
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",

        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",

        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    # -------------------------------------------------------------------------
    # Franka
    # -------------------------------------------------------------------------

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",

        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    # -------------------------------------------------------------------------
    # Joint Wrench Sensor
    #
    # IMPORTANT:
    #
    # The sensor is replicated together with the robot because its prim path
    # also uses {ENV_REGEX_NS}.
    #
    # Each environment therefore gets:
    #
    #   /World/envs/env_0/Robot
    #   /World/envs/env_1/Robot
    #   ...
    #
    # and its corresponding JointWrenchSensor.
    # -------------------------------------------------------------------------

    joint_wrench = JointWrenchSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",

        update_period=0.0,

        debug_vis=True,
    )

    # -------------------------------------------------------------------------
    # Katao
    # -------------------------------------------------------------------------

    katao: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/katao",

        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.katao,

            # rigid_props=RigidBodyPropertiesCfg(
            #     rigid_body_enabled=True,
            #     kinematic_enabled=False,
            #     disable_gravity=False,
            # ),
            rigid_props=PhysxRigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,

                solver_position_iteration_count=32,
            ),

            # collision_props=CollisionPropertiesCfg(
            #     collision_enabled=True,
            # ),
            collision_props=PhysxCollisionPropertiesCfg(
                collision_enabled=True,

                contact_offset=0.001,
                rest_offset=0.000,
            ),

            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=1,
                dynamic_friction=0.8,
                restitution=0.0,
                friction_combine_mode="average",
            ),

        ),

        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.6, 0.037057),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # -------------------------------------------------------------------------
    # Stick part
    # -------------------------------------------------------------------------

    stick_part: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stick_part",

        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.stick_part,

            # rigid_props=RigidBodyPropertiesCfg(
            #     rigid_body_enabled=True,
            #     kinematic_enabled=True,
            #     disable_gravity=False,
            # ),

            # collision_props=CollisionPropertiesCfg(
            #     collision_enabled=True,
            # ),

            rigid_props=PhysxRigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=False,

                solver_position_iteration_count=32,
            ),

            collision_props=PhysxCollisionPropertiesCfg(
                collision_enabled=True,

                contact_offset=0.0008,
                rest_offset=0.000,
            ),

            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.4,
                restitution=0.0,
                friction_combine_mode="average",
            ),

        ),

        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.131),
            rot=(0.0, -0.707, 0.707, 0.0),
        ),
    )


# -----------------------------------------------------------------------------
# Task phases
# -----------------------------------------------------------------------------

class TaskPhase(IntEnum):

    PREGRASP = 0

    GRASP_APPROACH = 1

    CLOSE_GRIPPER = 2

    LIFT = 3

    TRANSPORT = 4

    PLACE = 5

    OPEN_GRIPPER = 6

    DONE = 7

    FAILED = 8


# -----------------------------------------------------------------------------
# USD / robot helpers
# -----------------------------------------------------------------------------

def get_env_origin(
    scene,
    env_idx=0,
):

    origin = (
        scene.env_origins[env_idx]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return origin


def get_prim_pose(
    object01,
    scene,
    env_idx=0,
):

    object01_state = (
        object01.data.root_link_pose_w[
            env_idx
        ]
    )

    world_position = (
        object01_state[:3]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    quat_wxyz = (
        object01_state[3:7]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    quat_xyzw = np.array(
        [
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0],
        ],
        dtype=np.float32,
    )

    quat_xyzw /= np.linalg.norm(
        quat_xyzw
    )

    env_origin = get_env_origin(
        scene,
        env_idx,
    )

    env_position = (
        world_position
        - env_origin
    )

    return (
        env_position,
        quat_xyzw,
        world_position,
        env_origin,
    )


def get_panda_indices(
    robot,
):

    arm_ids = []

    for i, name in enumerate(
        robot.joint_names
    ):

        if name.startswith(
            "panda_joint"
        ):

            suffix = name[
                len("panda_joint"):
            ]

            if suffix.isdigit():
                arm_ids.append(i)

    arm_ids.sort(
        key=lambda i: int(
            robot.joint_names[i][
                len("panda_joint"):
            ]
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
        for i, name in enumerate(
            robot.joint_names
        )
        if name in (
            "panda_finger_joint1",
            "panda_finger_joint2",
        )
    ]

    hand_ids = [
        i
        for i, name in enumerate(
            robot.body_names
        )
        if name == "panda_hand"
    ]

    if len(hand_ids) != 1:

        raise RuntimeError(
            "Could not uniquely find panda_hand: "
            f"{hand_ids}"
        )

    return (
        arm_ids,
        finger_ids,
        hand_ids[0],
    )


def get_ee_pose(
    robot,
    scene,
    ee_body_id,
    env_idx=0,
):

    pose = robot.data.body_pose_w[
        env_idx,
        ee_body_id,
    ]

    world_position = (
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

    quat /= np.linalg.norm(
        quat
    )

    env_position = (
        world_position
        - get_env_origin(
            scene,
            env_idx,
        )
    )

    return (
        env_position,
        quat,
        world_position,
    )


# -----------------------------------------------------------------------------
# cuMotion RobotState helpers
# -----------------------------------------------------------------------------

def make_estimated_state(
    robot,
    joint_space,
    env_idx=0,
):

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

                wp.array(
                    q,
                    dtype=wp.float32,
                ),
            ),

            velocities=(
                joint_space,

                wp.array(
                    dq,
                    dtype=wp.float32,
                ),
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
    ).reshape(
        1,
        3,
    )

    orientation = np.asarray(
        orientation,
        dtype=np.float32,
    ).reshape(
        1,
        4,
    )

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

    positions = (
        desired_state
        .joints
        .positions
    )

    if hasattr(
        positions,
        "numpy",
    ):

        positions_np = (
            positions.numpy()
        )

    else:

        positions_np = np.asarray(
            positions
        )

    positions_np = np.asarray(
        positions_np,
        dtype=np.float32,
    ).reshape(
        1,
        -1,
    )

    indices = (
        desired_state
        .joints
        .position_indices
    )

    if hasattr(
        indices,
        "numpy",
    ):

        indices = indices.numpy()

    indices = np.asarray(
        indices,
        dtype=np.int64,
    ).reshape(
        -1
    ).tolist()

    target = torch.as_tensor(
        positions_np,
        dtype=torch.float32,
        device=robot.device,
    )

    env_ids = torch.tensor(
        [env_idx],
        dtype=torch.long,
        device=robot.device,
    )

    robot.set_joint_position_target_index(
        target=target,
        joint_ids=indices,
        env_ids=env_ids,
    )


# -----------------------------------------------------------------------------
# Gripper helpers
# -----------------------------------------------------------------------------

def set_gripper_env(
    robot,
    finger_ids,
    value,
    env_idx,
):

    if not finger_ids:
        return

    target = torch.full(
        (
            1,
            len(finger_ids),
        ),

        float(value),

        dtype=torch.float32,

        device=robot.device,
    )

    env_ids = torch.tensor(
        [env_idx],
        dtype=torch.long,
        device=robot.device,
    )

    robot.set_joint_position_target_index(
        target=target,
        joint_ids=finger_ids,
        env_ids=env_ids,
    )


def set_gripper_all(
    robot,
    finger_ids,
    value,
    num_envs,
):

    if not finger_ids:
        return

    target = torch.full(
        (
            num_envs,
            len(finger_ids),
        ),

        float(value),

        dtype=torch.float32,

        device=robot.device,
    )

    robot.set_joint_position_target_index(
        target=target,
        joint_ids=finger_ids,
    )


# -----------------------------------------------------------------------------
# JointWrenchSensor helpers
# -----------------------------------------------------------------------------

def initialize_wrench_sensor(
    wrench_sensor,
):

    print(
        "\n================ JOINT WRENCH SENSOR ======================"
    )

    print(
        "[INFO] Sensor body names:"
    )

    for i, name in enumerate(
        wrench_sensor.body_names
    ):

        print(
            f"  body[{i:02d}] = {name}"
        )

    sensor_body_ids, sensor_body_names = (
        wrench_sensor.find_bodies(
            "panda_link7"
        )
    )

    if len(sensor_body_ids) != 1:

        raise RuntimeError(
            "JointWrenchSensor did not find exactly "
            f"one panda_link7: "
            f"names={sensor_body_names}, "
            f"ids={sensor_body_ids}"
        )

    sensor_body_id = int(
        sensor_body_ids[0]
    )

    print(
        "[OK] JointWrenchSensor body:"
        f" {wrench_sensor.body_names[sensor_body_id]}"
    )

    print(
        "[INFO] Sensor body id:"
        f" {sensor_body_id}"
    )

    print(
        "[INFO] JointWrenchSensor convention:"
    )

    print(
        "  force  : incoming joint reaction force [N]"
    )

    print(
        "  torque : incoming joint reaction torque [N*m]"
    )

    print(
        "  frame  : child-side incoming joint frame"
    )

    print(
        "  point  : child-side joint anchor"
    )

    print(
        "============================================================\n"
    )

    return sensor_body_id


def publish_wrench_for_all_envs(
    wrench_sensor,
    sensor_body_id,
    ros_publishers,
    sim_time,
    next_ros_time,
    ros_period,
):

    if sim_time + 1e-12 < next_ros_time:
        return next_ros_time

    for env_idx in range(
        len(ros_publishers)
    ):

        force = (
            wrench_sensor
            .data
            .force
            .torch[
                env_idx,
                sensor_body_id,
            ]
        )

        torque = (
            wrench_sensor
            .data
            .torque
            .torch[
                env_idx,
                sensor_body_id,
            ]
        )

        frame_id = (
            f"env_{env_idx}/panda_link7_joint_incoming"
        )

        ros_publishers[
            env_idx
        ].publish(
            force_xyz=force,
            torque_xyz=torque,
            frame_id=frame_id,
        )

    while next_ros_time <= sim_time:

        next_ros_time += ros_period

    return next_ros_time


def print_wrench_for_all_envs(
    wrench_sensor,
    sensor_body_id,
    phases,
):

    for env_idx in range(
        len(phases)
    ):

        force = (
            wrench_sensor
            .data
            .force
            .torch[
                env_idx,
                sensor_body_id,
            ]
        )

        torque = (
            wrench_sensor
            .data
            .torque
            .torch[
                env_idx,
                sensor_body_id,
            ]
        )

        print(
            f"[Wrench ENV {env_idx}] "
            f"phase={phases[env_idx].name} | "
            f"F=["
            f"{float(force[0]):9.3f}, "
            f"{float(force[1]):9.3f}, "
            f"{float(force[2]):9.3f}"
            f"] N | "
            f"T=["
            f"{float(torque[0]):9.3f}, "
            f"{float(torque[1]):9.3f}, "
            f"{float(torque[2]):9.3f}"
            f"] N*m"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    print(
        "\n=========================================================="
    )

    print(
        "  FRANKA + cuMotion RMPFlow MULTI-ENV TEST"
    )

    print(
        "  + JointWrenchSensor -> ROS 2"
    )

    print(
        "=========================================================="
    )

    print(
        "[INFO] Starting main()"
    )

    # -------------------------------------------------------------------------
    # Simulation
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

    # -------------------------------------------------------------------------
    # Scene
    # -------------------------------------------------------------------------

    print(
        "[INFO] Creating InteractiveScene..."
    )

    scene = InteractiveScene(
        SceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=2.0,
        )
    )

    print(
        "[INFO] Resetting simulation..."
    )

    sim.reset()

    scene.update(
        sim.get_physics_dt()
    )

    robot = scene["robot"]

    katao = scene["katao"]

    stick_part = scene["stick_part"]

    wrench_sensor = scene["joint_wrench"]

    # -------------------------------------------------------------------------
    # Joint wrench sensor
    # -------------------------------------------------------------------------

    sensor_body_id = (
        initialize_wrench_sensor(
            wrench_sensor
        )
    )

    # -------------------------------------------------------------------------
    # Franka indices
    # -------------------------------------------------------------------------

    arm_ids, finger_ids, ee_body_id = (
        get_panda_indices(
            robot
        )
    )

    print(
        "\n================ FRANKA CHECK ============================="
    )

    print(
        f"[INFO] DOFs       : {robot.num_joints}"
    )

    print(
        f"[INFO] Bodies     : {robot.num_bodies}"
    )

    print(
        "[INFO] Arm joints : "
        f"{[robot.joint_names[i] for i in arm_ids]}"
    )

    print(
        f"[INFO] Finger ids : {finger_ids}"
    )

    print(
        "[INFO] EE body    : "
        f"{robot.body_names[ee_body_id]}"
    )

    # -------------------------------------------------------------------------
    # ROS 2 publishers
    # -------------------------------------------------------------------------

    ros_publishers = []

    for env_idx in range(
        args_cli.num_envs
    ):

        publisher = WrenchPublisher(
            topic_prefix=args_cli.wrench_topic_prefix,
            env_idx=env_idx,
        )

        ros_publishers.append(
            publisher
        )

    ros_period = 1.0 / max(
        args_cli.ros_rate,
        1e-6,
    )

    next_ros_time = 0.0

    next_wrench_print_time = 0.0

    sim_time = 0.0

    print(
        "\n================ ROS 2 ===================================="
    )

    print(
        f"[INFO] ROS 2 wrench rate : "
        f"{args_cli.ros_rate:.1f} Hz"
    )

    print(
        "[INFO] Wrench topics:"
    )

    for env_idx in range(
        args_cli.num_envs
    ):

        print(
            f"  ENV {env_idx}: "
            f"{ros_publishers[env_idx].topic}"
        )

    print(
        "============================================================\n"
    )

    # -------------------------------------------------------------------------
    # Read object poses
    # -------------------------------------------------------------------------

    katao_positions = []

    katao_quats = []

    stick_part_positions = []

    stick_part_quats = []

    print(
        "\n================ TARGET READ ==============================="
    )

    print(
        f"[INFO] num_envs   : {args_cli.num_envs}"
    )

    print(
        f"[INFO] katao prim : {args_cli.katao}"
    )

    for env_idx in range(
        args_cli.num_envs
    ):

        (
            katao_pos_env,
            katao_quat,
            katao_pos_world,
            env_origin,
        ) = get_prim_pose(
            katao,
            scene,
            env_idx=env_idx,
        )

        (
            stick_pos_env,
            stick_quat,
            stick_pos_world,
            _,
        ) = get_prim_pose(
            stick_part,
            scene,
            env_idx=env_idx,
        )

        katao_positions.append(
            katao_pos_env
        )

        katao_quats.append(
            katao_quat
        )

        stick_part_positions.append(
            stick_pos_env
        )

        stick_part_quats.append(
            stick_quat
        )

        print(
            f"[ENV {env_idx}] "
            f"origin={np.round(env_origin, 5)} | "
            f"katao_world={np.round(katao_pos_world, 5)} -> "
            f"katao_env={np.round(katao_pos_env, 5)} | "
            f"stick_world={np.round(stick_pos_world, 5)} -> "
            f"stick_env={np.round(stick_pos_env, 5)}"
        )

    katao_positions = np.stack(
        katao_positions,
        axis=0,
    )

    katao_quats = np.stack(
        katao_quats,
        axis=0,
    )

    stick_part_positions = np.stack(
        stick_part_positions,
        axis=0,
    )

    stick_part_quats = np.stack(
        stick_part_quats,
        axis=0,
    )

    # -------------------------------------------------------------------------
    # Initial EE pose
    # -------------------------------------------------------------------------

    (
        initial_ee_pos_env,
        initial_ee_quat,
        initial_ee_pos_world,
    ) = get_ee_pose(
        robot,
        scene,
        ee_body_id,
        env_idx=0,
    )

    print(
        f"[INFO] env0 EE world : "
        f"{np.round(initial_ee_pos_world, 5)}"
    )

    print(
        f"[INFO] env0 EE local : "
        f"{np.round(initial_ee_pos_env, 5)}"
    )

    print(
        f"[INFO] env0 EE quat  : "
        f"{initial_ee_quat} (x,y,z,w)"
    )

    # -------------------------------------------------------------------------
    # Target EE orientation
    # -------------------------------------------------------------------------

    target_quat = np.asarray(
        args_cli.ee_quat,
        dtype=np.float32,
    )

    target_quat /= np.linalg.norm(
        target_quat
    )

    print(
        "[INFO] commanded EE quat : "
        f"{target_quat} (x,y,z,w)"
    )

    # -------------------------------------------------------------------------
    # cuMotion model
    # -------------------------------------------------------------------------

    print(
        "\n[STEP] Loading built-in cuMotion Franka configuration..."
    )

    cumotion_robot = load_cumotion_robot(
        directory="/home/yh/cumotion_robots/franka",
        urdf_filename="robot.urdf",
        xrdf_filename="robot.xrdf",
    )

    print(
        "[OK] cuMotion Franka configuration loaded."
    )

    robot_joint_space = list(
        robot.joint_names
    )

    robot_site_space = (
        cumotion_robot
        .robot_description
        .tool_frame_names()
    )

    if len(robot_joint_space) != robot.num_joints:

        raise RuntimeError(
            f"Joint-space mismatch: "
            f"{len(robot_joint_space)} names for "
            f"{robot.num_joints} articulation DOFs."
        )

    if not robot_site_space:

        raise RuntimeError(
            "cuMotion Franka configuration has no tool frames."
        )

    tool_frame = robot_site_space[0]

    print(
        "[INFO] Isaac Lab joint space : "
        f"{robot_joint_space}"
    )

    print(
        "[INFO] cuMotion controlled joints : "
        f"{cumotion_robot.controlled_joint_names}"
    )

    print(
        "[INFO] cuMotion tool frames : "
        f"{robot_site_space}"
    )

    print(
        "[INFO] selected tool frame : "
        f"{tool_frame}"
    )

    # -------------------------------------------------------------------------
    # Independent cuMotion controller per environment
    # -------------------------------------------------------------------------

    print(
        f"\n[STEP] Creating "
        f"{args_cli.num_envs} independent cuMotion controllers..."
    )

    controllers = []

    for env_idx in range(
        args_cli.num_envs
    ):

        world_interface = (
            CumotionWorldInterface()
        )

        controller = RmpFlowController(
            cumotion_robot=cumotion_robot,
            cumotion_world_interface=world_interface,
            robot_joint_space=robot_joint_space,
            robot_site_space=robot_site_space,
            tool_frame=tool_frame,
        )

        cfg = (
            controller
            .get_rmp_flow_config()
        )

        cfg.set_param(
            "cspace_target_rmp/metric_scalar",
            1.0,
        )

        cfg.set_param(
            "collision_rmp/metric_scalar",
            0.0,
        )

        cfg.set_param(
            "target_rmp/max_metric_scalar",
            1000.0,
        )

        controllers.append(
            controller
        )

        print(
            f"[OK] controller[{env_idx}] created."
        )

    print(
        f"[OK] Created "
        f"{len(controllers)} independent cuMotion controllers."
    )

    # -------------------------------------------------------------------------
    # Build Cartesian targets
    # -------------------------------------------------------------------------

    pregrasp = (
        katao_positions
        + np.asarray(
            [
                0.0,
                0.0,
                args_cli.pregrasp_height,
            ],
            dtype=np.float32,
        )
    )

    grasp = (
        katao_positions
        + np.asarray(
            [
                0.0,
                0.0,
                args_cli.grasp_height,
            ],
            dtype=np.float32,
        )
    )

    lift = (
        grasp
        + np.asarray(
            [
                0.0,
                0.0,
                args_cli.lift_height,
            ],
            dtype=np.float32,
        )
    )

    transport = (
        stick_part_positions
        + np.asarray(
            [
                0.0,
                0.0,
                args_cli.lift_height,
            ],
            dtype=np.float32,
        )
    )

    place = (
        transport
        + np.asarray(
            [
                0.0,
                0.0,
                args_cli.place_height,
            ],
            dtype=np.float32,
        )
    )

    print(
        "\n================ MOTION TARGETS ============================"
    )

    for e in range(
        args_cli.num_envs
    ):

        print(
            f"[ENV {e}] "
            f"PREGRASP={np.round(pregrasp[e], 4)} | "
            f"GRASP={np.round(grasp[e], 4)} | "
            f"LIFT={np.round(lift[e], 4)} | "
            f"TRANSPORT={np.round(transport[e], 4)} | "
            f"PLACE={np.round(place[e], 4)}"
        )

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------

    print(
        f"\n[STEP] Warmup for "
        f"{args_cli.warmup} simulation frames..."
    )

    for i in range(
        args_cli.warmup
    ):

        set_gripper_all(
            robot,
            finger_ids,
            args_cli.open,
            args_cli.num_envs,
        )

        scene.write_data_to_sim()

        sim.step()

        scene.update(
            sim.get_physics_dt()
        )

        # Keep ROS node responsive during warmup.
        ros_spin_some()

    print(
        "[OK] Warmup complete."
    )

    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------

    dt = sim.get_physics_dt()

    timeout_steps = max(
        1,
        int(
            args_cli.phase_timeout
            / dt
        ),
    )

    grasp_wait_steps = max(
        1,
        int(
            args_cli.grasp_wait
            / dt
        ),
    )

    # -------------------------------------------------------------------------
    # Cartesian targets dictionary
    # -------------------------------------------------------------------------

    phase_targets = {

        TaskPhase.PREGRASP:
            np.asarray(
                pregrasp,
                dtype=np.float32,
            ),

        TaskPhase.GRASP_APPROACH:
            np.asarray(
                grasp,
                dtype=np.float32,
            ),

        TaskPhase.LIFT:
            np.asarray(
                lift,
                dtype=np.float32,
            ),

        TaskPhase.TRANSPORT:
            np.asarray(
                transport,
                dtype=np.float32,
            ),

        TaskPhase.PLACE:
            np.asarray(
                place,
                dtype=np.float32,
            ),
    }

    # -------------------------------------------------------------------------
    # Per-environment state
    # -------------------------------------------------------------------------

    num_envs = args_cli.num_envs

    phase = np.full(
        num_envs,
        TaskPhase.PREGRASP,
        dtype=np.int32,
    )

    phase_step = np.zeros(
        num_envs,
        dtype=np.int32,
    )

    phase_time = np.zeros(
        num_envs,
        dtype=np.float32,
    )

    phase_initialized = np.zeros(
        num_envs,
        dtype=bool,
    )

    phase_error = np.full(
        num_envs,
        np.inf,
        dtype=np.float32,
    )

    completed_phases = np.zeros(
        num_envs,
        dtype=np.int32,
    )

    # -------------------------------------------------------------------------
    # FSM helpers
    # -------------------------------------------------------------------------

    def reset_phase_timer(
        env_idx,
    ):

        phase_step[env_idx] = 0

        phase_time[env_idx] = 0.0

        phase_error[env_idx] = np.inf

        phase_initialized[env_idx] = False


    def transition_to(
        env_idx,
        next_phase,
    ):

        old_phase = TaskPhase(
            int(phase[env_idx])
        )

        phase[env_idx] = int(
            next_phase
        )

        completed_phases[
            env_idx
        ] += 1

        reset_phase_timer(
            env_idx
        )

        print(
            f"[ENV {env_idx}] "
            f"{old_phase.name} -> "
            f"{TaskPhase(next_phase).name}"
        )


    def initialize_cartesian_phase(
        env_idx,
    ):

        current_phase = TaskPhase(
            int(phase[env_idx])
        )

        if current_phase not in phase_targets:
            return

        controller = controllers[
            env_idx
        ]

        target = phase_targets[
            current_phase
        ][env_idx]

        estimated = make_estimated_state(
            robot,
            robot_joint_space,
            env_idx=env_idx,
        )

        setpoint = make_setpoint_state(
            tool_frame,
            robot_site_space,
            target,
            target_quat,
        )

        print(
            f"[ENV {env_idx}] "
            f"{current_phase.name}: "
            f"controller.reset() -> "
            f"target={np.round(target, 5)}"
        )

        reset_ok = controller.reset(
            estimated,
            setpoint,
            t=0.0,
        )

        if not reset_ok:

            raise RuntimeError(
                f"cuMotion reset failed: "
                f"env={env_idx}, "
                f"phase={current_phase.name}"
            )

        phase_initialized[
            env_idx
        ] = True

        phase_step[
            env_idx
        ] = 0

        phase_time[
            env_idx
        ] = 0.0

        phase_error[
            env_idx
        ] = np.inf


    def get_env_ee_error(
        env_idx,
        target,
    ):

        ee_pos, _, _ = get_ee_pose(
            robot,
            scene,
            ee_body_id,
            env_idx=env_idx,
        )

        return float(
            np.linalg.norm(
                ee_pos - target
            )
        )


    # -------------------------------------------------------------------------
    # Execute independent task FSM
    # -------------------------------------------------------------------------

    print(
        "\n=========================================================="
    )

    print(
        "  PER-ENVIRONMENT TASK EXECUTION"
    )

    print(
        "=========================================================="
    )

    print(
        f"[INFO] num_envs={num_envs}"
    )

    print(
        "[INFO] Each environment has an independent task state."
    )

    print(
        "[INFO] Each environment has an independent wrench stream."
    )

    print(
        "[INFO] Physics is advanced once per simulation step."
    )

    global_step = 0

    # -------------------------------------------------------------------------
    # Main FSM loop
    # -------------------------------------------------------------------------

    while simulation_app.is_running():

        # =====================================================================
        # 1. Execute current phase for every environment
        # =====================================================================

        for env_idx in range(
            num_envs
        ):

            current_phase = TaskPhase(
                int(phase[env_idx])
            )

            # -------------------------------------------------------------
            # Finished environment
            # -------------------------------------------------------------

            if current_phase in (
                TaskPhase.DONE,
                TaskPhase.FAILED,
            ):

                continue

            # -------------------------------------------------------------
            # Initialize phase
            # -------------------------------------------------------------

            if not phase_initialized[
                env_idx
            ]:

                if current_phase in phase_targets:

                    initialize_cartesian_phase(
                        env_idx
                    )

                elif current_phase == TaskPhase.CLOSE_GRIPPER:

                    print(
                        f"[ENV {env_idx}] "
                        "[PHASE] CLOSE_GRIPPER"
                    )

                    phase_initialized[
                        env_idx
                    ] = True

                    phase_step[
                        env_idx
                    ] = 0

                    phase_time[
                        env_idx
                    ] = 0.0

                elif current_phase == TaskPhase.OPEN_GRIPPER:

                    print(
                        f"[ENV {env_idx}] "
                        "[PHASE] OPEN_GRIPPER"
                    )

                    phase_initialized[
                        env_idx
                    ] = True

                    phase_step[
                        env_idx
                    ] = 0

                    phase_time[
                        env_idx
                    ] = 0.0

            # -------------------------------------------------------------
            # Cartesian motion phases
            # -------------------------------------------------------------

            if current_phase in phase_targets:

                controller = controllers[
                    env_idx
                ]

                target = phase_targets[
                    current_phase
                ][env_idx]

                estimated = make_estimated_state(
                    robot,
                    robot_joint_space,
                    env_idx=env_idx,
                )

                setpoint = make_setpoint_state(
                    tool_frame,
                    robot_site_space,
                    target,
                    target_quat,
                )

                desired = controller.forward(
                    estimated,
                    setpoint,
                    phase_time[env_idx],
                )

                if desired is None:

                    raise RuntimeError(
                        f"cuMotion returned None: "
                        f"env={env_idx}, "
                        f"phase={current_phase.name}"
                    )

                apply_cumotion_joint_target(
                    robot,
                    desired,
                    env_idx=env_idx,
                )

                phase_time[
                    env_idx
                ] += dt

                phase_step[
                    env_idx
                ] += 1

            # -------------------------------------------------------------
            # Close gripper
            # -------------------------------------------------------------

            elif current_phase == TaskPhase.CLOSE_GRIPPER:

                set_gripper_env(
                    robot,
                    finger_ids,
                    args_cli.close,
                    env_idx,
                )

                phase_time[
                    env_idx
                ] += dt

                phase_step[
                    env_idx
                ] += 1

            # -------------------------------------------------------------
            # Open gripper
            # -------------------------------------------------------------

            elif current_phase == TaskPhase.OPEN_GRIPPER:

                set_gripper_env(
                    robot,
                    finger_ids,
                    args_cli.open,
                    env_idx,
                )

                phase_time[
                    env_idx
                ] += dt

                phase_step[
                    env_idx
                ] += 1

        # =====================================================================
        # 2. Physics step
        #
        # All environments advance together physically.
        #
        # This is the ONLY sim.step() in the FSM iteration.
        # =====================================================================

        scene.write_data_to_sim()

        sim.step()

        scene.update(
            dt
        )

        sim_time += dt

        global_step += 1

        # =====================================================================
        # 3. Publish JointWrenchSensor for every environment
        # =====================================================================

        next_ros_time = publish_wrench_for_all_envs(
            wrench_sensor=wrench_sensor,
            sensor_body_id=sensor_body_id,
            ros_publishers=ros_publishers,
            sim_time=sim_time,
            next_ros_time=next_ros_time,
            ros_period=ros_period,
        )

        # Keep ROS 2 callbacks responsive.
        ros_spin_some()

        # =====================================================================
        # 4. Evaluate each environment independently
        # =====================================================================

        for env_idx in range(
            num_envs
        ):

            current_phase = TaskPhase(
                int(phase[env_idx])
            )

            if current_phase in (
                TaskPhase.DONE,
                TaskPhase.FAILED,
            ):

                continue

            # -------------------------------------------------------------
            # Cartesian phase completion
            # -------------------------------------------------------------

            if current_phase in phase_targets:

                target = phase_targets[
                    current_phase
                ][env_idx]

                error = get_env_ee_error(
                    env_idx,
                    target,
                )

                phase_error[
                    env_idx
                ] = error

                # Require at least 20 simulation steps before success.
                if (
                    error < args_cli.tolerance
                    and phase_step[env_idx] >= 20
                ):

                    print(
                        f"[ENV {env_idx}] "
                        f"[OK] {current_phase.name} "
                        f"reached | "
                        f"error={error:.5f} m | "
                        f"steps={phase_step[env_idx]}"
                    )

                    # PREGRASP -> GRASP
                    if current_phase == TaskPhase.PREGRASP:

                        transition_to(
                            env_idx,
                            TaskPhase.GRASP_APPROACH,
                        )

                    # GRASP -> CLOSE
                    elif current_phase == TaskPhase.GRASP_APPROACH:

                        transition_to(
                            env_idx,
                            TaskPhase.CLOSE_GRIPPER,
                        )

                    # LIFT -> TRANSPORT
                    elif current_phase == TaskPhase.LIFT:

                        transition_to(
                            env_idx,
                            TaskPhase.TRANSPORT,
                        )

                    # TRANSPORT -> PLACE
                    elif current_phase == TaskPhase.TRANSPORT:

                        transition_to(
                            env_idx,
                            TaskPhase.PLACE,
                        )

                    # PLACE -> OPEN
                    elif current_phase == TaskPhase.PLACE:

                        transition_to(
                            env_idx,
                            TaskPhase.OPEN_GRIPPER,
                        )

            # -------------------------------------------------------------
            # Close gripper completion
            # -------------------------------------------------------------

            elif current_phase == TaskPhase.CLOSE_GRIPPER:

                if (
                    phase_step[env_idx]
                    >= grasp_wait_steps
                ):

                    print(
                        f"[ENV {env_idx}] "
                        "[OK] CLOSE_GRIPPER complete | "
                        f"wait={args_cli.grasp_wait:.2f}s"
                    )

                    transition_to(
                        env_idx,
                        TaskPhase.LIFT,
                    )

            # -------------------------------------------------------------
            # Open gripper completion
            # -------------------------------------------------------------

            elif current_phase == TaskPhase.OPEN_GRIPPER:

                if (
                    phase_step[env_idx]
                    >= grasp_wait_steps
                ):

                    print(
                        f"[ENV {env_idx}] "
                        "[OK] OPEN_GRIPPER complete"
                    )

                    transition_to(
                        env_idx,
                        TaskPhase.DONE,
                    )

                    print(
                        f"[ENV {env_idx}] "
                        "[DONE] Complete pick/place sequence."
                    )

            # -------------------------------------------------------------
            # Per-environment timeout
            # -------------------------------------------------------------

            if (
                phase_step[env_idx]
                >= timeout_steps
            ):

                print(
                    f"[ENV {env_idx}] "
                    f"[WARN] {current_phase.name} timeout | "
                    f"error={phase_error[env_idx]:.5f} m"
                )

                phase[
                    env_idx
                ] = TaskPhase.FAILED

                phase_initialized[
                    env_idx
                ] = False

                print(
                    f"[ENV {env_idx}] "
                    f"[FAILED] Task stopped at "
                    f"{current_phase.name}"
                )

        # =====================================================================
        # 5. Periodic task + wrench logging
        # =====================================================================

        if (
            global_step
            % args_cli.print_every
            == 0
        ):

            state_strings = []

            for env_idx in range(
                num_envs
            ):

                current_phase = TaskPhase(
                    int(phase[env_idx])
                )

                if current_phase in phase_targets:

                    state_strings.append(
                        f"e{env_idx}="
                        f"{current_phase.name}"
                        f"({phase_error[env_idx]:.4f}m)"
                    )

                else:

                    state_strings.append(
                        f"e{env_idx}="
                        f"{current_phase.name}"
                    )

            print(
                f"[STEP {global_step:6d}] "
                + " | ".join(
                    state_strings
                )
            )

            print_wrench_for_all_envs(
                wrench_sensor=wrench_sensor,
                sensor_body_id=sensor_body_id,
                phases=[
                    TaskPhase(
                        int(phase[e])
                    )
                    for e in range(num_envs)
                ],
            )

        # =====================================================================
        # 6. Stop only after every environment is DONE or FAILED
        # =====================================================================

        all_finished = all(
            TaskPhase(
                int(phase[e])
            )
            in (
                TaskPhase.DONE,
                TaskPhase.FAILED,
            )
            for e in range(num_envs)
        )

        if all_finished:

            print(
                "\n=========================================================="
            )

            print(
                "  ALL ENVIRONMENTS FINISHED"
            )

            print(
                "=========================================================="
            )

            for env_idx in range(
                num_envs
            ):

                final_phase = TaskPhase(
                    int(phase[env_idx])
                )

                print(
                    f"[ENV {env_idx}] "
                    f"final state = "
                    f"{final_phase.name}"
                )

            break

    # -------------------------------------------------------------------------
    # Final verification
    # -------------------------------------------------------------------------

    final_ee_positions = []

    final_ee_errors = []

    final_katao_positions = []

    katao_displacements = []

    print(
        "\n================ FINAL VERIFICATION ======================="
    )

    for env_idx in range(
        args_cli.num_envs
    ):

        (
            final_ee_pos_env,
            _,
            final_ee_pos_world,
        ) = get_ee_pose(
            robot,
            scene,
            ee_body_id,
            env_idx=env_idx,
        )

        (
            final_katao_pos_env,
            _,
            final_katao_pos_world,
            _,
        ) = get_prim_pose(
            katao,
            scene,
            env_idx=env_idx,
        )

        final_ee_positions.append(
            final_ee_pos_env
        )

        # Final Cartesian target is PLACE.
        final_ee_errors.append(
            float(
                np.linalg.norm(
                    final_ee_pos_env
                    - place[env_idx]
                )
            )
        )

        final_katao_positions.append(
            final_katao_pos_env
        )

        katao_displacements.append(
            float(
                np.linalg.norm(
                    final_katao_pos_env
                    - katao_positions[env_idx]
                )
            )
        )

        print(
            f"[ENV {env_idx}] "
            f"final EE local="
            f"{np.round(final_ee_pos_env, 5)} | "
            f"world="
            f"{np.round(final_ee_pos_world, 5)} | "
            f"place target="
            f"{np.round(place[env_idx], 5)} | "
            f"katao local="
            f"{np.round(final_katao_pos_env, 5)} | "
            f"world="
            f"{np.round(final_katao_pos_world, 5)}"
        )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    print(
        "\n=========================================================="
    )

    print(
        "  TEST FINISHED"
    )

    print(
        "=========================================================="
    )

    for env_idx in range(
        args_cli.num_envs
    ):

        final_state = TaskPhase(
            int(phase[env_idx])
        )

        print(
            f"[ENV {env_idx}] "
            f"state={final_state.name} | "
            f"final EE="
            f"{np.round(final_ee_positions[env_idx], 5)} | "
            f"place target="
            f"{np.round(place[env_idx], 5)} | "
            f"error="
            f"{final_ee_errors[env_idx]:.4f} m | "
            f"katao displacement="
            f"{katao_displacements[env_idx]:.4f} m"
        )

    max_final_ee_error = max(
        final_ee_errors
    )

    all_reached = (
        max_final_ee_error
        < args_cli.tolerance
    )

    if all_reached:

        print(
            f"\n[SUCCESS] All "
            f"{args_cli.num_envs} environments "
            "reached the PLACE target."
        )

    else:

        print(
            f"\n[FAIL] At least one environment "
            "missed the PLACE target. "
            f"max EE error="
            f"{max_final_ee_error:.4f} m"
        )

    for env_idx, displacement in enumerate(
        katao_displacements
    ):

        if displacement > 0.03:

            print(
                f"[ENV {env_idx}] "
                "katao moved with the gripper."
            )

        else:

            print(
                f"[ENV {env_idx}] "
                "katao did not move significantly. "
                "This is a grasp/contact issue unless "
                "the EE target itself also failed."
            )

    # -------------------------------------------------------------------------
    # Keep application alive for inspection.
    #
    # Wrench publishing continues during this period.
    # -------------------------------------------------------------------------

    print(
        "\n[INFO] Test is complete. "
        "Window will remain open until you close it."
    )

    while simulation_app.is_running():

        scene.update(
            dt
        )

        sim.step()

        sim_time += dt

        next_ros_time = publish_wrench_for_all_envs(
            wrench_sensor=wrench_sensor,
            sensor_body_id=sensor_body_id,
            ros_publishers=ros_publishers,
            sim_time=sim_time,
            next_ros_time=next_ros_time,
            ros_period=ros_period,
        )

        ros_spin_some()

    # -------------------------------------------------------------------------
    # ROS publisher cleanup
    # -------------------------------------------------------------------------

    for publisher in ros_publishers:

        try:
            publisher.destroy()
        except Exception:
            pass

    ros_shutdown()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            "\n=========================================================="
        )

        print(
            "  SCRIPT EXCEPTION"
        )

        print(
            "=========================================================="
        )

        print(
            f"[ERROR] {type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        try:

            while simulation_app.is_running():

                simulation_app.update()

        except Exception:

            pass

    finally:

        ros_shutdown()

        simulation_app.close()