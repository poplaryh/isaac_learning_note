# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

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
from isaaclab.sensors import JointWrenchSensorCfg, ContactSensorCfg

from .mdp import actions
from .mdp import observations
from .mdp import rewards
from .mdp import events
from .mdp import terminations
 
##
# Scene definition
##


@configclass
class KataoV1SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    ee_frame: FrameTransformerCfg = MISSING

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",

        # 启用接触传感器 API，否则 ContactSensorCfg 找不到带 contact reporter 的刚体
        spawn=FRANKA_PANDA_HIGH_PD_CFG.spawn.replace(activate_contact_sensors=False),

        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    joint_wrench = JointWrenchSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",

        update_period=0.0,

        debug_vis=True,
    )

    katao: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/katao",

        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/yh/tianji/mission_docs/mission01_v2/ihihi/katao02.usd",

            rigid_props=PhysxRigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,

                solver_position_iteration_count=32,
            ),

            collision_props=PhysxCollisionPropertiesCfg(
                collision_enabled=True,

                # contact_offset=0.001,
                # rest_offset=0.000,
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
            usd_path="/home/yh/tianji/mission_docs/mission01_v2/ihihi/stick_part1.usd",

            rigid_props=PhysxRigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=False,

                solver_position_iteration_count=32,
            ),

            collision_props=PhysxCollisionPropertiesCfg(
                collision_enabled=True,

                # contact_offset=0.001,
                # rest_offset=0.000,
            ),

        ),

        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.131),
            rot=(0.0, -0.707, 0.707, 0.0),
        ),
    )

    # gripper_left_contact = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
    #     filter_prim_paths_expr=[
    #         "{ENV_REGEX_NS}/katao",
    #     ],
    # )

    # gripper_right_contact = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
    #     filter_prim_paths_expr=[
    #         "{ENV_REGEX_NS}/katao",
    #     ],
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    robot_joints: ActionTermCfg = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"], 
        scale=1, use_zero_offset=True,
        clip={"panda_joint.*": (-0.04, 0.04)},
    )

    # gripper: ActionTermCfg = mdp.BinaryJointPositionActionCfg(
    #     asset_name="robot",

    #     joint_names=[
    #         "panda_finger_joint1",
    #         "panda_finger_joint2",
    #     ],

    #     open_command_expr={
    #         "panda_finger_joint1": 0.04,
    #         "panda_finger_joint2": 0.04,
    #     },

    #     close_command_expr={
    #         "panda_finger_joint1": 0.0,
    #         "panda_finger_joint2": 0.0,
    #     },
    # )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        # -----------------------------
        # Katao pose
        # -----------------------------

        katao_pose = ObsTerm(
            func=observations.katao_pose,
            params={
                "asset_cfg": SceneEntityCfg("katao"),
            },
        )

        # -----------------------------
        # Katao velocity
        # -----------------------------

        katao_velocity = ObsTerm(
            func=observations.katao_velocity,
            params={
                "asset_cfg": SceneEntityCfg("katao"),
            },
        )

        # -----------------------------
        # Robot joint position
        # -----------------------------

        robot_joint_position = ObsTerm(
            func=observations.robot_joint_position,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "panda_joint1",
                        "panda_joint2",
                        "panda_joint3",
                        "panda_joint4",
                        "panda_joint5",
                        "panda_joint6",
                        "panda_joint7",
                        "panda_finger_joint1",
                        "panda_finger_joint2",
                    ],
                ),
            },
        )

        # -----------------------------
        # Robot joint velocity
        # -----------------------------

        robot_joint_velocity = ObsTerm(
            func=observations.robot_joint_velocity,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "panda_joint1",
                        "panda_joint2",
                        "panda_joint3",
                        "panda_joint4",
                        "panda_joint5",
                        "panda_joint6",
                        "panda_joint7",
                        "panda_finger_joint1",
                        "panda_finger_joint2",
                    ],
                ),
            },
        )

        katao_stick_xy = ObsTerm(
            func=observations.katao_stick_xy,
            params={
                "asset_cfg": SceneEntityCfg("katao"),
            },
        )
        
        katao_stick_z = ObsTerm(
            func=observations.katao_stick_z,
            params={
                "asset_cfg": SceneEntityCfg("katao"),
            },
        )

        katao_orientation_pipe = ObsTerm(
            func=observations.katao_orientation_pipe,
            params={
                "asset_cfg": SceneEntityCfg("katao"),
            },
        )

        katao_orientation_z = ObsTerm(
            func=observations.katao_orientation_z,
            params={
                "asset_cfg": SceneEntityCfg("katao"),
            },
        )

        grasped = ObsTerm(
            func=observations.katao_grasped,
            params={
                "asset_cfg": SceneEntityCfg("katao"),
                "diff_threshold": 0.04,
            },
        )

        # -----------------------------
        # Robot end-effector joint wrench
        # -----------------------------

        robot_ee_joint_wrench = ObsTerm(
            func=observations.robot_ee_joint_wrench,
            params={
                "sensor_name": "joint_wrench",
                "body_name": "panda_hand",
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(
            func=events.reset_to_recorded_state,
            mode="reset",
    )



@configclass
class RewardsCfg:

    # 存活奖励
    alive = RewTerm(
        func=rewards.alive_reward,
        weight=1.0,
    )

    # 动作惩罚
    action_penalty = RewTerm(
        func=rewards.action_penalty,
        weight=-0.01,
    )

    # 关节速度惩罚
    joint_velocity_penalty = RewTerm(
        func=rewards.joint_velocity_penalty,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "panda_joint1",
                    "panda_joint2",
                    "panda_joint3",
                    "panda_joint4",
                    "panda_joint5",
                    "panda_joint6",
                    "panda_joint7",
                    # "panda_finger_joint1",
                    # "panda_finger_joint2",
                ],
            ),
        },
    )

    # -----------------------------
    # Stage 1: Alignment
    # -----------------------------

    katao_grasped_reward = RewTerm(
        func=rewards.katao_grasped,
        weight=1.0,
        params={"diff_threshold": 0.04},
    )

    katao_xy_position_reward = RewTerm(
        func=rewards.katao_xy_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        },
    )

    katao_orientation_pipe_reward = RewTerm(
        func=rewards.katao_orientation_pipe_reward,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        },
    )

    katao_orientation_z_reward = RewTerm(
        func=rewards.katao_orientation_z_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        },
    )

    # -----------------------------
    # Stage 2: Insertion
    # -----------------------------

    katao_insertion_reward = RewTerm(
        func=rewards.katao_insertion_reward,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        },
    )

    seated_reward = RewTerm(
        func=rewards.insertion_gate,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        },
    )

    # -----------------------------
    # Stage 3: Rotation
    # -----------------------------

    katao_rotation_mission_reward = RewTerm(
        func=rewards.rotation_mission_reward,   
        weight=6.0,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        },
    )

    # -----------------------------
    # Stage 4: Success
    # -----------------------------

    katao_success_reward = RewTerm(
        func=rewards.success_reward,
        weight=25.0,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Not grasped
    not_grasped = DoneTerm(
        func=terminations.not_grasped,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        },
    )

    # (3) Success
    success = DoneTerm(
        func=terminations.success,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        })


@configclass
class KataoV1EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: KataoV1SceneCfg = KataoV1SceneCfg(num_envs=1, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.device = "cuda:0"

        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
        # print('-----------------------')
        # print(mdp.RelativeJointPositionActionCfg.__annotations__)
        # print('-----------------------')