# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

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

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
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
from isaaclab.sensors import JointWrenchSensorCfg

from .mdp import actions
from .mdp import observations
from .mdp import rewards
from .mdp import events
from .mdp import terminations
 
##
# Scene definition
##


@configclass
class KataoV0SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",

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

                # contact_offset=0.001,
                # rest_offset=0.000,
            ),

            # physics_material=PhysxRigidBodyMaterialCfg(
            #     static_friction=0.8,
            #     dynamic_friction=0.6,
            #     restitution=0.0,
            #     friction_combine_mode="average",
            # ),

        ),

        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.6, 0.037057),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    stick_part: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/stick_part",

        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/yh/tianji/mission_docs/mission01_v2/ihihi/stick_part02.usd",

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

                # solver_position_iteration_count=16,
            ),

            collision_props=PhysxCollisionPropertiesCfg(
                collision_enabled=True,

                # contact_offset=0.001,
                # rest_offset=0.000,
            ),

            # physics_material=PhysxRigidBodyMaterialCfg(
            #     static_friction=0.1,
            #     dynamic_friction=0.05,
            #     restitution=0.0,
            #     friction_combine_mode="average",
            # ),

        ),

        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.131),
            rot=(0.0, -0.707, 0.707, 0.0),
        ),
    )

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    # robot_joints: ActionTermCfg = mdp.JointPositionActionCfg(
    #     asset_name="robot",

    #     joint_names=[
    #         "panda_joint1",
    #         "panda_joint2",
    #         "panda_joint3",
    #         "panda_joint4",
    #         "panda_joint5",
    #         "panda_joint6",
    #         "panda_joint7",
    #         # "panda_finger_joint1",
    #         # "panda_finger_joint2",
    #     ],

    #     scale=1,
    #     use_default_offset=False,
    # )

    robot_joints: ActionTermCfg = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"], 
        scale=1, use_zero_offset=True,
        clip={"panda_joint.*": (-0.04, 0.04)},
    )


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
                        # "panda_finger_joint1",
                        # "panda_finger_joint2",
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
                        # "panda_finger_joint1",
                        # "panda_finger_joint2",
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
    # alive = RewTerm(
    #     func=rewards.alive_reward,
    #     weight=1.0,
    # )

    # 动作惩罚
    # action_penalty = RewTerm(
    #     func=rewards.action_penalty,
    #     weight=-0.01,
    # )

    # 关节速度惩罚
    # joint_velocity_penalty = RewTerm(
    #     func=rewards.joint_velocity_penalty,
    #     weight=-0.001,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "panda_joint1",
    #                 "panda_joint2",
    #                 "panda_joint3",
    #                 "panda_joint4",
    #                 "panda_joint5",
    #                 "panda_joint6",
    #                 "panda_joint7",
    #                 # "panda_finger_joint1",
    #                 # "panda_finger_joint2",
    #             ],
    #         ),
    #     },
    # )

    # 位置奖励
    # katao_alignment_reward = RewTerm(
    #     func=rewards.katao_alignment_reward,
    #     weight=1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("katao"),
    #     },
    # )

    katao_xy_position_reward = RewTerm(
        func=rewards.katao_xy_reward,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        },
    )

    katao_z_position_reward = RewTerm(
        func=rewards.katao_z_reward,
        weight=5.0,
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
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        },
    )

    katao_success_reward = RewTerm(
        func=rewards.success_reward,
        weight=100.0,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Success
    success = DoneTerm(
        func=terminations.success,
        params={
            "asset_cfg": SceneEntityCfg("katao"),
        })



@configclass
class KataoV0EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: KataoV0SceneCfg = KataoV0SceneCfg(num_envs=1, env_spacing=2.0)
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
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        # print('-----------------------')
        # print(mdp.RelativeJointPositionActionCfg.__annotations__)
        # print('-----------------------')