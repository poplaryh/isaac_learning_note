# mdp/observations.py

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul


def katao_pose(
    env,
    asset_cfg: SceneEntityCfg,
):
    katao = env.scene[asset_cfg.name]

    # root_link_pose_w:
    #
    # [x, y, z, qw, qx, qy, qz]
    #

    pose_w = katao.data.root_link_pose_w

    position_w = pose_w[:, :3]

    # 转换成 environment local frame
    position_local = (
        position_w
        - env.scene.env_origins
    )

    orientation = pose_w[:, 3:7]

    return torch.cat(
        [
            position_local,
            orientation,
        ],
        dim=-1,
    )


# def katao_velocity(
#     env,
#     asset_cfg: SceneEntityCfg,
# ):
#     katao = env.scene[asset_cfg.name]

#     velocity = katao.data.root_link_vel_w

#     return velocity

def katao_velocity(
    env,
    asset_cfg: SceneEntityCfg,
):
    katao = env.scene[asset_cfg.name]

    velocity = katao.data.root_link_vel_w

    return velocity.reshape(
        env.num_envs,
        6,
    )

def robot_joint_position(
    env,
    asset_cfg: SceneEntityCfg,
):
    robot = env.scene[asset_cfg.name]

    return robot.data.joint_pos[
        :,
        asset_cfg.joint_ids,
    ]


def robot_joint_velocity(
    env,
    asset_cfg: SceneEntityCfg,
):
    robot = env.scene[asset_cfg.name]

    return robot.data.joint_vel[
        :,
        asset_cfg.joint_ids,
    ]


def previous_action(env):
    return env.action_manager.action

def stick_part_pose(
    env,
    asset_cfg: SceneEntityCfg,
):
    stick_part = env.scene[asset_cfg.name]

    # root_link_pose_w:
    #
    # [x, y, z, qw, qx, qy, qz]
    #

    pose_w = stick_part.data.root_link_pose_w

    position_w = pose_w[:, :3]

    # 转换成 environment local frame
    position_local = (
        position_w
        - env.scene.env_origins
    )

    orientation = pose_w[:, 3:7]

    return torch.cat(
        [
            position_local,
            orientation,
        ],
        dim=-1,
    )
