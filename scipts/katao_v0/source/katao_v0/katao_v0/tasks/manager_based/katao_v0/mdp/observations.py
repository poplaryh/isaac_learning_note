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

def katao_stick_xy(
    env,
    asset_cfg: SceneEntityCfg,
):
    katao = env.scene["katao"]
    stick_part = env.scene["stick_part"]

    katao_pos_w = katao.data.root_link_pose_w[:, :3]
    stick_pos_w = stick_part.data.root_link_pose_w[:, :3]

    return torch.linalg.norm(
            stick_pos_w[:, :2] - katao_pos_w[:, :2],
            dim=-1,
        ).unsqueeze(-1)

def katao_stick_z(
    env,
    asset_cfg: SceneEntityCfg,
):
    katao = env.scene["katao"]
    stick_part = env.scene["stick_part"]

    katao_pos_w = katao.data.root_link_pose_w[:, :3]
    stick_pos_w = stick_part.data.root_link_pose_w[:, :3]


    return (stick_pos_w[:, 2] - katao_pos_w[:, 2]).unsqueeze(-1)

def katao_orientation_pipe(env, asset_cfg: SceneEntityCfg):
    """Reward for alignment of the katao."""
    katao = env.scene["katao"]
    stick_part = env.scene["stick_part"]
    katao_quat_w = katao.data.root_link_pose_w[:, 3:7]
    stick_quat_w = stick_part.data.root_link_pose_w[:, 3:7]
    num_envs = katao_quat_w.shape[0]

    stick_quat_inv = quat_conjugate(stick_quat_w)
    relative_quat = quat_mul(
            stick_quat_inv,
            katao_quat_w,
        )
    # ==========================
    #  Rotation alignment (katao 豁口朝向 与 pipe 朝向 对齐角度)
    # ==========================
    # Katao insertion axis
    slot_axis_local = torch.tensor(
        [0.0, 1.0, 0.0],
        device=katao_quat_w.device,
    ).expand(num_envs, -1)

    pipe_axis_local = torch.tensor(
        [0.0, 0.0, 1.0],
        device=katao_quat_w.device,
    ).expand(num_envs, -1)

    slot_axis_stick = quat_apply(
        relative_quat,
        slot_axis_local,
    )

    cos_angle = torch.sum(
        slot_axis_stick * pipe_axis_local,
        dim=-1,
    )

    cos_angle = torch.clamp(
        cos_angle,
        -1.0,
        1.0,
    )

    return cos_angle.unsqueeze(-1)

def katao_orientation_z(env, asset_cfg: SceneEntityCfg):
    """Reward for alignment of the katao."""
    katao = env.scene["katao"]
    stick_part = env.scene["stick_part"]
    katao_quat_w = katao.data.root_link_pose_w[:, 3:7]
    stick_quat_w = stick_part.data.root_link_pose_w[:, 3:7]
    num_envs = katao_quat_w.shape[0]

    # ==========================
    # vertical Axis alignment (katao 竖直朝向 与 sitck_part 竖直朝向 对齐角度)
    # ==========================

    # Stick axis = local
    katao_axis_local = torch.tensor(
        [0.0, 0.0, 1.0],
        device=katao_quat_w.device,
    ).expand(num_envs, -1)
    stick_axis_local = torch.tensor(
        [0.0, 1.0, 0.0],
        device=stick_quat_w.device,
    ).expand(num_envs, -1)

    # Convert both axes to world frame
    katao_axis_w = quat_apply(
        katao_quat_w,
        katao_axis_local,
    )

    stick_axis_w = quat_apply(
        stick_quat_w,
        stick_axis_local,
    )

    cos_angle_z = torch.sum(
        katao_axis_w * stick_axis_w,
        dim=-1,
    )

    cos_angle_z = torch.clamp(
        cos_angle_z,
        -1.0,
        1.0,
    )

    return cos_angle_z.unsqueeze(-1)



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
