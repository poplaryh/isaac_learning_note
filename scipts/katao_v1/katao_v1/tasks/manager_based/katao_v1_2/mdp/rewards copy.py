# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv



def alive_reward(env):
    return torch.ones(
        env.num_envs,
        device=env.device,
    )


def action_penalty(env):

    action = env.action_manager.action

    return torch.sum(
        action ** 2,
        dim=-1,
    )


def joint_velocity_penalty(
    env,
    asset_cfg: SceneEntityCfg,
):

    robot = env.scene[asset_cfg.name]

    joint_vel = robot.data.joint_vel[
        :,
        asset_cfg.joint_ids,
    ]

    return torch.sum(
        joint_vel ** 2,
        dim=-1,
    )


def katao_alignment_reward(env, asset_cfg: SceneEntityCfg):
    """Reward for alignment of the katao."""
    katao = env.scene[asset_cfg.name]
    stick_part = env.scene["stick_part"]
    katao_pos_w = katao.data.root_link_pose_w[:, :3]
    katao_quat_w = katao.data.root_link_pose_w[:, 3:7]
    stick_pos_w = stick_part.data.root_link_pose_w[:, :3]
    stick_quat_w = stick_part.data.root_link_pose_w[:, 3:7]
    return function_katao_alignment(
        katao_pos_w,
        katao_quat_w,
        stick_pos_w,
        stick_quat_w,
    )


def katao_xy_reward(env, asset_cfg: SceneEntityCfg):
    """Reward for alignment of the katao."""
    katao = env.scene["katao"]
    stick_part = env.scene["stick_part"]
    katao_pos_w = katao.data.root_link_pose_w[:, :3]
    stick_pos_w = stick_part.data.root_link_pose_w[:, :3]

    pos_diff_w = katao_pos_w - stick_pos_w
    xy_error = torch.linalg.norm(
            pos_diff_w[:, :2],
            dim=-1,
        )
    reward_position = torch.exp(
            -20.0 * xy_error
        )

    return reward_position

def katao_z_reward(env, asset_cfg: SceneEntityCfg):
    """Reward for alignment of the katao."""
    katao = env.scene["katao"]
    stick_part = env.scene["stick_part"]
    katao_pos_w = katao.data.root_link_pose_w[:, :3]
    stick_pos_w = stick_part.data.root_link_pose_w[:, :3]

    pos_diff_w = katao_pos_w - stick_pos_w
    pos_diff_w[:, 2] += 0.002
    z_error = torch.abs(pos_diff_w[:, 2])
    reward_position = torch.exp(
            -20.0 * z_error
        )

    return reward_position

def katao_orientation_pipe_reward(env, asset_cfg: SceneEntityCfg):
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

    reward_rotation = (
        cos_angle + 1.0
    ) * 0.5

    return reward_rotation

def katao_orientation_z_reward(env, asset_cfg: SceneEntityCfg):
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

    reward_rotation = (
        cos_angle_z + 1.0
    ) * 0.5

    return reward_rotation

def success_reward(env, asset_cfg: SceneEntityCfg):
    success = (
        (katao_xy_reward(env, asset_cfg) > 0.9)
        & (katao_z_reward(env, asset_cfg) > 0.9)
        # & (katao_orientation_pipe_reward(env, asset_cfg) > 0.98)
        # & (katao_orientation_z_reward(env, asset_cfg) > 0.98)
    )
    success = success.float()
    # if success == 1.0:
    #     print('----------------------------------------------')
    #     print(f'xy error= {katao_xy_reward(env, asset_cfg)}, z error= {katao_z_reward(env, asset_cfg)}')
    #     print(f'cos angle= {katao_orientation_pipe_reward(env, asset_cfg)}, cos angle z= {katao_orientation_z_reward(env, asset_cfg)}')
    #     print(f'success= {success}')
    #     print('----------------------------------------------')

    return success

def function_katao_alignment(
    katao_pos_w,
    katao_quat_w,
    stick_pos_w,
    stick_quat_w,
):
    '''
    1, xy_error计算的是stick_part和katao世界坐标的xy偏差，我认为，如果两者偏差为0,就实现了水平对齐
    2, stick_part是固定在地面不动的一个物体
    3, z_error 计算的是katao和stick_part的世界坐标中的z的差值的绝对值，我认为，当两者的z_error，就实现了插入深度达到要求
    4, 当前reward文件里面，cos_angle评价的是katao上的豁口与stick_part上的侧面的管子伸出的方向的对齐情况，目前的参数设置情况下，初始为1,说明两者对齐了，符合实际情况
    5, 当前reward文件里面，cos_angle_z评价的是katao垂直于地面的轴与stick_part垂直于地面的轴的平行情况，用于约束katao竖着套入stick_part，当前的参数设置情况下，cos_angle_z为1,符合实际两者平行的情况
    '''
    num_envs = katao_quat_w.shape[0]

    # ==========================
    # 1. katao pose in stick frame
    # ==========================

    stick_quat_inv = quat_conjugate(stick_quat_w)

    pos_diff_w = katao_pos_w - stick_pos_w

    pos_diff_stick = quat_apply(
        stick_quat_inv,
        pos_diff_w,
    )

    relative_quat = quat_mul(
        stick_quat_inv,
        katao_quat_w,
    )

    # ==========================
    # 2. Position alignment
    # ==========================

    xy_error = torch.linalg.norm(
        pos_diff_w[:, :2],
        dim=-1,
    )

    z_error = torch.abs(pos_diff_w[:, 2])

    reward_position = torch.exp(
        -20.0 * xy_error
    )

    reward_position_z = torch.exp(
        -20.0 * z_error
    )

    # ==========================
    # 3. Rotation alignment (katao 豁口朝向 与 pipe 朝向 对齐角度)
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

    reward_rotation = (
        cos_angle + 1.0
    ) * 0.5

    # reward_rotation = - torch.exp(
    #     torch.abs(cos_angle - 1.0)
    # )

    # ==========================
    # 4. vertical Axis alignment (katao 竖直朝向 与 sitck_part 竖直朝向 对齐角度)
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

    reward_rotation_z = (
        cos_angle_z + 1.0
    ) * 0.5

    # reward_rotation_z = - torch.exp(
    #     torch.abs(cos_angle_z - 1.0)
    # )

    # ==========================
    # 5. Total
    # ==========================

    success = (
        (xy_error < 0.0005)
        & (z_error < 0.0005)
        & (cos_angle > 0.99)
        & (cos_angle_z > 0.99)
    )

    reward = (
        2.0 * reward_position
        + 2.0 * reward_position_z
        + 3.0 * reward_rotation
        + 3.0 * reward_rotation_z
        + 100.0 * success
    )
    print('--------------------------------------------------------------------------------------------------')
    print('the cos_angle:', cos_angle, 'the z-axis cos angle: ', cos_angle_z)
    print('the x_error:', pos_diff_w[:, 0], 'the y_error:', pos_diff_w[:, 1], 'the z_error:', pos_diff_w[:, 2])
    print('the xy_error:', xy_error, 'z_error:', z_error)
    print('--------------------------------------------------------------------------------------------------')

    return reward

