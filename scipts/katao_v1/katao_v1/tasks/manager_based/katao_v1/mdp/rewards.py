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

# ==========================
# stage 1: alignment
# ==========================

def katao_grasped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    diff_threshold: float = 0.04,
) -> torch.Tensor:
    """Reward when katao is grasped (close + gripper closed)."""
    robot = env.scene['robot']
    ee_frame = env.scene["ee_frame"]
    katao = env.scene["katao"]
    
    katao_pos = katao.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.norm(katao_pos - ee_pos, dim=1)
    
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        
        grasped = torch.logical_and(
            pose_diff < diff_threshold,
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[0]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            ) > env.cfg.gripper_threshold,
        )
        grasped = torch.logical_and(
            grasped,
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[1]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            ) > env.cfg.gripper_threshold,
        )
        # print('夹爪抓取到物体的判定为:', grasped)
        return grasped.float()
    
    return torch.zeros(env.num_envs, device=env.device)
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
            -50.0 * xy_error
        ) # normalize to [0, 1]
    # print('xy_error:', xy_error, 'reward_position:', reward_position, pos_diff_w[:, 0], pos_diff_w[:, 1], pos_diff_w[:, 2])

    return reward_position

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
    ) * 0.5 # normalize to [0, 1]

    # print('cos_angle_z:', cos_angle_z, 'reward_rotation:', reward_rotation)

    return reward_rotation
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
    ) * 0.5 # normalize to [0, 1]

    # print('cos_angle:', cos_angle, 'reward_rotation:', reward_rotation)
    return reward_rotation


def alignment_gate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    xy_threshold: float = 0.9,
    orientation_z_threshold: float = 0.9,
    orientation_pipe_threshold: float = 0.9,
    grasped_threshold: float = 0.5,
) -> torch.Tensor:
    """对齐完成判据：katao 与 stick_part 同时满足 4 个条件时返回 1.0，否则 0.0。

    各子奖励的阈值换算（方便按物理误差选阈值）：
    - katao_xy_reward = exp(-1000 * xy_error)
      -> xy_error = -ln(threshold) / 1000，如 0.9 ≈ 0.105mm，0.7 ≈ 0.36mm，0.5 ≈ 0.69mm
    - katao_orientation_z_reward / katao_orientation_pipe_reward = (cos + 1) / 2
      -> 角度 = arccos(2*threshold - 1)，如 0.98 ≈ 16.3°，0.99 ≈ 11.5°，0.998 ≈ 3.6°
    - katao_grasped 是 0/1 布尔值，阈值 0.5 即"已抓取"
    """
    aligned = (
        (katao_xy_reward(env, asset_cfg) > xy_threshold)
        & (katao_orientation_z_reward(env, asset_cfg) > orientation_z_threshold)
        & (katao_orientation_pipe_reward(env, asset_cfg) > orientation_pipe_threshold)
        & (katao_grasped(env, asset_cfg) > grasped_threshold)
    )
    # print('对齐判定结果:', aligned)
    return aligned.float()

# ==========================
#  插入深度奖励
# ==========================

def katao_insertion_reward(env, asset_cfg: SceneEntityCfg):
    """套入深度奖励，z_error 越小越好。"""
    katao = env.scene["katao"]
    stick_part = env.scene["stick_part"]
    katao_pos_w = katao.data.root_link_pose_w[:, :3]
    stick_pos_w = stick_part.data.root_link_pose_w[:, :3]

    pos_diff_w = katao_pos_w - stick_pos_w
    # pos_diff_w[:, 2] -= 0.001
    z_error = torch.abs(pos_diff_w[:, 2])
    reward_position = torch.exp(
            -100.0 * z_error
        )
    print('z_error:', z_error, 'reward_position:', reward_position)

    print('hhhhhhhhhhhh', reward_position * alignment_gate(env, asset_cfg))
    return reward_position * alignment_gate(env, asset_cfg)

def katao_insertion_reward_no_alignment(env, asset_cfg: SceneEntityCfg):
    """套入深度奖励，z_error 越小越好。"""
    katao = env.scene["katao"]
    stick_part = env.scene["stick_part"]
    katao_pos_w = katao.data.root_link_pose_w[:, :3]
    stick_pos_w = stick_part.data.root_link_pose_w[:, :3]

    pos_diff_w = katao_pos_w - stick_pos_w
    # pos_diff_w[:, 2] -= 0.001
    z_error = torch.abs(pos_diff_w[:, 2])
    reward_position = torch.exp(
            -100.0 * z_error
        )
    # print('z_error:', z_error, 'reward_position:', reward_position)

    # print('hhhhhhhhhhhh', reward_position * alignment_gate(env, asset_cfg))
    return reward_position

def insertion_gate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    z_threshold: float = 0.95,
) -> torch.Tensor:
    """套入深度完成判据：katao 与 stick_part 同时满足 1 个条件时返回 1.0，否则 0.0。

    各子奖励的阈值换算（方便按物理误差选阈值）：
    - katao_z_reward = exp(-100 * z_error)
      -> z_error = -ln(threshold) / 100，如 0.9 ≈ 1.05mm，0.7 ≈ 3.56mm，0.5 ≈ 6.93mm
    """
    inserted = (
        (katao_insertion_reward(env, asset_cfg) > z_threshold)
    )
    # print('套入深度判定结果:', inserted)
    return inserted.float()

# ==========================
# 旋拧奖励
# ==========================

def rotation_mission_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    rotation_axis_local: tuple[float, float, float] = (0.0, 1.0, 0.0),
    target_angle_deg: float = 45.0,
) -> torch.Tensor:
    """Reward for katao 轴与 pipe 轴呈目标夹角（katao 轴相对 pipe 轴旋转 target_angle_deg）。

    参考 katao_orientation_pipe_reward：把 katao 的豁口轴变换到 stick_part 坐标系，
    再与"目标方向"做点积。目标方向 = pipe 轴绕 rotation_axis_local 旋转 target_angle_deg
    后的方向（Rodrigues 公式）。

    - rotation_axis_local：旋转参考轴（stick_part 本体系）。默认 (0,1,0) 即 stick 竖直轴，
      相当于从竖直轴正方向往下看、绕它旋转 45 度。
    - target_angle_deg：目标夹角（度）。正负号决定旋转方向（绕 rotation_axis_local 的
      右手定则）；实际运行时若发现旋转方向与预期的"逆时针"相反，改成 -45.0 即可。

    注意：初始对齐状态（katao 轴与 pipe 轴重合）下该奖励约为 (cos45° + 1) / 2 ≈ 0.854，
    不是 1.0；只有 katao 轴转到目标方向时才为 1.0。
    """
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

    # Katao insertion axis
    slot_axis_local = torch.tensor(
        [0.0, 1.0, 0.0],
        device=katao_quat_w.device,
    ).expand(num_envs, -1)

    pipe_axis_local = torch.tensor(
        [0.0, 0.0, 1.0],
        device=katao_quat_w.device,
    ).expand(num_envs, -1)

    # katao 豁口轴 -> stick_part 坐标系
    slot_axis_stick = quat_apply(
        relative_quat,
        slot_axis_local,
    )

    # 目标方向：pipe 轴绕 rotation_axis_local 旋转 target_angle_deg（Rodrigues 公式）
    theta = torch.deg2rad(
        torch.tensor(float(target_angle_deg), device=katao_quat_w.device)
    )
    axis = torch.tensor(
        rotation_axis_local,
        dtype=torch.float32,
        device=katao_quat_w.device,
    )
    axis = axis / torch.linalg.norm(axis)
    axis_batched = axis.expand(num_envs, -1)

    target_axis = (
        pipe_axis_local * torch.cos(theta)
        + torch.linalg.cross(pipe_axis_local, axis_batched) * torch.sin(theta)
        + axis_batched
        * torch.sum(axis_batched * pipe_axis_local, dim=-1, keepdim=True)
        * (1.0 - torch.cos(theta))
    )

    cos_angle = torch.sum(
        slot_axis_stick * target_axis,
        dim=-1,
    )

    cos_angle = torch.clamp(
        cos_angle,
        -1.0,
        1.0,
    )

    reward_rotation = (
        cos_angle + 1.0
    ) * 0.5  # normalize to [0, 1]
    # print('the cos_angle:', cos_angle, 'the reward_rotation: ', reward_rotation)

    return reward_rotation * insertion_gate(env, asset_cfg)


# ==========================
# 成功奖励
# ==========================

def success_reward(env, asset_cfg: SceneEntityCfg):
    success = (
        (katao_xy_reward(env, asset_cfg) > 0.9)
        & (katao_insertion_reward(env, asset_cfg) > 0.9)
        & (rotation_mission_reward(env, asset_cfg) > 0.95)
        & (katao_orientation_z_reward(env, asset_cfg) > 0.95)
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
