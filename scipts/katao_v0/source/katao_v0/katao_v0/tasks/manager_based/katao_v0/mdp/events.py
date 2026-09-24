# mdp/events.py

import torch

from isaaclab.managers import SceneEntityCfg


# ============================================================
# 你的 recorded state
# ============================================================

RECORDED_JOINT_POS = torch.tensor(
    [
        -0.0275,
         0.7459,
         0.0415,
        -1.2212,
        -0.0327,
         1.9685,
        -0.2418,
         0.0188,
         0.0194,
    ],
    dtype=torch.float32,
)


RECORDED_JOINT_VEL = torch.tensor(
    [
        -0.0116,
         0.0170,
         0.0128,
         0.0344,
        -0.0072,
        -0.0289,
        -0.0144,
        -0.0003,
        -0.0043,
    ],
    dtype=torch.float32,
)


# [x, y, z, qw, qx, qy, qz]
RECORDED_KATAO_POSE = torch.tensor(
    [
        0.75,
        0,
        0.2318,
        1.0000,
        -0.0058467,
        0.0018718,
        0,
    ],
    dtype=torch.float32,
)


# [vx, vy, vz, wx, wy, wz]
# RECORDED_KATAO_VEL = torch.tensor(
#     [
#          0.0683,
#          0.0645,
#          0.06,
#          3.4959,
#          3.4079,
#          11.1351,
#     ],
#     dtype=torch.float32,
# )
RECORDED_KATAO_VEL = torch.zeros(
    6,
    dtype=torch.float32,
)

# ============================================================
# Reset
# ============================================================

def reset_to_recorded_state(
    env,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    katao_cfg: SceneEntityCfg = SceneEntityCfg("katao"),
):

    robot = env.scene[robot_cfg.name]
    katao = env.scene[katao_cfg.name]

    num_reset = len(env_ids)

    device = env.device

    # --------------------------------------------------------
    # Robot
    # --------------------------------------------------------

    joint_pos = (
        RECORDED_JOINT_POS
        .to(device)
        .unsqueeze(0)
        .repeat(num_reset, 1)
    )

    joint_vel = (
        RECORDED_JOINT_VEL
        .to(device)
        .unsqueeze(0)
        .repeat(num_reset, 1)
    )

    # --------------------------------------------------------
    # Katao
    # --------------------------------------------------------

    katao_pose = (
        RECORDED_KATAO_POSE
        .to(device)
        .unsqueeze(0)
        .repeat(num_reset, 1)
    )

    katao_pose[:, :3] += env.scene.env_origins[env_ids]

    katao_vel = (
        RECORDED_KATAO_VEL
        .to(device)
        .unsqueeze(0)
        .repeat(num_reset, 1)
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # 你的原代码记录的是 root_link_pose_w
    #
    # 因此这里首先按照 world pose 写入。
    # --------------------------------------------------------

    robot.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
    )

    # katao.write_root_state_to_sim(
    #     katao_pose,
    #     katao_vel,
    #     env_ids=env_ids,
    # )
    katao.write_root_pose_to_sim(
        katao_pose,
        env_ids=env_ids,
    )

    katao.write_root_velocity_to_sim(
        katao_vel,
        env_ids=env_ids,
    )

    # --------------------------------------------------------
    # 清空/重置内部 buffer
    # --------------------------------------------------------

    robot.reset(env_ids)
    katao.reset(env_ids)
    
    print('-----------------------')
    print('reset to recorded state!!!')
    print('-----------------------')

    # robot.set_joint_position_target_index(
    #     target=joint_pos,
    #     env_ids=env_ids,
    # )

    # robot.set_joint_velocity_target_index(
    #     target=joint_vel,
    #     env_ids=env_ids,
    # )