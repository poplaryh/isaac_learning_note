# mdp/terminations.py
from .rewards import *

def time_out(env):
    return env.episode_length_buf >= env.max_episode_length

def success(env, asset_cfg: SceneEntityCfg):
    success = (
        (katao_xy_reward(env, asset_cfg) > 0.9)
        & (katao_insertion_reward(env, asset_cfg) > 0.9)
        & (rotation_mission_reward(env, asset_cfg) > 0.95)
        & (katao_orientation_z_reward(env, asset_cfg) > 0.95)
        & (katao_grasped(env, asset_cfg) > 0.5)
    )
    return success

def not_grasped(env, asset_cfg: SceneEntityCfg, grace_steps: int = 0):
    """判断是否未抓取，若未抓取则终止。

    Args:
        env: 环境对象。
        asset_cfg: 场景实体配置。
        grace_steps: 宽限步数。episode 开始后前 grace_steps 步内不触发终止，
            给机械臂留出接近并抓取 katao 的时间（初始状态下物体未被夹持）。
    Returns:
        bool: 若未抓取则返回 True，否则返回 False。
    """
    grasped = katao_grasped(env, asset_cfg)
    not_grasped = (grasped < 0.5)  # 若抓取状态小于 0.5，则认为未抓取
    # 宽限期：episode 开始的前 grace_steps 步不终止
    if grace_steps > 0:
        not_grasped = not_grasped & (env.episode_length_buf > grace_steps)
    return not_grasped