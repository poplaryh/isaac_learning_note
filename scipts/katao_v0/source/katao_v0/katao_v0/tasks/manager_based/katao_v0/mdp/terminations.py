# mdp/terminations.py
from .rewards import *

def time_out(env):
    return env.episode_length_buf >= env.max_episode_length

def success(env, asset_cfg: SceneEntityCfg):
    success = (
        (katao_xy_reward(env, asset_cfg) > 0.9)
        & (katao_z_reward(env, asset_cfg) > 0.9)
        # & (katao_orientation_pipe_reward(env, asset_cfg) > 0.98)
        # & (katao_orientation_z_reward(env, asset_cfg) > 0.98)
    )
    return success