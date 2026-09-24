# mdp/terminations.py

def time_out(env):
    return env.episode_length_buf >= env.max_episode_length