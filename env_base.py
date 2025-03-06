from typing import Dict, List, Any
import numpy as np

class EnvBase:
    """Base class for vectorized environments"""
    def __init__(self, num_envs, render=False):
        self.num_envs = num_envs
        self.render = render
        self.dones = [False] * num_envs
        self.episode_counts = [0] * num_envs

    def step(self, actions_dict_list):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError