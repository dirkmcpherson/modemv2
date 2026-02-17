# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import numpy as np
import torch
import random
import gym
import warnings
from maniskill_adapter import ManiSkillEnvAdapter
from pusht_adapter import PushTEnvAdapter

warnings.filterwarnings("ignore", category=DeprecationWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DefaultDictWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.success = False

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, **kwargs):
        self.success = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = defaultdict(float, info)
        self.success = self.success or bool(info["success"])
        info["success"] = float(self.success)
        return obs, reward, done, info


class SqueezeBatchWrapper(gym.Wrapper):
    """
    ManiskillEnvAdapter returns (B, ...) obs.
    If B=1, we want (...,) to match other ModemV2 envs.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._squeeze(obs)

    def step(self, action):
        # Allow (act_dim,) action input by adding batch dim
        if action.ndim == 1:
            action = action[None, ...] 
            
        obs, reward, done, info = self.env.step(action)
        return self._squeeze(obs), reward, done, info

    def _squeeze(self, x):
        # rgb is (B, C, H, W) -> (C, H, W)
        if x.ndim == 4 and x.shape[0] == 1:
            return x[0]
        return x


def make_env(cfg):
    """
    Make environment for experiments.
    """
    domain, _ = cfg.task.split("-", 1)
    if domain == "mw":  # Meta-World
        from tasks.metaworld import make_metaworld_env

        env = make_metaworld_env(cfg)
    elif domain == "adroit":  # Adroit
        from tasks.adroit import make_adroit_env

        env = make_adroit_env(cfg)
    elif domain == 'franka': # Franka
        from tasks.franka import make_franka_env


        env = make_franka_env(cfg)
    elif domain == 'ms': # Maniskill
        channels_per_frame = 4 if getattr(cfg, "use_depth", True) else 3
        c = channels_per_frame * getattr(cfg, "frame_stack", 1)
        h = getattr(cfg, "img_size", 128)
        w = h
        
        env = ManiSkillEnvAdapter(
            cfg, 
            expected_B=1, 
            expected_tail=(c, h, w)
        )
    elif domain == 'pusht': # PushT
        c = 3 * getattr(cfg, "frame_stack", 1)
        h = getattr(cfg, "img_size", 128)
        w = h
        
        env = PushTEnvAdapter(
            cfg, 
            expected_B=1, 
            expected_tail=(c, h, w)
        )

        
    else:  # DMControl
        from tasks.dmcontrol import make_dmcontrol_env

        env = make_dmcontrol_env(cfg)

    env = DefaultDictWrapper(env)
    cfg.domain = domain
    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]
    if hasattr(env, 'state_dim'):
        cfg.state_dim = env.state_dim
    return env
