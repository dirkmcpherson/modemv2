import numpy as np
import torch
import gymnasium as gym
import gym_pusht
from collections import deque

class PushTEnvAdapter:
    def __init__(self, cfg, expected_B: int, expected_tail):
        """
        expected_tail is (C, H, W) where C is stacked total.
        """
        self.B = int(expected_B)
        self.C_stacked, self.H, self.W = map(int, expected_tail)
        self.C = 3 # RGB
        
        self.cfg = cfg
        self._num_frames = cfg.get("frame_stack", 1)
        self._frames = deque([], maxlen=self._num_frames)
        
        assert self.C_stacked == self.C * self._num_frames

        # Strip 'pusht-' prefix if present
        task_id = cfg.task
        if task_id.startswith("pusht-"):
            task_id = task_id[6:] 
        # In this repo it seems to be gym_pusht/PushT-v0
        if not task_id.startswith("gym_pusht/"):
            task_id = f"gym_pusht/{task_id}"

        self.env = gym.make(
            task_id,
            obs_type="pixels_state",
            render_mode="rgb_array",
            observation_width=self.W,
            observation_height=self.H
        )

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(self.B, self.C_stacked, self.H, self.W), 
            dtype=np.uint8
        )
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {"render.modes": ["rgb_array"]}
        
        # Determine state_dim
        obs, _ = self.env.reset(seed=0)
        self.state = self._extract_state(obs)
        self.state_dim = self.state.shape[0]
        cfg.state_dim = self.state_dim

        self._current_step = 0
        self._terminated_early = False
        self._final_obs = None
        self._final_rew = 0.0
        self._final_info = {}
        self.last_eef_cmd = None

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def base_env(self):
        return self

    def reset(self, seed=0):
        obs, info = self.env.reset(seed=seed)
        self._current_step = 0
        self._terminated_early = False
        self._final_obs = None
        self._final_rew = 0.0
        self._final_info = {}

        img = self._extract_obs_image(obs)
        self.last_img = img
        
        for _ in range(self._num_frames):
            self._frames.append(img)
            
        stacked_img = self._stacked_obs()
        self.state = self._extract_state(obs)
        return stacked_img

    def step(self, action_np):
        self._current_step += 1
        
        if self._terminated_early:
            self._frames.append(self._final_obs)
            stacked_img = self._stacked_obs()
            done = self._current_step >= self.cfg.episode_length
            return stacked_img, self._final_rew, done, self._final_info

        action_np = np.asarray(action_np, dtype=np.float32)
        if action_np.ndim == 2 and action_np.shape[0] == 1:
            action_np = action_np[0] # PushT expects (2,) for single env
        
        self.last_eef_cmd = action_np

        obs, reward, terminated, truncated, info = self.env.step(action_np)
        env_done = terminated or truncated
        
        img = self._extract_obs_image(obs)
        self.last_img = img
        self._frames.append(img)
        stacked_img = self._stacked_obs()
        
        self.state = self._extract_state(obs)
        rew = float(reward)
        
        if env_done and self._current_step < self.cfg.episode_length:
            self._terminated_early = True
            self._final_obs = img
            self._final_rew = rew
            self._final_info = info
            done = False
        else:
            done = env_done or (self._current_step >= self.cfg.episode_length)
            
        return stacked_img, rew, done, info

    def _stacked_obs(self):
        obs = np.concatenate(list(self._frames), axis=1) # (B, 3*S, H, W)
        return obs

    def _extract_obs_image(self, obs):
        # obs is dict with 'pixels' and 'state'
        pixels = obs['pixels'] # (H, W, 3) uint8
        # Add B dimension and transpose to (B, C, H, W)
        img = pixels[None, ...].transpose(0, 3, 1, 2)
        if img.shape[0] != self.B:
             img = np.repeat(img, self.B, axis=0)
        return img

    def _extract_state(self, obs):
        # obs['state'] is [agent_x, agent_y, block_x, block_y, block_angle]
        s = np.asarray(obs['state'], dtype=np.float32).reshape(-1)
        return s

    def render(self, mode="rgb_array", **kwargs):
        if mode == "rgb_array":
            return self.env.render()
        return None
