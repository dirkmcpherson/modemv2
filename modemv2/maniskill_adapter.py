import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # registers envs

class ManiSkillEnvAdapter:
    def __init__(self, cfg, expected_B: int, expected_tail):
        """
        expected_tail is (C, H, W) from ReplayBuffer._obs.shape[2:].
        We will emit obs shaped (B, C, H, W) as float32.
        """
        self.B = int(expected_B)
        self.C_stacked, self.H, self.W = map(int, expected_tail)
        self.C = 3 # Mono-camera RGB
        
        from collections import deque
        self._num_frames = cfg.get("frame_stack", 1)
        self._frames = deque([], maxlen=self._num_frames)
        
        assert self.C_stacked == self.C * self._num_frames

        # Pick task + modes
        task_id = cfg.task
        if task_id.startswith("ms-"):
            task_id = task_id[3:]
        self.env = gym.make(
            task_id,
            num_envs=self.B,
            obs_mode="rgbd",                 # or "rgb"
            control_mode="pd_ee_delta_pose", # or whatever you want
            render_mode=None,                # keep None for headless smoke test
        )

        self.action_space = self.env.action_space
        # Create observation space matching the expected output shape (B, C_stacked, H, W)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(self.B, self.C_stacked, self.H, self.W), 
            dtype=np.uint8
        )
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {"render.modes": []}
        self.cfg = cfg
        
        # Determine state_dim
        obs, _ = self.env.reset(seed=0)
        self.state = self._extract_state(obs)
        self.state_dim = self.state.shape[0]
        cfg.state_dim = self.state_dim

        self.debug_once = True

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def base_env(self):
        return self

    def reset(self, seed=0):
        obs, _ = self.env.reset(seed=seed)
        self._current_step = 0
        self._terminated_early = False
        self._final_obs = None
        self._final_rew = 0.0
        self._final_info = {}

        # Always compute outputs
        img = self._extract_obs_image(obs)   # should be (B,3,H,W) uint8
        self.last_img = img  # Cache for render (unstacked)
        
        for _ in range(self._num_frames):
            self._frames.append(img)
            
        stacked_img = self._stacked_obs()
        self.state = self._extract_state(obs)  # should be (state_dim,) float32

        if self.debug_once:
            print("obs type:", type(obs))
            print("obs keys:", list(obs.keys()) if isinstance(obs, dict) else None)
            print("agent type:", type(obs["agent"]))
            if isinstance(obs["agent"], dict):
                print("agent keys:", obs["agent"].keys())
                print("qpos shape:", self._to_cpu_numpy(obs["agent"]["qpos"]).shape)
                print("qvel shape:", self._to_cpu_numpy(obs["agent"]["qvel"]).shape)

            sd = obs["sensor_data"]
            cam = sd["base_camera"]
            print("base_camera type:", type(cam))
            if isinstance(cam, dict):
                print("base_camera keys:", cam.keys())
                print("rgb shape:", self._to_cpu_numpy(cam["rgb"]).shape, "dtype:", self._to_cpu_numpy(cam["rgb"]).dtype)
                print("depth shape:", self._to_cpu_numpy(cam["depth"]).shape, "dtype:", self._to_cpu_numpy(cam["depth"]).dtype)

            print("stacked img shape:", stacked_img.shape, "dtype:", stacked_img.dtype)
            print("state len/dtype:", len(self.state), self.state.dtype)
            self.debug_once = False

        return stacked_img


    def _any_done(self, x):
        # x can be bool, numpy, torch CPU/CUDA, or vector
        if torch.is_tensor(x):
            return bool(x.detach().any().item())
        x = np.asarray(x)
        return bool(x.any())
    
    def _to_scalar_reward(self, r):
        if torch.is_tensor(r):
            r = r.detach()
            return float(r.mean().item()) if r.numel() > 1 else float(r.item())
        r = np.asarray(r, dtype=np.float32)
        return float(r.mean()) if r.size > 1 else float(r)

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        # Each frame is (B, 3, H, W). Concatenate on channel dim -> (B, 3*Stack, H, W)
        obs = np.concatenate(list(self._frames), axis=1)
        assert obs.shape[1] == self.C_stacked
        return obs

    def step(self, action_np):
        """
        action_np can be (action_dim,) or (B, action_dim).
        ManiSkill expects (B, action_dim) when num_envs=B.
        """
        self._current_step += 1
        
        if self._terminated_early:
            self._frames.append(self._final_obs)
            stacked_img = self._stacked_obs()
            done = self._current_step >= self.cfg.episode_length
            return stacked_img, self._final_rew, done, self._final_info

        action_np = np.asarray(action_np, dtype=np.float32)
        if action_np.ndim == 1:
            action_np = np.repeat(action_np[None, :], self.B, axis=0)

        self.last_eef_cmd = action_np # Store for logger

        obs, reward, terminated, truncated, info = self.env.step(action_np)
        env_done = self._any_done(terminated) or self._any_done(truncated)
        
        img = self._extract_obs_image(obs)
        self.last_img = img # Cache for render (unstacked)
        self._frames.append(img)
        stacked_img = self._stacked_obs()
        
        self.state = self._extract_state(obs)
        # reward from vector env might be shape (B,), make scalar
        rew = self._to_scalar_reward(reward)

        # If sparse rewards requested, use success signal from info
        if not self.cfg.dense_reward:
            if isinstance(info, dict) and "success" in info:
                rew = self._to_scalar_reward(info["success"])
        
        # persistence logic
        if env_done and self._current_step < self.cfg.episode_length:
            self._terminated_early = True
            self._final_obs = img
            self._final_rew = rew
            self._final_info = info
            done = False
        else:
            done = env_done or (self._current_step >= self.cfg.episode_length)
            
        return stacked_img, rew, done, info

    def render(self, mode="rgb_array", **kwargs):
        """
        Return the last observation image as a render.
        Logger expects (H, W, 3) for rgb_array.
        Our image is (B, 3, H, W).
        """
        if mode == "rgb_array" and hasattr(self, 'last_img') and self.last_img is not None:
            # Take first env, transpose (3, H, W) -> (H, W, 3)
            img = self.last_img[0].transpose(1, 2, 0)
            return img
        return None

    def _extract_state(self, obs):
        """
        obs['agent'] is a dict with qpos/qvel (and possibly more).
        Return UNBATCHED (state_dim,) float32.
        """
        agent = obs["agent"]
        assert isinstance(agent, dict), f"Expected obs['agent'] dict, got {type(agent)}"

        qpos = agent.get("qpos", None)
        qvel = agent.get("qvel", None)
        if qpos is None or qvel is None:
            raise KeyError(f"agent keys are {list(agent.keys())}, expected qpos/qvel")

        qpos = self._to_cpu_numpy(qpos).astype(np.float32).reshape(-1)
        qvel = self._to_cpu_numpy(qvel).astype(np.float32).reshape(-1)

        s = np.concatenate([qpos, qvel], axis=0)

        # Optional: include a few low-dim extras (ONLY if small)
        extra = obs.get("extra", None)
        if isinstance(extra, dict):
            s_extra = self._extract_extra_lowdim(extra)
            if s_extra is not None:
                s = np.concatenate([s, s_extra], axis=0)

        # Pad/trim to cfg.state_dim
        import omegaconf
        if omegaconf.OmegaConf.is_missing(self.cfg, "state_dim"):
            target = s.shape[0]
        else:
            target = int(self.cfg.state_dim)
            
        if s.shape[0] < target:
            s = np.pad(s, (0, target - s.shape[0]), mode="constant")
        elif s.shape[0] > target:
            s = s[:target]

        return s.astype(np.float32)

    def _extract_extra_lowdim(self, extra):
        parts = []
        for k, v in extra.items():
            a = self._to_cpu_numpy(v)
            if a.dtype.kind in "biuf":
                a = a.reshape(-1)
                if a.size <= 256:   # keep it small
                    parts.append(a.astype(np.float32))
        if not parts:
            return None
        out = np.concatenate(parts, axis=0)
        if out.size > 512:
            out = out[:512]
        return out


    def _extract_obs_image(self, obs):
        cam = obs["sensor_data"]["base_camera"]
        assert isinstance(cam, dict), f"Expected base_camera dict, got {type(cam)}"

        # common keys: rgb, depth, segmentation, etc
        if "rgb" not in cam:
            raise KeyError(f"base_camera keys are {list(cam.keys())}, expected 'rgb'")

        rgb = self._to_cpu_numpy(cam["rgb"])

        # likely (H,W,3) or (B,H,W,3). We need (B,3,H,W).
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            rgb = rgb[None, ...]  # add batch dim -> (1,H,W,3)

        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"Unexpected rgb shape: {rgb.shape}")

        # keep uint8 in 0..255 for ReplayBuffer
        if rgb.dtype != np.uint8:
            # ManiSkill might already return uint8; if float, convert carefully
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        rgb = np.transpose(rgb, (0, 3, 1, 2))  # (B,3,H,W) uint8

        # If you asked for num_envs=B, rgb might already be (B,H,W,3).
        # Ensure batch matches expected:
        if rgb.shape[0] != self.B:
            # If B>1 but camera only returns 1, replicate (temporary hack)
            if rgb.shape[0] == 1:
                rgb = np.repeat(rgb, self.B, axis=0)
            else:
                raise ValueError(f"RGB batch {rgb.shape[0]} != expected B {self.B}")

        # Must match buffer expected C/H/W exactly
        if (rgb.shape[1], rgb.shape[2], rgb.shape[3]) != (self.C, self.H, self.W):
            # Attempt resize using torch for convenience (since we have torch)
            import torch.nn.functional as F
            
            # (B, C, H_in, W_in) -> (B, C, H_out, W_out)
            t_in = torch.as_tensor(rgb).float() # interpolate needs float
            t_out = F.interpolate(t_in, size=(self.H, self.W), mode='bilinear', align_corners=False)
            
            # Back to uint8
            rgb = t_out.clamp(0, 255).byte().numpy()

        if (rgb.shape[1], rgb.shape[2], rgb.shape[3]) != (self.C, self.H, self.W):
            raise ValueError(
                f"RGB is (B,{rgb.shape[1]},{rgb.shape[2]},{rgb.shape[3]}) "
                f"but buffer expects (B,{self.C},{self.H},{self.W}). "
                "Adjust +obs_shape to match ManiSkill camera resolution."
            )

        return rgb



    def _state_to_image(self, state):
        # Minimal synthetic image fallback: (B, C, H, W)
        # state may be (B,D) or (D,)
        s = np.asarray(state, dtype=np.float32)
        if s.ndim == 1:
            s = np.repeat(s[None, :], self.B, axis=0)
        img = np.zeros((self.B, self.C, self.H, self.W), dtype=np.float32)
        flat = img.reshape(self.B, -1)
        m = min(flat.shape[1], s.shape[1])
        flat[:, :m] = 1.0 / (1.0 + np.exp(-s[:, :m]))
        return img

    def _flatten_dict_numeric(self, d):
        parts = []

        def rec(x):
            if isinstance(x, dict):
                for v in x.values():
                    rec(v)
            elif torch.is_tensor(x):
                a = x.detach().cpu().numpy()
                parts.append(a.reshape(-1))
            else:
                try:
                    a = np.asarray(x)
                    if a.dtype.kind in "biuf":
                        parts.append(a.reshape(-1))
                except Exception:
                    pass

        rec(d)

        if not parts:
            return np.zeros((self.B, 1), dtype=np.float32)

        out = np.concatenate(parts, axis=0)
        return out
    
    def _to_cpu_numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

