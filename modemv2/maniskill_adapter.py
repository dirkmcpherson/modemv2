import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # registers envs

class ManiSkillEnvAdapter:
    def __init__(self, cfg, expected_B: int, expected_tail):
        """
        expected_tail is (C, H, W) — per-view channels (with frame_stack), height, width.
        obs_shape will be (num_cameras, C, H, W) matching the franka wrapper convention.
        """
        self.B = int(expected_B)
        self.C_stacked, self.H, self.W = map(int, expected_tail)
        self.use_depth = getattr(cfg, "use_depth", True)
        self.C = 4 if self.use_depth else 3  # RGBD or RGB per frame per camera

        from collections import deque
        self._num_frames = cfg.get("frame_stack", 1)
        self._frames = deque([], maxlen=self._num_frames)

        assert self.C_stacked == self.C * self._num_frames

        # Camera views from config
        self.camera_names = list(cfg.get("camera_views", ["base_camera"]))
        self._num_cameras = len(self.camera_names)

        # Use panda_wristcam if hand_camera is requested
        needs_wristcam = any("hand" in c for c in self.camera_names)
        robot_uids = "panda_wristcam" if needs_wristcam else "panda"

        # Pick task + modes
        task_id = cfg.task
        if task_id.startswith("ms-"):
            task_id = task_id[3:]
        self.env = gym.make(
            task_id,
            num_envs=self.B,
            obs_mode="rgbd",
            control_mode="pd_ee_delta_pose",
            render_mode=None,
            robot_uids=robot_uids,
        )

        self.action_space = self.env.action_space
        # observation_space: (num_cameras, C_stacked, H, W) — matches franka convention
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self._num_cameras, self.C_stacked, self.H, self.W),
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

        img = self._extract_obs_image(obs)  # (num_cameras, C, H, W) uint8
        self.last_img = img

        for _ in range(self._num_frames):
            self._frames.append(img)

        stacked_img = self._stacked_obs()
        self.state = self._extract_state(obs)

        if self.debug_once:
            print(f"ManiSkillAdapter cameras: {self.camera_names}")
            print(f"  per-frame img shape (RGBD): {img.shape}")
            print(f"  stacked img shape:          {stacked_img.shape}")
            print(f"  state shape:                {self.state.shape}")
            sd = obs["sensor_data"]
            for cam_name in self.camera_names:
                if cam_name in sd:
                    cam = sd[cam_name]
                    print(f"  {cam_name} rgb:   {self._to_cpu_numpy(cam['rgb']).shape}")
                    print(f"  {cam_name} depth: {self._to_cpu_numpy(cam['depth']).shape}")
            self.debug_once = False

        return stacked_img

    def _any_done(self, x):
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
        # Each frame is (num_cameras, C, H, W). Concatenate on channel dim -> (num_cameras, C*Stack, H, W)
        obs = np.concatenate(list(self._frames), axis=1)
        assert obs.shape[1] == self.C_stacked
        return obs

    def step(self, action_np):
        self._current_step += 1

        if self._terminated_early:
            self._frames.append(self._final_obs)
            stacked_img = self._stacked_obs()
            done = self._current_step >= self.cfg.episode_length
            return stacked_img, self._final_rew, done, self._final_info

        action_np = np.asarray(action_np, dtype=np.float32)
        if action_np.ndim == 1:
            action_np = np.repeat(action_np[None, :], self.B, axis=0)

        self.last_eef_cmd = action_np

        obs, reward, terminated, truncated, info = self.env.step(action_np)
        env_done = self._any_done(terminated) or self._any_done(truncated)

        img = self._extract_obs_image(obs)
        self.last_img = img
        self._frames.append(img)
        stacked_img = self._stacked_obs()

        self.state = self._extract_state(obs)
        rew = self._to_scalar_reward(reward)

        if not self.cfg.dense_reward:
            if isinstance(info, dict) and "success" in info:
                rew = self._to_scalar_reward(info["success"])

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
        if mode == "rgb_array" and hasattr(self, 'last_img') and self.last_img is not None:
            # Take first camera view, RGB only (first 3 channels), transpose (C, H, W) -> (H, W, C)
            img = self.last_img[0, :3].transpose(1, 2, 0)
            return img
        return None

    def _extract_state(self, obs):
        agent = obs["agent"]
        assert isinstance(agent, dict), f"Expected obs['agent'] dict, got {type(agent)}"

        qpos = agent.get("qpos", None)
        qvel = agent.get("qvel", None)
        if qpos is None or qvel is None:
            raise KeyError(f"agent keys are {list(agent.keys())}, expected qpos/qvel")

        qpos = self._to_cpu_numpy(qpos).astype(np.float32).reshape(-1)
        qvel = self._to_cpu_numpy(qvel).astype(np.float32).reshape(-1)

        s = np.concatenate([qpos, qvel], axis=0)

        extra = obs.get("extra", None)
        if isinstance(extra, dict):
            s_extra = self._extract_extra_lowdim(extra)
            if s_extra is not None:
                s = np.concatenate([s, s_extra], axis=0)

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
                if a.size <= 256:
                    parts.append(a.astype(np.float32))
        if not parts:
            return None
        out = np.concatenate(parts, axis=0)
        if out.size > 512:
            out = out[:512]
        return out

    def _extract_obs_image(self, obs):
        """Extract RGB or RGBD from all configured cameras.
        Returns (num_cameras, C, H, W) — C is 3 (RGB) or 4 (RGBD) based on use_depth.
        When depth is included, np.concatenate upcasts to float32 (matching franka sim).
        """
        import torch.nn.functional as F

        views = []
        sd = obs["sensor_data"]
        for cam_name in self.camera_names:
            if cam_name not in sd:
                raise KeyError(f"Camera '{cam_name}' not in sensor_data. Available: {list(sd.keys())}")
            cam = sd[cam_name]
            if "rgb" not in cam:
                raise KeyError(f"{cam_name} keys are {list(cam.keys())}, expected 'rgb'")

            # --- RGB ---
            rgb = self._to_cpu_numpy(cam["rgb"])
            if rgb.ndim == 4:
                rgb = rgb[0]  # take first env: (H, W, 3)
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"Unexpected rgb shape from {cam_name}: {rgb.shape}")
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb = np.transpose(rgb, (2, 0, 1))  # (3, H, W)

            # Resize RGB if needed
            if (rgb.shape[1], rgb.shape[2]) != (self.H, self.W):
                t_in = torch.as_tensor(rgb).float().unsqueeze(0)
                t_out = F.interpolate(t_in, size=(self.H, self.W), mode='bilinear', align_corners=False)
                rgb = t_out.squeeze(0).clamp(0, 255).byte().numpy()

            if not self.use_depth:
                views.append(rgb)
                continue

            # --- Depth: scale meters to [0, 255] to match franka convention ---
            # Robohive scales MuJoCo [0,1] depth by 255. ManiSkill depth is in meters.
            # Scale: 1m = 100, clip to [0, 255] (max ~2.55m), matching franka's [0, 255] range.
            depth = self._to_cpu_numpy(cam["depth"]).astype(np.float32)
            if depth.ndim == 4:
                depth = depth[0]  # (H, W, 1)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[:, :, 0]  # (H, W)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth * 100.0, 0, 255).astype(np.uint8)
            depth = depth[np.newaxis, :, :]  # (1, H, W)

            # Resize depth if needed
            if (depth.shape[1], depth.shape[2]) != (self.H, self.W):
                t_in = torch.as_tensor(depth).float().unsqueeze(0)
                t_out = F.interpolate(t_in, size=(self.H, self.W), mode='bilinear', align_corners=False)
                depth = t_out.squeeze(0).clamp(0, 255).byte().numpy()

            rgbd = np.concatenate([rgb, depth], axis=0)  # (4, H, W)
            views.append(rgbd)

        return np.stack(views, axis=0)  # (num_cameras, C, H, W)

    def _to_cpu_numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
