import os, sys, time
from pathlib import Path

from maniskill_adapter import ManiSkillEnvAdapter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gym
import hydra

from cfg_parse import parse_cfg
from env import set_seed
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer

torch.backends.cudnn.benchmark = True

import types
import torch

def patch_replaybuffer_sample():
    import algorithm.helper as h

    orig_sample = h.ReplayBuffer.sample

    def sample_patched(self, *args, **kwargs):
        """
        Same as ReplayBuffer.sample(), but fixes the uint8/float mismatch
        in the mask write by forcing computations to match obs dtype.
        """
        try:
            return orig_sample(self, *args, **kwargs)
        except RuntimeError as e:
            msg = str(e)
            if "Index put requires the source and destination dtypes match" not in msg:
                raise

            # Re-run a slightly modified version: make next_obs float so assignments work.
            # Easiest approach: temporarily view internal obs storage as float for the sample.
            # We do this by cloning/casting the tensors used inside sample.

            # ---- Minimal reimplementation approach ----
            # We call the original sample, but we need to intercept *inside*.
            # Since we can't, we reproduce the function with just the dtype fix.
            raise RuntimeError(
                "ReplayBuffer.sample dtype mismatch hit. "
                "Use Option B below (copy/paste patched sample) or patch helper.py."
            ) from e

    h.ReplayBuffer.sample = sample_patched


class ToyEnv:
    def __init__(self, cfg, batch_size: int, obs_tail_shape):
        """
        obs_tail_shape is the shape AFTER the batch dim, e.g. (C,H,W) or (K,C,H,W) etc.
        We'll return obs shaped (B, *obs_tail_shape).
        """
        self.B = int(batch_size)
        self.obs_tail_shape = tuple(int(x) for x in obs_tail_shape)

        self.action_dim = int(getattr(cfg, "action_dim", 2))
        self.state_dim = int(getattr(cfg, "state_dim", 8))
        self.horizon = int(getattr(cfg, "episode_length", 50))
        self.t = 0
        self.rng = np.random.RandomState(int(getattr(cfg, "seed", 0)))

        self.state = None
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

    def reset(self):
        self.t = 0
        self.state = self.rng.randn(self.state_dim).astype(np.float32)
        return self._obs()

    def _obs(self):
        # Make a deterministic-ish float32 observation that matches buffer shape exactly.
        # (If the code later expects uint8, we can switch, but wait for an error.)
        obs = self.rng.rand(self.B, *self.obs_tail_shape).astype(np.float32)

        # Encode a little state signal into the first few elements (safe for any shape)
        flat = obs.reshape(self.B, -1)
        for i in range(min(self.state_dim, flat.shape[1])):
            flat[:, i] = 1.0 / (1.0 + np.exp(-self.state[i]))
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # accept either (action_dim,) or (B, action_dim)
        if action.ndim == 2:
            action = action[0]  # use first element -> (action_dim,)

        action = np.clip(action, -1.0, 1.0)

        self.t += 1

        pad = np.zeros((self.state_dim,), dtype=np.float32)
        pad[: min(self.action_dim, self.state_dim)] = action[: min(self.action_dim, self.state_dim)]

        noise = 0.01 * self.rng.randn(self.state_dim).astype(np.float32)
        self.state = 0.95 * self.state + 0.1 * pad + noise

        norm = float(np.linalg.norm(self.state))
        reward = -norm
        done = self.t >= self.horizon
        info = {"success": float(norm < 0.5)}
        return self._obs(), reward, done, info



@hydra.main(config_path="cfgs", config_name="config", version_base="1.1")
def main(cfg):
    assert torch.cuda.is_available(), "CUDA not available (your train() asserts this too)."

    # Convert Hydra config -> your internal config object/dict
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    # Override task label so you don't collide with real logs
    cfg.task = "toy-hydra"
    cfg.exp_name = "toy-hydra"

    agent = TDMPC(cfg)
    buf = ReplayBuffer(cfg)

    # Ask the buffer what observation shape it expects.
    # It should have shape like (capacity_or_T, B, ...obs_tail...)
    expected_B = int(buf._obs.shape[1])
    expected_tail = tuple(buf._obs.shape[2:])

    print("Buffer expects obs shape:", (expected_B, *expected_tail))

    from maniskill_adapter import ManiSkillEnvAdapter
    # env = ToyEnv(cfg, batch_size=expected_B, obs_tail_shape=expected_tail)
    env = ManiSkillEnvAdapter(cfg, expected_B, expected_tail)

    # Collect 2 episodes, then do a handful of updates
    total_steps = 0
    for ep in range(2):
        obs = env.reset()
        episode = Episode(cfg, obs, env.state)
        t = 0
        succ = 0.0

        while not episode.done:
            action, _ = agent.plan(obs, env.state, step=total_steps, t=t)

            # Convert ONCE for env
            action_np = action.detach().cpu().numpy()

            # Step env with numpy action
            obs2, reward, done, info = env.step(action_np)

            # Store unbatched torch action for Episode
            if action_np.ndim == 2:     # (B, action_dim)
                a_ep = action_np[0]     # (action_dim,)
            else:
                a_ep = action_np
            a_ep = torch.as_tensor(a_ep, device=action.device, dtype=action.dtype)

            episode += (obs2, env.state, a_ep, reward, done)


            s = info.get("success", 0)
            if torch.is_tensor(s):
                succ += float(s.detach().float().mean().item())   # or .any().item() if you prefer
            else:
                s = np.asarray(s, dtype=np.float32)
                succ += float(s.mean()) if s.size > 1 else float(s)
            obs = obs2
            t += 1
            total_steps += 1

        buf += episode
        print(f"ep {ep} len={len(episode)} rew={episode.cumulative_reward:.3f} succ_count={succ}")

    # Do a small number of updates (this is the real “does tdmpc plumbing work?” test)
    for i in range(20000):
        metrics = agent.update(buf, total_steps + i, demo_buffer=None, train_pi=True)
        if (i + 1) % 100 == 0:
            print("update", i + 1, "total_loss", float(metrics.get("total_loss", 0.0)))

    if isinstance(metrics, dict):
        print("Update OK. Metric keys sample:", list(metrics.keys())[:12])
    else:
        print("Update OK. Metrics type:", type(metrics))

    print("✅ Hydra-config-compatible toy test passed.")


if __name__ == "__main__":
    main()
