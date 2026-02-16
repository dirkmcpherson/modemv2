import h5py
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add robohive to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT / "modemv2" / "tasks" / "robohive"))

# ManiSkill PickCube-v1 flat state layout (42-dim):
#   [0:9]   qpos
#   [9:18]  qvel
#   [18]    is_grasped
#   [19:26] tcp_pose (pos=3 + quat=4)
#   [26:29] goal_pos
#   [29:32] obj_pos
#   [32:36] obj_quat
#   [36:39] obj_to_tcp_pos
#   [39:42] obj_to_goal_pos

def convert_ms_to_modemv2(h5_path, output_dir, two_cameras=False, depth=False):
    print(f"Opening {h5_path}...")
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        print(f"Found {len(traj_keys)} trajectories.")

        # Check available sensors in sample trajectory
        sample_traj = f[traj_keys[0]]
        has_hand_camera = "obs/sensor_data/hand_camera/rgb" in sample_traj
        has_depth = "obs/sensor_data/base_camera/depth" in sample_traj
        if two_cameras and not has_hand_camera:
            print("WARNING: --two_cameras requested but hand_camera not found in h5. Using base_camera only.")
            two_cameras = False
        if depth and not has_depth:
            print("WARNING: --depth requested but depth not found in h5. Skipping depth.")
            print("  (Regenerate demos with obs_mode='state+rgbd' to include depth.)")
            depth = False

        for k in traj_keys:
            traj = f[k]

            base_rgb = traj["obs/sensor_data/base_camera/rgb"][:]
            state = traj["obs/state"][:]
            actions = traj["actions"][:]
            success = traj["success"][:]
            rewards = traj["rewards"][:]

            # Decompose 42-dim flat state to match dreamerv3 env's _extract_state:
            # 29-dim = qpos(9) + qvel(9) + is_grasped(1) + tcp_pose(7) + goal_pos(3)
            qp = state[:, 0:9]
            qv = state[:, 9:18]
            is_grasped = state[:, 18:19]     # (N, 1)
            tcp_pos = state[:, 19:22]        # (N, 3)
            tcp_rot = state[:, 22:26]        # (N, 4)
            goal_pos = state[:, 26:29]       # (N, 3)

            # Dummy errors (unused but expected by helper.py for non-PickCube tasks)
            obj_err = np.zeros((state.shape[0], 3))
            tar_err = np.zeros((state.shape[0], 3))

            trial_data = {
                "time": np.arange(state.shape[0]),
                "observations": state,
                "actions": actions,
                "rewards": rewards,
                "env_infos/time": np.arange(state.shape[0]),
                "env_infos/solved": success,
                "env_infos/rwd_dense": rewards,
                "env_infos/done": success,
                "env_infos/obs_dict/qp": qp,
                "env_infos/obs_dict/qv": qv,
                "env_infos/obs_dict/grasp_pos": tcp_pos,
                "env_infos/obs_dict/grasp_rot": tcp_rot,
                "env_infos/obs_dict/is_grasped": is_grasped,
                "env_infos/obs_dict/goal_pos": goal_pos,
                "env_infos/obs_dict/object_err": obj_err,
                "env_infos/obs_dict/target_err": tar_err,
                "env_infos/visual_dict/rgb:base_camera:128x128:2d": base_rgb,
                "success": success,
            }

            if depth:
                # ManiSkill depth: (N, H, W, 1) float16/32 → store as (N, 1, H, W) float32
                base_depth = traj["obs/sensor_data/base_camera/depth"][:].astype(np.float32)
                if base_depth.ndim == 4 and base_depth.shape[-1] == 1:
                    base_depth = base_depth.transpose(0, 3, 1, 2)  # (N, 1, H, W)
                trial_data["env_infos/visual_dict/d:base_camera:128x128:2d"] = base_depth

            if two_cameras:
                hand_rgb = traj["obs/sensor_data/hand_camera/rgb"][:]
                trial_data["env_infos/visual_dict/rgb:hand_camera:128x128:2d"] = hand_rgb
                if depth:
                    hand_depth = traj["obs/sensor_data/hand_camera/depth"][:].astype(np.float32)
                    if hand_depth.ndim == 4 and hand_depth.shape[-1] == 1:
                        hand_depth = hand_depth.transpose(0, 3, 1, 2)
                    trial_data["env_infos/visual_dict/d:hand_camera:128x128:2d"] = hand_depth

            full_data = {
                "ManiSkill_Demos": {
                    "Trial0": trial_data
                }
            }

            out_path = os.path.join(output_dir, f"{k}.pickle")
            print(f"Saving {out_path}...")
            with open(out_path, "wb") as pf:
                pickle.dump(full_data, pf)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--teleop', action='store_true')
    parser.add_argument('--two_cameras', action='store_true',
                        help='Include hand_camera RGB alongside base_camera (6-channel obs)')
    parser.add_argument('--depth', action='store_true',
                        help='Include depth maps (requires h5 generated with rgbd obs mode)')
    args = parser.parse_args()

    if args.teleop:
        H5_PATH = "/home/j/.maniskill/demos/PickCube-v1/teleop/trajectory.state+rgb+depth.pd_ee_delta_pose.physx_cpu.h5" if args.depth else "/home/j/.maniskill/demos/PickCube-v1/teleop/trajectory.state+rgb.pd_ee_delta_pose.physx_cpu.h5"
        OUT_DIR = "/home/j/workspace/modemv2/demonstrations/ms-PickCube-v1-teleop"
    else:
        H5_PATH = "/home/j/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state+rgb+depth.pd_ee_delta_pose.physx_cpu.h5" if args.depth else "/home/j/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state+rgb.pd_ee_delta_pose.physx_cpu.h5"
        OUT_DIR = "/home/j/workspace/modemv2/demonstrations/ms-PickCube-v1"
    convert_ms_to_modemv2(H5_PATH, OUT_DIR, two_cameras=args.two_cameras, depth=args.depth)
