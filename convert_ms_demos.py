import h5py
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add robohive to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT / "modemv2" / "tasks" / "robohive"))

from robohive.logger.grouped_datasets import Trace

def convert_ms_to_modemv2(h5_path, output_dir, task_name):
    print(f"Opening {h5_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        print(f"Found {len(traj_keys)} trajectories.")
        
        for k in traj_keys:
            traj = f[k]
            # obs/sensor_data/base_camera/rgb (N+1, 128, 128, 3)
            # obs/state (N+1, 42)
            # actions (N, 7)
            # success (N+1,)
            
            rgb = traj["obs/sensor_data/base_camera/rgb"][:]
            state = traj["obs/state"][:]
            actions = traj["actions"][:]
            success = traj["success"][:]
            
            # Decompose state (Assuming ManiSkill Panda 3 state structure)
            # qpos: 0-9, qvel: 9-18, tcp_pos: 18-21, tcp_quat: 21-25
            qp = state[:, :9]
            qv = state[:, 9:18]
            grasp_pos = state[:, 18:21]
            grasp_rot = state[:, 21:25]
            
            # Dummy errors and other fields expected by helper.py
            obj_err = np.zeros((state.shape[0], 3))
            tar_err = np.zeros((state.shape[0], 3))
            
            # Create the data structure matching RoboHive Trace
            # We'll save it as a dictionary that Trace.load can read, 
            # but it's easier to just use Trace directly if possible.
            
            trial_data = {
                "Trial0": {
                    "time": np.arange(state.shape[0]),
                    "observations": state, # placeholder
                    "actions": actions,
                    "rewards": np.zeros(state.shape[0]), # placeholder
                    "env_infos/time": np.arange(state.shape[0]),
                    "env_infos/solved": success,
                    "env_infos/done": success, # or similar
                    "env_infos/obs_dict/qp": qp,
                    "env_infos/obs_dict/qv": qv,
                    "env_infos/obs_dict/grasp_pos": grasp_pos,
                    "env_infos/obs_dict/grasp_rot": grasp_rot,
                    "env_infos/obs_dict/object_err": obj_err,
                    "env_infos/obs_dict/target_err": tar_err,
                    "env_infos/visual_dict/rgb:base_camera:128x128:2d": rgb,
                    "env_infos/visual_dict/d:base_camera:128x128:2d": np.zeros((rgb.shape[0], 1, rgb.shape[1], rgb.shape[2]), dtype=np.float32), # Dummy depth (N, 1, H, W)
                    "success": success
                }
            }
            
            # MoDemV2 Trace loader expects a specific structure: { Root: { Trial0: { ... } } }
            full_data = {
                "ManiSkill_Demos": {
                    "Trial0": trial_data["Trial0"]
                }
            }
            
            out_path = os.path.join(output_dir, f"{k}.pickle")
            print(f"Saving {out_path}...")
            with open(out_path, "wb") as pf:
                pickle.dump(full_data, pf)

if __name__ == "__main__":
    H5_PATH = "/home/j/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state+rgb.pd_ee_delta_pose.physx_cpu.h5"
    OUT_DIR = "/home/j/workspace/modemv2/demonstrations/ms-PickCube-v1"
    convert_ms_to_modemv2(H5_PATH, OUT_DIR, "ms-PickCube-v1")
