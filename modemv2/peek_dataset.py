"""
Peek at MoDemV2 datasets: .pickle demo files or .npz episode files.

Usage:
    # Print value ranges for all arrays in demo pickles
    python peek_dataset.py demonstrations/ms-PickCube-v1 --ranges

    # Play back RGB frames with reward/action overlays
    python peek_dataset.py demonstrations/ms-PickCube-v1

    # Show per-dimension action histograms
    python peek_dataset.py demonstrations/ms-PickCube-v1 --actions
"""

import argparse
import glob
import os
import pickle
import time
import numpy as np


# --------------------------------------------------------------------------- #
#  Helpers for navigating MoDemV2 pickle structure
# --------------------------------------------------------------------------- #

def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_trial(data):
    """
    MoDemV2 pickles have structure: {root_key: {Trial0: {...}}}.
    Drill down to the trial-level dict.
    """
    if not isinstance(data, dict):
        return data
    # If any key looks like env_infos/* or 'actions', we're already at trial level
    if any(k.startswith("env_infos") or k in ("actions", "rewards", "observations") for k in data.keys()):
        return data
    # Otherwise drill into the first nested dict
    for v in data.values():
        if isinstance(v, dict):
            result = _find_trial(v)
            if result is not None:
                return result
    return data


def _find_visual_keys(trial):
    """Find all RGB and depth keys grouped by camera name.
    Returns dict: {camera_name: {'rgb': key, 'depth': key or None}}
    """
    cameras = {}
    for k in trial.keys():
        if 'visual_dict' not in k:
            continue
        # Keys look like: env_infos/visual_dict/rgb:base_camera:128x128:2d
        parts = k.split('/')[-1]  # e.g. rgb:base_camera:128x128:2d
        tokens = parts.split(':')
        if len(tokens) < 2:
            continue
        modality = tokens[0]  # 'rgb' or 'd'
        cam_name = tokens[1]  # 'base_camera', 'hand_camera', etc.
        if cam_name not in cameras:
            cameras[cam_name] = {'rgb': None, 'depth': None}
        if modality == 'rgb':
            cameras[cam_name]['rgb'] = k
        elif modality == 'd':
            cameras[cam_name]['depth'] = k
    return cameras


def _depth_to_colormap(depth_img):
    """Convert a single-channel depth image to a BGR colormap for display.
    depth_img: (H, W) float array.
    """
    import cv2
    valid = depth_img[np.isfinite(depth_img)]
    if len(valid) == 0:
        return np.zeros((*depth_img.shape, 3), dtype=np.uint8)
    vmin, vmax = valid.min(), valid.max()
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    normalized = np.clip((depth_img - vmin) / (vmax - vmin), 0, 1)
    gray = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


# --------------------------------------------------------------------------- #
#  Ranges mode
# --------------------------------------------------------------------------- #

def print_ranges(directory, max_files=None):
    files = _gather_files(directory)
    if not files:
        return

    count = 0
    for fp in files:
        if max_files and count >= max_files:
            break
        count += 1
        print(f"\nFile: {os.path.basename(fp)}")
        try:
            if fp.endswith(".pickle"):
                data = _load_pickle(fp)
                trial = _find_trial(data)
                _print_dict_ranges(trial)
            elif fp.endswith(".npz"):
                with np.load(fp) as data:
                    _print_dict_ranges(dict(data))
        except Exception as e:
            print(f"  Error: {e}")


def _print_dict_ranges(d):
    for key in sorted(d.keys()):
        val = np.asarray(d[key])
        if np.issubdtype(val.dtype, np.number) or np.issubdtype(val.dtype, np.bool_):
            print(f"  {key:55s} | shape: {str(val.shape):20s} | dtype: {str(val.dtype):8s} | min: {val.min():10.4f} | max: {val.max():10.4f} | mean: {val.mean():10.4f}")
        else:
            print(f"  {key:55s} | shape: {str(val.shape):20s} | dtype: {val.dtype}")


# --------------------------------------------------------------------------- #
#  Video playback mode
# --------------------------------------------------------------------------- #

def _build_grid_frame_pickle(trial, frame_idx, cameras, cell_size=256):
    """Build a grid image for one frame from a pickle trial.
    Layout: rows=cameras, cols=[RGB, Depth] (depth column only if any depth exists).
    """
    import cv2

    has_any_depth = any(v['depth'] is not None for v in cameras.values())
    cam_names = sorted(cameras.keys())
    num_cols = 2 if has_any_depth else 1
    num_rows = len(cam_names)

    grid = np.zeros((num_rows * cell_size, num_cols * cell_size, 3), dtype=np.uint8)

    for row, cam_name in enumerate(cam_names):
        info = cameras[cam_name]
        y0 = row * cell_size

        # RGB
        if info['rgb'] is not None:
            rgb = np.asarray(trial[info['rgb']])
            if frame_idx < len(rgb):
                img = rgb[frame_idx]
                # May be (H, W, 3) or (3, H, W)
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                img = cv2.resize(img, (cell_size, cell_size))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                grid[y0:y0+cell_size, 0:cell_size] = img

        # Label
        cv2.putText(grid, cam_name, (4, y0 + 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Depth
        if has_any_depth and info['depth'] is not None:
            depth = np.asarray(trial[info['depth']])
            if frame_idx < len(depth):
                d = depth[frame_idx]
                # Stored as (1, H, W) float
                if d.ndim == 3 and d.shape[0] == 1:
                    d = d[0]
                elif d.ndim == 3 and d.shape[-1] == 1:
                    d = d[:, :, 0]
                colored = _depth_to_colormap(d)
                colored = cv2.resize(colored, (cell_size, cell_size))
                grid[y0:y0+cell_size, cell_size:2*cell_size] = colored

    # Column headers
    cv2.putText(grid, "RGB", (cell_size // 2 - 15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    if has_any_depth:
        cv2.putText(grid, "Depth", (cell_size + cell_size // 2 - 20, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return grid


def play_files(directory, pause_time=2.0, fps=30, max_files=None):
    import cv2

    files = _gather_files(directory)
    if not files:
        return

    print(f"Found {len(files)} files. Press 'q' to quit.")
    count = 0

    for fp in files:
        if max_files and count >= max_files:
            break
        count += 1
        print(f"Playing: {os.path.basename(fp)}")

        try:
            if fp.endswith(".pickle"):
                data = _load_pickle(fp)
                trial = _find_trial(data)
                cameras = _find_visual_keys(trial)
                if not cameras:
                    print(f"  Skipping: no visual keys found. Keys: {list(trial.keys())}")
                    continue

                # Determine number of frames from first available RGB key
                first_rgb = next((v['rgb'] for v in cameras.values() if v['rgb']), None)
                if first_rgb is None:
                    print(f"  Skipping: no RGB key found.")
                    continue
                num_frames = len(np.asarray(trial[first_rgb]))

                rewards = np.asarray(trial.get("rewards", trial.get("env_infos/rwd_dense", [])))
                actions = np.asarray(trial["actions"]) if "actions" in trial else None
                success = np.asarray(trial["success"]) if "success" in trial else None

                for i in range(num_frames):
                    frame = _build_grid_frame_pickle(trial, i, cameras)

                    # Overlays on bottom of grid
                    h = frame.shape[0]
                    if rewards is not None and len(rewards) > 0 and i < len(rewards):
                        cv2.putText(frame, f"Reward: {float(rewards[i]):.2f}", (10, h - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    if actions is not None and i < len(actions):
                        a_str = "[" + ", ".join(f"{x:.2f}" for x in actions[i]) + "]"
                        cv2.putText(frame, f"Act: {a_str}", (10, h - 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                    if success is not None and i < len(success):
                        s = bool(success[i])
                        color = (0, 255, 0) if s else (0, 0, 255)
                        cv2.putText(frame, f"Success: {s}", (10, h - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                    cv2.imshow("MoDemV2 Dataset Peek", frame)
                    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                        print("Quitting...")
                        cv2.destroyAllWindows()
                        return

            elif fp.endswith(".npz"):
                with np.load(fp) as npz:
                    if "image" not in npz:
                        print(f"  Skipping: no 'image' key.")
                        continue
                    images = npz["image"]
                    rewards = npz.get("reward", None)
                    actions = npz.get("action", None)

                    # images: (T, H, W, C) — may be 3ch or 6ch
                    if images.ndim == 4 and images.shape[1] in (1, 3, 6):
                        images = np.transpose(images, (0, 2, 3, 1))

                    num_frames = len(images)
                    for i in range(num_frames):
                        frame = images[i]
                        if frame.shape[-1] == 6:
                            frame = frame[:, :, :3]
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame = cv2.resize(frame, (512, 512))

                        if rewards is not None and i < len(rewards):
                            cv2.putText(frame, f"Reward: {float(rewards[i]):.2f}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        if actions is not None and i < len(actions):
                            a_str = "[" + ", ".join(f"{x:.2f}" for x in actions[i]) + "]"
                            cv2.putText(frame, f"Action: {a_str}", (10, 70),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                        cv2.imshow("MoDemV2 Dataset Peek", frame)
                        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                            print("Quitting...")
                            cv2.destroyAllWindows()
                            return

            print(f"  Finished {os.path.basename(fp)}. Pausing {pause_time}s...")
            time.sleep(pause_time)

        except Exception as e:
            print(f"  Error reading {fp}: {e}")

    cv2.destroyAllWindows()
    print("Done processing all files.")


# --------------------------------------------------------------------------- #
#  Action histogram mode
# --------------------------------------------------------------------------- #

def plot_action_histograms(directory, max_files=None):
    import matplotlib.pyplot as plt

    files = _gather_files(directory)
    if not files:
        return

    print(f"Aggregating actions from {len(files)} files...")
    all_actions = []
    count = 0

    for fp in files:
        if max_files and count >= max_files:
            break
        count += 1
        try:
            if fp.endswith(".pickle"):
                data = _load_pickle(fp)
                trial = _find_trial(data)
                if "actions" in trial:
                    all_actions.append(np.asarray(trial["actions"]))
            elif fp.endswith(".npz"):
                with np.load(fp) as npz:
                    if "action" in npz:
                        all_actions.append(npz["action"])
        except Exception as e:
            print(f"  Error reading {fp}: {e}")

    if not all_actions:
        print("No actions found.")
        return

    all_actions = np.concatenate(all_actions, axis=0)
    if all_actions.ndim == 1:
        all_actions = all_actions[:, np.newaxis]
    if all_actions.ndim != 2:
        print(f"Unexpected action shape: {all_actions.shape}")
        return

    num_dims = all_actions.shape[1]
    print(f"Plotting histograms for {num_dims} action dims. Total samples: {len(all_actions)}")

    cols = min(4, num_dims)
    rows = (num_dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    axes = axes.flatten()

    for i in range(num_dims):
        ax = axes[i]
        dim_actions = all_actions[:, i]
        ax.hist(dim_actions, bins=50, color="skyblue", edgecolor="black")
        ax.set_title(f"Action Dim {i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        stats = f"min: {dim_actions.min():.2f}\nmax: {dim_actions.max():.2f}\nmean: {dim_actions.mean():.2f}"
        ax.text(0.95, 0.95, stats, transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    for i in range(num_dims, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
#  File discovery
# --------------------------------------------------------------------------- #

def _gather_files(directory):
    """Find .pickle and .npz files in a directory (non-recursive)."""
    pickles = sorted(glob.glob(os.path.join(directory, "*.pickle")))
    npzs = sorted(glob.glob(os.path.join(directory, "*.npz")))
    files = pickles + npzs
    if not files:
        print(f"No .pickle or .npz files found in {directory}")
    else:
        print(f"Found {len(pickles)} pickle + {len(npzs)} npz files in {directory}")
    return files


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Peek at MoDemV2 datasets (.pickle demos or .npz episodes)."
    )
    parser.add_argument("path", type=str, help="Directory containing .pickle or .npz files")
    parser.add_argument("--pause", type=float, default=2.0, help="Seconds between files (default: 2.0)")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
    parser.add_argument("--ranges", action="store_true", help="Print value ranges instead of playing video")
    parser.add_argument("--actions", action="store_true", help="Show action histograms")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files to process")

    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: '{args.path}' is not a directory.")
    elif args.actions:
        plot_action_histograms(args.path, args.max_files)
    elif args.ranges:
        print_ranges(args.path, args.max_files)
    else:
        play_files(args.path, args.pause, args.fps, args.max_files)
