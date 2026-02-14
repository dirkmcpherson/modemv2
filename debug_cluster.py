import numpy as np
import torch
import sys

# Monkey-patch before anything else
orig_from_numpy = torch.from_numpy
def patched_from_numpy(arr):
    try:
        return orig_from_numpy(arr)
    except TypeError:
        return torch.as_tensor(arr)

torch.from_numpy = patched_from_numpy

print(f"Python: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Torch version: {torch.__version__}")

arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

print("\nTesting Patched from_numpy:")

try:
    t = torch.from_numpy(arr)
    print("  torch.from_numpy(arr): SUCCESS (using patch)")
except Exception as e:
    print(f"  torch.from_numpy(arr): FAILED - {e}")

# Now try to import ManiSkill and see if it crashes during env creation
# (This part is for the user to verify if they can)
print("\nPatch applied. User should try running the training script now.")
