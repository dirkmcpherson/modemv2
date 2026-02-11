import numpy as np
import torch
import sapien
import sys
import os

print(f"Python: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"NumPy file: {np.__file__}")
print(f"Torch version: {torch.__version__}")
print(f"Sapien version: {sapien.__version__}")

# Try a small test
try:
    arr = np.array([1, 2, 3])
    t = torch.from_numpy(arr)
    print("torch.from_numpy test: SUCCESS")
except Exception as e:
    print(f"torch.from_numpy test: FAILED - {type(e).__name__}: {e}")

# Check sys.path
print("\nsys.path:")
for p in sys.path:
    print(f"  {p}")
