import numpy as np
import numpy
import torch
import sapien
import sys
import os

print(f"Python: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"NumPy file: {np.__file__}")
print(f"NumPy alias match: {np is numpy}")
print(f"Torch version: {torch.__version__}")
print(f"Torch file: {torch.__file__}")
print(f"Sapien version: {sapien.__version__}")

# Try to see where the internal np.ndarray of torch comes from
try:
    import torch._C
    # This is internal, but might give a hint
    print(f"Torch internal numpy check: {hasattr(torch._C, '_from_numpy')}")
except:
    pass

# Try a small test
try:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    print(f"Array type: {type(arr)}")
    print(f"Array module: {type(arr).__module__}")
    t = torch.from_numpy(arr)
    print("torch.from_numpy test: SUCCESS")
except Exception as e:
    print(f"torch.from_numpy test: FAILED - {type(e).__name__}: {e}")
    # Print type identities if possible
    try:
        from numpy import ndarray
        print(f"ndarray type in script: {id(ndarray)}")
        print(f"type(arr) in script: {id(type(arr))}")
    except:
        pass

# Check sys.path
print("\nsys.path:")
for p in sys.path:
    print(f"  {p}")
