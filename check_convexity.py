import numpy as np
from data import sample_convex_hulls
from pathlib import Path
import os

npz_dir = Path("/scr/aunag/objaverse/coacd/")
files = list(npz_dir.glob("*.npz"))
print(f"Number of files: {len(files)}")
num_non_convex = 0

for f in files:
    coacd = np.load(f, allow_pickle=True)['vertices']
    try:
        hulls = sample_convex_hulls(coacd, 1000)
    except Exception as e:
        print(f"Error processing {f}, error was: {e}")
        num_non_convex += 1

print(f"Number of non-convex objects: {num_non_convex}")
print(f"Percentage of non-convex objects: {num_non_convex / len(files) * 100:.2f}%")
