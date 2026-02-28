import numpy as np
from pathlib import Path
import os

npz_dir = Path("/vision/group/objaverse/affogato_subset/coacd/")
files = list(npz_dir.glob("*.npz"))
print(f"Number of files: {len(files)}")

num = 0
large_files = []
small_files = []

for file in files:  
    data = np.load(file, allow_pickle=True)
    vertices = data['vertices']

    if vertices.shape[0] > 32:
        large_files.append(os.path.basename(file))
        num += 1
    else:
        small_files.append(os.path.basename(file))
        

print(f"Number of files with more than 32 vertices: {num}")

# save lists
with open("submeshes_more_than_32.txt", "w") as f:
    f.write("\n".join(large_files))

with open("regular_submeshes.txt", "w") as f:
    f.write("\n".join(small_files))

print("Lists saved to submeshes_more_than_32.txt and regular_submeshes.txt")



