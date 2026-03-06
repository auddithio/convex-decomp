import torch
import numpy as np
import trimesh
from pathlib import Path
import argparse

# TODO: confirm with Cem
# 1. we're doing surface voxelisation (hollowed) for the input meshes, not occupancy grids
# 2. this normalisation does NOT preserve aspect ratio!

def voxelize_mesh_with_info(mesh: trimesh.Trimesh, resolution: int = 64):
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0])

    mesh = mesh.copy()
    mesh.vertices = (mesh.vertices - center) / scale  # → [-0.5, 0.5]

    if not mesh.is_watertight:
        filled = mesh.fill_holes()
        if filled is not None:
            mesh = filled

    pitch = 1.0 / resolution
    vg = mesh.voxelized(pitch=pitch)

    occupied = vg.sparse_indices
    origin = vg.origin
    world_pts = occupied * pitch + origin
    coords = np.floor((world_pts + 0.5) * resolution).astype(int)

    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    valid = np.all((coords >= 0) & (coords < resolution), axis=1)
    grid[coords[valid, 0], coords[valid, 1], coords[valid, 2]] = 1.0

    # Return grid + the transform needed to align CoACD targets
    return torch.from_numpy(grid), center, scale


def voxelize_mesh(mesh: trimesh.Trimesh, resolution: int = 64) -> torch.Tensor:
    """TODO: Remove, faulty
    Convert a mesh to voxel grid tensor of shape [D, H, W]."""
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0])
    vertices = (mesh.vertices - center) / scale # in [-0.5, 0.5]
    mesh_normalized = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

    voxel_grid = mesh_normalized.voxelized(pitch=1.0 / resolution)
    voxel_matrix = voxel_grid.matrix

    # pad or crop to exact resolution
    pad_width = [(0, max(0, resolution - s)) for s in voxel_matrix.shape]
    voxel_matrix = np.pad(voxel_matrix, pad_width, mode='constant')
    voxel_matrix = voxel_matrix[:resolution, :resolution, :resolution]

    return torch.from_numpy(voxel_matrix.astype(np.float32))


def preprocess_voxels(input_dir: str, output_dir: str, resolution: int = 64):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(input_path.glob("*.npz")))
    print(f"Found {len(npz_files)} NPZ files to process.")

    for i, npz_file in enumerate(npz_files):
        try:
            if (output_path / f"{npz_file.stem}.npz").exists():
                print(f"[{i+1}/{len(npz_files)}] Skipping {npz_file.name}, already processed.")
                continue

            data = np.load(npz_file, allow_pickle=True)
            if 'vertices' not in data:
                print(f"Skipping {npz_file.name}: missing 'vertices' key.")
                continue

            vertices = data['vertices']
            if isinstance(vertices, np.ndarray) and vertices.dtype == object:
                vertices = np.concatenate(vertices, axis=0)

            mesh = trimesh.Trimesh(vertices=vertices)
            voxels = voxelize_mesh_with_info(mesh, resolution)
            np.savez_compressed(output_path / f"{npz_file.stem}.npz", voxels=voxels.numpy())

            print(f"[{i+1}/{len(npz_files)}] Saved: {npz_file.stem}.npz")

        except Exception as e:
            print(f"Error processing {npz_file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute voxel grids from NPZ meshes.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .npz mesh files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save voxel files.")
    parser.add_argument("--resolution", type=int, default=64, help="Voxel grid resolution (default: 64).")

    args = parser.parse_args()
    preprocess_voxels(args.input_dir, args.output_dir, args.resolution)
