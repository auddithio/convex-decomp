import numpy as np
import trimesh
from pathlib import Path
import argparse
import multiprocessing as mp
from functools import partial
import traceback
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Worker result
# ---------------------------------------------------------------------------

@dataclass
class Result:
    name:     str
    tag:      Literal["OK", "SKIP", "ERROR"]
    reason:   str = ""
    over_32:  bool = False   # mesh has more than 32 submeshes
    bad_hull: bool = False   # mesh fails build_tets_array (would cause dataset retry)


# ---------------------------------------------------------------------------
# Voxelisation
# ---------------------------------------------------------------------------

def voxelize_convex_submesh(
    vertices: np.ndarray,
    resolution: int,
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Voxelise a single convex submesh using its convex hull.
    Normalisation uses the global center/scale so all parts share
    the same coordinate frame as the CoACD targets in VoxelDataset.

    Returns a bool occupancy grid of shape [R, R, R].
    """
    mesh = trimesh.Trimesh(vertices=vertices).convex_hull  # watertight by construction
    mesh = mesh.copy()
    mesh.vertices = (mesh.vertices - center) / scale       # → [-0.5, 0.5]

    pitch = 1.0 / resolution
    vg = mesh.voxelized(pitch=pitch)

    occupied  = vg.sparse_indices
    world_pts = occupied * pitch + vg.transform[:3, 3] # last part is origin
    coords    = np.floor((world_pts + 0.5) * resolution).astype(int)

    grid  = np.zeros((resolution, resolution, resolution), dtype=bool)
    valid = np.all((coords >= 0) & (coords < resolution), axis=1)
    coords = coords[valid]
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return grid


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def process_one(npz_file: Path, output_path: Path, resolution: int) -> Result:
    from data import build_tets_array
    name     = npz_file.name
    out_file = output_path / name

    if out_file.exists():
        return Result(name=name, tag="SKIP", reason="already processed")

    try:
        data = np.load(npz_file, allow_pickle=True)

        if 'vertices' not in data:
            return Result(name=name, tag="SKIP", reason="missing 'vertices' key")

        raw = data['vertices']
        submeshes = [s.astype(np.float32) for s in raw] if raw.dtype == object \
                    else [raw.astype(np.float32)]

        if not submeshes:
            return Result(name=name, tag="SKIP", reason="no submeshes")

        # --- flag: more than 32 submeshes (will be truncated by dataset) ---
        over_32 = len(submeshes) > 32

        # --- global normalisation (must match VoxelDataset exactly) ---
        all_verts  = np.concatenate(submeshes, axis=0)
        bounds_min = all_verts.min(axis=0)
        bounds_max = all_verts.max(axis=0)
        center     = ((bounds_min + bounds_max) / 2.0).astype(np.float64)
        scale      = float(np.max(bounds_max - bounds_min))

        if scale == 0:
            return Result(name=name, tag="SKIP", reason="degenerate mesh (scale=0)")

        normalized = []
        for v in submeshes:
            normalized.append((v - center) / scale)

        # --- flag: dry-run build_tets_array to catch QhullError / zero-volume ---
        # This mirrors exactly what VoxelDataset does, so anything that would
        # cause a retry in training is caught here instead.
        bad_hull = False
        try:
            _, probs = build_tets_array(normalized)
            if not np.isfinite(probs).all():
                bad_hull = True
        except Exception:
            bad_hull = True

        # --- voxelise and union submesh grids ---
        grid = np.zeros((resolution, resolution, resolution), dtype=bool)
        for verts in submeshes:
            if len(verts) < 4:
                continue
            part  = voxelize_convex_submesh(verts, resolution, center, scale)
            grid |= part

        np.savez_compressed(
            out_file,
            voxels=grid.astype(np.float32),
            center=center,
            scale=np.array([scale]),
        )
        return Result(name=name, tag="OK", over_32=over_32, bad_hull=bad_hull)

    except Exception:
        tb = traceback.format_exc().strip().splitlines()[-1]
        return Result(name=name, tag="ERROR", reason=tb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess_voxels(
    input_dir:   str,
    output_dir:  str,
    resolution:  int,
    num_workers: int,
    stats_dir:   str,
):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    stats_path  = Path(stats_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stats_path.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_path.glob("*.npz"))
    total     = len(npz_files)
    print(f"Found {total} NPZ files. Launching {num_workers} workers at resolution={resolution}.")

    worker = partial(process_one, output_path=output_path, resolution=resolution)

    ok = skip = err = 0
    over_32_names  = []   # meshes with >32 submeshes
    bad_hull_names = []   # meshes that fail build_tets_array

    with mp.Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, npz_files, chunksize=4), 1):
            if result.tag == "OK":
                ok += 1
                if result.over_32:
                    over_32_names.append(result.name)
                if result.bad_hull:
                    bad_hull_names.append(result.name)
            elif result.tag == "SKIP":
                skip += 1
            else:
                err += 1

            status_str = f"[{i:>6}/{total}] {result.tag:<5} {result.name}"
            if result.reason:
                status_str += f"  [{result.reason}]"
            if result.over_32:
                status_str += "  [>32 submeshes]"
            if result.bad_hull:
                status_str += "  [bad hull]"
            print(status_str, flush=True)

    # write stats files — done in main process so no locking needed
    over_32_path  = stats_path / "over_32_submeshes.txt"
    bad_hull_path = stats_path / "bad_hulls.txt"

    over_32_path.write_text("\n".join(sorted(over_32_names)) + ("\n" if over_32_names else ""))
    bad_hull_path.write_text("\n".join(sorted(bad_hull_names)) + ("\n" if bad_hull_names else ""))

    print(f"\nDone.  OK={ok}  Skipped={skip}  Errors={err}")
    print(f"  >32 submeshes : {len(over_32_names):>6}  →  {over_32_path}")
    print(f"  Bad hulls     : {len(bad_hull_names):>6}  →  {bad_hull_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute voxel grids from CoACD NPZ meshes.")
    parser.add_argument("--input_dir",   type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--stats_dir",   type=str, required=True,
                        help="Directory to write over_32_submeshes.txt and bad_hulls.txt")
    parser.add_argument("--resolution",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    preprocess_voxels(
        args.input_dir,
        args.output_dir,
        args.resolution,
        args.num_workers,
        args.stats_dir,
    )