# ============================================================================
# Dataset and Data Processing
# ============================================================================
import numpy as np
from TRELLIS.trellis.representations import mesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

import trimesh
from scipy.spatial import ConvexHull

# TODO: copy dataset to /scr 

def sample_tetrahedra(tets, probs, N):
    """
    tets:  (T, 4, 3)
    probs: (T,)
    N:     number of samples

    returns: (N, 3)
    """

    # Choose tetrahedra
    idx = np.random.choice(len(tets), size=N, p=probs)

    # Gather vertices
    V = tets[idx]           # (N, 4, 3)

    # Generate barycentric coordinates
    r = np.random.rand(N, 3)
    r.sort(axis=1)

    w0 = r[:, 0:1]
    w1 = r[:, 1:2] - r[:, 0:1]
    w2 = r[:, 2:3] - r[:, 1:2]
    w3 = 1.0 - r[:, 2:3]

    # Weighted sum
    pts = (
        w0 * V[:, 0] +
        w1 * V[:, 1] +
        w2 * V[:, 2] +
        w3 * V[:, 3]
    )

    return pts


def build_tets_array(hulls):
    tets = []
    vols = []

    for pts in hulls:
        pts = np.asarray(pts, dtype=np.float64)
        hull = ConvexHull(pts)
        c = pts.mean(axis=0)

        for f in hull.simplices:
            v0, v1, v2 = pts[f]
            v3 = c

            vol = abs(np.dot(v0 - v3, np.cross(v1 - v3, v2 - v3))) / 6.0
            if vol > 0:
                tets.append([v0, v1, v2, v3])
                vols.append(vol)

    tets = np.asarray(tets, dtype=np.float64)     # (T, 4, 3)
    vols = np.asarray(vols)

    probs = vols / vols.sum()
    return tets, probs



class VoxelDataset(Dataset):
    """Dataset for loading convex decompositions and converting to voxels + mesh sequences"""
    
    def __init__(self, model_list: str, voxel_directory: str, coacd_directory: str, sample_points=1024):
        # TODO: add augmentation?
        
        # model list is a .txt file with model names (with .npz)
        self.files = [line.strip() for line in Path(model_list).open()]
        self.voxel_directory = Path(voxel_directory)
        self.coacd_directory = Path(coacd_directory)
        self.sample_points = sample_points
    
        print(f"Found {len(self.files)} NPZ files")
    
    def __len__(self):
        return len(self.files)
    
    def _pad_coacd(self, coacd, max_submeshes=32, max_vertices=60):
        padded = []
        for submesh in coacd[:max_submeshes]:  # truncate if more than 32
            v = submesh
            # pad vertices along dim 0 (up to 60)
            if v.size(0) < max_vertices:
                pad_len = max_vertices - v.size(0)
                v = F.pad(v, (0, 0, 0, pad_len))  # pad only along vertex dim
            elif v.size(0) > max_vertices:
                v = v[:max_vertices]  # truncate if too many
            padded.append(v)
        
        # pad missing submeshes (if less than 32)
        while len(padded) < max_submeshes:
            padded.append(torch.zeros(max_vertices, 3))
        
        return torch.stack(padded)  # (32, 60, 3)
    
    
    def __getitem__(self, idx):
        
        for _ in range(10):  # retry
            npz_path = self.files[idx]
            try:
                voxels = np.load(self.voxel_directory / npz_path)['voxels']
                coacd = np.load(self.coacd_directory / npz_path, allow_pickle=True)['vertices']

                # TODO: normalize the coacd vertices to be in [0, 1] range, since the voxels are in a 64^3 grid
                all_vertices = np.concatenate(coacd, axis=0).astype(np.float32)

                bounds_min = all_vertices.min(axis=0)
                bounds_max = all_vertices.max(axis=0)

                center = (bounds_min + bounds_max) / 2.0
                scale = np.max(bounds_max - bounds_min)

                normalized_submeshes = []
                for submesh in coacd:
                    v = submesh.astype(np.float32)
                    v = (v - center) / scale      # → [-0.5, 0.5]
                    # v = v + 0.5                   # → [0, 1] TODO: pred can't take this value?
                    normalized_submeshes.append(v)

                points = sample_convex_hulls(normalized_submeshes, self.sample_points)
                break
            
            except Exception as e:
                idx = np.random.randint(0, len(self.files))
        else:
            raise RuntimeError("Too many corrupted files encountered in a row.")
        
        item = {
            'voxels': torch.from_numpy(voxels).float(),
            'sequence': torch.from_numpy(points).float()
        }

        # print(f"Loaded item {idx}: voxels shape {item['voxels'].shape}, sequence shape {item['sequence'].shape}")
        return item


        # # find max length for each dim
        # max_submeshes = 32 # number of submeshes
        # max_vertices = 60 # number of vertices per submesh
        
        # # pad each array to max size
        # padded = []
        # for item in coacd:
        #     try: 
        #         if item.shape[0] < max_vertices:
        #             padding = np.zeros((max_vertices - item.shape[0], 3))
        #             padded.append(np.vstack([item, padding]))
        #         else:
        #             padded.append(item)

        #     except Exception as e:
        #         print(f"Item shape: {item.shape}")
        #         print(f"Error processing item in COACD for {npz_path}, error was: {e}")

        # while len(padded) < max_submeshes:
        #     padded.append(np.zeros((max_vertices, 3)))
        
        # coacd = np.stack(padded).astype(np.float32)  # (num_submeshes, 32, 60)

        # the coacd is a (possibly ragged) array, do what's below!
        # combined_collision_mesh = trimesh.boolean.union([cm for cm in target_object.collision_meshes])
        # points = combined_collision_mesh.sample(self.sample_points) 
        
        # convert the ragged array into a flat array of shape (-1, 3), and then sample N points from it.
        # flat_coacd = np.vstack(coacd)
        # if len(flat_coacd) > self.sample_points:
        #     indices = np.random.choice(len(flat_coacd), self.sample_points, replace=False)
        # elif len(flat_coacd) < self.sample_points:
        #     indices = np.random.choice(len(flat_coacd), self.sample_points, replace=True)
        # coacd = flat_coacd[indices]

        # coacd = self._pad_coacd(coacd)
        # print(type(coacd), coacd.shape)

        # meshes = [hull_to_trimesh(h) for h in coacd]
        # meshes = [m for m in meshes if not m.is_empty and m.is_watertight and m.is_volume]  # filter out bad meshes
        # TODO: check how many bad samples there are
        # print(f"Loaded {len(meshes)} valid meshes for {npz_path} out of {len(coacd)} submeshes")


        # union_mesh = trimesh.boolean.union(
        #     meshes,
        # )

        # TODO: consider returning the convex meshes and do random sampling directly in the loss computation for more variety
        # points, _ = trimesh.sample.volume_mesh(union_mesh, 1024) # TODO: look up how to sample points fast from an already convex mesh

        

def hull_to_trimesh(points):
    return trimesh.convex.convex_hull(points)

def hull_to_trimesh_scipy(points):
    hull = ConvexHull(points)

    # get the equations to sample points from the convex hull manually
    # make sure the sample points are uniformly distributed across the convex hull based on the volume
    # later, can move to loss computation side

    vertices = points[hull.vertices]
    faces = hull.simplices
    return trimesh.Trimesh(vertices=vertices, faces=faces)

# TODO: scipy gives you planes for each convex hull, can simplify sampling

def sample_convex_hulls(hulls, N):
    tets, probs = build_tets_array(hulls)
    pts = sample_tetrahedra(tets, probs, N)
    return pts


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Find max sequence length
    max_seq_len = max(item['sequence'].shape[0] for item in batch)
    
    # Pad sequences
    voxels = torch.stack([item['voxels'] for item in batch])
    
    sequences = []

    for item in batch:
        seq = item['sequence']
        pad_len = max_seq_len - len(seq)
        
        if pad_len > 0:
            seq = torch.cat([seq, torch.zeros(pad_len, 3)], dim=0)
        
        sequences.append(seq)
    
    return {
        'voxels': voxels,
        'sequence': torch.stack(sequences),
    }

