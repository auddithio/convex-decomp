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
    
    def __init__(self, model_list: str, voxel_directory: str, coacd_directory: str, stats_dir: str = None, sample_points=1024):

        # TODO: add augmentation?
        all_files = [f.name for f in Path(voxel_directory).glob("*.npz")]
        
        skip = set()
        if stats_dir is not None:
            stats_path = Path(stats_dir)
            for fname in ("over_32_submeshes.txt", "bad_hulls.txt", "large_voxels.txt"):
                fpath = stats_path / fname
                if fpath.exists():
                    skip.update(line.strip() for line in fpath.open())

        self.files = [f for f in all_files if f not in skip]
        self.voxel_directory = Path(voxel_directory)
        self.coacd_directory = Path(coacd_directory)
        self.sample_points = sample_points

        print(f"Loaded {len(self.files)} files ({len(all_files) - len(self.files)} excluded by stats filter)")

    
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
        npz_path = self.files[idx]

        voxel_data = np.load(self.voxel_directory / npz_path)
        voxels = voxel_data['voxels']
        center = voxel_data['center']            
        scale  = voxel_data['scale'].item()      

        coacd = np.load(self.coacd_directory / npz_path, allow_pickle=True)['vertices']

        normalized_submeshes = []
        for submesh in coacd:
            v = submesh.astype(np.float32)
            v = (v - center) / scale  
            v += 0.5  # shift to [0, 1]?           
            normalized_submeshes.append(v)

        points = sample_convex_hulls(normalized_submeshes, self.sample_points)

        return {
            'voxels':   torch.from_numpy(voxels).float(),
            'sequence': torch.from_numpy(points).float(),
        }


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

