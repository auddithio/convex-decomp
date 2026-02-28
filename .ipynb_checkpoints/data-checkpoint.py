# ============================================================================
# Dataset and Data Processing
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import trimesh
from scipy.spatial import ConvexHull
from typing import List, Tuple, Optional

from trellis.modules.sparse import SparseTensor  

def dense_to_sparse(voxel_grid: torch.Tensor, batch_idx: int = 0):
    """
    Convert dense voxel grid [D, H, W] into a SparseTensor.
    Nonzero voxels are treated as active coordinates.
    """
    assert voxel_grid.ndim == 3
    coords = (voxel_grid > 0).nonzero(as_tuple=False).int()  # [N, 3]
    
    # add batch index (required by TRELLIS/torchsparse)
    batch_col = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32)
    coords = torch.cat([batch_col, coords], dim=1)  # [N, 4]
    
    # Feature can be occupancy (1.0), or you can add other per-voxel info
    feats = voxel_grid[voxel_grid > 0].unsqueeze(1)  # [N, 1]
    
    return SparseTensor(feats=feats, coords=coords)


class VoxelDataset(Dataset):
    """Dataset for loading convex decompositions and converting to voxels + mesh sequences"""
    
    def __init__(self, glb_directory: str, voxel_resolution: int = 64, 
                 max_meshes: int = 32, max_vertices_per_mesh: int = 60,
                 augment: bool = True):
        self.glb_directory = Path(glb_directory)
        self.voxel_resolution = voxel_resolution
        self.max_meshes = max_meshes
        self.max_vertices_per_mesh = max_vertices_per_mesh
        self.augment = augment
        
        # find all files
        self.glb_files = sorted(list(self.glb_directory.glob("*.glb")))
        if len(self.glb_files) == 0:
            raise ValueError(f"No GLB files found in {glb_directory}")
        
        print(f"Found {len(self.glb_files)} GLB files")
    
    def __len__(self):
        return len(self.glb_files)
    
    def voxelize_mesh(self, mesh: trimesh.Trimesh, resolution: int = 64) -> torch.Tensor:
        """Convert mesh to voxel grid"""
        # Normalize mesh to unit cube centered at origin
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        scale = np.max(bounds[1] - bounds[0])
        
        vertices = (mesh.vertices - center) / scale
        mesh_normalized = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
        
        # Voxelize
        voxel_grid = mesh_normalized.voxelized(pitch=1.0/resolution)
        voxel_matrix = voxel_grid.matrix
        
        # Pad to exact resolution
        pad_width = [(0, max(0, resolution - s)) for s in voxel_matrix.shape]
        voxel_matrix = np.pad(voxel_matrix, pad_width, mode='constant')
        voxel_matrix = voxel_matrix[:resolution, :resolution, :resolution]
        
        return torch.from_numpy(voxel_matrix.astype(np.float32))
    
    def extract_mesh_sequence(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        """Extract vertices as sequence with mesh separators"""
        # Normalize mesh
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        scale = np.max(bounds[1] - bounds[0])
        vertices = (mesh.vertices - center) / scale
        
        # Apply rotation augmentation
        if self.augment:
            vertices = self.apply_rotation(vertices)

        # for simplicity, just return all vertices as a single sequence
        # return torch.from_numpy(vertices.astype(np.float32))
    
        split_meshes = mesh.split()
        print(f"Split into {len(split_meshes)} submeshes")
        
        if len(split_meshes) > self.max_meshes:
            # Keep largest meshes
            split_meshes = sorted(split_meshes, key=lambda m: len(m.vertices), reverse=True)
            split_meshes = split_meshes[:self.max_meshes]
        
        # Shuffle mesh order if augmenting
        if self.augment:
            np.random.shuffle(split_meshes)
        
        sequence = []
        for submesh in split_meshes:
            submesh_verts = submesh.vertices
            if self.augment:
                # Shuffle vertex order
                perm = np.random.permutation(len(submesh_verts))
                submesh_verts = submesh_verts[perm]
            
            # Limit vertices per mesh
            if len(submesh_verts) > self.max_vertices_per_mesh:
                indices = np.random.choice(len(submesh_verts), self.max_vertices_per_mesh, replace=False)
                submesh_verts = submesh_verts[indices]
            
            # add newmesh token (represented as [0, 0, 0])
            sequence.append([0.0, 0.0, 0.0])
            sequence.extend(submesh_verts.tolist())
        
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
               
        return sequence_tensor
    
    def __getitem__(self, idx):
        glb_path = self.glb_files[idx]
        
        try:
            # Load mesh
            mesh = trimesh.load(glb_path, force='mesh')
            
            # Handle multi-mesh GLB
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            if self.augment:
                angles = np.random.uniform(0, 2*np.pi, 3)
                R = trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2])
                mesh.apply_transform(R)
            
            voxels = self.voxelize_mesh(mesh, self.voxel_resolution)
            sequence = self.extract_mesh_sequence(mesh)
            
            return {
                'voxels': voxels,
                'sequence': sequence,
            }
        
        except Exception as e:
            print(f"Error loading {glb_path}: {e}")
            # Return empty sample
            return {
                'voxels': torch.zeros(self.voxel_resolution, self.voxel_resolution, self.voxel_resolution),
                'sequence': torch.zeros(1, 3),
                'mesh': None
            }


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

