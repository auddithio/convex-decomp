# Dataset Verification Script for Jupyter Notebook
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

# ============================================================================
# Visualize Voxels
# ============================================================================
def visualize_mesh(mesh: trimesh.Trimesh):
    """Visualize a trimesh mesh in 3D"""
    if mesh is None:
        print("No mesh to visualize")
        return
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize mesh to unit cube centered at origin
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0])
    
    vertices = (mesh.vertices - center) / scale
    
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='green', s=1, alpha=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Mesh Visualization ({len(vertices)} vertices)')
    
    plt.tight_layout()
    plt.show()

    

def visualize_voxels(voxels, threshold=0.5):
    """Visualize voxel grid in 3D"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get occupied voxel coordinates
    occupied = voxels > threshold
    x, y, z = torch.where(occupied)
    
    ax.scatter(x, y, z, c='blue', marker='s', alpha=0.3, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Voxel Visualization ({len(x)} occupied voxels)')
    
    plt.tight_layout()
    plt.show()

# print("\n=== Voxel Visualization ===")
# visualize_voxels(sample['voxels'])

# ============================================================================
# 4. Visualize Vertex Sequence
# ============================================================================

def visualize_vertex_sequence(sequence, mask):
    """Visualize predicted vertex sequence with mesh separators"""
    fig = plt.figure(figsize=(12, 5))
    
    # Filter valid vertices
    valid_verts = sequence[mask].numpy()
    
    # Identify newmesh tokens
    norms = np.linalg.norm(valid_verts, axis=1)
    is_newmesh = norms < 1e-6
    
    # Separate into meshes
    mesh_indices = np.where(is_newmesh)[0]
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(mesh_indices) + 1))
    
    start = 0
    for i, end in enumerate(list(mesh_indices) + [len(valid_verts)]):
        mesh_verts = valid_verts[start:end]
        # Skip newmesh token itself
        mesh_verts = mesh_verts[np.linalg.norm(mesh_verts, axis=1) > 1e-6]
        
        if len(mesh_verts) > 0:
            ax1.scatter(mesh_verts[:, 0], mesh_verts[:, 1], mesh_verts[:, 2],
                       c=[colors[i]], label=f'Mesh {i}', s=20, alpha=0.7)
        start = end
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Vertex Sequence (colored by mesh)')
    ax1.legend()
    
    # 2D plot showing sequence structure
    ax2 = fig.add_subplot(122)
    ax2.plot(norms, marker='o', markersize=2, linestyle='-', alpha=0.5)
    ax2.axhline(y=1e-6, color='r', linestyle='--', label='<newmesh> threshold')
    ax2.set_xlabel('Sequence Index')
    ax2.set_ylabel('Vertex Norm')
    ax2.set_title('Sequence Structure (newmesh tokens at 0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Total vertices: {len(valid_verts)}")
    print(f"Newmesh tokens: {is_newmesh.sum()}")
    print(f"Actual vertices: {(~is_newmesh).sum()}")

# print("\n=== Vertex Sequence Visualization ===")
# visualize_vertex_sequence(sample['sequence'], sample['mask'])

# ============================================================================
# 5. Compare Original Mesh vs Voxelized
# ============================================================================

def compare_original_vs_sequence(original_mesh, sequence, mask):
    """Compare original mesh with extracted sequence"""
    if original_mesh is None:
        print("No original mesh available")
        return
    
    fig = plt.figure(figsize=(15, 5))
    
    # Original mesh
    ax1 = fig.add_subplot(131, projection='3d')
    vertices = original_mesh.vertices
    # Normalize
    bounds = original_mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0])
    vertices = (vertices - center) / scale
    
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='green', s=1, alpha=0.3)
    ax1.set_title(f'Original Mesh\n({len(vertices)} vertices)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Extracted sequence
    ax2 = fig.add_subplot(132, projection='3d')
    valid_verts = sequence[mask].numpy()
    # Remove newmesh tokens
    valid_verts = valid_verts[np.linalg.norm(valid_verts, axis=1) > 1e-6]
    
    ax2.scatter(valid_verts[:, 0], valid_verts[:, 1], valid_verts[:, 2],
               c='blue', s=10, alpha=0.5)
    ax2.set_title(f'Extracted Sequence\n({len(valid_verts)} vertices)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Voxels
    ax3 = fig.add_subplot(133, projection='3d')
    occupied = sample['voxels'] > 0.5
    x, y, z = torch.where(occupied)
    ax3.scatter(x, y, z, c='red', marker='s', alpha=0.2, s=1)
    ax3.set_title(f'Voxelized\n({len(x)} voxels)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

# print("\n=== Original vs Extracted Comparison ===")
# compare_original_vs_sequence(sample['mesh'], sample['sequence'], sample['mask'])

# ============================================================================
# 6. Test DataLoader with Batching
# ============================================================================

# from torch.utils.data import DataLoader

# dataloader = DataLoader(
#     dataset,
#     batch_size=4,
#     shuffle=True,
#     collate_fn=collate_fn
# )

# print("\n=== DataLoader Batch ===")
# batch = next(iter(dataloader))

# print(f"Batch voxels shape: {batch['voxels'].shape}")
# print(f"Batch sequence shape: {batch['sequence'].shape}")
# print(f"Batch mask shape: {batch['mask'].shape}")
# print(f"Number of meshes: {len(batch['meshes'])}")

# # Show per-sample statistics
# for i in range(len(batch['meshes'])):
#     valid = batch['mask'][i].sum()
#     print(f"  Sample {i}: {valid} valid tokens")

# # ============================================================================
# # 7. Test Augmentation
# # ============================================================================

# print("\n=== Testing Augmentation ===")
# print("Getting same sample 3 times to see augmentation effects:")

# for i in range(3):
#     sample_aug = dataset[0]
#     seq = sample_aug['sequence'][sample_aug['mask']]
#     # Remove newmesh tokens
#     seq = seq[seq.norm(dim=1) > 1e-6]
    
#     print(f"\nIteration {i+1}:")
#     print(f"  First 3 vertices: {seq[:3]}")
#     print(f"  Last 3 vertices: {seq[-3:]}")
#     print(f"  Mean position: {seq.mean(dim=0)}")
#     print(f"  Total vertices: {len(seq)}")

# # ============================================================================
# # 8. Visualize Multiple Samples Side-by-Side
# # ============================================================================

# def visualize_multiple_samples(dataset, n_samples=4):
#     """Visualize multiple samples in a grid"""
#     fig = plt.figure(figsize=(15, 4*n_samples))
    
#     for i in range(n_samples):
#         sample = dataset[i]
        
#         # Voxels
#         ax1 = fig.add_subplot(n_samples, 3, i*3 + 1, projection='3d')
#         occupied = sample['voxels'] > 0.5
#         x, y, z = torch.where(occupied)
#         ax1.scatter(x, y, z, c='red', marker='s', alpha=0.2, s=1)
#         ax1.set_title(f'Sample {i} - Voxels')
        
#         # Sequence
#         ax2 = fig.add_subplot(n_samples, 3, i*3 + 2, projection='3d')
#         valid_verts = sample['sequence'][sample['mask']].numpy()
#         valid_verts = valid_verts[np.linalg.norm(valid_verts, axis=1) > 1e-6]
#         ax2.scatter(valid_verts[:, 0], valid_verts[:, 1], valid_verts[:, 2],
#                    c='blue', s=10, alpha=0.5)
#         ax2.set_title(f'Sample {i} - Vertices')
        
#         # Original mesh
#         ax3 = fig.add_subplot(n_samples, 3, i*3 + 3, projection='3d')
#         if sample['mesh'] is not None:
#             vertices = sample['mesh'].vertices
#             bounds = sample['mesh'].bounds
#             center = (bounds[0] + bounds[1]) / 2
#             scale = np.max(bounds[1] - bounds[0])
#             vertices = (vertices - center) / scale
#             ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
#                        c='green', s=1, alpha=0.3)
#         ax3.set_title(f'Sample {i} - Original')
    
#     plt.tight_layout()
#     plt.show()

# print("\n=== Multiple Samples Visualization ===")
# visualize_multiple_samples(dataset, n_samples=min(4, len(dataset)))