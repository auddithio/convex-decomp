import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import random

# compute minimum across different submeshes

def chamfer_loss(pred, gt):
    # pred: (N, 3)
    # gt: (M, 3)
    dist1 = torch.cdist(pred, gt)  # (N, M)
    loss1 = dist1.min(dim=1).values.mean()
    loss2 = dist1.min(dim=0).values.mean()
    return loss1 + loss2


def batched_chamfer_loss(pred, gt):
    # pred: (B, N, 3)
    # gt: (B, M, 3)
    dist = torch.cdist(pred, gt)  # (B, N, M)
    loss1 = dist.min(dim=2).values.mean(dim=1)  # (B,)
    loss2 = dist.min(dim=1).values.mean(dim=1)  # (B,)
    return (loss1 + loss2).mean()  # scalar


# ============================================================================
# Loss Functions
# ============================================================================

# class ConvexHullLoss(nn.Module):
#     """Loss based on convex hull comparison"""
    
#     def __init__(self, over_approx_weight: float = 1.5, 
#                  under_approx_weight: float = 1.0,
#                  redundancy_weight: float = 0.5):
#         super().__init__()
#         self.over_approx_weight = over_approx_weight
#         self.under_approx_weight = under_approx_weight
#         self.redundancy_weight = redundancy_weight
        
#     def point_to_mesh_distance(self, points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
#         """Compute signed distance from points to mesh (negative = inside)"""
#         # Use trimesh's proximity query
#         closest_points, distances, _ = mesh.nearest.on_surface(points)
        
#         # Determine sign (inside vs outside)
#         # Points inside have negative distance
#         is_inside = mesh.contains(points)
#         signed_distances = np.where(is_inside, -distances, distances)
        
#         return signed_distances
    
#     def forward(self, pred_vertices: torch.Tensor, gt_mesh, mask: torch.Tensor = None):
#         """
#         Args:
#             pred_vertices: (B, seq_len, 3) - predicted vertices
#             gt_mesh: list of trimesh.Trimesh - ground truth meshes
#             mask: (B, seq_len) - mask for valid vertices
#         """
#         B = pred_vertices.shape[0]
#         total_loss = 0.0
        
#         for b in range(B):
#             if gt_mesh[b] is None:
#                 continue
                
#             # Get valid predicted vertices
#             if mask is not None:
#                 valid_mask = mask[b]
#                 verts = pred_vertices[b, valid_mask].detach().cpu().numpy()
#             else:
#                 verts = pred_vertices[b].detach().cpu().numpy()
            
#             # Filter out newmesh tokens (zeros)
#             non_zero_mask = np.linalg.norm(verts, axis=1) > 1e-6
#             verts = verts[non_zero_mask]
            
#             if len(verts) < 4:  # Need at least 4 points for convex hull
#                 total_loss += 10.0  # Penalty for too few points
#                 continue
            
#             try:
#                 # Compute convex hull of predictions
#                 pred_hull = ConvexHull(verts)
#                 pred_hull_mesh = trimesh.Trimesh(
#                     vertices=verts,
#                     faces=pred_hull.simplices
#                 )
                
#                 # Normalize GT mesh to same scale as predictions
#                 bounds = gt_mesh[b].bounds
#                 center = (bounds[0] + bounds[1]) / 2
#                 scale = np.max(bounds[1] - bounds[0])
#                 gt_verts = (gt_mesh[b].vertices - center) / scale
#                 gt_mesh_normalized = trimesh.Trimesh(vertices=gt_verts, faces=gt_mesh[b].faces)
                
#                 # Sample points from GT mesh surface
#                 gt_points, _ = trimesh.sample.sample_surface(gt_mesh_normalized, 1000)
                
#                 # Compute distances from GT points to predicted hull
#                 distances = self.point_to_mesh_distance(gt_points, pred_hull_mesh)
                
#                 # Over-approximation: GT points outside pred hull (positive distance)
#                 over_approx = np.maximum(0, distances)
#                 over_loss = np.mean(over_approx ** 2) * self.over_approx_weight
                
#                 # Under-approximation: GT points inside that should be covered
#                 under_approx = np.maximum(0, -distances)
#                 under_loss = np.mean(under_approx ** 2) * self.under_approx_weight
                
#                 # Redundancy penalty: predicted points inside their own convex hull
#                 redundancy_loss = 0.0
#                 for i, vert in enumerate(verts):
#                     # Check if point is redundant (removing it doesn't change hull)
#                     other_verts = np.delete(verts, i, axis=0)
#                     if len(other_verts) >= 4:
#                         try:
#                             other_hull = ConvexHull(other_verts)
#                             other_hull_mesh = trimesh.Trimesh(
#                                 vertices=other_verts,
#                                 faces=other_hull.simplices
#                             )
#                             dist = self.point_to_mesh_distance(vert.reshape(1, -1), other_hull_mesh)[0]
#                             if dist < 0:  # Point is inside hull of others
#                                 redundancy_loss += (-dist) ** 2
#                         except:
#                             pass
                
#                 redundancy_loss = redundancy_loss / len(verts) * self.redundancy_weight
                
#                 batch_loss = over_loss + under_loss + redundancy_loss
#                 total_loss += batch_loss
                
#             except Exception as e:
#                 print(f"Error computing hull loss: {e}")
#                 total_loss += 10.0  # Penalty for invalid hull
        
#         return total_loss / B


import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def cvx_layer():
    N = 60  # number of mesh points in cluster
    x = cp.Parameter(3)
    P = cp.Parameter((N, 3))
    lam = cp.Variable(N)

    objective = cp.Minimize(cp.sum_squares(P.T @ lam - x))
    constraints = [lam >= 0, cp.sum(lam) == 1]
    problem = cp.Problem(objective, constraints)
    cvx_layer = CvxpyLayer(problem, parameters=[x, P], variables=[lam])

    return cvx_layer

def compute_cluster_probabilities(sample_points, mesh_points, soft_cluster_assignments, threshold=0.1):
    # Suppose the sample points were generated as a convex combination of mesh points.
    # In each cluster, we compute the probability of the sample points being inside that cluster
    # for cluster in range(soft_cluster_assignments.shape[1]):
    #     cluster_inclusion_probabilities = soft_cluster_assignments[:, cluster]  # (N, )
    #     cluster_distribution = torch.distributions.Dirichlet(cluster_inclusion_probabilities + 1e-6)

        # TODO: Figure out how to compute the probability of the sample point being inside the convex hull
        # Probably requires something like cvxpylayers etc.
        # ???


    distances = []
    for cluster in range(soft_cluster_assignments.shape[1]):
        cluster_points = mesh_points[soft_cluster_assignments[:, cluster] > threshold]
        for x_i in sample_points:
            lam_star, _ = cvx_layer(x_i, cluster_points)
            reconstructed = (cluster_points.T @ lam_star).T
            dist = torch.norm(reconstructed - x_i)
            distances.append(dist)

    cluster_probabilities = torch.exp(-torch.stack(distances))

    return cluster_probabilities


def sample_random_point_from_single_soft_convex_hull(mesh_points, inclusion_probabilities, num_samples):
  # mesh_points: (N, 3)
  # inclusion_probabilities: (N, )
  distribution = torch.distributions.Dirichlet(inclusion_probabilities + 1e-6)
  weights = distribution.sample((num_samples,))  # (num_samples, N)
  sampled_points = (weights.unsqueeze(2) * mesh_points).sum(dim=1)
  return sampled_points

def compute_loss_sample_option_1_sample_in_unit_bbox(embedding_model, mesh, coacd_meshes):
  # Pick M/2 points inside the convex decompositions and M/2 points outside
  M = 100
  random_coacd_meshes = random.choices(coacd_meshes, weights=[mesh.volume() for mesh in coacd_meshes], k=M // 2)
  points_inside_coacd = [mesh.sample_point_in_mesh() for mesh in random_coacd_meshes]  # (M/2, 3)

  points_outside_coacd = []
  while len(points_outside_coacd) < M // 2:
    point = mesh.bounding_box.sample_random_point()
    if not any(coacd_mesh.contains_point(point) for coacd_mesh in coacd_meshes):
      points_outside_coacd.append(point)

  sample_points = torch.tensor(points_inside_coacd + points_outside_coacd)  # (M, 3)

  # Compute embedding for all points on the mesh
  mesh_points = mesh.sample_points_on_surface(number_of_points=1000)  # (N, 3)
  mesh_embeddings = embedding_model(mesh_points)  # (N, 448)

  # Check if sampled point should be inside the label meshes
  inside = torch.tensor(
    [
      any(coacd_mesh.contains_point(sample_point) for coacd_mesh in coacd_meshes)
      for sample_point in sample_points
    ]
  )  # (M, )

  # Compute probability of point being inside one of the convex decompositions
  # First, soft-kmeans cluster the embeddings
  num_clusters = len(coacd_meshes)
  soft_cluster_assignments = soft_kmeans(mesh_embeddings, num_clusters)  # (N, num_clusters)

  # Compute the probability of the sampled point being inside each cluster
  # TODO: How do we implement this???
  cluster_probabilities = compute_cluster_probabilities(sample_points, mesh_points, soft_cluster_assignments)  # (M, num_clusters)

  # Total probability of the point being inside any cluster
  # TODO: Should we do some kind of softmax over axis 1 prior to summing?
  total_probability = torch.softmax(cluster_probabilities, axis=1).sum(axis=1)  # (M,)

  # Compute binary cross-entropy loss between predicted probability and ground truth
  loss = torch.binary_cross_entropy(total_probability, inside.float())  # (M,)
  return loss.mean()

def compute_loss_sample_option_2_sample_in_clusters(embedding_model, mesh, coacd_meshes):
  # Compute embedding for all points on the mesh
  mesh_points = mesh.sample_points_on_surface(number_of_points=1000)  # (N, 3)
  mesh_embeddings = embedding_model(mesh_points)  # (N, 448)

  # Compute probability of point being inside one of the convex decompositions
  # First, soft-kmeans cluster the embeddings
  num_clusters = len(coacd_meshes)
  soft_cluster_assignments = soft_kmeans(mesh_embeddings, num_clusters)  # (N, num_clusters)

  # Make sure that each point is assigned to at least one cluster.
  # TODO: Maybe soft-kmeans already does this - Check
  # soft_cluster_assignments = softmax(soft_cluster_assignments, dim=1)

  # Sample M points from the clusters
  M = 100
  points_per_cluster = M // num_clusters
  sample_points = []
  for cluster in range(num_clusters):
    cluster_inclusion_probabilities = soft_cluster_assignments[:, cluster]  # (N, )
    sampled_points = sample_random_point_from_single_soft_convex_hull(mesh_points, cluster_inclusion_probabilities, points_per_cluster)  # (points_per_cluster, 3)
    sample_points.append(sampled_points)
  sample_points = th.vstack(sample_points)  # (M, 3)

  # Compute signed distance to the convex decomposition at each of the sampled points
  signed_distances = th.minimum(0, sdf(sample_points))  # (M, )  

  # Compute binary cross-entropy loss between predicted probability and ground truth
  return signed_distances.mean()