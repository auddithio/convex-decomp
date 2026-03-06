import sys
import os
sys.path.append("/sailhome/aunag/code/convex_decomp/TRELLIS")  

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from data import VoxelDataset, collate_fn  
from loss import chamfer_loss, batched_chamfer_loss 

from trellis.models.structured_latent_vae.decoder_gs import SLatGaussianDecoder
from trellis.modules.sparse import SparseTensor  

import wandb
from pytorch_lightning.loggers import WandbLogger

HIGH_LOSS_THRESHOLD = 1e5
NUM_BAD_EXAMPLES_TO_LOG = 5
SKIP_NUM = 50000

# TODO: move importance model
# scale errors by the importance model for each voxel
# look at cost for the CoACD
# what are the costs for the CoACD decomposition? maybe we can use that as a signal for which samples are harder and should be weighted more in the loss?
# it's a C++ code, we need to compile it and run it on the dataset to get the costs. We can then save those costs in a file and load them in the training loop to weight the loss for each sample accordingly. This way, we can focus more on the harder samples that have higher costs in the CoACD decomposition, which might lead to better overall performance of our model.
# look into the VHACD. convexdecomp
# how to incorporate

# convert dense voxel grids into SparseTensor
def dense_to_sparse(voxel_grid, threshold=0.0):
    """
    Convert a dense voxel grid (B, D, H, W)
    into a sparse tensor accepted by SLatMeshDecoder.
    """
    voxel_grid = voxel_grid.unsqueeze(1)  # add channel dim

    batch_size, channels, D, H, W = voxel_grid.shape
    coords_list, feats_list = [], []

    for b in range(batch_size):
        mask = voxel_grid[b].sum(0) > threshold  # (D,H,W)
        idxs = mask.nonzero(as_tuple=False)  # [N, 3]
        if len(idxs) == 0:
            continue
        coords = torch.cat([torch.full((idxs.shape[0], 1), b, device=idxs.device), idxs], dim=1)  # [N,4]
        feats = voxel_grid[b, :, idxs[:, 0], idxs[:, 1], idxs[:, 2]].T  # [N,C]
        coords_list.append(coords)
        feats_list.append(feats)

    coords = torch.cat(coords_list, dim=0)
    feats = torch.cat(feats_list, dim=0)

    return SparseTensor(feats=feats, coords=coords.int())


class SLatMeshTrainer(L.LightningModule):
    def __init__(self, lr=1e-4, res=64):
        super().__init__()
        self.biggest_input_size = 0
        self.save_hyperparameters()
        self.model = SLatGaussianDecoder(resolution=res, model_channels=768, latent_channels=1, num_blocks=12,
                                          representation_config={
                                            "lr": {
                                                "_xyz": 1.0,
                                                "_features_dc": 1.0,
                                                "_opacity": 1.0,
                                                "_scaling": 1.0,
                                                "_rotation": 0.1
                                            },
                                            "perturb_offset": True,
                                            "voxel_size": 1.5,
                                            "num_gaussians": 32,
                                            "2d_filter_kernel_size": 0.1,
                                            "3d_filter_kernel_size": 9e-4,
                                            "scaling_bias": 4e-3,
                                            "opacity_bias": 0.1,
                                            "scaling_activation": "softplus"
                                            }
                                        )
        

    def forward(self, sparse_voxels):
        """
        Forward pass through the decoder.
        Returns a sparse tensor with predicted mesh points and membership scores
        """
        return self.model(sparse_voxels)
    
    def sample_random_point_from_single_soft_convex_hull(self, mesh_points, inclusion_probabilities, num_samples):
        # mesh_points: (N, 3)
        # inclusion_probabilities: (N, )
        distribution = torch.distributions.Dirichlet(inclusion_probabilities + 1e-6)
        weights = distribution.rsample((num_samples,))  # (num_samples, N)
        sampled_points = (weights.unsqueeze(2) * mesh_points).sum(dim=1)
        return sampled_points
    
    def sample_random_points_from_soft_convex_hulls_vectorized(self, mesh_points, inclusion_probabilities, num_samples):
        """
        Vectorized version of sample_random_point_from_single_soft_convex_hull.

        Args:
            mesh_points: (N, 3)
            inclusion_probabilities: (N, K) 
            num_samples: int (number of samples per cluster)
        Returns:
            sampled_points: (K * num_samples, 3)
        """
        # mesh_points: (N, 3)
        # inclusion_probabilities: (N, K)
        N, K = inclusion_probabilities.shape

        # Create K independent Dirichlet distributions in one go
        dist = torch.distributions.Dirichlet((inclusion_probabilities.T + 1e-6))
        # Sample weights: (num_samples, K, N)
        weights = dist.rsample((num_samples,))

        # Compute convex combinations for all clusters
        # (num_samples, K, N) x (N, 3) into (num_samples, K, 3)
        sampled_points = torch.einsum('skn,nc->skc', weights, mesh_points)

        # Flatten to (K * num_samples, 3)
        sampled_points = sampled_points.reshape(K * num_samples, 3)
        return sampled_points.to(torch.float32)

    def training_step_cvx(self, batch, batch_idx):
        voxels = batch["voxels"]
        gt_vertices = batch["sequence"]

        # convert to sparse
        sparse_vox = dense_to_sparse(voxels)
        m = self(sparse_vox)

        # sample M points from the clusters
        points_per_cluster = 1
        loss = 0.0

        for i in range(m.shape[0]): # batch

            sample_points = []
            m_coords = (m.coords[m.layout[i]][:, 1:].float() + 0.5) / 64
            m_feats = m.feats[m.layout[i]]  # (N, 32)

            sample_points = self.sample_random_points_from_soft_convex_hulls_vectorized(
                m_coords, m_feats, points_per_cluster
            )  # (32*P, 3)

            # 1: manual loopy way
            # loss += self.convex_hull_reconstruction_loss(sample_points, gt_vertices[i])

            # 2: batch way
            label_convex_hulls = gt_vertices[i]
            K, N, _ = label_convex_hulls.shape
            B = sample_points.shape[0]

            # print("(1) K:", K, "(2) N:", N, "(3) B:", B)

            # repeat n tile
            V_batch = label_convex_hulls.permute(0, 2, 1).repeat(B, 1, 1)  # (B*K, 3, N)
            q_batch = sample_points.repeat_interleave(K, dim=0)              # (B*K, 3)

            # TODO: debugging, remove
            # print("V_batch:", V_batch.shape, V_batch.stride(), V_batch.is_contiguous())
            # print("q_batch:", q_batch.shape, q_batch.stride(), q_batch.is_contiguous())
            

            # TODO: create a dummy w_star for debugging, remove
            # w_star = torch.ones((B*K, N), device=V_batch.device)
            # project all (B*K) pairs at once
            w_star, = self.convex_proj_layer(V_batch, q_batch)

            w_star = w_star.to(V_batch.device)
            proj_points = torch.bmm(V_batch, w_star.unsqueeze(-1)).squeeze(-1)

            # compute distances and reshape
            distances = torch.norm(q_batch - proj_points, dim=1).view(B, K)
            min_dist, _ = distances.min(dim=1)
            loss += min_dist.sum() / B

            del w_star, proj_points, distances
            torch.cuda.empty_cache()


        self.log("train_loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _log_bad_example(self, voxels, pred_pts, gt_pts, loss_val, batch_idx, item_idx):
        """
        Log a high-loss example to W&B with detailed diagnostics.
        🔴 Red → predicted sampled points
        🟢 Green → GT vertices
        🔵 Blue → voxelized input
        """
        if not isinstance(self.logger, WandbLogger):
            return

        # avoid flooding W&B
        if not hasattr(self, "_bad_example_count"):
            self._bad_example_count = 0
        if self._bad_example_count > NUM_BAD_EXAMPLES_TO_LOG: 
            return

        self._bad_example_count += 1

        pred = pred_pts.detach().float().cpu().numpy()
        gt   = gt_pts.detach().float().cpu().numpy()
        vox  = voxels.detach().cpu().numpy()

        # ---- Sanity diagnostics ----
        print("\n===== HIGH LOSS EXAMPLE =====")
        print("Loss:", loss_val.item())
        print("Pred min/max:", pred.min(), pred.max())
        print("GT min/max:", gt.min(), gt.max())
        print("Vox min/max:", vox.min(), vox.max())
        print("Pred NaNs:", np.isnan(pred).any())
        print("GT NaNs:", np.isnan(gt).any())
        print("=============================\n")

        # ---- Convert voxel grid to point cloud ----
        if vox.ndim == 3:
            voxel_points = np.argwhere(vox > 0)
        else:
            voxel_points = np.argwhere(vox[0] > 0)

        voxel_colors = np.tile(np.array([[0, 0, 255]]), (voxel_points.shape[0], 1))  # blue

        # ---- Color prediction & GT ----
        pred_colors = np.tile(np.array([[255, 0, 0]]), (pred.shape[0], 1)) # red
        gt_colors   = np.tile(np.array([[0, 255, 0]]), (gt.shape[0], 1)) # green

        pred_pc = np.concatenate([pred, pred_colors], axis=1)
        gt_pc   = np.concatenate([gt, gt_colors], axis=1)

        if voxel_points.shape[0] > 0:
            voxel_pc = np.concatenate([voxel_points / 64.0, voxel_colors], axis=1)
            combined = np.concatenate([pred_pc, gt_pc, voxel_pc], axis=0)
        else:
            combined = np.concatenate([pred_pc, gt_pc], axis=0)

        self.logger.experiment.log({
            f"debug/high_loss_example_{batch_idx}_{item_idx}":
                wandb.Object3D(combined),
            "debug/high_loss_value": loss_val.item(),
        })
    
    def training_step(self, batch, batch_idx):

        voxels = batch["voxels"]
        gt_vertices = batch["sequence"]

        sparse_vox = dense_to_sparse(voxels)

        if sparse_vox.feats.shape[0] > SKIP_NUM:
            print(f"Skipping batch {batch_idx}: {sparse_vox.feats.shape[0]} occupied voxels")
            return None # skip this batch to avoid OOM, TODO: handle better, maybe by sub-sampling the voxels?

        if sparse_vox.feats.shape[0] > self.biggest_input_size:
            self.biggest_input_size = sparse_vox.feats.shape[0]
            print(f"New biggest input size: {self.biggest_input_size}")

        m = self(sparse_vox)

        # sample M points from the clusters
        points_per_cluster = 32
        loss = 0.0

        for i in range(m.shape[0]): # each batch has different numbers of active points

            sample_points = []
            m_coords = (m.coords[m.layout[i]][:, 1:].float() + 0.5) / 64
            m_feats = m.feats[m.layout[i]]  # (N, 32)

            sample_points = self.sample_random_points_from_soft_convex_hulls_vectorized(
                m_coords, m_feats, points_per_cluster
            )  # (32*P, 3)

            # chamfer loss between sampled points and gt vertices
            chamfer_loss_value = chamfer_loss(
                sample_points,  # (M, 3)
                gt_vertices[i],   # (N, 3)
            )

            if (chamfer_loss_value.detach().item() > HIGH_LOSS_THRESHOLD or torch.isnan(chamfer_loss_value)):
                self._log_bad_example(
                    voxels[i],
                    sample_points,
                    gt_vertices[i],
                    chamfer_loss_value,
                    batch_idx,
                    i,
                )

            loss += chamfer_loss_value

        self.log("train_loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=True)
        return loss

    
    def project_to_convex_hull(self, points_torch, query_point):
        """
        points_torch: (N, 3)
        query_point: (3,)
        Returns: projected_point (3,), convex weights (N,)
        """
        V_t = points_torch.T  # shape (3, N)
        w_star, = self.convex_proj_layer(V_t, query_point)
        projected_point = V_t @ w_star
        return projected_point, w_star
    
    def min_distance_to_label_hulls(self, x, label_convex_hulls):
        distances = []
        for V in label_convex_hulls:
            proj, _ = self.project_to_convex_hull(V, x)
            distances.append(torch.norm(x - proj, p=2))
        distances = torch.stack(distances)
        return torch.min(distances)
    
    def convex_hull_reconstruction_loss(self, predicted_points, label_convex_hulls):
        losses = []
        for x in predicted_points:  # x: (3,)
            dist = self.min_distance_to_label_hulls(x, label_convex_hulls)
            losses.append(dist)
        losses = torch.stack(losses)
        return losses.mean()

    def validation_step(self, batch, batch_idx):
        voxels = batch["voxels"]
        gt_vertices = batch["sequence"]

        sparse_vox = dense_to_sparse(voxels)
        pred_mesh = self(sparse_vox)

        pred_vertices = pred_mesh.vertices
        gt_vertices = gt_vertices.to(pred_vertices.device)

        val_loss = self.chamfer(
            pred_vertices.unsqueeze(0),
            gt_vertices.unsqueeze(0),
        )

        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


class CUDAMemorySnapshotCallback(L.Callback):
    def __init__(self, every_n_steps=50, out_dir="titanrtx_snapshots"):
        self.every_n_steps = every_n_steps
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.every_n_steps == 0:
            path = os.path.join(self.out_dir, f"snapshot_step_{step}.pickle")
            torch.cuda.memory._dump_snapshot(path)


if __name__ == "__main__":

    # torch.cuda.memory._record_memory_history()

    wandb_logger = WandbLogger(
        project="convex_decomposition",
        name="exp_001_sampled_1024",
        log_model=True,   # uploads checkpoints
    )


    train_ds = VoxelDataset(
        model_list="regular_submeshes.txt",
        # voxel_directory="/vision/group/objaverse/voxels_2.0",
        # coacd_directory="/vision/group/objaverse/affogato_subset/coacd",
        voxel_directory="/scr/aunag/objaverse/voxels_2.0/",
        coacd_directory="/scr/aunag/objaverse/coacd/",
        stats_dir="./mesh_stats/"
    ) 
    
    # TODO: remove later, for debugging, use only 10 samples
    # train_ds = torch.utils.data.Subset(train_ds, range(10))

    print(f"Training on {len(train_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True) 

    model = SLatMeshTrainer(lr=1e-4)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_scaled/",
        filename="epoch_{epoch:04d}",
        every_n_epochs=1,
        save_top_k=-1,      # keep all checkpoints
        save_last=True,     # also keep last.ckpt
    )

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=100,
        log_every_n_steps=1,
        precision=16,
        callbacks=[checkpoint_callback], # [CUDAMemorySnapshotCallback(every_n_steps=1)],
        logger=wandb_logger, 
        # profiler="pytorch",
    )

    print("Starting training...")

    import traceback

    try:
        trainer.fit(model, train_loader)
        print("Training complete.")

    except Exception as e:
        traceback.print_exc()
        raise


