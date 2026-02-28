# ============================================================================
# Model Architecture:

# Latents (SparseTensor)
#    ↓
# SLatMeshDecoder (model)
#    ↓
# Predicted 3D Mesh (MeshExtractResult)
#    ↓
# Trainer renders the mesh → compares to ground truth
#    ↓
# Losses are computed (depth, normal, TSDF, etc.)
#    ↓
# Backprop through the renderer and decoder
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis.modules.utils import convert_module_to_f16, convert_module_to_f32
from trellis.modules import sparse as sp
from trellis.modules.transformer import AbsolutePositionEmbedder
from trellis.modules.sparse.transformer import SparseTransformerBlock


class VoxelEncoder(nn.Module):
    """3D CNN encoder for voxel input"""
    
    def __init__(self, voxel_resolution: int = 64, embed_dim: int = 512):
        super().__init__()
        
        # 3D convolutions to encode voxel grid
        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)  # 64 -> 32
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)  # 32 -> 16
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)  # 16 -> 8
        self.conv4 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)  # 8 -> 4
        
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(16, 128)
        self.norm4 = nn.GroupNorm(32, 256)
        
        # Project to transformer dimension
        self.proj = nn.Linear(256 * 4 * 4 * 4, embed_dim)
        
    def forward(self, x):
        # x: (B, 64, 64, 64)
        x = x.unsqueeze(1)  # (B, 1, 64, 64, 64)
        
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        x = F.gelu(self.norm3(self.conv3(x)))
        x = F.gelu(self.norm4(self.conv4(x)))
        
        # Flatten
        x = x.flatten(1)  # (B, 256*4*4*4)
        x = self.proj(x)  # (B, embed_dim)
        
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder for sequence generation"""
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, 
                 num_layers: int = 6, ff_dim: int = 2048):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Input projection for vertices
        self.vertex_proj = nn.Linear(3, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 5000, embed_dim) * 0.02)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, 3)
        )
        
    def forward(self, voxel_encoding, tgt_sequence, tgt_mask=None):
        """
        Args:
            voxel_encoding: (B, embed_dim) - encoded voxel features
            tgt_sequence: (B, seq_len, 3) - target sequence
            tgt_mask: (B, seq_len) - mask for padding
        """
        B, seq_len = tgt_sequence.shape[0], tgt_sequence.shape[1]
        
        # Project vertices to embedding space
        tgt_embed = self.vertex_proj(tgt_sequence)  # (B, seq_len, embed_dim)
        
        # Add positional encoding
        tgt_embed = tgt_embed + self.pos_encoding[:, :seq_len, :]
        
        # Expand voxel encoding as memory
        memory = voxel_encoding.unsqueeze(1)  # (B, 1, embed_dim)
        
        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt_embed.device)
        
        # Apply transformer
        output = self.transformer(
            tgt_embed,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~tgt_mask if tgt_mask is not None else None
        )
        
        # Generate vertices
        vertices = self.output_head(output)  # (B, seq_len, 3)
        
        return vertices


class VoxelToMeshModel(nn.Module):
    """Complete model for voxel to mesh conversion"""
    
    def __init__(self, voxel_resolution: int = 64, embed_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 6, ff_dim: int = 2048):
        super().__init__()
        
        self.encoder = VoxelEncoder(voxel_resolution, embed_dim)
        self.decoder = TransformerDecoder(embed_dim, num_heads, num_layers, ff_dim)
        
    def forward(self, voxels, tgt_sequence, tgt_mask=None):
        """
        Args:
            voxels: (B, 64, 64, 64)
            tgt_sequence: (B, seq_len, 3)
            tgt_mask: (B, seq_len)
        """
        voxel_encoding = self.encoder(voxels)
        vertices = self.decoder(voxel_encoding, tgt_sequence, tgt_mask)
        return vertices
    
    def generate(self, voxels, max_length: int = 1000, temperature: float = 1.0):
        """Autoregressive generation"""
        B = voxels.shape[0]
        device = voxels.device
        
        # Encode voxels
        voxel_encoding = self.encoder(voxels)
        
        # Start with newmesh token
        sequence = torch.zeros(B, 1, 3).to(device)
        
        for _ in range(max_length - 1):
            # Predict next vertex
            output = self.decoder(voxel_encoding, sequence)
            next_vertex = output[:, -1:, :]  # (B, 1, 3)
            
            # Add to sequence
            sequence = torch.cat([sequence, next_vertex], dim=1)
            
            # Simple stopping criterion: if last vertex is very close to origin
            if torch.all(torch.norm(next_vertex, dim=-1) < 0.01):
                break
        
        return sequence
