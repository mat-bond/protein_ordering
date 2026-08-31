import math
from typing import Optional, TypeAlias

import numpy as np
import pydantic
import torch
from torch import nn

from layers.ect import EctConfig
from scipy.sparse import coo_matrix

import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import torch_geometric
import math
from .Equiformer.equiformer.nets.registry import register_model
from .Equiformer.equiformer.nets.instance_norm import EquivariantInstanceNorm
from .Equiformer.equiformer.nets.graph_norm import EquivariantGraphNorm
from .Equiformer.equiformer.nets.layer_norm import EquivariantLayerNormV2
from .Equiformer.equiformer.nets.fast_layer_norm import EquivariantLayerNormFast
from .Equiformer.equiformer.nets.radial_func import RadialProfile
from .Equiformer.equiformer.nets.tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from .Equiformer.equiformer.nets.fast_activation import Activation, Gate
from .Equiformer.equiformer.nets.drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .Equiformer.equiformer.nets.gaussian_rbf import GaussianRadialBasisLayer
from .Equiformer.equiformer.nets.graph_attention_transformer import (
    get_norm_layer,
    EdgeDegreeEmbeddingNetwork,
    TransBlock,
)
from modules.graph_encoder import PointCloudEquiformerEncoder
from modules.edge_layers import EdgeGraphLayer

Tensor: TypeAlias = torch.Tensor
### Gumbell-Softmax operation taken from https://arxiv.org/pdf/1802.08665 and https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/permutations.html


class ModelConfig(pydantic.BaseModel):
    module: str
    L_max: int
    learning_rate: float
    ectlossconfig: EctConfig
    ectconfig: EctConfig

    d_model: int = 512 
    nhead: int = 8
    num_encoder_layers: int = 8
    num_decoder_layers: int = 3
    dim_feedforward: int = 2048
    dropout: float = 0.1

    use_spatial_features: bool = True
    memory_token_dropout: float = 0.1
    tau: float = 0.3
    n_sink_iter: int = 20
    edge_radius: float = 0.1
    d_max: float = 0.1

    equiformer_num_heads: int = 4
    equiformer_max_neighbors: int = 64
    equiformer_avg_degree: float = 18.0
    equiformer_num_basis: int = 32

    equiformer_num_global_tokens: int = 4 
    equiformer_global_mixer_heads: int = 4
    equiformer_global_mixer_layers: int = 4 

    num_edgescorer_layers: int = 3


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1), :]

Tensor: TypeAlias = torch.Tensor

class EdgeScorer(nn.Module):
    def __init__(self, d_model, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        half_d = self.d_model//2 
        self.edge_dim = 4*half_d
        self.node_dim = half_d

        self.node_proj = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model,self.node_dim)
        )

        self.edge_node_proj = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model,self.node_dim)
        )

        self.graph_layers = nn.ModuleList([
            EdgeGraphLayer(self.node_dim, self.edge_dim, d_model, dropout)
            for _ in range(num_layers)
        ])

        self.score_head = nn.Sequential(
            nn.Linear(self.edge_dim,d_model),
            nn.GELU(),
            nn.Linear(d_model,2)
        )

        self.node_out_proj = nn.Sequential(
            nn.Linear(self.node_dim,d_model),
            nn.GELU(),
            nn.Linear(d_model,d_model)
        )

    def forward(self, memory, edge_src, edge_dst, mask):

        B,L,D = memory.shape
        flat_mask = mask.reshape(-1)
        flat_mem = memory.reshape(B*L,-1)[flat_mask] # [B*L,D]

        if edge_dst.numel() != edge_src.numel():
            raise ValueError("Incompatible edge lists")
        
        node_feat = self.node_proj(flat_mem)
        edge_node_feat = self.edge_node_proj(flat_mem) 
        h_i = edge_node_feat[edge_src]
        h_j = edge_node_feat[edge_dst]
        dh_ij = torch.abs(h_i-h_j) 
        prod_h = torch.mul(h_i,h_j)
        edge_feat = torch.cat([h_i,h_j,dh_ij,prod_h],dim=-1)
        E, _ = edge_feat.shape
        logits = torch.zeros((E,2), dtype=edge_feat.dtype, device=edge_feat.device, requires_grad=True)

        for layer in self.graph_layers:
            # Removed message passing weights for now. 
            # logits = self.score_head(edge_feat) # [E, 2]
            # score_ij = logits[:, 1]
            # g_ij = torch.sigmoid(score_ij)
            # g_sum = torch.zeros(node_feat.size(0), device=g_ij.device, dtype=g_ij.dtype)
            # g_sum.index_add_(0,edge_dst,g_ij)
            # w_ij = 2.0 * g_ij / g_sum[edge_dst].clamp_min(1e-6)
            node_feat, edge_feat = layer(node_feat, edge_feat, edge_src, edge_dst)

        logits = self.score_head(edge_feat) # [E, 2]

        node_out = self.node_out_proj(node_feat)

        return logits, node_out

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.L_max = config.L_max
        self.tau = config.tau
        self.n_sink_iter = config.n_sink_iter

        D = config.d_model
        self.D = D
        self.score_scale = D ** -0.5

        self.unknown_coord_embed = nn.Parameter(torch.zeros(1, 1, D))

        self.padding_embed = nn.Parameter(torch.zeros(1, 1, D))

        self.graph_encoder = PointCloudEquiformerEncoder(
            d_model=D,
            num_layers=config.num_encoder_layers,
            max_radius=config.edge_radius,
            max_num_neighbors=config.equiformer_max_neighbors,
            number_of_basis=config.equiformer_num_basis,
            basis_type="gaussian",
            num_heads=config.equiformer_num_heads,
            avg_degree=config.equiformer_avg_degree,
            nonlinear_message=True,
            rescale_degree=False,
            norm_layer="layer",
            alpha_drop=config.dropout,
            proj_drop=config.dropout,
            out_drop=config.dropout,
            drop_path_rate=0.0,
            num_global_tokens=config.equiformer_num_global_tokens,
            global_mixer_heads=config.equiformer_global_mixer_heads,
            global_mixer_layers=config.equiformer_global_mixer_layers,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.edge_scorer = EdgeScorer(D, num_layers=config.num_edgescorer_layers, dropout=config.dropout)
        
        self.refine_norm = nn.LayerNorm(self.D)

        self.query_embed = nn.Parameter(torch.randn(1, self.L_max, D) * 0.02)
        self.query_pos = SinusoidalPositionalEncoding(D, max_len=self.L_max)
        self.query_dropout = nn.Dropout(config.dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=D,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.residue_decoder = nn.TransformerDecoder(
            dec_layer, num_layers=config.num_decoder_layers
        )
        
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
    

    def _log_sinkhorn(self, log_alpha: Tensor, n_iter: int) -> Tensor:
        # log_alpha: [T, I] or [B, T, I]
        squeeze = False
        if log_alpha.ndim == 2:
            log_alpha = log_alpha.unsqueeze(0)
            squeeze = True

        for _ in range(n_iter):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)

        out = log_alpha.exp()
        return out.squeeze(0) if squeeze else out


    def _masked_assignment(
        self,
        logits: Tensor,        # [B, T, I]
        point_mask: Tensor,    # [B, I] True = valid input point,
        tau: float
    ) -> Tensor:
        B, T, I = logits.shape
        P_full = logits.new_zeros(B, T, I)

        for b in range(B):
            valid_t = point_mask[b].nonzero(as_tuple=True)[0]
            valid_i = point_mask[b].nonzero(as_tuple=True)[0]

            nt = valid_t.numel()
            ni = valid_i.numel()

            if nt == 0 and ni == 0:
                continue
            if nt != ni:
                raise ValueError(
                    f"Example {b}: permutation requires same number of valid "
                    f"targets and inputs, got {nt} targets and {ni} inputs."
                )
    
            sub_logits = logits[b].index_select(0, valid_t).index_select(1, valid_i)

            # Deterministic, differentiable Sinkhorn.
            sub_P = self._log_sinkhorn(
                sub_logits / tau,
                self.n_sink_iter,
            )

            rr, cc = torch.meshgrid(valid_t, valid_i, indexing="ij")
            P_full[b, rr, cc] = sub_P

        return P_full
    
    def forward(
        self,
        xyz_cloud: Tensor,                    # [B, L, 3]
        point_mask: Optional[Tensor] = None, # [B, L], True = valid point 
        padding_mask: Optional[Tensor] = None, # [B, L], True = not padding
        assignment_tau: Optional[float] = None,
    ):
        tau = self.tau if assignment_tau is None else float(assignment_tau)
        B, L, _ = xyz_cloud.shape
        if L > self.L_max:
            raise ValueError(f"Input length {L} exceeds L_max={self.L_max}")

        device = xyz_cloud.device

        if point_mask is None:
            point_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        else:
            point_mask = point_mask.bool()

        if padding_mask is None:
            padding_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        else:
            padding_mask = padding_mask.bool()

        mask = padding_mask & point_mask

        point_mask = mask

        # Encoder sees input points
        src_key_padding_mask = ~mask

        xyz_safe = torch.where(point_mask.unsqueeze(-1), xyz_cloud, torch.zeros_like(xyz_cloud))
        memory, _, edge_src, edge_dst = self.graph_encoder(xyz_safe, mask)

        memory = self.dropout(memory)

        # Predict edge scores and refine memory
        edge_logits, refined_valid = self.edge_scorer(memory, edge_src, edge_dst, mask)

        # Residual add of refined memory
        flat_mask = mask.reshape(-1)
        memory_delta = memory.new_zeros(B * L, self.D)
        memory_delta[flat_mask] = refined_valid
        memory = memory + memory_delta.view(B, L, self.D)
        memory = self.refine_norm(memory)
        memory = memory.masked_fill(~mask.unsqueeze(-1), 0.0)
        memory = self.dropout(memory)

        # Decoder queries are target residue positions
        queries = self.query_embed[:, :L, :].expand(B, -1, -1)
        queries = self.query_dropout(self.query_pos(queries))

        residue_repr = self.residue_decoder(
            tgt=queries,
            memory=memory,
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # [B, L, D]

        residue_repr = self.dropout(residue_repr)
        
        q = self.q_proj(residue_repr) # [B, T, D]
        k = self.k_proj(memory) # [B, I, D]
        assignment_logits = torch.einsum("btd,bid->bti", q, k)*self.score_scale

        # Run Sinkhorn / Hungarian only on valid submatrices
        permutation_matrices = self._masked_assignment(
            assignment_logits,
            point_mask=mask,
            tau=tau
        )  # [B, L, L], row=target, col=input

        # Reorder scrambled input points into backbone order
        ordered_xyz = torch.einsum("bti,bid->btd", permutation_matrices, xyz_safe)

        # Zero out invalid target positions
        ordered_xyz = ordered_xyz.masked_fill(~mask.unsqueeze(-1), 0.0) # Should this zero out?

        return ordered_xyz, permutation_matrices, assignment_logits, edge_logits, edge_src, edge_dst