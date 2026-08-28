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

Tensor: TypeAlias = torch.Tensor
_RESCALE = True
_USE_BIAS = True
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

    equiformer_num_global_tokens: int = 4 # dont raise, is bad.
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


class PointCloudEquiformerEncoder(nn.Module):
    """
    Dense padded input  [B, L, 3] + mask [B, L]
        -> sparse radius graph over valid points
        -> Equiformer-style equivariant transformer blocks
        -> dense scalar memory [B, L, D] for the downstream decoder

    This is task-adapted:
      - no atom/residue types are assumed
      - node initialization uses a constant 0e feature
      - final output is purely scalar so the existing Transformer decoder
        can consume it directly
    """

    def __init__(
            self,
            d_model: int,
            num_layers: int = 6,
            max_radius: float = 0.1,
            max_num_neighbors: int = 64,
            number_of_basis: int = 32,
            basis_type: str = "gaussian",
            num_heads: int = 4,
            lmax: int = 2,
            avg_degree: float = 16.0,
            irreps_node_embedding: Optional[str] = None,
            irreps_head: Optional[str] = None,
            irreps_mlp_mid: Optional[str] = None,
            irreps_pre_attn: Optional[str] = None,
            nonlinear_message: bool = True,
            rescale_degree: bool = False,
            norm_layer: str = "layer",
            alpha_drop: float = 0.1,
            proj_drop: float = 0.1,
            out_drop: float = 0.1,
            drop_path_rate: float = 0.0,
            num_global_tokens: int = 4,
            global_mixer_heads: int = 4,
            global_mixer_layers: int = 2,
        ):
        super().__init__()

        self.d_model = d_model
        self.max_radius = max_radius
        self.max_num_neighbors = max_num_neighbors
        self.number_of_basis = number_of_basis
        self.num_layers = num_layers
        self.basis_type = basis_type
        self.norm_layer = norm_layer

        if irreps_node_embedding is None or irreps_head is None or irreps_mlp_mid is None:
            hidden_irreps, head_irreps, mlp_irreps = self._default_irreps(d_model)
            if irreps_node_embedding is None:
                irreps_node_embedding = hidden_irreps
            if irreps_head is None:
                irreps_head = head_irreps
            if irreps_mlp_mid is None:
                irreps_mlp_mid = mlp_irreps

        self.irreps_node_attr = o3.Irreps("1x0e")
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        self.irreps_head = o3.Irreps(irreps_head)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        self.irreps_out = o3.Irreps(f"{d_model}x0e")

        # Constant node seed (since inputs are unlabeled 3D points).
        self.input_proj = LinearRS(
            self.irreps_node_attr,
            self.irreps_node_embedding,
            bias=True,
            rescale=_RESCALE,
        )

        if basis_type == "gaussian":
            self.rbf = GaussianRadialBasisLayer(number_of_basis, cutoff=max_radius)
        else:
            raise ValueError(f"Unknown basis_type={basis_type}") # For now, only Gaussian implemented

        self.fc_neurons = [number_of_basis, 64, 64]

        # Equiformer-style degree embedding; here it becomes a geometry-only
        # initialization because we do not have atom types.
        self.edge_degree_embed = EdgeDegreeEmbeddingNetwork(
            irreps_node_embedding=self.irreps_node_embedding,
            irreps_edge_attr=self.irreps_edge_attr,
            fc_neurons=self.fc_neurons,
            avg_aggregate_num=avg_degree,
        )

        self.blocks = nn.ModuleList([
            TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_node_output=self.irreps_node_embedding,
                fc_neurons=self.fc_neurons,
                irreps_head=self.irreps_head,
                num_heads=num_heads,
                irreps_pre_attn=irreps_pre_attn,
                rescale_degree=rescale_degree,
                nonlinear_message=nonlinear_message,
                alpha_drop=alpha_drop,
                proj_drop=proj_drop,
                drop_path_rate=drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=norm_layer,
            )
            for _ in range(num_layers)
        ])

        self.norm = get_norm_layer(norm_layer)(self.irreps_node_embedding)
        self.out_proj = LinearRS(
            self.irreps_node_embedding,
            self.irreps_out,
            bias=True,
            rescale=_RESCALE,
        )

        self.out_dropout = (
            EquivariantDropout(self.irreps_out, drop_prob=out_drop)
            if out_drop > 0.0 else None
        )

        self.padding_embed = nn.Parameter(torch.zeros(1, d_model))

        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(
            torch.randn(1, self.num_global_tokens, d_model) * 0.02
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=global_mixer_heads,
            dim_feedforward=4 * d_model,
            dropout=out_drop,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.global_mixer = nn.TransformerEncoder(
            enc_layer,
            num_layers=global_mixer_layers,
        )
        self.global_mixer_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _default_irreps(d_model: int):
        # Mixed-parity hidden irreps sized so the total representation dimension
        # stays in the same ballpark as d_model.
        n0e = max(16, d_model // 4)
        n0o = max(4,  d_model // 16)
        n1e = max(4,  d_model // 16)
        n1o = max(4,  d_model // 16)
        n2e = max(2,  d_model // 32)
        n2o = max(2,  d_model // 32)

        hidden = (
            f"{n0e}x0e+{n0o}x0o+"
            f"{n1e}x1e+{n1o}x1o+"
            f"{n2e}x2e+{n2o}x2o"
        )

        h0e = max(8, d_model // 32)
        h0o = max(4, d_model // 64)
        h1e = max(4, d_model // 64)
        h1o = max(4, d_model // 64)
        h2e = max(2, d_model // 128)
        h2o = max(2, d_model // 128)

        head = (
            f"{h0e}x0e+{h0o}x0o+"
            f"{h1e}x1e+{h1o}x1o+"
            f"{h2e}x2e+{h2o}x2o"
        )

        mid = (
            f"{3*n0e}x0e+{3*n0o}x0o+"
            f"{3*n1e}x1e+{3*n1o}x1o+"
            f"{3*n2e}x2e+{3*n2o}x2o"
        )
        return hidden, head, mid

    def _encode_sparse(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        pos:   [N, 3]
        batch: [N]
        returns: [N, d_model] scalar memory
        """
        N = pos.shape[0]
        dtype = pos.dtype
        device = pos.device

        node_attr = torch.ones(N, 1, device=device, dtype=dtype)
        node_feat = self.input_proj(node_attr)

        if N > 1:
            edge_src, edge_dst = radius_graph(
                pos,
                r=self.max_radius,
                batch=batch,
                loop=False,
                max_num_neighbors=self.max_num_neighbors,
            )

            if edge_src.numel() > 0:
                edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
                edge_len = edge_vec.norm(dim=-1)

                edge_sh = o3.spherical_harmonics(
                    l=self.irreps_edge_attr,
                    x=edge_vec,
                    normalize=True,
                    normalization="component",
                )
                edge_rbf = self.rbf(edge_len)

                node_feat = node_feat + self.edge_degree_embed(
                    node_feat,
                    edge_sh,
                    edge_rbf,
                    edge_src,
                    edge_dst,
                    batch,
                )

                for blk in self.blocks:
                    node_feat = blk(
                        node_input=node_feat,
                        node_attr=node_attr,
                        edge_src=edge_src,
                        edge_dst=edge_dst,
                        edge_attr=edge_sh,
                        edge_scalars=edge_rbf,
                        batch=batch,
                    )
        else: 
            edge_src = torch.empty(0, dtype=torch.long, device=device)
            edge_dst = torch.empty(0, dtype=torch.long, device=device)

        node_feat = self.norm(node_feat, batch=batch)
        node_feat = self.out_proj(node_feat)

        if self.out_dropout is not None:
            node_feat = self.out_dropout(node_feat)

        return node_feat, edge_src, edge_dst

    def forward(self, xyz: torch.Tensor, mask: torch.Tensor):
        """
        xyz:  [B, L, 3]
        mask: [B, L] bool
        returns:
            memory: [B, L, D]
            aux:    None  
        """
        B, L, _ = xyz.shape
        flat_mask = mask.reshape(-1)
        edge_src = torch.empty(0, dtype=torch.long, device=xyz.device)
        edge_dst = torch.empty(0, dtype=torch.long, device=xyz.device)
        dense_out = self.padding_embed.expand(B * L, -1).clone()

        if flat_mask.any():
            flat_xyz = xyz.reshape(B * L, 3)[flat_mask]
            batch = torch.arange(B, device=xyz.device).repeat_interleave(L)[flat_mask]

            # Translation invariance already comes from relative edge vectors,
            # but centering per graph helps numerics a bit.
            centers = []
            for b in range(B):
                xb = xyz[b][mask[b]]
                if xb.numel() == 0:
                    centers.append(torch.zeros(1, 3, device=xyz.device, dtype=xyz.dtype))
                else:
                    centers.append(xb.mean(dim=0, keepdim=True))
            centers = torch.cat(centers, dim=0)
            flat_xyz = flat_xyz - centers.index_select(0, batch)

            valid_out, edge_src, edge_dst = self._encode_sparse(flat_xyz, batch)
            dense_out[flat_mask] = valid_out

        dense_out = dense_out.view(B, L, self.d_model)

        # zero padded slots before global attention
        dense_out = dense_out.masked_fill(~mask.unsqueeze(-1), 0.0)

        g = self.global_tokens.expand(B, -1, -1)              # [B, G, D]
        mixed = torch.cat([g, dense_out], dim=1)              # [B, G+L, D]

        global_mask = torch.zeros(B, self.num_global_tokens, dtype=torch.bool, device=mask.device)
        mixed_key_padding_mask = torch.cat([global_mask, ~mask], dim=1)

        mixed = self.global_mixer(
            mixed,
            src_key_padding_mask=mixed_key_padding_mask,
        )
        mixed = self.global_mixer_norm(mixed)

        # keep only node memory
        dense_out = mixed[:, self.num_global_tokens:, :]
        dense_out = dense_out.masked_fill(~mask.unsqueeze(-1), 0.0)

        return dense_out, mixed[:, :self.num_global_tokens, :], edge_src, edge_dst

class EdgeGraphLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, d_model, dropout = 0.1):
        super().__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*node_dim+edge_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, edge_dim),
            nn.Dropout(dropout)
            )
        self.edge_norm = nn.LayerNorm(edge_dim)

        self.msg_mlp = nn.Sequential(
            nn.Linear(2*node_dim+edge_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, node_dim),
            nn.Dropout(dropout)
        )
        self.msg_norm = nn.LayerNorm(node_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(2*node_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, node_dim),
            nn.Dropout(dropout)
        )
        self.node_norm = nn.LayerNorm(node_dim)
        
    def forward(self, node_feat, edge_feat, edge_src, edge_dst, weights_ij = None):
        z_i = node_feat[edge_src]
        z_j = node_feat[edge_dst]

        # Update Edges
        edge_in = torch.cat([z_i,z_j,edge_feat],dim=-1)
        edge_feat = self.edge_norm(self.edge_mlp(edge_in)+edge_feat)

        # Message passing according to updated edges
        msg_in = torch.cat([z_i,z_j,edge_feat],dim=-1)
        m_ij = weights_ij.unsqueeze(-1)*self.msg_norm(self.msg_mlp(msg_in)) if weights_ij is not None else self.msg_norm(self.msg_mlp(msg_in))
        m_i = node_feat.new_zeros(node_feat.shape)
        m_i.index_add_(0,edge_dst,m_ij)

        # Update nodes according to messages
        node_in = torch.cat([node_feat,m_i],dim=-1)
        node_feat = self.node_norm(self.node_mlp(node_in)+node_feat)

        return node_feat, edge_feat

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