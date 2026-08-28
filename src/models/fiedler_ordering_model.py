"""
End-to-end differentiable protein point-cloud ordering via a learned backbone
adjacency followed by a Fiedler spectral ordering layer and Sinkhorn.

Expected placement:
    src/models/fiedler_ordering_model.py

Set/override rmsd_modelconfig.module to:
    models.fiedler_ordering_model
"""

import math
from typing import Optional, TypeAlias

import pydantic
import torch
import torch.nn.functional as F
from torch import nn
from torch_cluster import radius_graph

from layers.ect import EctConfig

from e3nn import o3
from .Equiformer.equiformer.nets.tensor_product_rescale import LinearRS
from .Equiformer.equiformer.nets.drop import EquivariantDropout
from .Equiformer.equiformer.nets.gaussian_rbf import GaussianRadialBasisLayer
from .Equiformer.equiformer.nets.graph_attention_transformer import (
    get_norm_layer,
    EdgeDegreeEmbeddingNetwork,
    TransBlock,
)

Tensor: TypeAlias = torch.Tensor
_RESCALE = True


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

    # tau is only a fallback; the training script passes a scheduled tau.
    tau: float = 0.2
    n_sink_iter: int = 50

    # Coordinates in the existing code are in nm-like units: 3.8 A = 0.038.
    edge_radius: float = 0.1
    d_max: float = 0.1
    bond_length: float = 0.038
    bond_sigma: float = 0.01

    equiformer_num_heads: int = 4
    equiformer_max_neighbors: int = 64
    equiformer_avg_degree: float = 18.0
    equiformer_num_basis: int = 32
    equiformer_num_global_tokens: int = 4
    equiformer_global_mixer_heads: int = 4
    equiformer_global_mixer_layers: int = 4

    num_edgescorer_layers: int = 3

    # Candidate graph for the learned backbone-edge head.  This is separate from
    # the Equiformer message-passing graph.
    candidate_radius: Optional[float] = None
    candidate_knn: int = 16

    # Spectral layer.
    fiedler_eigh_dtype: str = "float64"
    spectral_background_eps: float = 1e-7
    spectral_rank_clamp_eps: float = 1e-6

    # torch.linalg.eigh has a singular eigenvector derivative when lambda_2
    # collides with lambda_1 or lambda_3.  The custom backward below uses the
    # exact derivative when the Fiedler eigengap is healthy and returns zero
    # spectral gradient for ill-defined near-degenerate examples.  EdgeCE still
    # trains those examples, so they can become path-like and re-enter the
    # end-to-end spectral objective later.
    spectral_backward_min_gap: float = 1e-6



class _StableFiedlerVector(torch.autograd.Function):
    """
    Return the second eigenvector of a real symmetric matrix with a stabilized
    backward pass.

    PyTorch's generic eigh eigenvector backward contains factors
        1 / (lambda_k - lambda_j)
    and becomes undefined when eigenvalues collide.  For a learned soft graph,
    this happens naturally while the graph is disconnected / nearly disconnected.

    Forward:
        exact torch.linalg.eigh result.

    Backward:
        exact symmetric eigenvector derivative when the separation of lambda_2
        from lambda_1 and lambda_3 is >= min_gap;
        otherwise return zero gradient for this example.

    Returning zero in the degenerate case is intentional: the Fiedler vector is
    not uniquely defined there, so there is no mathematically meaningful
    eigenvector gradient to follow.  EdgeCE remains active and can move the graph
    out of the degenerate regime.
    """

    @staticmethod
    def forward(ctx, matrix: Tensor, min_gap: float):
        eigvals, eigvecs = torch.linalg.eigh(matrix)

        if matrix.shape[0] < 2:
            raise ValueError("Stable Fiedler vector requires an NxN matrix with N>=2")

        v = eigvecs[:, 1]

        ctx.save_for_backward(eigvals, eigvecs)
        ctx.min_gap = float(min_gap)

        # Eigenvalues are used only for diagnostics / scheduling in this model.
        # Do not route a second gradient through them.
        ctx.mark_non_differentiable(eigvals)
        return v, eigvals

    @staticmethod
    def backward(ctx, grad_v: Tensor, grad_eigvals: Tensor | None):
        eigvals, eigvecs = ctx.saved_tensors
        n = eigvals.numel()
        k = 1

        grad_matrix = torch.zeros(
            n, n,
            device=eigvecs.device,
            dtype=eigvecs.dtype,
        )

        if grad_v is None or not torch.isfinite(grad_v).all():
            return grad_matrix, None

        # Only separations involving the eigenvector we differentiated matter.
        gaps = eigvals[k] - eigvals
        keep = torch.ones(n, dtype=torch.bool, device=eigvals.device)
        keep[k] = False

        min_sep = gaps[keep].abs().min()

        if (
            not torch.isfinite(min_sep)
            or float(min_sep.detach().item()) < ctx.min_gap
        ):
            # lambda_2 is not uniquely defined as a 1-D eigenspace here.
            return grad_matrix, None

        # Exact derivative for a simple eigenvalue:
        #
        # dv_k = sum_{j!=k} v_j (v_j^T dA v_k)/(lambda_k-lambda_j)
        #
        # Convert the vector-Jacobian product into a symmetric matrix gradient.
        coeff = eigvecs.transpose(0, 1) @ grad_v
        coeff = coeff.clone()
        coeff[k] = 0.0

        denom = gaps.clone()
        denom[k] = 1.0  # coefficient[k] is zero; avoid 0/0.

        # This clamp is inactive whenever the min-gap condition above passes,
        # but leaves a second numerical guard against subnormal denominators.
        signed = torch.where(
            denom >= 0,
            torch.ones_like(denom),
            -torch.ones_like(denom),
        )
        denom = signed * denom.abs().clamp_min(ctx.min_gap)
        denom[k] = 1.0

        z = eigvecs @ (coeff / denom)
        vk = eigvecs[:, k]

        grad_matrix = 0.5 * (
            torch.outer(z, vk) + torch.outer(vk, z)
        )

        grad_matrix = torch.nan_to_num(
            grad_matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return grad_matrix, None



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
    """
    Scores candidate edges as non-backbone/backbone.

    It keeps the original learned node-pair features and adds explicit geometric
    scalars so the 3.8 A inductive bias does not need to be rediscovered.
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        bond_length: float = 0.038,
        bond_sigma: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.node_dim = d_model // 2
        self.edge_dim = 4 * self.node_dim
        self.bond_length = float(bond_length)
        self.bond_sigma = float(bond_sigma)

        self.node_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, self.node_dim),
        )

        self.edge_node_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, self.node_dim),
        )

        # Original pair representation is 4*node_dim. Add four geometry scalars,
        # then project back to the same edge width expected by EdgeGraphLayer.
        self.edge_init = nn.Sequential(
            nn.Linear(4 * self.node_dim + 4, self.edge_dim),
            nn.LayerNorm(self.edge_dim),
            nn.GELU(),
            nn.Linear(self.edge_dim, self.edge_dim),
        )

        self.graph_layers = nn.ModuleList([
            EdgeGraphLayer(self.node_dim, self.edge_dim, d_model, dropout)
            for _ in range(num_layers)
        ])

        self.score_head = nn.Sequential(
            nn.Linear(self.edge_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        self.node_out_proj = nn.Sequential(
            nn.Linear(self.node_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        memory: Tensor,
        packed_xyz: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        mask: Tensor,
    ):
        B, L, D = memory.shape
        flat_mask = mask.reshape(-1)
        flat_mem = memory.reshape(B * L, D)[flat_mask]

        if edge_src.numel() != edge_dst.numel():
            raise ValueError("Incompatible edge lists")

        node_feat = self.node_proj(flat_mem)
        edge_node_feat = self.edge_node_proj(flat_mem)

        if edge_src.numel() == 0:
            logits = flat_mem.new_zeros((0, 2))
            return logits, self.node_out_proj(node_feat)

        h_i = edge_node_feat[edge_src]
        h_j = edge_node_feat[edge_dst]
        dh_ij = torch.abs(h_i - h_j)
        prod_h = h_i * h_j

        d_ij = (packed_xyz[edge_src] - packed_xyz[edge_dst]).norm(dim=-1)
        bond_dev = torch.abs(d_ij - self.bond_length) / max(self.bond_sigma, 1e-8)
        bond_gauss = torch.exp(-0.5 * bond_dev.square())
        d_over_bond = d_ij / max(self.bond_length, 1e-8)
        inv_d = self.bond_length / d_ij.clamp_min(1e-6)

        geom = torch.stack(
            [d_over_bond, bond_dev, bond_gauss, inv_d],
            dim=-1,
        ).to(h_i.dtype)

        raw_edge = torch.cat([h_i, h_j, dh_ij, prod_h, geom], dim=-1)
        edge_feat = self.edge_init(raw_edge)

        for layer in self.graph_layers:
            node_feat, edge_feat = layer(
                node_feat, edge_feat, edge_src, edge_dst
            )

        logits = self.score_head(edge_feat)
        node_out = self.node_out_proj(node_feat)
        return logits, node_out


def _log_sinkhorn(logits: Tensor, n_iter: int) -> Tensor:
    log_p = logits
    for _ in range(n_iter):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p.exp()


class Model(nn.Module):
    """
    Learned adjacency -> differentiable Fiedler ordering -> Sinkhorn permutation.

    No Transformer sequence decoder and no Hungarian/Hamiltonian solver are used.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.L_max = config.L_max
        self.D = config.d_model
        self.default_tau = float(config.tau)
        self.n_sink_iter = int(config.n_sink_iter)

        self.candidate_radius = (
            float(config.candidate_radius)
            if config.candidate_radius is not None
            else float(config.edge_radius)
        )
        self.candidate_knn = int(config.candidate_knn)

        if config.fiedler_eigh_dtype not in {"float32", "float64"}:
            raise ValueError(
                "fiedler_eigh_dtype must be 'float32' or 'float64'"
            )
        self.eigh_dtype = (
            torch.float64
            if config.fiedler_eigh_dtype == "float64"
            else torch.float32
        )
        self.spectral_background_eps = float(config.spectral_background_eps)
        self.spectral_rank_clamp_eps = float(config.spectral_rank_clamp_eps)
        self.spectral_backward_min_gap = float(
            config.spectral_backward_min_gap
        )

        D = config.d_model
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
        self.edge_scorer = EdgeScorer(
            D,
            num_layers=config.num_edgescorer_layers,
            dropout=config.dropout,
            bond_length=config.bond_length,
            bond_sigma=config.bond_sigma,
        )

    @staticmethod
    def _packed_xyz_and_batch(xyz: Tensor, mask: Tensor):
        B, L, _ = xyz.shape
        flat_mask = mask.reshape(-1)
        packed_xyz = xyz.reshape(B * L, 3)[flat_mask]
        packed_batch = (
            torch.arange(B, device=xyz.device)
            .repeat_interleave(L)[flat_mask]
        )
        return packed_xyz, packed_batch

    def _candidate_edges(
        self,
        packed_xyz: Tensor,
        packed_batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Undirected union of radius candidates and kNN fallback, returned as both
        directed orientations.  Selection depends only on the fixed input points.
        """
        device = packed_xyz.device
        src_chunks = []
        dst_chunks = []

        if packed_xyz.shape[0] <= 1:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        batch_ids = torch.unique(packed_batch)

        # Candidate selection is discrete, but the input coordinates are not model
        # parameters. The learned edge scores and everything downstream remain
        # differentiable w.r.t. network parameters.
        with torch.no_grad():
            for b in batch_ids.tolist():
                packed_idx = (packed_batch == b).nonzero(as_tuple=True)[0]
                n = int(packed_idx.numel())
                if n <= 1:
                    continue

                pos = packed_xyz.index_select(0, packed_idx)
                dist = torch.cdist(pos.float(), pos.float())

                tri = torch.triu_indices(n, n, offset=1, device=device)
                radius_keep = dist[tri[0], tri[1]] <= self.candidate_radius
                ru = tri[0][radius_keep]
                rv = tri[1][radius_keep]

                k = min(self.candidate_knn, n - 1)
                if k > 0:
                    d_knn = dist.clone()
                    d_knn.fill_diagonal_(float("inf"))
                    nbr = torch.topk(
                        d_knn, k=k, dim=-1, largest=False
                    ).indices
                    ku = (
                        torch.arange(n, device=device)
                        .unsqueeze(1)
                        .expand(n, k)
                        .reshape(-1)
                    )
                    kv = nbr.reshape(-1)
                    kmin = torch.minimum(ku, kv)
                    kmax = torch.maximum(ku, kv)
                else:
                    kmin = torch.empty(0, dtype=torch.long, device=device)
                    kmax = torch.empty(0, dtype=torch.long, device=device)

                u = torch.cat([ru, kmin], dim=0)
                v = torch.cat([rv, kmax], dim=0)

                if u.numel() == 0:
                    continue

                keys = torch.unique(u * n + v)
                u = keys // n
                v = keys % n

                gu = packed_idx[u]
                gv = packed_idx[v]

                # Both orientations so EdgeScorer message passing is symmetric in
                # graph support; logits themselves may still differ and are later
                # averaged into an undirected adjacency.
                src_chunks.extend([gu, gv])
                dst_chunks.extend([gv, gu])

        if not src_chunks:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        return torch.cat(src_chunks), torch.cat(dst_chunks)

    @staticmethod
    def _dense_soft_adjacency(
        edge_logits: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Scatter directed backbone probabilities into padded [B,L,L], average
        duplicate/directional predictions, then symmetrize.
        """
        B, L = mask.shape
        device = edge_logits.device

        # Compute probabilities in fp32 even under bf16/fp16 Fabric precision.
        edge_prob = F.softmax(edge_logits.float(), dim=-1)[:, 1]

        A_flat = torch.zeros(
            B * L * L, device=device, dtype=edge_prob.dtype
        )
        C_flat = torch.zeros_like(A_flat)

        if edge_prob.numel() > 0:
            flat_valid = mask.reshape(-1).nonzero(as_tuple=True)[0]
            src_flat = flat_valid[edge_src]
            dst_flat = flat_valid[edge_dst]

            src_b = torch.div(src_flat, L, rounding_mode="floor")
            dst_b = torch.div(dst_flat, L, rounding_mode="floor")
            if not torch.equal(src_b, dst_b):
                raise RuntimeError("Candidate edge crosses batch examples")

            src_i = src_flat % L
            dst_i = dst_flat % L
            lin = src_b * (L * L) + src_i * L + dst_i

            A_flat.index_add_(0, lin, edge_prob)
            C_flat.index_add_(0, lin, torch.ones_like(edge_prob))

        A = A_flat.view(B, L, L) / C_flat.view(B, L, L).clamp_min(1.0)
        A = 0.5 * (A + A.transpose(-1, -2))
        A = A - torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
        return A

    def _path_fiedler_to_rank(self, v: Tensor) -> Tensor:
        n = int(v.numel())
        if n <= 1:
            return torch.zeros_like(v)

        t = torch.arange(n, device=v.device, dtype=v.dtype)
        raw_canonical = torch.cos(
            torch.pi * (t + 0.5) / float(n)
        )
        v_cos = v * raw_canonical.norm().clamp_min(1e-12)

        eps = self.spectral_rank_clamp_eps
        if v.dtype == torch.float32:
            eps = max(eps, 1e-6)
        v_cos = v_cos.clamp(-1.0 + eps, 1.0 - eps)

        return float(n) * torch.acos(v_cos) / torch.pi - 0.5

    def _spectral_assignment(
        self,
        adjacency: Tensor,
        mask: Tensor,
        tau: float,
        output_dtype: torch.dtype,
    ):
        B, L, _ = adjacency.shape
        spectral_dtype = self.eigh_dtype
        device = adjacency.device

        P_full = torch.zeros(
            B, L, L, device=device, dtype=output_dtype
        )
        logits_full = torch.zeros(
            B, L, L, device=device, dtype=spectral_dtype
        )
        fiedler_full = torch.zeros(
            B, L, device=device, dtype=spectral_dtype
        )
        lambda2 = torch.full(
            (B,), float("nan"), device=device, dtype=spectral_dtype
        )
        lambda3 = torch.full_like(lambda2, float("nan"))

        for b in range(B):
            valid_idx = mask[b].nonzero(as_tuple=True)[0]
            n = int(valid_idx.numel())

            if n == 0:
                continue
            if n == 1:
                i = valid_idx[0]
                P_full[b, i, i] = 1.0
                logits_full[b, i, i] = 0.0
                fiedler_full[b, i] = 1.0
                lambda2[b] = 0.0
                continue

            A = adjacency[b].index_select(0, valid_idx).index_select(
                1, valid_idx
            ).to(spectral_dtype)
            A = 0.5 * (A + A.T)
            A = A - torch.diag_embed(torch.diagonal(A))

            # Adding a uniform complete-graph component makes the soft graph
            # connected without changing the non-constant eigenvectors.
            if self.spectral_background_eps > 0:
                eye = torch.eye(n, device=device, dtype=spectral_dtype)
                A = A + self.spectral_background_eps * (1.0 - eye)

            degree = A.sum(dim=-1)
            laplacian = torch.diag(degree) - A

            # IMPORTANT: do not backprop through torch.linalg.eigh directly.
            # Its eigenvector derivative is singular when the learned graph has
            # a repeated / nearly repeated Fiedler eigenvalue.  The custom op
            # below gives the exact derivative for a healthy eigengap and zero
            # spectral gradient for ill-defined degenerate examples.
            v, eigvals = _StableFiedlerVector.apply(
                laplacian,
                self.spectral_backward_min_gap,
            )
            v = v / v.norm().clamp_min(1e-12)

            spectral_rank = self._path_fiedler_to_rank(v)
            canonical_rank = torch.arange(
                n, device=device, dtype=spectral_dtype
            )

            logits = -(
                canonical_rank[:, None] - spectral_rank[None, :]
            ).square() / float(tau)

            P_local = _log_sinkhorn(logits, self.n_sink_iter)

            rr, cc = torch.meshgrid(valid_idx, valid_idx, indexing="ij")
            P_full[b, rr, cc] = P_local.to(output_dtype)
            logits_full[b, rr, cc] = logits
            fiedler_full[b, valid_idx] = v
            lambda2[b] = eigvals[1]
            if n > 2:
                lambda3[b] = eigvals[2]

        return P_full, logits_full, fiedler_full, lambda2, lambda3

    def forward(
        self,
        xyz_cloud: Tensor,
        point_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        spectral_tau: Optional[float] = None,
        compute_spectral: bool = True,
    ):
        B, L, _ = xyz_cloud.shape
        if L > self.L_max:
            raise ValueError(
                f"Input length {L} exceeds L_max={self.L_max}"
            )

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
        xyz_safe = torch.where(
            mask.unsqueeze(-1), xyz_cloud, torch.zeros_like(xyz_cloud)
        )

        memory, _, _, _ = self.graph_encoder(xyz_safe, mask)
        memory = self.dropout(memory)

        packed_xyz, packed_batch = self._packed_xyz_and_batch(
            xyz_safe, mask
        )
        edge_src, edge_dst = self._candidate_edges(
            packed_xyz, packed_batch
        )

        edge_logits, _ = self.edge_scorer(
            memory,
            packed_xyz,
            edge_src,
            edge_dst,
            mask,
        )

        adjacency = self._dense_soft_adjacency(
            edge_logits, edge_src, edge_dst, mask
        )

        result = {
            "edge_logits": edge_logits,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "adjacency": adjacency,
            "mask": mask,
            "ordered_xyz": None,
            "permutation_matrices": None,
            "assignment_logits": None,
            "fiedler": None,
            "lambda2": None,
            "lambda3": None,
        }

        if not compute_spectral:
            return result

        tau = self.default_tau if spectral_tau is None else float(spectral_tau)
        P, assignment_logits, fiedler, lambda2, lambda3 = (
            self._spectral_assignment(
                adjacency=adjacency,
                mask=mask,
                tau=tau,
                output_dtype=xyz_safe.dtype,
            )
        )

        ordered_xyz = torch.einsum(
            "bti,bid->btd", P, xyz_safe
        )
        ordered_xyz = ordered_xyz.masked_fill(
            ~mask.unsqueeze(-1), 0.0
        )

        result.update(
            {
                "ordered_xyz": ordered_xyz,
                "permutation_matrices": P,
                "assignment_logits": assignment_logits,
                "fiedler": fiedler,
                "lambda2": lambda2,
                "lambda3": lambda3,
            }
        )
        return result