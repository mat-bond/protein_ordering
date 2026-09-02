
import torch
from e3nn import o3
from torch import nn
from torch_cluster import radius_graph

from ..Equiformer.equiformer.nets.drop import (
    EquivariantDropout,
)
from ..Equiformer.equiformer.nets.gaussian_rbf import GaussianRadialBasisLayer
from ..Equiformer.equiformer.nets.graph_attention_transformer import (
    EdgeDegreeEmbeddingNetwork,
    TransBlock,
    get_norm_layer,
)
from ..Equiformer.equiformer.nets.tensor_product_rescale import (
    LinearRS,
)

_RESCALE = True
_USE_BIAS = True
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
            irreps_node_embedding: str | None = None,
            irreps_head: str | None = None,
            irreps_mlp_mid: str | None = None,
            irreps_pre_attn: str | None = None,
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