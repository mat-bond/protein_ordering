
from torch import nn
import torch

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