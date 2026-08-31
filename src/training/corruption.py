
import torch

def random_rotation_matrix(device=None, dtype=torch.float32):
    u1 = torch.rand((), device=device, dtype=dtype)
    u2 = torch.rand((), device=device, dtype=dtype)
    u3 = torch.rand((), device=device, dtype=dtype)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    x, y, z, w = q1, q2, q3, q4
    return torch.stack([
        torch.stack([
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ]),
        torch.stack([
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ]),
        torch.stack([
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ]),
    ])


def apply_rotation_masked(
    ca: torch.Tensor, valid: torch.Tensor, R: torch.Tensor
):
    out = ca.clone()
    out[valid] = out[valid] @ R.T
    return out


def add_masked_coord_jitter(
    x: torch.Tensor,
    mask: torch.Tensor,
    sigma_frac: float = 0.0,
):
    out = x.clone()
    B = x.shape[0]
    for b in range(B):
        valid = mask[b]
        pts = out[b, valid]
        if pts.shape[0] == 0:
            continue
        centered = pts - pts.mean(dim=0, keepdim=True)
        scale = (
            centered.pow(2).sum(dim=-1).mean().sqrt().clamp_min(1e-6)
        )
        out[b, valid] = pts + torch.randn_like(pts) * (
            sigma_frac * scale
        )
    return out * mask.unsqueeze(-1)


@torch.no_grad()
def make_uniformly_permuted_cloud(
    pcs_gt: torch.Tensor,
    mask: torch.Tensor,
):
    out = torch.zeros_like(pcs_gt)
    B, L, _ = pcs_gt.shape
    target_col = torch.full(
        (B, L), -100, dtype=torch.long, device=pcs_gt.device
    )

    for b in range(B):
        valid = mask[b]
        valid_idx = valid.nonzero(as_tuple=True)[0]
        pts = pcs_gt[b, valid]
        n = pts.shape[0]

        if n == 0:
            continue
        if n == 1:
            out[b, valid] = pts
            target_col[b, valid_idx] = valid_idx
            continue

        perm = torch.randperm(n, device=pcs_gt.device)
        pts_perm = pts[perm]

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(n, device=perm.device)

        out[b, valid] = pts_perm
        target_col[b, valid_idx] = valid_idx[inv_perm].long()

    return out * mask.unsqueeze(-1), target_col


@torch.no_grad()
def make_uniformly_permuted_cloud_deterministic(
    pcs_gt: torch.Tensor,
    mask: torch.Tensor,
    idx: torch.Tensor,
    step: int = 0,
):
    out = torch.zeros_like(pcs_gt)
    B, L, _ = pcs_gt.shape
    target_col = torch.full(
        (B, L), -100, dtype=torch.long, device=pcs_gt.device
    )

    for b in range(B):
        valid = mask[b]
        valid_idx = valid.nonzero(as_tuple=True)[0]
        pts = pcs_gt[b, valid]
        n = pts.shape[0]

        if n == 0:
            continue
        if n == 1:
            out[b, valid] = pts
            target_col[b, valid_idx] = valid_idx
            continue

        g = torch.Generator(device=pcs_gt.device)
        g.manual_seed(
            int(step * 100003 + int(idx[b].item()) * 9176 + 12345)
        )

        perm = torch.randperm(n, generator=g, device=pcs_gt.device)
        pts_perm = pts[perm]

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(n, device=perm.device)

        out[b, valid] = pts_perm
        target_col[b, valid_idx] = valid_idx[inv_perm].long()

    return out * mask.unsqueeze(-1), target_col


def reverse_target_col(
    target_col: torch.Tensor,
    mask: torch.Tensor,
):
    out = torch.full_like(target_col, -100)
    B, _ = target_col.shape
    for b in range(B):
        valid_idx = mask[b].nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            continue
        out[b, valid_idx] = target_col[b, valid_idx.flip(0)]
    return out