
import torch

from metrics.loss import chamfer

def masked_kabsch_mse(pred, target, mask, min_n: int = 4, eps: float = 1e-8):
    pred   = torch.nan_to_num(pred,   nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    B, L, _ = pred.shape
    mask_f = mask.to(dtype=pred.dtype)
    n = mask_f.sum(dim=1)  # [B]

    good = n >= min_n
    if not good.any():
        return pred.new_tensor(0.0)

    pred   = pred[good]
    target = target[good]
    mask_f = mask_f[good]
    n = n[good].clamp_min(1.0)

    pred_centroid = (pred * mask_f[..., None]).sum(dim=1) / n[:, None]
    tgt_centroid  = (target * mask_f[..., None]).sum(dim=1) / n[:, None]

    X = (pred - pred_centroid[:, None, :]) * mask_f[..., None]
    Y = (target - tgt_centroid[:, None, :]) * mask_f[..., None]

    H = X.transpose(1, 2) @ Y
    # optional stabilizer:
    # H = H + 1e-4 * torch.eye(3, device=H.device, dtype=H.dtype).unsqueeze(0)

    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)

    d = torch.det(V @ U.transpose(-2, -1))
    D = torch.eye(3, device=pred.device, dtype=pred.dtype).unsqueeze(0).repeat(V.shape[0], 1, 1)
    D[:, 2, 2] = torch.where(d < 0, -1.0, 1.0)

    R = V @ D @ U.transpose(-2, -1)

    X_rot = X @ R

    diff2 = ((X_rot - Y) ** 2).sum(dim=-1) * mask_f  # [Bg, L]
    return diff2.sum() / n.sum().clamp_min(1.0)

def masked_kabsch_mse_per_example(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_n: int = 4,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      mse_sum_per_ex: [B]
      count_per_ex:   [B]
    so global mse = mse_sum_per_ex.sum() / count_per_ex.sum()
    """
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    B, L, _ = pred.shape
    mask_f = mask.to(dtype=pred.dtype)
    n = mask_f.sum(dim=1)  # [B]

    mse_sum = pred.new_zeros(B)
    count = pred.new_zeros(B)

    good = n >= min_n
    if not good.any():
        return mse_sum, count

    pred_g = pred[good]
    target_g = target[good]
    mask_f_g = mask_f[good]
    n_g = n[good].clamp_min(1.0)

    pred_centroid = (pred_g * mask_f_g[..., None]).sum(dim=1) / n_g[:, None]
    tgt_centroid = (target_g * mask_f_g[..., None]).sum(dim=1) / n_g[:, None]

    X = (pred_g - pred_centroid[:, None, :]) * mask_f_g[..., None]
    Y = (target_g - tgt_centroid[:, None, :]) * mask_f_g[..., None]

    H = X.transpose(1, 2) @ Y

    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)

    d = torch.det(V @ U.transpose(-2, -1))
    D = torch.eye(3, device=pred.device, dtype=pred.dtype).unsqueeze(0).repeat(V.shape[0], 1, 1)
    D[:, 2, 2] = torch.where(d < 0, -1.0, 1.0)

    R = V @ D @ U.transpose(-2, -1)
    X_rot = X @ R

    diff2 = ((X_rot - Y) ** 2).sum(dim=-1) * mask_f_g   # [Bg, L]
    mse_sum_g = diff2.sum(dim=1)                        # [Bg]

    mse_sum[good] = mse_sum_g
    count[good] = n_g

    return mse_sum, count

def reverse_valid_order(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    B, L = mask.shape
    for b in range(B):
        valid_idx = mask[b].nonzero(as_tuple=True)[0]
        if valid_idx.numel() <= 1:
            continue
        out[b, valid_idx] = x[b, valid_idx.flip(0)]
    return out

def masked_kabsch_mse_bidirectional(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_n: int = 4,
    eps: float = 1e-8,
):
    target_rev = reverse_valid_order(target, mask)

    mse_sum_fwd, count_fwd = masked_kabsch_mse_per_example(
        pred, target, mask, min_n=min_n, eps=eps
    )
    mse_sum_rev, count_rev = masked_kabsch_mse_per_example(
        pred, target_rev, mask, min_n=min_n, eps=eps
    )

    mean_fwd = mse_sum_fwd / count_fwd.clamp_min(1.0)
    mean_rev = mse_sum_rev / count_rev.clamp_min(1.0)

    choose_rev = mean_rev < mean_fwd

    chosen_sum = torch.where(choose_rev, mse_sum_rev, mse_sum_fwd)
    chosen_count = torch.where(choose_rev, count_rev, count_fwd)

    per_example = chosen_sum / chosen_count.clamp_min(1.0)

    good = chosen_count > 0
    batch_mean = per_example[good].mean() if good.any() else pred.new_tensor(0.0)

    return batch_mean, per_example, chosen_count

def drmsd_loss_per_example(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    min_n: int = 2,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      drmsd_sum_per_ex: [B]   numerator per example
      pair_count_per_ex: [B]  denominator per example

    Global loss:
      drmsd = (drmsd_sum_per_ex / pair_count_per_ex.clamp_min(1)).sqrt()
      but for bidirectional selection we compare per-example drmsd values first.
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    B, L, _ = x.shape
    device = x.device
    dtype = x.dtype

    mask_f = mask.to(dtype=dtype)
    n = mask_f.sum(dim=1)  # [B]

    drmsd_sum = x.new_zeros(B)
    pair_count = x.new_zeros(B)

    good = n >= min_n
    if not good.any():
        return drmsd_sum, pair_count

    xg = x[good]          # [Bg, L, 3]
    yg = y[good]
    mg = mask_f[good]     # [Bg, L]

    dx = torch.cdist(xg, xg)   # [Bg, L, L]
    dy = torch.cdist(yg, yg)   # [Bg, L, L]

    pair = mg[:, :, None] * mg[:, None, :]  # [Bg, L, L]
    eye = torch.eye(L, device=device, dtype=dtype).unsqueeze(0)
    pair = pair * (1.0 - eye)  # exclude diagonal

    sq = ((dx - dy) ** 2) * pair                 # [Bg, L, L]
    drmsd_sum_g = sq.sum(dim=(1, 2))             # [Bg]
    pair_count_g = pair.sum(dim=(1, 2)).clamp_min(1.0)  # [Bg]

    drmsd_sum[good] = drmsd_sum_g
    pair_count[good] = pair_count_g

    return drmsd_sum, pair_count


def drmsd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    min_n: int = 2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Standard masked dRMSD averaged over examples.
    """
    drmsd_sum, pair_count = drmsd_loss_per_example(
        x, y, mask, min_n=min_n, eps=eps
    )
    good = pair_count > 0
    if not good.any():
        return x.new_tensor(0.0)

    drmsd_per_ex = torch.sqrt(drmsd_sum[good] / pair_count[good].clamp_min(eps))
    return drmsd_per_ex.mean()


def drmsd_loss_bidirectional(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    min_n: int = 2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Bidirectional masked dRMSD.
    For each example independently, choose the smaller of:
      - dRMSD(x, y)
      - dRMSD(x, reverse_valid_order(y))
    Then average over examples.
    """
    y_rev = reverse_valid_order(y, mask)

    sum_fwd, count_fwd = drmsd_loss_per_example(
        x, y, mask, min_n=min_n, eps=eps
    )
    sum_rev, count_rev = drmsd_loss_per_example(
        x, y_rev, mask, min_n=min_n, eps=eps
    )

    good_fwd = count_fwd > 0
    good_rev = count_rev > 0
    good = good_fwd | good_rev
    if not good.any():
        return x.new_tensor(0.0)

    loss_fwd = torch.full_like(sum_fwd, float("inf"))
    loss_rev = torch.full_like(sum_rev, float("inf"))

    loss_fwd[good_fwd] = torch.sqrt(sum_fwd[good_fwd] / count_fwd[good_fwd].clamp_min(eps))
    loss_rev[good_rev] = torch.sqrt(sum_rev[good_rev] / count_rev[good_rev].clamp_min(eps))

    chosen = torch.minimum(loss_fwd, loss_rev)
    return chosen[good].mean()

def masked_mse_no_align(pred, target, mask, pad_value=0.0):
    assert pred.shape == target.shape and pred.ndim == 3 and pred.shape[-1] == 3
    assert mask.shape == pred.shape[:2]

    pred   = torch.nan_to_num(pred,   nan=pad_value, posinf=pad_value, neginf=pad_value)
    target = torch.nan_to_num(target, nan=pad_value, posinf=pad_value, neginf=pad_value)

    mask_f = mask.to(dtype=pred.dtype)
    n = mask_f.sum(dim=1).clamp_min(1.0)

    diff2 = ((pred - target) ** 2).sum(dim=-1) * mask_f
    return (diff2.sum(dim=1) / n).mean()


def chamfer_masked(pred_pc: torch.Tensor,
                   ref_pc: torch.Tensor,
                   mask: torch.Tensor,
                   pad_value: float = 0.0) -> torch.Tensor:
    assert pred_pc.shape == ref_pc.shape and pred_pc.ndim == 3
    assert mask.shape == pred_pc.shape[:2]

    
    pred_pc = torch.nan_to_num(pred_pc, nan=pad_value, posinf=pad_value, neginf=pad_value)
    ref_pc  = torch.nan_to_num(ref_pc,  nan=pad_value, posinf=pad_value, neginf=pad_value)

    B, Lmax, D = pred_pc.shape
    pred_list = [pred_pc[b, mask[b]] for b in range(B)]
    ref_list  = [ref_pc[b,  mask[b]] for b in range(B)]

    Lb = max(int(x.shape[0]) for x in pred_list + ref_list)
    Lb = max(Lb, 1)

    pred_pad = pred_pc.new_full((B, Lb, D), pad_value)
    ref_pad  = ref_pc.new_full((B, Lb, D), pad_value)

    for b in range(B):
        lp = min(pred_list[b].shape[0], Lb)
        lr = min(ref_list[b].shape[0],  Lb)
        if lp > 0:
            pred_pad[b, :lp] = pred_list[b][:lp]
        if lr > 0:
            ref_pad[b, :lr]  = ref_list[b][:lr]

    return chamfer(pred_pad, ref_pad)

def compute_confidence_scores(
    assignment_logits: torch.Tensor,
    mask: torch.Tensor,
    target_mask: torch.Tensor | None = None,
):
    """
    Normalized-entropy confidence per ordered slot.

    Returns [B, T] in [0, 1], where 1 is most confident.
    Invalid target slots are set to 0.
    """
    if target_mask is None:
        target_mask = mask

    very_neg = torch.finfo(assignment_logits.dtype).min
    masked_logits = assignment_logits.masked_fill(
        ~mask[:, None, :], very_neg
    )
    p = torch.softmax(masked_logits, dim=-1)
    entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1)

    n_valid = mask.sum(dim=-1).clamp_min(1).to(assignment_logits.dtype)
    max_entropy = torch.log(n_valid)
    conf = torch.ones_like(entropy)

    multi = n_valid > 1
    if multi.any():
        conf[multi] = 1.0 - (
            entropy[multi] / max_entropy[multi].unsqueeze(-1)
        )

    conf = conf.masked_fill(~target_mask, 0.0)
    return conf.clamp(0.0, 1.0)