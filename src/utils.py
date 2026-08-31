

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