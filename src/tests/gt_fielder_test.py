"""
Validation-only ablation: ground-truth path adjacency -> Fiedler vector -> ordering.

This script intentionally does NOT load or run the neural network.  It keeps the
same validation dataloading / masking / deterministic uniform permutation used by
the training script, then asks a narrower question:

    If the neighbor graph were exactly correct, can a differentiable spectral
    layer convert that graph into the correct global residue ordering?

It reports:
  1) direct GT-permutation RMSD (sanity check; should be ~0),
  2) hard Fiedler-sort RMSD / permutation accuracy,
  3) soft Fiedler -> Sinkhorn RMSD / permutation accuracy for a tau sweep.

Run from the repository root, e.g.

    python test_gt_fiedler_validation.py \
        --config configs/encoder_airplane.yaml

For a quick smoke test:

    python test_gt_fiedler_validation.py \
        --config configs/encoder_airplane.yaml \
        --overfit128 --max-batches 2
"""

import argparse
import math
import os
import random
from typing import Any

# Keep the same deterministic/runtime environment choices as the source script.
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_USE_OFFSCREEN"] = "true"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import yaml
from lightning import seed_everything
from lightning.fabric import Fabric
from tqdm import tqdm

from loaders import load_datamodule, load_module
from metrics.protein_loss import masked_kabsch_mse_bidirectional


torch.set_float32_matmul_precision("medium")


def load_config(path: str):
    """Same config loading as the source training script."""
    with open(path, encoding="utf-8") as stream:
        config_dict: dict[str, Any] = yaml.safe_load(stream)

    dataconfig = load_module(config_dict["data"], classname="DataConfig")
    trainerconfig = load_module(config_dict["trainer"], classname="TrainerConfig")
    loggerconfig = load_module(config_dict["logger"], classname="LogConfig")
    rmsd_modelconfig = load_module(
        config_dict["rmsd_modelconfig"], classname="ModelConfig"
    )
    cloud_modelconfig = load_module(
        config_dict["cloud_modelconfig"], classname="ModelConfig"
    )

    return (
        dataconfig,
        rmsd_modelconfig,
        cloud_modelconfig,
        trainerconfig,
        loggerconfig,
    )


@torch.no_grad()
def make_uniformly_permuted_cloud_deterministic(
    pcs_gt: torch.Tensor,   # [B, L, 3]
    mask: torch.Tensor,     # [B, L] bool
    idx: torch.Tensor,      # [B]
    step: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Copied from the source validation code.

    Returns:
      cloud_in    [B, L, 3]
      target_col  [B, L], where target_col[b,t] = input column containing
                  target residue t; -100 means ignored/padded.
    """
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
        g.manual_seed(int(step * 100003 + int(idx[b].item()) * 9176 + 12345))

        perm = torch.randperm(n, generator=g, device=pcs_gt.device)
        pts_perm = pts[perm]

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(n, device=perm.device)

        out[b, valid] = pts_perm
        target_col[b, valid_idx] = valid_idx[inv_perm].long()

    out = out * mask.unsqueeze(-1)
    return out, target_col


def reverse_target_col(target_col: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Same reversal convention as the source bidirectional permutation metric."""
    out = torch.full_like(target_col, -100)
    B, _ = target_col.shape
    for b in range(B):
        valid_idx = mask[b].nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            continue
        out[b, valid_idx] = target_col[b, valid_idx.flip(0)]
    return out


def build_gt_path_adjacency(
    target_col: torch.Tensor,
    mask: torch.Tensor,
    dtype: torch.dtype,
    connect_across_gaps: bool = True,
) -> torch.Tensor:
    """
    Build an undirected GT path adjacency in scrambled-input index space.

    target_col[b,t] is the scrambled input column holding residue t.

    If connect_across_gaps=True (default), the valid observed residues are
    compressed into one path in sequence order.  This is the cleanest ablation
    of "correct graph -> global ordering" and guarantees one path over all
    observed points.

    If False, an edge is added only when the two valid target indices differ by
    exactly 1, matching the semantics of edge_ce_loss in the source script.
    Proteins with missing residues can then produce disconnected graphs.
    """
    B, L = target_col.shape
    A = torch.zeros(B, L, L, device=target_col.device, dtype=dtype)

    for b in range(B):
        valid_t = mask[b].nonzero(as_tuple=True)[0]
        n = valid_t.numel()
        if n <= 1:
            continue

        cols = target_col[b, valid_t]

        if connect_across_gaps:
            keep = torch.ones(n - 1, dtype=torch.bool, device=target_col.device)
        else:
            keep = (valid_t[1:] - valid_t[:-1]) == 1

        src = cols[:-1][keep]
        dst = cols[1:][keep]

        A[b, src, dst] = 1.0
        A[b, dst, src] = 1.0

    return A


def log_sinkhorn(logits: torch.Tensor, n_iter: int = 50) -> torch.Tensor:
    """Differentiable log-domain Sinkhorn for one square matrix [N,N]."""
    log_p = logits
    for _ in range(n_iter):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p.exp()


def path_fiedler_to_rank(v: torch.Tensor) -> torch.Tensor:
    """
    Convert the unit-norm Fiedler vector of an ideal N-node path into an
    approximately uniform rank coordinate 0..N-1.

    For a path graph, v_k is proportional to
        cos(pi * (k + 1/2) / N).
    Undoing the cosine with arccos avoids the severe endpoint compression of
    raw Fiedler coordinates, so a single Sinkhorn temperature behaves much more
    consistently across sequence lengths.

    The result may run N-1..0 instead of 0..N-1 because eigenvector sign is
    arbitrary; all reported ordering metrics are bidirectional.
    """
    n = v.numel()
    if n <= 1:
        return torch.zeros_like(v)

    t = torch.arange(n, device=v.device, dtype=v.dtype)
    raw_canonical = torch.cos(torch.pi * (t + 0.5) / float(n))

    # eigh returns a unit-norm eigenvector.  Scale it to the raw cosine's norm.
    v_cos = v * raw_canonical.norm().clamp_min(1e-12)

    # Keep arccos finite.  This script is an ablation; in a learned end-to-end
    # version you may prefer a smoother bounded transform near +/-1.
    eps = 1e-12 if v.dtype == torch.float64 else 1e-6
    v_cos = v_cos.clamp(-1.0 + eps, 1.0 - eps)

    rank = float(n) * torch.acos(v_cos) / torch.pi - 0.5
    return rank


def fiedler_vector_from_adjacency(
    A: torch.Tensor,
    eigh_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A: [N,N] symmetric adjacency.

    Returns:
      v2:      [N] second Laplacian eigenvector
      eigvals: [N]

    Eigensolve is done in the requested dtype.  float64 is the default for this
    diagnostic because long ideal paths have a small low-frequency spectral gap,
    and we do not want float32 eigensolver noise to masquerade as an architectural
    failure.  The returned eigenvector stays in the eigensolver dtype so hard
    sorting is not degraded by an unnecessary cast.
    """
    Aw = (0.5 * (A + A.T)).to(eigh_dtype)
    Aw = Aw - torch.diag_embed(torch.diagonal(Aw))

    degree = Aw.sum(dim=-1)
    laplacian = torch.diag(degree) - Aw

    eigvals, eigvecs = torch.linalg.eigh(laplacian)

    if A.shape[0] == 1:
        v2 = torch.ones(1, device=A.device, dtype=eigh_dtype)
    else:
        v2 = eigvecs[:, 1]

    v2 = v2 / v2.norm().clamp_min(1e-12)
    return v2, eigvals


def fiedler_assignments(
    adjacency: torch.Tensor,
    mask: torch.Tensor,
    taus: list[float],
    sinkhorn_iters: int,
    eigh_dtype: torch.dtype,
):
    """
    Produce both hard spectral ordering and soft Sinkhorn assignments.

    Returns:
      hard_P:   [B,L,L]
      soft_Ps:  dict[tau] -> [B,L,L]
      fiedler:  [B,L]
      lambda2:  [B]
      lambda3:  [B]
    """
    B, L, _ = adjacency.shape
    hard_P = adjacency.new_zeros(B, L, L)
    soft_Ps = {tau: adjacency.new_zeros(B, L, L) for tau in taus}

    # Spectral quantities must use the eigensolver dtype, not adjacency dtype.
    # adjacency/cloud data may be float32 or float64 independently of --eigh-dtype.
    fiedler_full = torch.zeros(
        B, L, device=adjacency.device, dtype=eigh_dtype
    )
    lambda2 = torch.full(
        (B,), float("nan"), device=adjacency.device, dtype=eigh_dtype
    )
    lambda3 = torch.full(
        (B,), float("nan"), device=adjacency.device, dtype=eigh_dtype
    )

    for b in range(B):
        valid_idx = mask[b].nonzero(as_tuple=True)[0]
        n = valid_idx.numel()

        if n == 0:
            continue

        if n == 1:
            i = valid_idx[0]
            hard_P[b, i, i] = 1.0
            for tau in taus:
                soft_Ps[tau][b, i, i] = 1.0
            fiedler_full[b, i] = 1.0
            lambda2[b] = 0.0
            continue

        A = adjacency[b].index_select(0, valid_idx).index_select(1, valid_idx)
        v, eigvals = fiedler_vector_from_adjacency(A, eigh_dtype=eigh_dtype)

        # Be explicit at the boundary so dtype changes elsewhere cannot break this.
        v = v.to(device=fiedler_full.device, dtype=fiedler_full.dtype)
        eigvals = eigvals.to(device=lambda2.device, dtype=lambda2.dtype)

        fiedler_full[b, valid_idx] = v

        lambda2[b] = eigvals[1]
        if n > 2:
            lambda3[b] = eigvals[2]

        # Hard spectral ordering.  Direction is arbitrary; bidirectional metrics
        # below explicitly accept either orientation.
        local_cols_asc = torch.argsort(v, dim=0)
        target_rows = valid_idx
        input_cols = valid_idx[local_cols_asc]
        hard_P[b, target_rows, input_cols] = 1.0

        spectral_rank = path_fiedler_to_rank(v)
        canonical_rank = torch.arange(
            n, device=v.device, dtype=eigh_dtype
        )

        for tau in taus:
            logits = -(
                canonical_rank[:, None] - spectral_rank[None, :]
            ).square() / float(tau)

            P_local = log_sinkhorn(logits, n_iter=sinkhorn_iters)
            rr, cc = torch.meshgrid(valid_idx, valid_idx, indexing="ij")
            soft_Ps[tau][b, rr, cc] = P_local.to(adjacency.dtype)

    return hard_P, soft_Ps, fiedler_full, lambda2, lambda3


def direct_gt_assignment(
    target_col: torch.Tensor,
    mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Exact permutation matrix implied by target_col; sanity-check baseline."""
    B, L = target_col.shape
    P = torch.zeros(B, L, L, device=target_col.device, dtype=dtype)
    for b in range(B):
        rows = mask[b].nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        cols = target_col[b, rows]
        P[b, rows, cols] = 1.0
    return P


def bidirectional_perm_accuracy(
    P: torch.Tensor,
    target_col: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-example bidirectional row-argmax permutation accuracy.

    Returns:
      acc:   [B] max(forward accuracy, reverse accuracy)
      exact: [B] 1 if every valid row is correct in one orientation
    """
    B, _, _ = P.shape
    target_rev = reverse_target_col(target_col, mask)
    pred_col = P.argmax(dim=-1)

    acc = P.new_zeros(B)
    exact = P.new_zeros(B)

    for b in range(B):
        rows = mask[b].nonzero(as_tuple=True)[0]
        n = rows.numel()
        if n == 0:
            continue

        pred = pred_col[b, rows]
        fwd = (pred == target_col[b, rows]).float().mean()
        rev = (pred == target_rev[b, rows]).float().mean()
        best = torch.maximum(fwd, rev)
        acc[b] = best
        exact[b] = (best == 1.0).to(P.dtype)

    return acc, exact


def valid_mask_is_contiguous(mask: torch.Tensor) -> torch.Tensor:
    """Per-example indicator that valid target indices form one contiguous span."""
    B, _ = mask.shape
    out = torch.ones(B, dtype=torch.bool, device=mask.device)
    for b in range(B):
        idx = mask[b].nonzero(as_tuple=True)[0]
        if idx.numel() <= 1:
            continue
        out[b] = bool(torch.all((idx[1:] - idx[:-1]) == 1))
    return out


def rmsd_angstrom_per_example(
    xyz_pred: torch.Tensor,
    pcs_gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Uses the exact same bidirectional Kabsch metric as the source script."""
    _, per_example_mse, count = masked_kabsch_mse_bidirectional(
        xyz_pred, pcs_gt, mask
    )
    rmsd = torch.full_like(per_example_mse, float("nan"))
    good = count > 0
    rmsd[good] = 100.0 * torch.sqrt(per_example_mse[good].clamp_min(1e-12))
    return rmsd


def summarize(values: list[float]) -> tuple[float, float, float, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(a)),
        float(np.median(a)),
        float(np.quantile(a, 0.90)),
        float(np.max(a)),
    )


@torch.no_grad()
def run_validation_test(
    fabric: Fabric,
    valdataloader,
    taus: list[float],
    sinkhorn_iters: int,
    no_progressbar: bool,
    max_batches: int | None,
    adjacency_mode: str,
    print_examples: int,
    eigh_dtype: torch.dtype,
):
    direct_rmsd_all: list[float] = []
    hard_rmsd_all: list[float] = []
    hard_acc_all: list[float] = []
    hard_exact_all: list[float] = []

    soft_rmsd_all = {tau: [] for tau in taus}
    soft_acc_all = {tau: [] for tau in taus}
    soft_exact_all = {tau: [] for tau in taus}

    lambda2_all: list[float] = []
    spectral_gap_all: list[float] = []

    n_examples = 0
    n_gapped = 0
    examples_printed = 0

    connect_across_gaps = adjacency_mode == "compressed"

    for batch_idx, (pcs_gt, lengths, _, _, valid_mask, pad_mask, idx) in enumerate(
        tqdm(valdataloader, disable=no_progressbar)
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Same validation preparation as the source script.
        pcs_gt = pcs_gt.to(fabric.device)
        lengths = lengths.to(fabric.device)
        valid_mask = valid_mask.to(fabric.device)
        pad_mask = pad_mask.to(fabric.device)
        idx = idx.to(fabric.device)

        mask = (pad_mask & valid_mask).bool()

        cloud_in, target_col = make_uniformly_permuted_cloud_deterministic(
            pcs_gt=pcs_gt,
            mask=mask,
            idx=idx,
            step=0,
        )

        contiguous = valid_mask_is_contiguous(mask)
        n_gapped += int((~contiguous).sum().item())

        A_gt = build_gt_path_adjacency(
            target_col=target_col,
            mask=mask,
            dtype=cloud_in.dtype,
            connect_across_gaps=connect_across_gaps,
        )

        direct_P = direct_gt_assignment(target_col, mask, dtype=cloud_in.dtype)
        hard_P, soft_Ps, fiedler, lambda2, lambda3 = fiedler_assignments(
            adjacency=A_gt,
            mask=mask,
            taus=taus,
            sinkhorn_iters=sinkhorn_iters,
            eigh_dtype=eigh_dtype,
        )

        xyz_direct = torch.bmm(direct_P, cloud_in).masked_fill(
            ~mask.unsqueeze(-1), 0.0
        )
        xyz_hard = torch.bmm(hard_P, cloud_in).masked_fill(
            ~mask.unsqueeze(-1), 0.0
        )

        direct_rmsd = rmsd_angstrom_per_example(xyz_direct, pcs_gt, mask)
        hard_rmsd = rmsd_angstrom_per_example(xyz_hard, pcs_gt, mask)
        hard_acc, hard_exact = bidirectional_perm_accuracy(hard_P, target_col, mask)

        good = torch.isfinite(direct_rmsd)
        direct_rmsd_all.extend(direct_rmsd[good].cpu().tolist())
        hard_rmsd_all.extend(hard_rmsd[good].cpu().tolist())
        hard_acc_all.extend(hard_acc[good].cpu().tolist())
        hard_exact_all.extend(hard_exact[good].cpu().tolist())

        valid_lambda2 = torch.isfinite(lambda2)
        lambda2_all.extend(lambda2[valid_lambda2].cpu().tolist())
        valid_gap = torch.isfinite(lambda2) & torch.isfinite(lambda3)
        spectral_gap_all.extend((lambda3 - lambda2)[valid_gap].cpu().tolist())

        for tau in taus:
            P = soft_Ps[tau]
            xyz_soft = torch.bmm(P, cloud_in).masked_fill(
                ~mask.unsqueeze(-1), 0.0
            )
            soft_rmsd = rmsd_angstrom_per_example(xyz_soft, pcs_gt, mask)
            soft_acc, soft_exact = bidirectional_perm_accuracy(P, target_col, mask)

            soft_rmsd_all[tau].extend(soft_rmsd[good].cpu().tolist())
            soft_acc_all[tau].extend(soft_acc[good].cpu().tolist())
            soft_exact_all[tau].extend(soft_exact[good].cpu().tolist())

        if examples_printed < print_examples:
            B = pcs_gt.shape[0]
            for b in range(B):
                if examples_printed >= print_examples:
                    break
                rows = mask[b].nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue

                pred_hard = hard_P[b].argmax(dim=-1)[rows]
                gt = target_col[b, rows]
                gt_rev = target_col[b, rows.flip(0)]
                v = fiedler[b, rows]

                print("\n--- example", int(idx[b].item()), "---")
                print("n_valid:", int(rows.numel()))
                print("contiguous valid mask:", bool(contiguous[b].item()))
                print("GT target_col first 20:      ", gt[:20].cpu().tolist())
                print("GT reversed first 20:        ", gt_rev[:20].cpu().tolist())
                print("hard Fiedler first 20:       ", pred_hard[:20].cpu().tolist())
                print("Fiedler v first 10 inputs:   ", [round(float(x), 5) for x in v[:10].cpu()])
                print("direct GT RMSD (A):          ", float(direct_rmsd[b].item()))
                print("hard Fiedler RMSD (A):       ", float(hard_rmsd[b].item()))
                print("hard bidir perm acc:         ", float(hard_acc[b].item()))
                for tau in taus:
                    P = soft_Ps[tau]
                    pred = P[b].argmax(dim=-1)[rows]
                    print(
                        f"soft tau={tau:g} first 20:       ",
                        pred[:20].cpu().tolist(),
                    )
                examples_printed += 1

        n_examples += int(good.sum().item())

    print("\n" + "=" * 88)
    print("GT PATH ADJACENCY -> FIEDLER -> ORDERING: VALIDATION ABLATION")
    print("=" * 88)
    print(f"examples evaluated:       {n_examples}")
    print(f"examples with mask gaps:  {n_gapped}")
    print(f"adjacency mode:           {adjacency_mode}")
    if adjacency_mode == "compressed":
        print("  compressed = connect consecutive observed residues into one path")
    else:
        print("  strict = only connect target indices differing by exactly 1")
    print(f"Sinkhorn iterations:      {sinkhorn_iters}")
    print(f"eigh dtype:               {str(eigh_dtype).replace('torch.', '')}")

    mean, med, p90, mx = summarize(direct_rmsd_all)
    print("\nSanity check: exact target_col permutation")
    print(f"  RMSD A: mean={mean:.6f}  median={med:.6f}  p90={p90:.6f}  max={mx:.6f}")

    mean, med, p90, mx = summarize(hard_rmsd_all)
    print("\nHard Fiedler sort")
    print(f"  RMSD A: mean={mean:.6f}  median={med:.6f}  p90={p90:.6f}  max={mx:.6f}")
    print(f"  mean bidir permutation accuracy: {np.mean(hard_acc_all):.6f}")
    print(f"  exact-order fraction:             {np.mean(hard_exact_all):.6f}")

    print("\nSoft Fiedler -> Sinkhorn")
    print(
        f"{'tau':>10}  {'mean RMSD A':>14}  {'median':>10}  {'p90':>10}  "
        f"{'perm acc':>10}  {'exact frac':>11}"
    )
    print("-" * 76)
    for tau in taus:
        mean, med, p90, _ = summarize(soft_rmsd_all[tau])
        perm_acc = float(np.mean(soft_acc_all[tau])) if soft_acc_all[tau] else float("nan")
        exact_frac = float(np.mean(soft_exact_all[tau])) if soft_exact_all[tau] else float("nan")
        print(
            f"{tau:10.5g}  {mean:14.6f}  {med:10.6f}  {p90:10.6f}  "
            f"{perm_acc:10.6f}  {exact_frac:11.6f}"
        )

    if lambda2_all:
        print("\nGT graph spectral diagnostics")
        print(f"  mean lambda2:                 {np.mean(lambda2_all):.8f}")
    if spectral_gap_all:
        print(f"  mean (lambda3-lambda2):       {np.mean(spectral_gap_all):.8f}")
        print(f"  median (lambda3-lambda2):     {np.median(spectral_gap_all):.8f}")

    print("\nInterpretation:")
    print("  - Exact target_col should be ~0 A; otherwise the test harness is wrong.")
    print("  - Hard Fiedler should also be ~0 A / ~100% bidirectional order if the")
    print("    GT adjacency-to-order conversion is valid for your masks.")
    print("  - If hard is perfect but soft RMSD is worse, the remaining issue is the")
    print("    continuous relaxation / temperature, not graph recovery.")
    print("  - If hard Fiedler is bad in strict mode and masks contain gaps, the GT")
    print("    graph is disconnected; use compressed mode for this clean ablation.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation-only GT adjacency -> Fiedler ordering ablation"
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default="configs/encoder_airplane.yaml",
        type=str,
    )
    parser.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="Pass dev=True to the same datamodule loader.",
    )
    parser.add_argument(
        "--overfit128",
        default=False,
        action="store_true",
        help="Match the source script's debug behavior: validation subset of first 32 items.",
    )
    parser.add_argument(
        "--no-progressbar",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--max-batches",
        default=None,
        type=int,
        help="Optional quick-test cap; default evaluates the full validation loader.",
    )
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=[2.0, 1.0, 0.5, 0.2, 0.1, 0.05],
        help=(
            "Sinkhorn temperature sweep on the arccos-unwarped spectral rank "
            "coordinate (adjacent ideal ranks are ~1 apart)."
        ),
    )
    parser.add_argument(
        "--sinkhorn-iters",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--adjacency-mode",
        choices=["compressed", "strict"],
        default="compressed",
        help=(
            "compressed: connect consecutive observed residues into one path; "
            "strict: exactly match edge_ce_loss adjacency and do not bridge mask gaps."
        ),
    )
    parser.add_argument(
        "--eigh-dtype",
        choices=["float64", "float32"],
        default="float64",
        help=(
            "Use float64 by default so long-path eigensolver roundoff does not "
            "confound this GT-adjacency diagnostic."
        ),
    )
    parser.add_argument(
        "--print-examples",
        default=2,
        type=int,
        help="Print detailed orderings for the first N validation examples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    (
        dataconfig,
        _rmsd_modelconfig,
        _cloud_modelconfig,
        trainerconfig,
        _loggerconfig,
    ) = load_config(args.config_path)

    # Same seeding / determinism setup as the source script.
    seed_everything(trainerconfig.seed, workers=True)
    random.seed(trainerconfig.seed)
    np.random.seed(trainerconfig.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trainerconfig.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    fabric = Fabric(
        accelerator=trainerconfig.accelerator,
        precision=trainerconfig.precision,
    )

    dm = load_datamodule(dataconfig, dev=args.dev)

    # Same validation dataloader path as the source script.
    if args.overfit128:
        from torch.utils.data import DataLoader, Subset

        val_dl_full = (
            dm.val_dataloader()
            if callable(dm.val_dataloader)
            else dm.val_dataloader
        )
        val_ds = val_dl_full.dataset
        n_subset = min(32, len(val_ds))
        val_dl = DataLoader(
            Subset(val_ds, list(range(n_subset))),
            batch_size=dataconfig.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=dataconfig.pin_memory,
            collate_fn=val_dl_full.collate_fn,
        )
    else:
        val_dl = (
            dm.val_dataloader()
            if callable(dm.val_dataloader)
            else dm.val_dataloader
        )

    valdataloader = fabric.setup_dataloaders(val_dl)

    print("Device:", fabric.device)
    print("Validation batches:", len(valdataloader))
    print("Taus:", args.taus)

    eigh_dtype = torch.float64 if args.eigh_dtype == "float64" else torch.float32

    run_validation_test(
        fabric=fabric,
        valdataloader=valdataloader,
        taus=args.taus,
        sinkhorn_iters=args.sinkhorn_iters,
        no_progressbar=args.no_progressbar,
        max_batches=args.max_batches,
        adjacency_mode=args.adjacency_mode,
        print_examples=args.print_examples,
        eigh_dtype=eigh_dtype,
    )


if __name__ == "__main__":
    main()