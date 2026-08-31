import os
import random
from typing import Any
import torch.nn.functional as F
from numpy import sqrt
import numpy as np
import yaml

from metrics.protein_loss import masked_kabsch_mse_bidirectional, chamfer_masked, drmsd_loss_bidirectional
from tests.hamiltonian_path_test import run_hamiltonian_benchmark

os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_USE_OFFSCREEN"] = "true"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse

import lightning
import pydantic
import torch
from lightning import seed_everything
from lightning.fabric import Fabric
from torchinfo import summary
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from loaders import (
    load_datamodule,
    load_logger,
    load_model,
    load_config
)
from protein_plot import plot_recon_3d
from protein_inspection import write_to_pdb, plot_rmsd_histogram
torch.set_float32_matmul_precision("medium")

from training.corruption import (
    random_rotation_matrix,
    apply_rotation_masked,
    add_masked_coord_jitter,
    make_uniformly_permuted_cloud,
    make_uniformly_permuted_cloud_deterministic,
    reverse_target_col,
)
from metrics.protein_loss import compute_confidence_scores

def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def make_checkpoint_state(model, optimizer, scheduler, epoch, step, best_val):
    return {
        "model": model,
        "optimizer": optimizer,
        # save scheduler explicitly as metadata
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
        "rng_state": get_rng_state(),
    }


def load_checkpoint_if_available(
    fabric,
    ckpt_path,
    model,
    optimizer,
    scheduler,
    restore_rng_state: bool = True,
    fine_tune: bool = False
):
    start_epoch = 0
    step = 0
    best_val = float("inf")

    if ckpt_path is None or not os.path.exists(ckpt_path):
        return start_epoch, step, best_val

    if not fine_tune:
        # restore model + optimizer in-place
        remainder = fabric.load(
            ckpt_path,
            state={
                "model": model,
                "optimizer": optimizer,
            },
        )
        # restore everything else manually from returned metadata
        scheduler.load_state_dict(remainder["scheduler"])
        start_epoch = int(remainder["epoch"]) + 1
        step = int(remainder["step"])
        best_val = float(remainder.get("best_val", float("inf")))

    else: 
        remainder = fabric.load(
            ckpt_path,
            state={
                "model": model,
            },
        )

    if restore_rng_state and "rng_state" in remainder and not fine_tune:
            set_rng_state(remainder["rng_state"])

    return start_epoch, step, best_val

def rotation_matrix_from_seed(seed: int, device=None, dtype=torch.float32):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0xFFFFFFFF)

    # sample on CPU generator for determinism across devices
    u1 = torch.rand((), generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    u2 = torch.rand((), generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    u3 = torch.rand((), generator=g, dtype=torch.float32).to(device=device, dtype=dtype)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1)     * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1)     * torch.cos(2 * torch.pi * u3)

    x, y, z, w = q1, q2, q3, q4
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)]),
        torch.stack([    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)]),
        torch.stack([    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]),
    ])
    return R

def fill_nans_for_ect(ca: torch.Tensor) -> torch.Tensor:
    # simplest: doesn’t crash ECT; mask will exclude invalid residues in losses
    return torch.nan_to_num(ca, nan=0.0, posinf=0.0, neginf=0.0)

def corruption_fraction(
    step: int,
    warmup_steps: int,
    transition_steps: int,
) -> float:
    """
    Returns corruption strength alpha in [0, 1].
    """
    if step < warmup_steps:
        return 0.0
    if step >= warmup_steps + transition_steps:
        return 1.0
    return float(step - warmup_steps) / float(max(1, transition_steps))


def disorder_strength_schedule(
    alpha: float,
    *,
    min_noise_frac: float = 0.005,
    max_noise_frac: float = 0.08,
    min_index_sigma: float = 0.10,
    max_index_sigma_frac: float = 0.35,
) -> tuple[float, float]:
    """
    Convert alpha in [0,1] to:
      - coordinate noise fraction
      - index perturbation sigma as a fraction of sequence length

    Small sigma => very local reorderings
    Large sigma => increasingly global reorderings
    """
    noise_frac = min_noise_frac + alpha * (max_noise_frac - min_noise_frac)
    index_sigma_frac = min_index_sigma + alpha * (max_index_sigma_frac - min_index_sigma)
    return noise_frac, index_sigma_frac


def make_locally_disordered_cloud(
    pcs_gt: torch.Tensor,   # [B, L, 3]
    mask: torch.Tensor,     # [B, L] bool
    step: int,
    *,
    warmup_steps: int = 0,
    transition_steps: int = 80_000,
    min_noise_frac: float = 0.005,
    max_noise_frac: float = 0.08,
    min_index_sigma: float = 0.10,
    max_index_sigma_frac: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    alpha = corruption_fraction(
        step=step,
        warmup_steps=warmup_steps,
        transition_steps=transition_steps,
    )

    noise_frac, index_sigma_frac = disorder_strength_schedule(
        alpha=alpha,
        min_noise_frac=min_noise_frac,
        max_noise_frac=max_noise_frac,
        min_index_sigma=min_index_sigma,
        max_index_sigma_frac=max_index_sigma_frac,
    )

    out = torch.zeros_like(pcs_gt)

    # target_col[b, t] = correct input column i for target row t
    # -100 means "ignore"
    B, L, _ = pcs_gt.shape
    target_col = torch.full(
        (B, L), -100, dtype=torch.long, device=pcs_gt.device
    )

    for b in range(B):
        valid = mask[b]
        valid_idx = valid.nonzero(as_tuple=True)[0]   # absolute valid positions in padded tensor

        pts = pcs_gt[b, valid]  # [n, 3]
        n = pts.shape[0]

        if n == 0:
            continue
        if n == 1:
            out[b, valid] = pts
            target_col[b, valid_idx] = valid_idx
            continue

        base_idx = torch.arange(n, device=pcs_gt.device, dtype=pts.dtype)
        sigma_idx = max(index_sigma_frac * float(n), 1e-6)
        noisy_idx = base_idx + torch.randn_like(base_idx) * sigma_idx

        # perm[i] = target-row index (local, 0..n-1) that got moved to input-column i
        perm = torch.argsort(noisy_idx)
        pts_perm = pts[perm]

        # inverse permutation:
        # inv_perm[t] = input-column i containing target-row t
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(n, device=perm.device)

        # map local indices back to absolute padded indices
        target_col[b, valid_idx] = valid_idx[inv_perm].long()

        centered = pts - pts.mean(dim=0, keepdim=True)
        scale = centered.pow(2).sum(dim=-1).mean().sqrt().clamp_min(1e-6)
        pts_perm = pts_perm + torch.randn_like(pts_perm) * (noise_frac * scale)

        out[b, valid] = pts_perm

    out = out * mask.unsqueeze(-1)
    return out, target_col, alpha, noise_frac, index_sigma_frac


def permutation_ce_and_acc_per_example(
    assignment_logits: torch.Tensor,  # [B, T, I]
    target_col: torch.Tensor,         # [B, T]
    mask: torch.Tensor,               # [B, T]
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, I = assignment_logits.shape

    valid_rows = mask & (target_col >= 0)  # [B, T]

    ce_sum = assignment_logits.new_zeros(B)
    chance_sum = assignment_logits.new_zeros(B)
    correct_sum = assignment_logits.new_zeros(B)
    count = assignment_logits.new_zeros(B)

    if valid_rows.any():
        b_idx, t_idx = valid_rows.nonzero(as_tuple=True)   # [N], [N]

        logits_sel = assignment_logits[b_idx, t_idx]       # [N, I]
        target_sel = target_col[b_idx, t_idx]              # [N]
        valid_cols_sel = mask[b_idx]                       # [N, I]

        very_neg = torch.finfo(logits_sel.dtype).min
        logits_sel = logits_sel.masked_fill(~valid_cols_sel, very_neg)

        pred_col = logits_sel.argmax(dim=-1)
        row_correct = (pred_col == target_sel).float()

        valid_cols_f = valid_cols_sel.to(logits_sel.dtype)
        n_valid = valid_cols_f.sum(dim=-1).clamp_min(1.0)   # [N]
        row_chance_ce = torch.log(n_valid)                  # [N]

        if label_smoothing <= 0.0:
            row_ce = F.cross_entropy(logits_sel, target_sel, reduction="none")
        else:
            log_probs = F.log_softmax(logits_sel, dim=-1)  # [N, I]
            target_logp = log_probs.gather(1, target_sel.unsqueeze(1)).squeeze(1)

            n_other = n_valid - 1.0
            row_ce = -target_logp.clone()  # fallback when only one valid class

            multi = n_other > 0
            if multi.any():
                sum_valid_logp = (log_probs[multi] * valid_cols_f[multi]).sum(dim=-1)
                mean_other_logp = (sum_valid_logp - target_logp[multi]) / n_other[multi]
                row_ce[multi] = (
                    -(1.0 - label_smoothing) * target_logp[multi]
                    - label_smoothing * mean_other_logp
                )

        ce_sum.scatter_add_(0, b_idx, row_ce)
        chance_sum.scatter_add_(0, b_idx, row_chance_ce)
        correct_sum.scatter_add_(0, b_idx, row_correct)
        count.scatter_add_(0, b_idx, torch.ones_like(row_ce))

    acc = correct_sum / count.clamp_min(1.0)
    return ce_sum, chance_sum, correct_sum, count, acc

def permutation_ce_and_acc_bidirectional(
    assignment_logits: torch.Tensor,  # [B, T, I]
    target_col: torch.Tensor,         # [B, T]
    mask: torch.Tensor,               # [B, T]
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_col_rev = reverse_target_col(target_col, mask)

    ce_sum_fwd, chance_sum_fwd, correct_fwd, count_fwd, acc_fwd = permutation_ce_and_acc_per_example(
        assignment_logits, target_col, mask, label_smoothing=label_smoothing
    )
    ce_sum_rev, chance_sum_rev, correct_rev, count_rev, acc_rev = permutation_ce_and_acc_per_example(
        assignment_logits, target_col_rev, mask, label_smoothing=label_smoothing
    )
    
    mean_ce_fwd = ce_sum_fwd / count_fwd.clamp_min(1.0)
    mean_ce_rev = ce_sum_rev / count_rev.clamp_min(1.0)

    choose_rev = mean_ce_rev < mean_ce_fwd

    chosen_ce_sum = torch.where(choose_rev, ce_sum_rev, ce_sum_fwd)
    chosen_chance_sum = torch.where(choose_rev, chance_sum_rev, chance_sum_fwd)
    chosen_correct = torch.where(choose_rev, correct_rev, correct_fwd)
    chosen_count = torch.where(choose_rev, count_rev, count_fwd)

    total_count = chosen_count.sum().clamp_min(1.0)
    perm_ce = chosen_ce_sum.sum() / total_count
    chance_ce = chosen_chance_sum.sum() / total_count
    perm_acc = chosen_correct.sum() / total_count

    return perm_ce, perm_acc, chance_ce

def permutation_ce_and_acc(
    assignment_logits: torch.Tensor,  # [B, T, I]
    target_col: torch.Tensor,         # [B, T], absolute input-column target, -100 ignore
    mask: torch.Tensor,               # [B, T] valid rows; same mask used for valid cols
) -> tuple[torch.Tensor, torch.Tensor]:
    # Mask invalid columns so CE only considers valid inputs
    very_neg = torch.finfo(assignment_logits.dtype).min
    logits_masked = assignment_logits.masked_fill(~mask[:, None, :], very_neg)

    valid_rows = mask & (target_col >= 0)
    if not valid_rows.any():
        zero = assignment_logits.new_zeros(())
        return zero, zero

    logits_sel = logits_masked[valid_rows]   # [N_valid, I]
    target_sel = target_col[valid_rows]      # [N_valid]

    perm_ce = F.cross_entropy(logits_sel, target_sel)

    pred_col = logits_sel.argmax(dim=-1)
    perm_acc = (pred_col == target_sel).float().mean()

    return perm_ce, perm_acc

def edge_ce_loss(edge_logits, target_col, mask, edge_src, edge_dst):
    if edge_logits.shape[0] == 0:
        zero = edge_logits.new_zeros(())
        return zero, zero

    B, L = target_col.shape
    flat_mask = mask.reshape(-1)

    input_to_col = torch.full_like(target_col, -100)
    for b in range(B):
        valid_idx = mask[b].nonzero(as_tuple=True)[0]
        input_cols = target_col[b, valid_idx]
        input_to_col[b, input_cols] = valid_idx

    input_to_col_flat = input_to_col.reshape(-1)[flat_mask]
    edge_gt = (torch.abs(input_to_col_flat[edge_src] - input_to_col_flat[edge_dst]) == 1).long()

    pos = edge_gt.sum()
    total = edge_gt.numel()
    neg = total - pos

    # mild inverse-frequency weighting
    w_neg = 1.0
    w_pos = (neg.float() / pos.clamp_min(1).float()).sqrt()

    class_weights = torch.tensor([w_neg, w_pos], device=edge_logits.device, dtype=edge_logits.dtype)

    edge_ce = F.cross_entropy(edge_logits, edge_gt, weight=class_weights)
    edge_pos_frac = edge_gt.float().mean()

    return edge_ce, edge_pos_frac

def train(
    fabric,
    dataloader,
    valdataloader,
    rmsd_model,
    optimizer,
    scheduler,
    trainerconfig,
    logger,
    no_progressbar,
    results_base_dir,
    dev,
    start_epoch=0,
    start_step=0,
    best_val=float("inf"),
    warmup_steps: int = 0,
    transition_steps: int = 80_000,
    min_noise_frac: float = 0.005,
    max_noise_frac: float = 0.08,
    min_index_sigma: float = 0.10,
    max_index_sigma_frac: float = 0.35,
    N_sigma: int = 0,
    W_MSE: float =1,
    W_DR: float =1,
    W_CE: float =1,
    W_ECE: float =1,
    W_DIST: float = 1,
    label_smoothing: float = 0.05,
    train_tau: float = 0.05,
    val_tau: float = 0.05
    
):
    step = start_step


    for epoch in range(start_epoch, trainerconfig.max_epochs):
        rmsd_model.train()
        loss_sum = 0.0
        n_sum = 0

        for batch_idx, (pcs_gt, lengths, _, _, valid_mask, pad_mask, idx) in enumerate(
            tqdm(dataloader, disable=no_progressbar)
        ):
            optimizer.zero_grad(set_to_none=True)

            pcs_gt = pcs_gt.to(fabric.device)
            lengths = lengths.to(fabric.device)
            valid_mask = valid_mask.to(fabric.device)
            pad_mask = pad_mask.to(fabric.device)
            idx = idx.to(fabric.device)

            mask = (pad_mask & valid_mask).bool()

            pcs_gt = torch.stack(
                    [
                        apply_rotation_masked(
                            pcs_gt[i],
                            mask[i],
                            random_rotation_matrix(
                                device=fabric.device,
                                dtype=pcs_gt.dtype,
                            ),
                        )
                        for i in range(pcs_gt.shape[0])
                    ],
                    dim=0,
                )

            pcs_gt = add_masked_coord_jitter(
                pcs_gt,
                mask,
                sigma_frac=N_sigma,
            )

            cloud_in, target_col = make_uniformly_permuted_cloud(
                pcs_gt=pcs_gt,
                mask=mask,
            )

            xyz_ord, permutation_matrices, assignment_logits, edge_logits, edge_src, edge_dst = rmsd_model(cloud_in, 
                                                                                                           point_mask=valid_mask, 
                                                                                                           padding_mask=pad_mask,
                                                                                                           assignment_tau=train_tau)
            xyz_ord = xyz_ord * mask.unsqueeze(-1)

            if not (torch.isfinite(xyz_ord).all() and torch.isfinite(cloud_in).all()):
                raise RuntimeError("Non-finite tensor in forward pass")

            mse, per_example_mse, count = masked_kabsch_mse_bidirectional(xyz_ord, pcs_gt, mask)
            dr = drmsd_loss_bidirectional(xyz_ord, pcs_gt, mask)
            cd_in = chamfer_masked(cloud_in, pcs_gt, mask)
            edge_ce, edge_pos_frac = edge_ce_loss(edge_logits,target_col,mask,edge_src,edge_dst)
            good = count > 0
            rmsd = torch.mean(100.0 * torch.sqrt(per_example_mse[good].clamp_min(1e-12)))

            diff = xyz_ord[:, 1:]-xyz_ord[:,:-1]
            dist = diff.norm(dim=-1)
            pair_mask = mask[:, 1: ]& mask[:,:-1]
            dist_loss = ((dist-0.038)**2)[pair_mask].mean()

            perm_ce, perm_acc, chance_ce = permutation_ce_and_acc_bidirectional(
                assignment_logits=assignment_logits,
                target_col=target_col,
                mask=mask,
                label_smoothing=label_smoothing
            )

            gap = chance_ce-perm_ce
            
            loss = W_MSE * mse + W_DR * dr + W_CE*perm_ce + W_ECE*edge_ce + W_DIST*dist_loss

            if not torch.isfinite(loss):
                print("non-finite loss")
                print("mse", mse.item())
                print("dr", dr.item())
                raise RuntimeError("non-finite loss")

            fabric.backward(loss)

            for name, p in rmsd_model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print("bad grad in", name)
                    raise RuntimeError("Non-finite gradient")

            fabric.clip_gradients(rmsd_model, optimizer, max_norm=0.5)
            optimizer.step()
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            B = pcs_gt.shape[0]
            loss_sum += loss.item() * B
            n_sum += B

            logger.log_metrics(
            {
                    "loss": loss.item(),
                    "lr": lr,
                    "dr": dr.item(),
                    "mse": mse.item(),
                    "perm_ce": perm_ce.item(),
                    "perm_acc": perm_acc.item(),
                    "Chance gap": gap.item(),
                    "RMSD (Å)": rmsd.item(),
                    "CD_input_vs_gt": cd_in.item(),
                    "Edge CE": edge_ce.item(),
                    "Edge pos frac": edge_pos_frac.item(),
                    "Dist loss": dist_loss.item()
                },
                step=step,
            )
            step += 1

            if batch_idx == 0 and epoch % trainerconfig.checkpoint_interval == 0:
                    with torch.no_grad():
                        pcs_recon_cpu = xyz_ord.detach().cpu()
                        pcs_gt_cpu = pcs_gt.detach().cpu()
                        mask_cpu = mask.detach().cpu()
                        cloud_in_cpu = cloud_in.detach().cpu()

                        Bvis = min(8, pcs_recon_cpu.shape[0])

                        recon_list = [pcs_recon_cpu[i][mask_cpu[i]] for i in range(Bvis)]
                        gt_list = [pcs_gt_cpu[i][mask_cpu[i]] for i in range(Bvis)]
                        cloud_list = [cloud_in_cpu[i][mask_cpu[i]] for i in range(Bvis)]

                        plot_recon_3d(
                            recon_list,
                            gt_list,
                            num_pc=Bvis,
                            filename=f"{results_base_dir}/pcs_train_pred_vs_gt_{epoch:04}.png",
                            align_to_ref=True,
                        )
                        plot_recon_3d(
                            cloud_list,
                            gt_list,
                            num_pc=Bvis,
                            filename=f"{results_base_dir}/pcs_train_input_vs_gt_{epoch:04}.png",
                            align_to_ref=True,
                        )

        val_metrics = validate(
            fabric=fabric,
            dataloader=valdataloader,
            rmsd_model=rmsd_model,
            no_progressbar=no_progressbar,
            results_base_dir=results_base_dir,
            epoch=epoch,
            checkpoint_interval=trainerconfig.checkpoint_interval,
            logger=logger,
            step=step,
            warmup_steps=warmup_steps,
            transition_steps=transition_steps,
            min_noise_frac=min_noise_frac,
            max_noise_frac=max_noise_frac,
            min_index_sigma=min_index_sigma,
            max_index_sigma_frac=max_index_sigma_frac,
            W_MSE=W_MSE,
            W_CE=W_CE,
            W_DR=W_DR,
            W_ECE=W_ECE,
            W_DIST=W_DIST,
            assignment_tau=val_tau
        )
        
        val_mean_rsmd = val_metrics["mean_rmsd"]

        train_epoch_loss = loss_sum / max(n_sum, 1)
        logger.log_metrics({"train_epoch_loss": train_epoch_loss}, step=step)

        is_best = val_mean_rsmd < best_val
        if is_best:
            best_val = val_mean_rsmd

        if not dev:
            fabric.save(
                f"{results_base_dir}/rmsd_last.ckpt",
                make_checkpoint_state(
                    model=rmsd_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=step,
                    best_val=best_val,
                ),
            )

        if is_best:
            best_val = val_mean_rsmd
            if not dev:
                fabric.save(
                    f"{results_base_dir}/rmsd_best.ckpt",
                    make_checkpoint_state(
                        model=rmsd_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=step,
                        best_val=best_val,
                    ),
                )


    if not dev:
        fabric.save(
            f"{results_base_dir}/rmsd_final.ckpt",
            make_checkpoint_state(
                model=rmsd_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=trainerconfig.max_epochs - 1,
                step=step,
                best_val=best_val,
            ),
        )

    fabric.save(f"{results_base_dir}/rmsd_model_final.ckpt", {"model": rmsd_model})

@torch.no_grad()
def validate(
    fabric,
    dataloader,
    rmsd_model,
    no_progressbar,
    results_base_dir,
    epoch,
    checkpoint_interval,
    logger=None,
    step: int | None = None,
    warmup_steps: int = 0,
    transition_steps: int = 80_000,
    min_noise_frac: float = 0.005,
    max_noise_frac: float = 0.08,
    min_index_sigma: float = 0.10,
    max_index_sigma_frac: float = 0.35,
    W_MSE: float =1,
    W_DR: float =1,
    W_CE: float =1,
    W_ECE: float =1,
    W_DIST: float = 1,
    inspect_best_only=False,
    pdb_inspect_amount: int =0,
    label_smoothing: float = 0.05,
    assignment_tau: float = 0.05,
    split_name: str = "val"
):
    rmsd_model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_cd_in = 0.0
    total_items = 0
    total_dr = 0.0
    total_perm_ce = 0.0
    total_perm_acc = 0.0
    total_perm_items = 0
    total_chance = 0.0
    total_ece = 0.0
    total_edge_items = 0
    total_dist_loss = 0.0
    all_rmsd = []
    last_batch = None
    alpha_corrupt = 0.0
    noise_frac = 0.0
    index_sigma_frac = 0.0

    for pcs_gt, lengths, _, _, valid_mask, pad_mask, idx in tqdm(
        dataloader, disable=no_progressbar
    ):
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
            step=0 # Each validation example gets the same deterministic permutation every epoch 
        )

        xyz_ord, permutation_matrices, assignment_logits, edge_logits, edge_src, edge_dst  = rmsd_model(cloud_in, 
                                                                                                        point_mask=valid_mask, 
                                                                                                        padding_mask=pad_mask,
                                                                                                        assignment_tau=assignment_tau)
        xyz_ord = xyz_ord * mask.unsqueeze(-1)

        mse, per_example_mse, count = masked_kabsch_mse_bidirectional(xyz_ord, pcs_gt, mask)
        dr = drmsd_loss_bidirectional(xyz_ord, pcs_gt, mask)
        cd_in = chamfer_masked(cloud_in, pcs_gt, mask)
        edge_ce, _ = edge_ce_loss(edge_logits,target_col,mask,edge_src,edge_dst)

        perm_ce, perm_acc, chance_ce = permutation_ce_and_acc_bidirectional(
            assignment_logits=assignment_logits,
            target_col=target_col,
            mask=mask, 
            label_smoothing=label_smoothing
        )

        good = count > 0 
        rmsd_values = 100.0 * torch.sqrt(per_example_mse[good].clamp_min(1e-12))
        all_rmsd.extend(rmsd_values.detach().cpu().tolist())    
        diff = xyz_ord[:, 1:] - xyz_ord[:, :-1]
        dist = diff.norm(dim=-1)

        pair_mask = mask[:, 1:] & mask[:, :-1]

        dist_loss = ((dist - 0.038) ** 2)[pair_mask].mean()

        loss = W_MSE*mse + W_DR*dr + W_CE * perm_ce + W_ECE*edge_ce + W_DIST*dist_loss

        num_edges = edge_logits.shape[0]
        gap = chance_ce - perm_ce
        B = pcs_gt.shape[0]
        total_loss += float(loss.item()) * B
        total_mse += float(mse.item()) * B
        total_cd_in += float(cd_in.item()) * B
        total_dr += float(dr.item()) * B
        n_valid_rows = int(mask.sum().item())
        total_perm_ce += float(perm_ce.item()) * n_valid_rows
        total_perm_acc += float(perm_acc.item()) * n_valid_rows
        total_perm_items += n_valid_rows 
        total_chance += float(gap.item()) * n_valid_rows
        total_ece += float(edge_ce.item()) * num_edges
        total_edge_items += num_edges
        total_items += B
        total_dist_loss += float(dist_loss.item())*B

        last_batch = (
            xyz_ord.detach().cpu(),
            pcs_gt.detach().cpu(),
            cloud_in.detach().cpu(),
            mask.detach().cpu(),
        )
    
    mean_loss = total_loss / max(total_items, 1)
    mean_mse = total_mse / max(total_items, 1)
    mean_cd_in = total_cd_in / max(total_items, 1)
    mean_dr = total_dr / max(total_items, 1)
    mean_perm_ce = total_perm_ce / max(total_perm_items, 1)
    mean_perm_acc = total_perm_acc / max(total_perm_items, 1) 
    mean_chance = total_chance / max(total_perm_items, 1) 
    mean_ece = total_ece / max(total_edge_items, 1) 
    mean_rmsd = np.mean(all_rmsd)
    mean_dist_loss = total_dist_loss / max(total_items, 1)

    if logger is not None and step is not None:
        logger.log_metrics(
            {
                f"{split_name}_loss": mean_loss,
                f"{split_name}_mse": mean_mse,
                f"{split_name} RMSD (Å)": mean_rmsd,
                f"{split_name} CD_input_vs_gt": mean_cd_in,
                f"{split_name} DR": mean_dr,
                f"{split_name}_perm_ce": mean_perm_ce,
                f"{split_name}_perm_acc": mean_perm_acc,
                f"{split_name} chance gap": mean_chance,
                f"{split_name} mean ECE": mean_ece,
                f"{split_name} dist loss": mean_dist_loss
            },
            step=step,
        )

    if not inspect_best_only:
        if last_batch is not None and step is not None and epoch % checkpoint_interval == 0:
            pcs_recon_b, pcs_gt_b, cloud_in_b, mask_b = last_batch
            Bvis = min(8, pcs_recon_b.shape[0])

            recon_list = [pcs_recon_b[i][mask_b[i]] for i in range(Bvis)]
            gt_list = [pcs_gt_b[i][mask_b[i]] for i in range(Bvis)]
            cloud_list = [cloud_in_b[i][mask_b[i]] for i in range(Bvis)]

            plot_recon_3d(
                recon_list,
                gt_list,
                num_pc=Bvis,
                filename=f"{results_base_dir}/pcs_val_pred_vs_gt_{epoch:04}.png",
                align_to_ref=True,
            )
            plot_recon_3d(
                cloud_list,
                gt_list,
                num_pc=Bvis,
                filename=f"{results_base_dir}/pcs_val_input_vs_gt_{epoch:04}.png",
                align_to_ref=True,
            )
    elif last_batch is not None:
        pcs_recon_b, pcs_gt_b, cloud_in_b, mask_b = last_batch
        confidence=100*compute_confidence_scores(assignment_logits=assignment_logits,mask=mask)
        write_to_pdb(directory=f"{results_base_dir}/pdb_files",
                        xyz_gt=100*pcs_gt_b,
                        xyz_pred=100*pcs_recon_b,
                        xyz_in=100*cloud_in_b,
                        confidence=confidence,
                        mask=mask_b,
                        amount=pdb_inspect_amount
        )
        plot_rmsd_histogram(
            all_rmsd,
            filename=f"{results_base_dir}/{split_name}_rmsd_histogram.png",
            bins=50,
            title=f"{split_name} per-protein RMSD distribution",
        )
        

    return {
        "mean_rmsd": mean_rmsd,
        "mean_loss": mean_loss,
        "perm_acc": mean_perm_acc,
    }

def run_benchmarks(dataloader, no_progressbar, fabric, n_starts):
    short_perm_scores_tot = []
    bond_perm_scores_tot = []
    short_edge_scores_tot = []
    bond_edge_scores_tot = []
    short_rmsd_tot = []
    bond_rmsd_tot = []

    for pcs_gt, lengths, _, _, valid_mask, pad_mask, idx in tqdm(
        dataloader, disable=no_progressbar
    ):
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
        step=0
        )
        (
            short_perm_scores,
            bond_perm_scores,
            short_edge_scores,
            bond_edge_scores,
            short_rmsd,
            bond_rmsd,
        ) = run_hamiltonian_benchmark(
            pcs_gt=pcs_gt,
            target_col=target_col,
            cloud_in=cloud_in,
            mask=mask,
            n_starts=n_starts,
        )
        short_perm_scores_tot.extend(short_perm_scores)
        bond_perm_scores_tot.extend(bond_perm_scores)
        short_edge_scores_tot.extend(short_edge_scores)
        bond_edge_scores_tot.extend(bond_edge_scores)
        short_rmsd_tot.extend(short_rmsd)
        bond_rmsd_tot.extend(bond_rmsd)

    short_perm_score = np.mean(short_perm_scores_tot)
    bond_perm_score = np.mean(bond_perm_scores_tot)
    short_edge_score = np.mean(short_edge_scores_tot)
    bond_edge_score = np.mean(bond_edge_scores_tot)
    short_rmsd_mean = np.mean(short_rmsd_tot)
    bond_rmsd_mean = np.mean(bond_rmsd)

    print(
            "Nearest-neighbor permutation accuracy:",
            np.mean(short_perm_score)
        )
    
    print(
        "3.8A permutation accuracy:",
        np.mean(bond_perm_score)
    )

    print(
        "Nearest-neighbor edge accuracy:",
        np.mean(short_edge_score)
    )

    print(
        "3.8A edge accuracy:",
        np.mean(bond_edge_score)
    )
    print("Nearest-neighbor mean RMSD (Å):", np.mean(short_rmsd_tot))
    print("3.8A mean RMSD (Å):", np.mean(bond_rmsd_tot))
    return 
            

def main():
    
    # GT-only corruption curriculum knobs
    CURRIC_WARMUP_STEPS = 0
    CURRIC_TRANSITION_STEPS = 80_000

    MIN_NOISE_FRAC = 0 # 0.005
    MAX_NOISE_FRAC = 0 #0.08
    N_SIGMA = 0.0

    # As a fraction of sequence length
    # 0.10 => small local reorderings
    # 0.35 => much broader disorder
    MIN_INDEX_SIGMA = 0.10
    MAX_INDEX_SIGMA_FRAC = 0.35

    parser = argparse.ArgumentParser(description="GT-only ordering-model training")
    parser.add_argument(
        "--config",
        dest="config_path",
        default="configs/rmsd.yaml",
        type=str,
    )
    parser.add_argument(
        "--compile",
        default=False,
        action="store_true",
        help="Compile the model (placeholder; currently unused).",
    )
    parser.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="Run a small subset.",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Resume flag placeholder.",
    )
    parser.add_argument(
        "--resume_best",
        default=False,
        action="store_true",
        help="Resume flag placeholder.",
    )
    parser.add_argument(
        "--no-progressbar",
        default=False,
        action="store_true",
        help="Disable tqdm.",
    )
    parser.add_argument(
        "--overfit128",
        default=False,
        action="store_true",
        help="Debug: overfit on small train/val subsets.",
    )
    parser.add_argument(
        "--inspect_best_only",
        default=False,
        action="store_true",
        help="Plots performance statistics of the best performing model on validation dataset. "
    )
    parser.add_argument(
        "--pdb_inspect_amount",
        default=0,
        type=int
    )
    parser.add_argument(
        "--fine_tune",
        default=False,
        action="store_true",
        help=".",
    )
    parser.add_argument(
        "--benchmark_only",
        default=False,
        action="store_true"
    )
    parser.add_argument("--w-perm", type=float, default=1.0)
    parser.add_argument("--w-mse", type=float, default=0.0)
    parser.add_argument("--w-edge", type=float, default=0.0)
    parser.add_argument("--w-dr", type=float, default=0.0)
    parser.add_argument("--w-dist", type=float, default=0.0)

    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--train-tau", type=float, default=0.1)
    parser.add_argument("--val-tau", type=float, default=None)
    parser.add_argument("--test-tau", type=float, default=None)

    parser.add_argument("--test-best", action="store_true")

    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    compile: bool = args.compile
    dev: bool = args.dev
    resume: bool = args.resume
    resume_best: bool = args.resume_best
    no_progressbar = args.no_progressbar
    inspect_best_only: bool =args.inspect_best_only
    fine_tune: bool = args.fine_tune
    benchmark_only: bool = args.benchmark_only
    test_best: bool = args.test_best

    if fine_tune and not resume_best:
        resume = True
        
    if inspect_best_only or test_best: 
        resume = False
        resume_best = True
    (
        dataconfig,
        modelconfig,
        trainerconfig,
        loggerconfig,
    )= load_config(args.config_path)

    results_base_dir = "results"
    if dev:
        trainerconfig.max_epochs = 5000
        results_base_dir += "_dev"

    results_base_dir += f"/{loggerconfig.results_dir}_gt_only_curriculum"
    os.makedirs(results_base_dir, exist_ok=True)


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

    logger = load_logger(loggerconfig)
    dm = load_datamodule(dataconfig, dev=dev)

    from torch.utils.data import DataLoader, Subset


    test_dl = (
        dm.test_dataloader()
        if callable(dm.test_dataloader)
        else dm.test_dataloader
    )
    train_dl_full = dm.train_dataloader() if callable(dm.train_dataloader) else dm.train_dataloader
    val_dl_full = dm.val_dataloader() if callable(dm.val_dataloader) else dm.val_dataloader

    train_dl = train_dl_full
    val_dl = val_dl_full

    if args.overfit128:
        train_ds = train_dl_full.dataset
        val_ds = val_dl_full.dataset

        train_dl = DataLoader(
            Subset(train_ds, list(range(32))),
            batch_size=dataconfig.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=dataconfig.pin_memory,
            collate_fn=train_dl_full.collate_fn,
        )
        val_dl = DataLoader(
            Subset(val_ds, list(range(32))),
            batch_size=dataconfig.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=dataconfig.pin_memory,
            collate_fn=val_dl_full.collate_fn,
        )

    testdataloader = fabric.setup_dataloaders(test_dl)
    dataloader = fabric.setup_dataloaders(train_dl)
    valdataloader = fabric.setup_dataloaders(val_dl)

    rmsd_model = load_model(modelconfig)
    
    print(summary(rmsd_model))
    
    optimizer = torch.optim.AdamW(
        rmsd_model.parameters(),
        lr=modelconfig.learning_rate,
        weight_decay=0.01,
    )

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * trainerconfig.max_epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6,
    )

    # set up everything through Fabric
    rmsd_model, optimizer, scheduler = fabric.setup(rmsd_model, optimizer, scheduler=scheduler)

    resume_path = None
    if resume_best:
        resume_path = f"{results_base_dir}/rmsd_best.ckpt"
    elif resume:
        resume_path = f"{results_base_dir}/rmsd_last.ckpt"

    start_epoch, start_step, best_val = load_checkpoint_if_available(
        fabric=fabric,
        ckpt_path=resume_path,
        model=rmsd_model,
        optimizer=optimizer,
        scheduler=scheduler,
        fine_tune=fine_tune
    )

    val_tau = (
        args.val_tau
        if args.val_tau is not None
        else args.train_tau
    )

    if not inspect_best_only and not benchmark_only:
        train(
            fabric=fabric,
            dataloader=dataloader,
            valdataloader=valdataloader,
            rmsd_model=rmsd_model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainerconfig=trainerconfig,
            logger=logger,
            no_progressbar=no_progressbar,
            results_base_dir=results_base_dir,
            dev=dev,
            start_epoch=start_epoch,
            start_step=start_step,
            best_val=best_val,
            warmup_steps=CURRIC_WARMUP_STEPS,
            transition_steps=CURRIC_TRANSITION_STEPS,
            min_noise_frac=MIN_NOISE_FRAC,
            max_noise_frac=MAX_NOISE_FRAC,
            min_index_sigma=MIN_INDEX_SIGMA,
            max_index_sigma_frac=MAX_INDEX_SIGMA_FRAC,
            N_sigma=N_SIGMA,
            W_MSE=args.w_mse,
            W_CE=args.w_perm,
            W_DR=args.w_dr,
            W_ECE=args.w_edge,
            W_DIST=args.w_dist,
            train_tau=args.train_tau,
            val_tau=val_tau
        )
    elif not benchmark_only:
        eval_loader = (
            testdataloader
            if args.test_best
            else valdataloader
        )
        split_name = "test" if args.test_best else "val"

        rmsd_model.eval()
        with torch.no_grad():
            validate(
            fabric=fabric,
            dataloader=eval_loader,
            rmsd_model=rmsd_model,
            no_progressbar=no_progressbar,
            results_base_dir=results_base_dir,
            epoch=0,
            checkpoint_interval=trainerconfig.checkpoint_interval,
            logger=None,
            step=None,
            warmup_steps=CURRIC_WARMUP_STEPS,
            transition_steps=CURRIC_TRANSITION_STEPS,
            min_noise_frac=MIN_NOISE_FRAC,
            max_noise_frac=MAX_NOISE_FRAC,
            min_index_sigma=MIN_INDEX_SIGMA,
            max_index_sigma_frac=MAX_INDEX_SIGMA_FRAC,
            W_MSE=args.w_mse,
            W_CE=args.w_perm,
            W_DR=args.w_dr,
            W_ECE=args.w_edge,
            inspect_best_only=inspect_best_only,
            pdb_inspect_amount=args.pdb_inspect_amount,
            assignment_tau=val_tau
        )
    else:
        run_benchmarks(valdataloader, no_progressbar, fabric, n_starts=500)


if __name__ == "__main__":
    main()
    
