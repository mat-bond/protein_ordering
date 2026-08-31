"""
End-to-end training for:
    point cloud -> Equiformer -> learned backbone adjacency
    -> differentiable Fiedler vector -> Sinkhorn permutation -> ordered coordinates

Expected model file:
    src/models/fiedler_ordering_model.py

This script preserves the existing datamodule/Fabric/config/checkpoint conventions and
the same training/validation point-cloud preparation used by the supplied setup.

Typical run from repository root:
    uv run python src/train_fiedler_e2e.py \
        --config configs/proteins_sep.yaml \

Quick debug:
    uv run python src/train_fiedler_e2e.py \
        --config configs/proteins_sep.yaml \
        --overfit128 --max-train-batches 4 --max-val-batches 4
"""

import argparse
import math
import os
import random
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from lightning import seed_everything
from lightning.fabric import Fabric
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary
from tqdm import tqdm

from loaders import (
    load_datamodule,
    load_logger,
    load_model,
    load_config
)
from metrics.loss import (
    masked_kabsch_mse_bidirectional,
    drmsd_loss_bidirectional,
)
from plotting import plot_recon_3d
from protein_inspection import write_to_pdb, plot_rmsd_histogram

from training.corruption import (
    random_rotation_matrix,
    apply_rotation_masked,
    add_masked_coord_jitter,
    make_uniformly_permuted_cloud,
    make_uniformly_permuted_cloud_deterministic,
    reverse_target_col,
)
from metrics.loss import compute_confidence_scores

torch.set_float32_matmul_precision("medium")

# -----------------------------------------------------------------------------
# Config / checkpoint plumbing: same conventions as the supplied trainer
# -----------------------------------------------------------------------------

def config_to_dict(config):
    if hasattr(config, "model_dump"):  # pydantic v2
        return config.model_dump()
    if hasattr(config, "dict"):        # pydantic v1
        return config.dict()
    return vars(config)


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


def make_checkpoint_state(
    model, optimizer, scheduler, epoch, step, best_val_rmsd
):
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_rmsd": best_val_rmsd,
        "rng_state": get_rng_state(),
    }


def load_checkpoint_if_available(
    fabric,
    ckpt_path,
    model,
    optimizer,
    scheduler,
    restore_rng_state: bool = True,
):
    start_epoch = 0
    step = 0
    best_val_rmsd = float("inf")

    if ckpt_path is None or not os.path.exists(ckpt_path):
        return start_epoch, step, best_val_rmsd

    remainder = fabric.load(
        ckpt_path,
        state={"model": model, "optimizer": optimizer},
    )
    scheduler.load_state_dict(remainder["scheduler"])
    start_epoch = int(remainder["epoch"]) + 1
    step = int(remainder["step"])
    best_val_rmsd = float(
        remainder.get("best_val_rmsd", float("inf"))
    )

    if restore_rng_state and "rng_state" in remainder:
        set_rng_state(remainder["rng_state"])

    return start_epoch, step, best_val_rmsd

def permutation_ce_per_example(
    assignment_logits: torch.Tensor,
    target_col: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = 0.0,
):
    B, T, I = assignment_logits.shape
    valid_rows = mask & (target_col >= 0)

    ce_sum = assignment_logits.new_zeros(B)
    correct_sum = assignment_logits.new_zeros(B)
    count = assignment_logits.new_zeros(B)

    if valid_rows.any():
        b_idx, t_idx = valid_rows.nonzero(as_tuple=True)
        logits_sel = assignment_logits[b_idx, t_idx]
        target_sel = target_col[b_idx, t_idx]
        valid_cols = mask[b_idx]

        very_neg = torch.finfo(logits_sel.dtype).min
        logits_sel = logits_sel.masked_fill(~valid_cols, very_neg)

        row_ce = F.cross_entropy(
            logits_sel,
            target_sel,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        row_correct = (
            logits_sel.argmax(dim=-1) == target_sel
        ).to(logits_sel.dtype)

        ce_sum.scatter_add_(0, b_idx, row_ce)
        correct_sum.scatter_add_(0, b_idx, row_correct)
        count.scatter_add_(
            0, b_idx, torch.ones_like(row_ce)
        )

    return ce_sum, correct_sum, count


def permutation_ce_and_acc_bidirectional(
    assignment_logits: torch.Tensor,
    target_col: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = 0.0,
):
    target_rev = reverse_target_col(target_col, mask)

    ce_f, corr_f, count_f = permutation_ce_per_example(
        assignment_logits,
        target_col,
        mask,
        label_smoothing,
    )
    ce_r, corr_r, count_r = permutation_ce_per_example(
        assignment_logits,
        target_rev,
        mask,
        label_smoothing,
    )

    mean_f = ce_f / count_f.clamp_min(1.0)
    mean_r = ce_r / count_r.clamp_min(1.0)
    choose_r = mean_r < mean_f

    ce = torch.where(choose_r, ce_r, ce_f)
    corr = torch.where(choose_r, corr_r, corr_f)
    count = torch.where(choose_r, count_r, count_f)

    total_count = count.sum().clamp_min(1.0)
    return ce.sum() / total_count, corr.sum() / total_count


def compressed_edge_ce_loss(
    edge_logits: torch.Tensor,
    target_col: torch.Tensor,
    mask: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
):
    """
    Positive iff two candidate input points are consecutive in the *observed*
    sequence after masking.

    This deliberately differs from the old abs(absolute_index_i-index_j)==1
    target.  With mask gaps, the old target is disconnected; the Fiedler layer
    requires one path through all observed points.
    """
    if edge_logits.shape[0] == 0:
        zero = edge_logits.new_zeros(())
        return zero, zero, zero, zero, zero

    B, L = target_col.shape
    flat_mask = mask.reshape(-1)

    input_to_rank = torch.full_like(target_col, -100)
    possible_directed = 0

    for b in range(B):
        valid_rows = mask[b].nonzero(as_tuple=True)[0]
        n = int(valid_rows.numel())
        if n == 0:
            continue

        input_cols = target_col[b, valid_rows]
        ranks = torch.arange(
            n, device=target_col.device, dtype=target_col.dtype
        )
        input_to_rank[b, input_cols] = ranks
        possible_directed += 2 * max(n - 1, 0)

    packed_rank = input_to_rank.reshape(-1)[flat_mask]
    edge_gt = (
        torch.abs(
            packed_rank[edge_src] - packed_rank[edge_dst]
        ) == 1
    ).long()

    pos = edge_gt.sum()
    total = edge_gt.numel()
    neg = total - pos

    w_neg = 1.0
    w_pos = (
        neg.float() / pos.clamp_min(1).float()
    ).sqrt()
    class_weights = torch.tensor(
        [w_neg, w_pos],
        device=edge_logits.device,
        dtype=edge_logits.dtype,
    )

    edge_ce = F.cross_entropy(
        edge_logits, edge_gt, weight=class_weights
    )

    pred = edge_logits.argmax(dim=-1)
    edge_acc = (pred == edge_gt).float().mean()
    pos_recall = (
        (pred[edge_gt == 1] == 1).float().mean()
        if pos > 0
        else edge_logits.new_tensor(1.0)
    )

    # Candidate support recall: before learning, does the candidate graph even
    # contain the GT compressed path edges?
    candidate_recall = edge_logits.new_tensor(
        float(pos.item()) / max(possible_directed, 1)
    )
    edge_pos_frac = edge_gt.float().mean()

    return (
        edge_ce,
        edge_pos_frac,
        edge_acc,
        pos_recall,
        candidate_recall,
    )


def exact_bidir_fraction(
    P: torch.Tensor,
    target_col: torch.Tensor,
    mask: torch.Tensor,
):
    target_rev = reverse_target_col(target_col, mask)
    pred = P.argmax(dim=-1)
    vals = []

    for b in range(P.shape[0]):
        rows = mask[b].nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        p = pred[b, rows]
        fwd = torch.all(p == target_col[b, rows])
        rev = torch.all(p == target_rev[b, rows])
        vals.append((fwd | rev).float())

    if not vals:
        return P.new_zeros(())
    return torch.stack(vals).mean()

# -----------------------------------------------------------------------------
# Schedule
# -----------------------------------------------------------------------------

def spectral_schedule(
    step: int,
    warmup_steps: int,
    ramp_steps: int,
    tau_start: float,
    tau_end: float,
):
    if step < warmup_steps:
        return 0.0, float(tau_start)

    if ramp_steps <= 0:
        frac = 1.0
    else:
        frac = min(
            1.0,
            max(0.0, (step - warmup_steps) / float(ramp_steps)),
        )

    # Geometric temperature anneal is more natural over a wide tau range.
    tau = tau_start * ((tau_end / tau_start) ** frac)
    return frac, float(tau)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

@torch.no_grad()
def validate(
    fabric,
    dataloader,
    model,
    no_progressbar,
    spectral_tau: float,
    label_smoothing: float,
    max_batches: int | None = None,
    results_base_dir: str | None = None,
    inspect_best_only: bool = False,
    pdb_inspect_amount: int = 0,
    split_name: str = "val",
):
    model.eval()

    rmsd_all = []
    sample_ids_all = [] 
    perm_acc_weighted = 0.0
    perm_count = 0
    exact_vals = []
    edge_ce_vals = []
    edge_pos_recall_vals = []
    candidate_recall_vals = []
    lambda2_vals = []
    gap_vals = []

    last_batch = None
    last_confidence = None

    for batch_idx, (
        pcs_gt,
        lengths,
        _,
        _,
        valid_mask,
        pad_mask,
        idx,
    ) in enumerate(tqdm(dataloader, disable=no_progressbar)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        pcs_gt = pcs_gt.to(fabric.device)
        valid_mask = valid_mask.to(fabric.device)
        pad_mask = pad_mask.to(fabric.device)
        idx = idx.to(fabric.device)

        mask = (pad_mask & valid_mask).bool()
        cloud_in, target_col = (
            make_uniformly_permuted_cloud_deterministic(
                pcs_gt=pcs_gt,
                mask=mask,
                idx=idx,
                step=0,
            )
        )

        out = model(
            cloud_in,
            point_mask=valid_mask,
            padding_mask=pad_mask,
            spectral_tau=spectral_tau,
            compute_spectral=True,
        )

        xyz_ord = out["ordered_xyz"] * mask.unsqueeze(-1)

        _, per_example_mse, count = (
            masked_kabsch_mse_bidirectional(
                xyz_ord, pcs_gt, mask
            )
        )
        good = count > 0
        rmsd = (
            100.0
            * torch.sqrt(
                per_example_mse[good].clamp_min(1e-12)
            )
        )
        rmsd_all.extend(rmsd.cpu().tolist())

        sample_ids_all.extend(
            idx[good].detach().cpu().tolist()
        )

        edge_ce, _, _, edge_pos_recall, candidate_recall = (
            compressed_edge_ce_loss(
                out["edge_logits"],
                target_col,
                mask,
                out["edge_src"],
                out["edge_dst"],
            )
        )
        edge_ce_vals.append(float(edge_ce.item()))
        edge_pos_recall_vals.append(float(edge_pos_recall.item()))
        candidate_recall_vals.append(float(candidate_recall.item()))

        _, perm_acc = permutation_ce_and_acc_bidirectional(
            out["assignment_logits"],
            target_col,
            mask,
            label_smoothing=label_smoothing,
        )
        nrows = int(mask.sum().item())
        perm_acc_weighted += float(perm_acc.item()) * nrows
        perm_count += nrows

        exact_vals.append(
            float(
                exact_bidir_fraction(
                    out["permutation_matrices"],
                    target_col,
                    mask,
                ).item()
            )
        )

        lam2 = out["lambda2"]
        lam3 = out["lambda3"]
        good2 = torch.isfinite(lam2)
        lambda2_vals.extend(lam2[good2].cpu().tolist())
        goodgap = torch.isfinite(lam2) & torch.isfinite(lam3)
        gap_vals.extend(
            (lam3 - lam2)[goodgap].cpu().tolist()
        )

        last_batch = (
            xyz_ord.detach().cpu(),
            pcs_gt.detach().cpu(),
            cloud_in.detach().cpu(),
            mask.detach().cpu(),
        )
        if inspect_best_only:
            last_confidence = (
                100.0
                * compute_confidence_scores(
                    assignment_logits=out["assignment_logits"],
                    mask=mask,
                )
            ).detach().cpu()

    mean_rmsd = (
        float(np.mean(rmsd_all)) if rmsd_all else float("inf")
    )
    median_rmsd = (
        float(np.median(rmsd_all)) if rmsd_all else float("inf")
    )

    metrics = {
        f"{split_name} RMSD (A)": mean_rmsd,
        f"{split_name} median RMSD (A)": median_rmsd,
        f"{split_name} perm acc": perm_acc_weighted / max(perm_count, 1),
        f"{split_name} exact order frac": (
            float(np.mean(exact_vals)) if exact_vals else 0.0
        ),
        f"{split_name} Edge CE": (
            float(np.mean(edge_ce_vals)) if edge_ce_vals else 0.0
        ),
        f"{split_name} edge positive recall": (
            float(np.mean(edge_pos_recall_vals))
            if edge_pos_recall_vals
            else 0.0
        ),
        f"{split_name} candidate GT recall": (
            float(np.mean(candidate_recall_vals))
            if candidate_recall_vals
            else 0.0
        ),
        f"{split_name} mean lambda2": (
            float(np.mean(lambda2_vals))
            if lambda2_vals
            else float("nan")
        ),
        f"{split_name} mean spectral gap": (
            float(np.mean(gap_vals))
            if gap_vals
            else float("nan")
        ),
    }

    if inspect_best_only:
        import csv
        import json
        if results_base_dir is None:
            raise ValueError(
                "results_base_dir is required when inspect_best_only=True"
            )

        if last_batch is not None and last_confidence is not None:
            pcs_recon_b, pcs_gt_b, cloud_in_b, mask_b = last_batch
            write_to_pdb(
                directory=f"{results_base_dir}/pdb_files",
                xyz_gt=100.0 * pcs_gt_b,
                xyz_pred=100.0 * pcs_recon_b,
                xyz_in=100.0 * cloud_in_b,
                confidence=last_confidence,
                mask=mask_b,
                amount=pdb_inspect_amount,
            )
        if rmsd_all:
            rmsd_np = np.asarray(rmsd_all)

            metrics[f"{split_name} p90 RMSD (A)"] = float(
                np.percentile(rmsd_np, 90)
            )
            metrics[f"{split_name} p95 RMSD (A)"] = float(
                np.percentile(rmsd_np, 95)
            )
            metrics[f"{split_name} frac RMSD < 0.1A"] = float(
                np.mean(rmsd_np < 0.1)
            )
            metrics[f"{split_name} frac RMSD > 10A"] = float(
                np.mean(rmsd_np > 10.0)
            )

        if rmsd_all:
            plot_rmsd_histogram(
                rmsd_all,
                filename=f"{results_base_dir}/{split_name}_rmsd_histogram.png",
                bins=50,
                title=f"{split_name} per-protein RMSD distribution",
            )

        with open(
                f"{results_base_dir}/{split_name}_per_protein.csv",
                "w",
                newline="",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", "rmsd_A"])

                for sample_id, rmsd_A in zip(
                    sample_ids_all,
                    rmsd_all,
                ):
                    writer.writerow([sample_id, rmsd_A])

        with open(
            f"{results_base_dir}/{split_name}_metrics.json",
            "w",
        ) as f:
            json.dump(metrics, f, indent=2)

    return metrics, last_batch


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(
    fabric,
    dataloader,
    valdataloader,
    model,
    optimizer,
    scheduler,
    trainerconfig,
    logger,
    no_progressbar,
    results_base_dir,
    dev,
    start_epoch,
    start_step,
    best_val_rmsd,
    args,
):
    step = start_step

    for epoch in range(start_epoch, trainerconfig.max_epochs):
        model.train()
        loss_sum = 0.0
        n_sum = 0

        for batch_idx, (
            pcs_gt,
            lengths,
            _,
            _,
            valid_mask,
            pad_mask,
            idx,
        ) in enumerate(tqdm(dataloader, disable=no_progressbar)):
            if (
                args.max_train_batches is not None
                and batch_idx >= args.max_train_batches
            ):
                break

            optimizer.zero_grad(set_to_none=True)

            pcs_gt = pcs_gt.to(fabric.device)
            valid_mask = valid_mask.to(fabric.device)
            pad_mask = pad_mask.to(fabric.device)
            idx = idx.to(fabric.device)
            mask = (pad_mask & valid_mask).bool()

            # Same random rotation augmentation as the supplied trainer.
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
                sigma_frac=args.coord_jitter,
            )

            cloud_in, target_col = make_uniformly_permuted_cloud(
                pcs_gt=pcs_gt,
                mask=mask,
            )

            global_scale, tau = spectral_schedule(
                step=step,
                warmup_steps=args.edge_warmup_steps,
                ramp_steps=args.global_ramp_steps,
                tau_start=args.tau_start,
                tau_end=args.tau_end,
            )

            compute_spectral = global_scale > 0.0

            out = model(
                cloud_in,
                point_mask=valid_mask,
                padding_mask=pad_mask,
                spectral_tau=tau,
                compute_spectral=compute_spectral,
            )

            (
                edge_ce,
                edge_pos_frac,
                edge_acc,
                edge_pos_recall,
                candidate_recall,
            ) = compressed_edge_ce_loss(
                out["edge_logits"],
                target_col,
                mask,
                out["edge_src"],
                out["edge_dst"],
            )

            loss = args.w_edge * edge_ce

            mse = edge_ce.new_zeros(())
            dr = edge_ce.new_zeros(())
            perm_ce = edge_ce.new_zeros(())
            perm_acc = edge_ce.new_zeros(())
            dist_loss = edge_ce.new_zeros(())
            rmsd = edge_ce.new_tensor(float("nan"))
            exact_frac = edge_ce.new_zeros(())
            spectral_grad_norm = float("nan")

            if compute_spectral:
                xyz_ord = out["ordered_xyz"] * mask.unsqueeze(-1)

                mse, per_example_mse, count = (
                    masked_kabsch_mse_bidirectional(
                        xyz_ord, pcs_gt, mask
                    )
                )
                dr = drmsd_loss_bidirectional(
                    xyz_ord, pcs_gt, mask
                )

                good = count > 0
                rmsd = torch.mean(
                    100.0
                    * torch.sqrt(
                        per_example_mse[good].clamp_min(1e-12)
                    )
                )

                diff = xyz_ord[:, 1:] - xyz_ord[:, :-1]
                dist = diff.norm(dim=-1)
                pair_mask = mask[:, 1:] & mask[:, :-1]
                if pair_mask.any():
                    dist_loss = (
                        (dist - 0.038).square()[pair_mask].mean()
                    )

                perm_ce, perm_acc = (
                    permutation_ce_and_acc_bidirectional(
                        out["assignment_logits"],
                        target_col,
                        mask,
                        label_smoothing=args.label_smoothing,
                    )
                )

                exact_frac = exact_bidir_fraction(
                    out["permutation_matrices"],
                    target_col,
                    mask,
                )

                global_loss = (
                    args.w_perm * perm_ce
                    + args.w_mse * mse
                    + args.w_dr * dr
                    + args.w_dist * dist_loss
                )
                loss = loss + global_scale * global_loss

                # Optional direct proof that the global spectral objective has a
                # gradient w.r.t. the predicted edge logits.
                if (
                    args.check_spectral_grad_every > 0
                    and step % args.check_spectral_grad_every == 0
                ):
                    g = torch.autograd.grad(
                        global_loss,
                        out["edge_logits"],
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if g is not None:
                        spectral_grad_norm = float(
                            g.detach().float().norm().item()
                        )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step}: "
                    f"edge_ce={edge_ce.item()} "
                    f"perm_ce={perm_ce.item()} mse={mse.item()}"
                )

            fabric.backward(loss)

            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError(
                        f"Non-finite gradient in {name} at step {step}"
                    )

            fabric.clip_gradients(
                model,
                optimizer,
                max_norm=args.grad_clip,
            )
            optimizer.step()
            scheduler.step()

            B = pcs_gt.shape[0]
            loss_sum += float(loss.item()) * B
            n_sum += B

            log_dict = {
                "loss": float(loss.item()),
                "lr": optimizer.param_groups[0]["lr"],
                "spectral/global_scale": global_scale,
                "spectral/tau": tau,
                "Edge CE": float(edge_ce.item()),
                "edge acc": float(edge_acc.item()),
                "edge positive recall": float(edge_pos_recall.item()),
                "edge positive frac": float(edge_pos_frac.item()),
                "candidate GT edge recall": float(candidate_recall.item()),
                "perm_ce": float(perm_ce.item()),
                "perm_acc": float(perm_acc.item()),
                "exact order frac": float(exact_frac.item()),
                "mse": float(mse.item()),
                "dr": float(dr.item()),
                "dist loss": float(dist_loss.item()),
                "RMSD (A)": float(rmsd.item()),
            }
            if math.isfinite(spectral_grad_norm):
                log_dict["spectral grad norm at edge_logits"] = (
                    spectral_grad_norm
                )

            if compute_spectral:
                lam2 = out["lambda2"]
                lam3 = out["lambda3"]
                goodgap = torch.isfinite(lam2) & torch.isfinite(lam3)
                if goodgap.any():
                    log_dict["spectral mean gap"] = float(
                        (lam3 - lam2)[goodgap].mean().item()
                    )

            logger.log_metrics(log_dict, step=step)
            step += 1

        train_epoch_loss = loss_sum / max(n_sum, 1)

        # Validation always runs the spectral branch, even during EdgeCE warmup,
        # so one can watch when the learned adjacency becomes globally orderable.
        _, val_tau = spectral_schedule(
            step=step,
            warmup_steps=args.edge_warmup_steps,
            ramp_steps=args.global_ramp_steps,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
        )

        if args.val_tau is not None: 
            val_tau = args.val_tau

        val_metrics, last_batch = validate(
            fabric=fabric,
            dataloader=valdataloader,
            model=model,
            no_progressbar=no_progressbar,
            spectral_tau=val_tau,
            label_smoothing=args.label_smoothing,
            max_batches=args.max_val_batches,
        )
        logger.log_metrics(
            {
                "train_epoch_loss": train_epoch_loss,
                **val_metrics,
            },
            step=step,
        )

        val_rmsd = float(val_metrics["val RMSD (A)"])
        is_best = val_rmsd < best_val_rmsd
        if is_best:
            best_val_rmsd = val_rmsd

        if (
            last_batch is not None
            and epoch % trainerconfig.checkpoint_interval == 0
        ):
            xyz_b, gt_b, cloud_b, mask_b = last_batch
            Bvis = min(8, xyz_b.shape[0])
            plot_recon_3d(
                [xyz_b[i][mask_b[i]] for i in range(Bvis)],
                [gt_b[i][mask_b[i]] for i in range(Bvis)],
                num_pc=Bvis,
                filename=(
                    f"{results_base_dir}/"
                    f"pcs_val_pred_vs_gt_{epoch:04}.png"
                ),
                align_to_ref=True,
            )
            plot_recon_3d(
                [cloud_b[i][mask_b[i]] for i in range(Bvis)],
                [gt_b[i][mask_b[i]] for i in range(Bvis)],
                num_pc=Bvis,
                filename=(
                    f"{results_base_dir}/"
                    f"pcs_val_input_vs_gt_{epoch:04}.png"
                ),
                align_to_ref=True,
            )

        if not dev:
            fabric.save(
                f"{results_base_dir}/fiedler_last.ckpt",
                make_checkpoint_state(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    best_val_rmsd,
                ),
            )
            if is_best:
                fabric.save(
                    f"{results_base_dir}/fiedler_best.ckpt",
                    make_checkpoint_state(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        step,
                        best_val_rmsd,
                    ),
                )

        print(
            f"[epoch {epoch}] train_loss={train_epoch_loss:.5f} "
            f"val_RMSD={val_rmsd:.4f} A "
            f"val_perm={val_metrics['val perm acc']:.4f} "
            f"val_edge_pos_recall="
            f"{val_metrics['val edge positive recall']:.4f} "
            f"candidate_recall="
            f"{val_metrics['val candidate GT recall']:.4f} "
            f"tau={val_tau:.4f}"
        )

    if not dev:
        fabric.save(
            f"{results_base_dir}/fiedler_final.ckpt",
            make_checkpoint_state(
                model,
                optimizer,
                scheduler,
                trainerconfig.max_epochs - 1,
                step,
                best_val_rmsd,
            ),
        )
    fabric.save(
        f"{results_base_dir}/fiedler_model_final.ckpt",
        {"model": model},
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end differentiable Fiedler ordering training"
    )
    p.add_argument(
        "--config",
        dest="config_path",
        default="configs/fiedler.yaml",
    )
    p.add_argument("--dev", action="store_true")
    p.add_argument("--overfit128", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-best", action="store_true")
    p.add_argument(
        "--inspect_best_only",
        "--inspect-best-only",
        dest="inspect_best_only",
        action="store_true",
        help=(
            "Load fiedler_best.ckpt, skip training, and inspect it on the "
            "validation set (PDB examples + RMSD histogram)."
        ),
    )
    p.add_argument(
        "--pdb_inspect_amount",
        "--pdb-inspect-amount",
        dest="pdb_inspect_amount",
        default=0,
        type=int,
        help="Number of validation examples to write as PDB files.",
    )
    p.add_argument("--no-progressbar", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)

    # Same default as your current clean GT-only experiments: no coordinate noise.
    p.add_argument("--coord-jitter", type=float, default=0.0)

    # Curriculum: EdgeCE first, then turn on the differentiable global path loss.
    p.add_argument("--edge-warmup-steps", type=int, default=2000)
    p.add_argument("--global-ramp-steps", type=int, default=8000)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.05)

    # Losses. Permutation CE is the main global supervision; MSE supplies a
    # coordinate-level end objective. Keep EdgeCE on throughout training.
    p.add_argument("--w-edge", type=float, default=1.0)
    p.add_argument("--w-perm", type=float, default=0.25)
    p.add_argument("--w-mse", type=float, default=10.0)
    p.add_argument("--w-dr", type=float, default=0.0)
    p.add_argument("--w-dist", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument(
        "--check-spectral-grad-every",
        type=int,
        default=200,
        help=(
            "Every N global-training steps, explicitly measure d(global loss)"
            "/d(edge_logits). Set 0 to disable."
        ),
    )

    # Fine-tuning
    p.add_argument("--finetune_from", type=str, default=None)
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override config learning rate, useful for fine-tuning.",
    )
    p.add_argument(
        "--val_tau",
        type=float,
        default=None,
    )

    # Testing
    p.add_argument(
        "--test_best",
        "--test-best",
        dest="test_best",
        action="store_true",
        )
    p.add_argument(
        "--test_tau",
        default=None,
        type=float
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Match the RMSD trainer: inspection means "load best and validate only".
    if args.inspect_best_only or args.test_best:
        args.resume = False
        args.resume_best = True

    (
        dataconfig,
        modelconfig,
        trainerconfig,
        loggerconfig,
    ) = load_config(
        args.config_path,
    )

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

    logger.log_hyperparams({
        **vars(args),
        "seed": trainerconfig.seed,
        "model_config": config_to_dict(modelconfig),
        "trainer_config": config_to_dict(trainerconfig),
        "data_config": config_to_dict(dataconfig),
    })

    dm = load_datamodule(dataconfig, dev=args.dev)

    from torch.utils.data import DataLoader, Subset

    train_dl = (
        dm.train_dataloader()
        if callable(dm.train_dataloader)
        else dm.train_dataloader
    )
    val_dl = (
        dm.val_dataloader()
        if callable(dm.val_dataloader)
        else dm.val_dataloader
    )
    test_dl = (
        dm.test_dataloader()
        if callable(dm.test_dataloader)
        else dm.test_dataloader
    )

    if args.overfit128:
        train_ds = train_dl.dataset
        val_ds = val_dl.dataset
        n_train = min(32, len(train_ds))
        n_val = min(32, len(val_ds))

        train_dl = DataLoader(
            Subset(train_ds, list(range(n_train))),
            batch_size=dataconfig.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=dataconfig.pin_memory,
            collate_fn=train_dl.collate_fn,
        )
        val_dl = DataLoader(
            Subset(val_ds, list(range(n_val))),
            batch_size=dataconfig.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=dataconfig.pin_memory,
            collate_fn=val_dl.collate_fn,
        )
    dataloader = fabric.setup_dataloaders(train_dl)
    valdataloader = fabric.setup_dataloaders(val_dl)
    testdataloader = fabric.setup_dataloaders(test_dl)

    model = load_model(modelconfig)
    print(summary(model))

    lr = args.lr if args.lr is not None else modelconfig.learning_rate

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
    )

    effective_steps_per_epoch = len(dataloader)
    if args.max_train_batches is not None:
        effective_steps_per_epoch = min(
            effective_steps_per_epoch,
            args.max_train_batches,
        )

    total_steps = max(
        1, effective_steps_per_epoch * trainerconfig.max_epochs
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6,
    )

    model, optimizer, scheduler = fabric.setup(
        model, optimizer, scheduler=scheduler
    )

    results_base_dir = (
        "results_dev" if args.dev else "results"
    )
    results_base_dir += (
        f"/{loggerconfig.results_dir}_fiedler_e2e"
    )
    if args.finetune_from is not None:
        results_base_dir += (
                f"_finetune"
            )
    os.makedirs(results_base_dir, exist_ok=True)

    resume_path = None
    if args.resume_best:
        resume_path = f"{results_base_dir}/fiedler_best.ckpt"
    elif args.resume:
        resume_path = f"{results_base_dir}/fiedler_last.ckpt"

    if args.test_best:
        results_base_dir += (
                f"_test"
            )
    os.makedirs(results_base_dir, exist_ok=True)

    if (
        (args.inspect_best_only or args.test_best)
        and resume_path is not None
        and not os.path.exists(resume_path)
    ):
        raise FileNotFoundError(
            f"Cannot inspect best model: checkpoint not found: {resume_path}"
        )

    if (args.inspect_best_only or args.test_best or (args.finetune_from is None)):
        start_epoch, start_step, best_val_rmsd = (
            load_checkpoint_if_available(
                fabric,
                resume_path,
                model,
                optimizer,
                scheduler,
            )
        )
    else: 
        fabric.load(
        args.finetune_from,
        state={"model": model},
        )

        start_epoch = 0
        start_step = 0
        best_val_rmsd = float("inf")

    print("Device:", fabric.device)
    print("Train batches:", len(dataloader))
    print("Validation batches:", len(valdataloader))
    print("Test batches: ", len(testdataloader))
    print("Model module:", args.model_module)
    print(
        "Spectral schedule:",
        f"warmup={args.edge_warmup_steps}",
        f"ramp={args.global_ramp_steps}",
        f"tau={args.tau_start}->{args.tau_end}",
    )

    if args.inspect_best_only or args.test_best:
        dt_ld = testdataloader if args.test_best else valdataloader

        split_name = "test" if args.test_best else "val"
        # Use the spectral temperature corresponding to the loaded best step.
        _, inspect_tau = spectral_schedule(
            step=start_step,
            warmup_steps=args.edge_warmup_steps,
            ramp_steps=args.global_ramp_steps,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
        )

        if args.test_best and (args.test_tau is not None):
            inspect_tau = args.test_tau 
        elif args.val_tau is not None:
            inspect_tau = args.val_tau

        max_batches=(
            None
            if args.test_best
            else args.max_val_batches
        )

        inspect_metrics, _ = validate(
            fabric=fabric,
            dataloader=dt_ld,
            model=model,
            no_progressbar=args.no_progressbar,
            spectral_tau=inspect_tau,
            label_smoothing=args.label_smoothing,
            max_batches=max_batches,
            results_base_dir=results_base_dir,
            inspect_best_only=True,
            pdb_inspect_amount=args.pdb_inspect_amount,
            split_name=split_name
        )

        print(f"Best-checkpoint {split_name} metrics:")
        for name, value in inspect_metrics.items():
            print(f"  {name}: {value}")
        print(
            "Inspection outputs:",
            f"{results_base_dir}/{split_name}_rmsd_histogram.png",
            f"{results_base_dir}/pdb_files",
        )

    else:
        train(
            fabric=fabric,
            dataloader=dataloader,
            valdataloader=valdataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainerconfig=trainerconfig,
            logger=logger,
            no_progressbar=args.no_progressbar,
            results_base_dir=results_base_dir,
            dev=args.dev,
            start_epoch=start_epoch,
            start_step=start_step,
            best_val_rmsd=best_val_rmsd,
            args=args,
        )


if __name__ == "__main__":
    main()