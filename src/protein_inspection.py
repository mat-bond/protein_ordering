#!/usr/bin/env python3
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_rmsd_histogram(
    all_rmsd,
    filename: str,
    bins: int = 50,
    title: str = "Validation RMSD distribution",
):
    """
    Plot histogram of per-protein RMSD values (in Å).

    Args:
        all_rmsd: list/array of RMSD values in Å
        filename: where to save the figure
        bins: number of histogram bins
        title: plot title
    """
    rmsd = np.asarray(all_rmsd, dtype=np.float64)

    if rmsd.size == 0:
        raise ValueError("all_rmsd is empty; nothing to plot.")

    rmsd = rmsd[np.isfinite(rmsd)]
    if rmsd.size == 0:
        raise ValueError("all_rmsd contains no finite values.")

    mean = rmsd.mean()
    median = np.median(rmsd)
    p90 = np.percentile(rmsd, 90)
    p95 = np.percentile(rmsd, 95)

    plt.figure(figsize=(8, 5))
    plt.hist(rmsd, bins=bins)
    plt.axvline(mean, linestyle="--", linewidth=2, label=f"mean = {mean:.2f} Å")
    plt.axvline(median, linestyle="--", linewidth=2, label=f"median = {median:.2f} Å")
    plt.axvline(p90, linestyle=":", linewidth=2, label=f"p90 = {p90:.2f} Å")
    plt.axvline(p95, linestyle=":", linewidth=2, label=f"p95 = {p95:.2f} Å")

    plt.xlabel("RMSD (Å)")
    plt.ylabel("Number of proteins")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def write_to_pdb(
    directory,
    xyz_gt: torch.Tensor,
    xyz_pred: torch.Tensor,
    mask: torch.Tensor,
    amount: int,
    confidence: torch.Tensor,
    xyz_in: torch.Tensor | None = None,
):
    """
    Write `amount` randomly chosen samples from the provided tensors to disk.

    For each chosen sample, create a folder:
      val_sample_{k:03d}_idx_{b:04d}/

    and inside write:
      - gt.pdb           : ground truth
      - pred.pdb         : prediction, B-factor = confidence
      - rev_gt.pdb       : reversed-order ground truth
      - scrambled_in.pdb : scrambled input (if xyz_in is provided)

    Expected shapes:
      xyz_gt      [B, L, 3]
      xyz_pred    [B, L, 3]
      mask        [B, L]    bool
      confidence  [B, L]
      xyz_in      [B, L, 3] or None

    Returns:
      list[str]: paths of created sample directories
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_gt = xyz_gt.detach().cpu()
    xyz_pred = xyz_pred.detach().cpu()
    mask = mask.detach().cpu().bool()
    confidence = confidence.detach().cpu()

    if xyz_in is not None:
        xyz_in = xyz_in.detach().cpu()

    if xyz_gt.ndim != 3 or xyz_gt.shape[-1] != 3:
        raise ValueError(f"xyz_gt must have shape [B, L, 3], got {tuple(xyz_gt.shape)}")
    if xyz_pred.shape != xyz_gt.shape:
        raise ValueError(f"xyz_pred must match xyz_gt shape, got {tuple(xyz_pred.shape)} vs {tuple(xyz_gt.shape)}")
    if mask.shape != xyz_gt.shape[:2]:
        raise ValueError(f"mask must have shape [B, L], got {tuple(mask.shape)}")
    if confidence.shape != xyz_gt.shape[:2]:
        raise ValueError(f"confidence must have shape [B, L], got {tuple(confidence.shape)}")
    if xyz_in is not None and xyz_in.shape != xyz_gt.shape:
        raise ValueError(f"xyz_in must match xyz_gt shape, got {tuple(xyz_in.shape)} vs {tuple(xyz_gt.shape)}")

    B = xyz_gt.shape[0]
    if B == 0 or amount <= 0:
        return []

    amount = min(amount, B)
    chosen = random.sample(range(B), k=amount)

    written_dirs: list[str] = []

    def _write_single_pdb(
        filepath: Path,
        coords: torch.Tensor,          # [N, 3]
        bfactors: torch.Tensor | None = None,
        header: str = "",
    ):
        n = coords.shape[0]

        if bfactors is None:
            bfactors = torch.zeros(n, dtype=torch.float32)
        else:
            bfactors = bfactors.to(torch.float32)

        with open(filepath, "w", encoding="utf-8") as fh:
            if header:
                fh.write(f"HEADER    {header}\n")

            serial = 1
            chain_id = "A"

            for resseq in range(n):
                x, y, z = coords[resseq].tolist()
                bfac = float(bfactors[resseq].item())

                line = (
                    f"ATOM  {serial:5d}  CA  GLY {chain_id}{resseq + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{1.00:6.2f}{bfac:6.2f}           C\n"
                )
                fh.write(line)
                serial += 1

            fh.write(f"TER   {serial:5d}      GLY {chain_id}{max(n,1):4d}\n")
            fh.write("END\n")

    for k, b in enumerate(chosen):
        valid_idx = mask[b].nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            continue

        gt = xyz_gt[b, valid_idx]
        pred = xyz_pred[b, valid_idx]
        rev_gt = torch.flip(gt, dims=[0])
        pred_conf = confidence[b, valid_idx]

        inp = None
        if xyz_in is not None:
            inp = xyz_in[b, valid_idx]

        sample_dir = out_dir / f"val_sample_{k:03d}_idx_{b:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        _write_single_pdb(
            sample_dir / "gt.pdb",
            gt,
            bfactors=None,
            header=f"VAL SAMPLE {b} GT",
        )
        _write_single_pdb(
            sample_dir / "pred.pdb",
            pred,
            bfactors=pred_conf,
            header=f"VAL SAMPLE {b} PRED",
        )
        _write_single_pdb(
            sample_dir / "rev_gt.pdb",
            rev_gt,
            bfactors=None,
            header=f"VAL SAMPLE {b} REVERSED GT",
        )

        if inp is not None:
            _write_single_pdb(
                sample_dir / "scrambled_in.pdb",
                inp,
                bfactors=None,
                header=f"VAL SAMPLE {b} SCRAMBLED INPUT",
            )

        written_dirs.append(str(sample_dir))

    return written_dirs


def main():
   # PLACEHOLDER
   ...

if __name__ == "__main__":
    main()