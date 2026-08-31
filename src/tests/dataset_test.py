from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from datasets.protein_clouds import DataConfig, _load_split_pt

def _graph_degree_stats_for_chain(
    xv: torch.Tensor,              # [Nv, 3], already valid-only
    cutoff: float,
    max_num_neighbors: Optional[int] = None,
) -> dict:
    """
    Computes raw radius-graph degree stats for one chain using torch.cdist.
    Self-edges are excluded.

    Returns dict with node-level raw degrees and truncation indicators.
    """
    Nv = xv.shape[0]
    if Nv == 0:
        return {
            "raw_degree": torch.empty(0, dtype=torch.long),
            "truncated_degree": torch.empty(0, dtype=torch.long),
            "isolated_mask": torch.empty(0, dtype=torch.bool),
            "truncated_mask": torch.empty(0, dtype=torch.bool),
        }

    if Nv == 1:
        raw_degree = torch.zeros(1, dtype=torch.long)
    else:
        dmat = torch.cdist(xv, xv)  # [Nv, Nv]
        dmat.fill_diagonal_(float("inf"))
        raw_degree = (dmat <= cutoff).sum(dim=-1).to(torch.long)

    if max_num_neighbors is None:
        truncated_degree = raw_degree.clone()
        truncated_mask = torch.zeros_like(raw_degree, dtype=torch.bool)
    else:
        truncated_degree = torch.clamp(raw_degree, max=max_num_neighbors)
        truncated_mask = raw_degree > max_num_neighbors

    isolated_mask = raw_degree == 0

    return {
        "raw_degree": raw_degree,
        "truncated_degree": truncated_degree,
        "isolated_mask": isolated_mask,
        "truncated_mask": truncated_mask,
    }


def test_coordinate_scaling(
    cfg: DataConfig,
    split: str = "train",
    n_show: int = 5,
    cutoff: Optional[float] = None,
    max_num_neighbors: Optional[int] = None,
):
    """
    Checks centering/scaling for one split and, if cutoff is provided,
    also reports radius-graph degree statistics in the scaled coordinates.

    Prints:
      - number of chains
      - min/mean/median/max of post-transform max radius
      - min/mean/median/max of post-transform RMS radius
      - min/mean/median/max of stored scales
      - first few example chains
      - if cutoff is not None:
          * mean / median / p90 / p95 degree
          * fraction of isolated nodes
          * fraction of nodes truncated by max_num_neighbors
    """
    root = Path(cfg.root_dir)
    split_file = root / f"{split}.pt"
    chains = _load_split_pt(split_file, cfg)

    if len(chains) == 0:
        print(f"[{split}] no chains loaded")
        return

    max_radii = []
    rms_radii = []
    scales = []

    all_raw_degrees = []
    all_truncated_degrees = []
    all_isolated_masks = []
    all_truncated_masks = []

    per_chain_mean_deg = []
    per_chain_isolated_frac = []
    per_chain_truncated_frac = []

    for ca_norm, L, centroid, scale, valid in chains:
        xv = ca_norm[valid]
        if xv.numel() == 0:
            continue

        r = torch.linalg.norm(xv, dim=-1)
        max_radii.append(r.max().item())
        rms_radii.append(torch.sqrt((r * r).mean()).item())
        scales.append(scale.item())

        if cutoff is not None:
            g = _graph_degree_stats_for_chain(
                xv,
                cutoff=cutoff,
                max_num_neighbors=max_num_neighbors,
            )
            raw_deg = g["raw_degree"]
            trunc_deg = g["truncated_degree"]
            isolated_mask = g["isolated_mask"]
            truncated_mask = g["truncated_mask"]

            if raw_deg.numel() > 0:
                all_raw_degrees.append(raw_deg)
                all_truncated_degrees.append(trunc_deg)
                all_isolated_masks.append(isolated_mask)
                all_truncated_masks.append(truncated_mask)

                per_chain_mean_deg.append(raw_deg.float().mean().item())
                per_chain_isolated_frac.append(isolated_mask.float().mean().item())
                per_chain_truncated_frac.append(truncated_mask.float().mean().item())

    max_radii = np.asarray(max_radii)
    rms_radii = np.asarray(rms_radii)
    scales = np.asarray(scales)

    print(f"=== Scaling check: {split} ===")
    print(f"num_chains: {len(max_radii)}")

    print(
        "post-transform max radius min/mean/median/max:",
        f"{max_radii.min():.4f} / {max_radii.mean():.4f} / "
        f"{np.median(max_radii):.4f} / {max_radii.max():.4f}"
    )
    print(
        "post-transform RMS radius min/mean/median/max:",
        f"{rms_radii.min():.4f} / {rms_radii.mean():.4f} / "
        f"{np.median(rms_radii):.4f} / {rms_radii.max():.4f}"
    )
    print(
        "stored scale min/mean/median/max:",
        f"{scales.min():.4f} / {scales.mean():.4f} / "
        f"{np.median(scales):.4f} / {scales.max():.4f}"
    )

    print(f"\nFirst {min(n_show, len(chains))} examples:")
    for i, (ca_norm, L, centroid, scale, valid) in enumerate(chains[:n_show]):
        xv = ca_norm[valid]
        r = torch.linalg.norm(xv, dim=-1)

        msg = (
            f"  idx={i:3d}  L={L:4d}  valid={int(valid.sum()):4d}  "
            f"scale={scale.item():8.4f}  "
            f"max_radius={r.max().item():7.4f}  "
            f"rms_radius={torch.sqrt((r * r).mean()).item():7.4f}"
        )

        if cutoff is not None and xv.shape[0] > 0:
            g = _graph_degree_stats_for_chain(
                xv,
                cutoff=cutoff,
                max_num_neighbors=max_num_neighbors,
            )
            raw_deg = g["raw_degree"]
            isolated_frac = g["isolated_mask"].float().mean().item()
            truncated_frac = g["truncated_mask"].float().mean().item()
            msg += (
                f"  mean_deg={raw_deg.float().mean().item():6.2f}"
                f"  iso_frac={isolated_frac:6.3f}"
                f"  trunc_frac={truncated_frac:6.3f}"
            )

        print(msg)

    if cfg.uniform_scale and not cfg.no_scale:
        expected = float(cfg.scale_by)
        unique_scales = np.unique(np.round(scales, 6))
        print("\nUniform-scale check:")
        print(f"  expected scale: {expected}")
        print(f"  unique stored scales (rounded): {unique_scales[:10]}")
        if len(unique_scales) == 1 and abs(unique_scales[0] - expected) < 1e-6:
            print("  PASS: all chains use the expected global scale")
        else:
            print("  WARNING: stored scales are not all equal to cfg.scale_by")

    if cutoff is not None:
        if len(all_raw_degrees) == 0:
            print("\nRadius-graph diagnostics:")
            print("  No valid nodes found for graph statistics.")
            return

        raw_deg = torch.cat(all_raw_degrees, dim=0).cpu().numpy()
        trunc_deg = torch.cat(all_truncated_degrees, dim=0).cpu().numpy()
        isolated = torch.cat(all_isolated_masks, dim=0).cpu().numpy().astype(np.float32)
        truncated = torch.cat(all_truncated_masks, dim=0).cpu().numpy().astype(np.float32)

        print("\nRadius-graph diagnostics:")
        print(f"  cutoff: {cutoff:.4f}")
        print(f"  max_num_neighbors: {max_num_neighbors}")

        print(f"  num_nodes_total: {raw_deg.shape[0]}")
        print(
            "  raw degree mean/median/p90/p95/max:",
            f"{raw_deg.mean():.3f} / {np.median(raw_deg):.3f} / "
            f"{np.percentile(raw_deg, 90):.3f} / {np.percentile(raw_deg, 95):.3f} / "
            f"{raw_deg.max():.3f}"
        )
        print(
            "  truncated degree mean/median/p90/p95/max:",
            f"{trunc_deg.mean():.3f} / {np.median(trunc_deg):.3f} / "
            f"{np.percentile(trunc_deg, 90):.3f} / {np.percentile(trunc_deg, 95):.3f} / "
            f"{trunc_deg.max():.3f}"
        )
        print(f"  fraction isolated nodes: {isolated.mean():.6f}")
        print(f"  fraction truncated nodes: {truncated.mean():.6f}")

        if len(per_chain_mean_deg) > 0:
            per_chain_mean_deg = np.asarray(per_chain_mean_deg)
            per_chain_isolated_frac = np.asarray(per_chain_isolated_frac)
            per_chain_truncated_frac = np.asarray(per_chain_truncated_frac)

            print("\n  Per-chain summaries:")
            print(
                "  mean degree across chains min/mean/median/max:",
                f"{per_chain_mean_deg.min():.3f} / {per_chain_mean_deg.mean():.3f} / "
                f"{np.median(per_chain_mean_deg):.3f} / {per_chain_mean_deg.max():.3f}"
            )
            print(
                "  isolated-node fraction across chains min/mean/median/max:",
                f"{per_chain_isolated_frac.min():.6f} / {per_chain_isolated_frac.mean():.6f} / "
                f"{np.median(per_chain_isolated_frac):.6f} / {per_chain_isolated_frac.max():.6f}"
            )
            print(
                "  truncated-node fraction across chains min/mean/median/max:",
                f"{per_chain_truncated_frac.min():.6f} / {per_chain_truncated_frac.mean():.6f} / "
                f"{np.median(per_chain_truncated_frac):.6f} / {per_chain_truncated_frac.max():.6f}"
            )

            
def sweep_radius_graph_cutoffs(
    cfg: DataConfig,
    split: str = "train",
    cutoffs: Sequence[float] = (0.08, 0.10, 0.12),
    max_num_neighbors: Optional[int] = 64,
):
    """
    Convenience function: run graph diagnostics for several cutoffs on one split.
    Useful for deciding whether 0.10 is too small / too large / about right.
    """
    print(f"\n=== Radius cutoff sweep on {split} ===")
    for cutoff in cutoffs:
        print("\n" + "=" * 72)
        test_coordinate_scaling(
            cfg=cfg,
            split=split,
            n_show=3,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
        )

def main():
    cfg = DataConfig()

    # Full scaling checks for all splits
    for name in ["train", "validation", "test"]:
        test_coordinate_scaling(cfg, split=name, n_show=5)

    # Hyperparameter diagnostics for the train split
    sweep_radius_graph_cutoffs(
        cfg,
        split="train",
        cutoffs=(0.08, 0.10, 0.12),
        max_num_neighbors=64,
    )


if __name__ == "__main__":
    main()