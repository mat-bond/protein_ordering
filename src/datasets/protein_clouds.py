# datasets/protein_clouds.py
from functools import partial
from pathlib import Path
from typing import Union, Tuple
import pydantic
import torch
from torch.utils.data import DataLoader, Dataset



def validate_dir(path: Union[str, Path], create_if_absent=True, parent=False) -> Path:
    p = Path(path)
    if parent:
        p = p.parent
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(p)
    if not p.exists() and not create_if_absent:
        raise FileNotFoundError(p)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


def masked_center_and_scale(
    ca: torch.Tensor,            # [L,3] float, may contain NaNs
    valid: torch.Tensor,         # [L] bool, True = residue is finite
    eps: float = 1e-8,
    scale_cloud: bool = True,
    uniform_scale: bool = True,
    uniform_scaling_factor: float = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      ca_norm:  [L,3] (NaNs preserved where invalid)
      centroid: [3]
      scale:    scalar tensor
    """
    # Ensure shapes
    assert ca.ndim == 2 and ca.shape[1] == 3
    assert valid.ndim == 1 and valid.shape[0] == ca.shape[0]

    # If too few valid points, bail (caller can skip)
    n_valid = valid.sum().item()
    if n_valid < 2:
        raise ValueError("Not enough valid residues to normalize.")

    # Compute centroid over valid only
    ca_valid = ca[valid]  # [Nv,3]
    centroid = ca_valid.mean(dim=0)  # [3]

    x = ca - centroid.unsqueeze(0)  # [L,3], invalid rows will remain NaN if they were NaN

    if scale_cloud:
        # RMS radius over valid only
        xv = x[valid]  # [Nv,3]
        scale = torch.tensor(uniform_scaling_factor, device=ca.device, dtype=ca.dtype) if uniform_scale else torch.sqrt((xv * xv).sum(dim=1).mean())
    else:
        scale = torch.tensor(1.0, device=ca.device, dtype=ca.dtype)

    scale = torch.clamp(scale, min=eps)
    ca_norm = x / scale

    return ca_norm, centroid, scale


class DataConfig(pydantic.BaseModel):
    module: str = "datasets.protein_clouds"
    root_dir: str = "./data/cath_npz_split"
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True

    seed: int = 14
    L_max: int = 500          # must match modelconfig.L_max
    min_len: int = 128        # your filter window
    max_len: int = 500        # your filter window

    no_scale: bool = False
    scale_by: float = 100.0
    uniform_scale: bool = True

class ProteinTraceDataset(Dataset):
    """
    Each item is:
      ca_norm: [L,3] float (may contain NaNs at invalid residues)
      L: int
      centroid: [3] float
      scale: scalar float
      valid: [L] bool (True where coords are finite)
    """
    def __init__(self, chains):
        self.chains = chains

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, i):
        ca, L, centroid, scale, valid = self.chains[i]
        return ca, L, centroid, scale, valid, i


def ca_list_collate(batch):
    cas, lengths, centroids, scales, valids, idxs = zip(*batch)
    return (
        list(cas),
        torch.tensor(lengths, dtype=torch.long),
        torch.stack(centroids, dim=0),
        torch.stack(scales, dim=0),
        list(valids),
        torch.tensor(idxs, dtype=torch.long),  
    )

def ca_pad_collate(batch, Lmax: int):
    cas, lengths, centroids, scales, valids, idxs = zip(*batch)

    lengths = torch.tensor(lengths, dtype=torch.long)
    B = len(cas)

    ca_pad = torch.full((B, Lmax, 3), float("nan"), dtype=cas[0].dtype)
    valid_mask = torch.zeros((B, Lmax), dtype=torch.bool)
    pad_mask = torch.zeros((B, Lmax), dtype=torch.bool)

    for i, (ca, valid) in enumerate(zip(cas, valids)):
        L = min(ca.shape[0], Lmax)
        ca_pad[i, :L] = ca[:L]
        valid_mask[i, :L] = valid[:L]
        pad_mask[i, :L] = True

    return (
        ca_pad,
        lengths,
        torch.stack(centroids, dim=0),
        torch.stack(scales, dim=0),
        valid_mask,
        pad_mask,
        torch.tensor(idxs, dtype=torch.long),
    )

def _load_split_pt(split_file: Path, cfg: DataConfig):
    """
    Loads split_file: train.pt / validation.pt / test.pt
    Format: list of dicts with keys: name, seq, ca (FloatTensor[L,3], may contain NaNs)
    """
    validate_dir(split_file, create_if_absent=False, parent=True)

    scale_cloud = (not cfg.no_scale)
    uniform_scale = cfg.uniform_scale
    data = torch.load(split_file, map_location="cpu")  # list[dict]
    scale_by = cfg.scale_by

    if scale_cloud and (scale_by == 0):
        raise ValueError("Passed scaling factor equal to zero")
    
    chains = []
    for rec in data:
        ca = rec["ca"]
        if not torch.is_tensor(ca):
            ca = torch.as_tensor(ca)
        ca = ca.to(dtype=torch.float32)  # [L,3]

        if ca.ndim != 2 or ca.shape[1] != 3:
            continue

        # NaN validity mask per residue
        valid = torch.isfinite(ca).all(dim=-1)  # [L]

        L = int(ca.shape[0])
        if not (cfg.min_len <= L <= cfg.max_len):
            continue

        # Normalize using only valid residues (keeps NaNs in-place)
        try:
            ca_norm, centroid, scale = masked_center_and_scale(
                ca=ca, valid=valid, scale_cloud=scale_cloud, 
                uniform_scale=uniform_scale, uniform_scaling_factor=scale_by
            )
        except ValueError:
            # e.g. too few valid points
            continue

        chains.append((ca_norm, L, centroid, scale, valid))

    return chains


def get_all_dataloaders(cfg: DataConfig, dev: bool = False):
    root = Path(cfg.root_dir)

    tr = _load_split_pt(root / "train.pt", cfg)
    vl = _load_split_pt(root / "validation.pt", cfg)
    te = _load_split_pt(root / "test.pt", cfg)

    if dev:
        tr, vl, te = tr[:64], vl[:64], te[:64]

    # Optional: mean/std over all *valid* train residues only
    if len(tr) > 0:
        all_valid = []
        for (ca, L, centroid, scale, valid) in tr:
            all_valid.append(ca[valid])  # [Nv,3]
        all_tr = torch.cat(all_valid, dim=0) if len(all_valid) else torch.empty((0, 3))
        if all_tr.numel() > 0:
            m = all_tr.mean(0, keepdim=True)
            s = all_tr.std(0, keepdim=True).clamp_min(1e-8)
        else:
            m = torch.zeros((1, 3))
            s = torch.ones((1, 3))
    else:
        m = torch.zeros((1, 3))
        s = torch.ones((1, 3))

    collate = partial(ca_pad_collate, Lmax=cfg.L_max)

    train_dl = DataLoader(
        ProteinTraceDataset(tr),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate,
    )
    val_dl = DataLoader(
        ProteinTraceDataset(vl),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate,
    )
    test_dl = DataLoader(
        ProteinTraceDataset(te),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate,
    )

    return train_dl, val_dl, test_dl, m, s



