import numpy as np
from pathlib import Path

def check_dir(split_dir, n=50):
    files = sorted(Path(split_dir).glob("*.npz"))[:n]
    for f in files:
        z = np.load(f)
        ca = z["ca"].astype(np.float32)  # raw
        d = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
        frac_bad = (d > 6.0).mean()
        print(f"{f.name:40s} L={len(ca):4d}  dmean={d.mean():.2f}  dstd={d.std():.2f}  max={d.max():.2f}  bad(>6A)={frac_bad:.3f}")

if __name__ == "__main__":
    check_dir("/work/lpdi/users/bondarev/inner-product-transforms/data/protein100_l128-256/train", n=50)