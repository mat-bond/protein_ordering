import numpy as np
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import matplotlib.pyplot as plt
import numpy as np
import torch


LIGHTRED = [255, 100, 100]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _mpl_color(color):
    if isinstance(color, (list, tuple, np.ndarray)) and len(color) == 3:
        c = np.asarray(color, dtype=float)
        if c.max() > 1.0:
            c = c / 255.0
        return tuple(c.tolist())
    return color


def _set_3d_equal(ax, pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    xmid, ymid, zmid = np.mean(x), np.mean(y), np.mean(z)

    r = max(np.ptp(x), np.ptp(y), np.ptp(z)) / 2.0
    if r == 0:
        r = 1.0

    ax.set_xlim(xmid - r, xmid + r)
    ax.set_ylim(ymid - r, ymid + r)
    ax.set_zlim(zmid - r, zmid + r)


def _style_3d_ax(ax):
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

def kabsch_align(P: np.ndarray, Q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Align P to Q with rigid transform (rotation + translation), no scaling.
    P, Q: [N,3]
    Returns: P_aligned [N,3]
    """
    assert P.ndim == 2 and Q.ndim == 2 and P.shape == Q.shape and P.shape[1] == 3
    n = P.shape[0]
    if n < 3:
        # Degenerate: just translate centroids
        Pc = P - P.mean(axis=0, keepdims=True)
        Qc = Q - Q.mean(axis=0, keepdims=True)
        return Pc + Q.mean(axis=0, keepdims=True)

    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    H = Pc.T @ Qc  # [3,3]
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    # reflection correction
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    P_aligned = Pc @ R + Q.mean(axis=0, keepdims=True)
    return P_aligned

def plot_recon_3d(
    recon_pcs,
    ref_pcs=None,
    num_pc=5,
    offset=0,
    filename=None,
    point_size=2,
    align_to_ref=True,
):
    """
    Same behavior as before, but accepts:
      - list/tuple of variable-length [Li,3] tensors/arrays, OR
      - batched arrays/tensors [B,L,3]

    No pre-padding needed.
    If ref_pcs is provided and align_to_ref=True:
      - Kabsch-align recon to ref per protein before plotting
      - Recon-only panel shows aligned recon
      - Overlay panel shows ref + aligned recon
    """

    def _get_pts(seq, i):
        pts = _to_numpy(seq[i + offset]).reshape(-1, 3)
        return pts

    total_available = len(recon_pcs)
    nplot = min(num_pc, max(0, total_available - offset))

    fig = plt.figure(figsize=(2 * nplot, 6))

    for col in range(nplot):
        recon_pts = _get_pts(recon_pcs, col)

        ref_pts = None
        recon_plot = recon_pts

        if ref_pcs is not None:
            ref_pts = _get_pts(ref_pcs, col)

            # Ensure same number of points for Kabsch (trim to common length)
            n = min(recon_pts.shape[0], ref_pts.shape[0])
            recon_trim = recon_pts[:n]
            ref_trim = ref_pts[:n]

            if align_to_ref and n > 0:
                recon_aligned = kabsch_align(recon_trim, ref_trim)
                # If recon/ref had unequal lengths, keep unmatched tail unaligned
                if recon_pts.shape[0] == n:
                    recon_plot = recon_aligned
                else:
                    recon_plot = np.concatenate([recon_aligned, recon_pts[n:]], axis=0)

        # 1) Recon (aligned if ref provided)
        ax = fig.add_subplot(3, nplot, 1 + col, projection="3d")
        ax.scatter(
            recon_plot[:, 0],
            recon_plot[:, 1],
            recon_plot[:, 2],
            s=point_size,
            c=_mpl_color("lightgray"),
        )
        _set_3d_equal(ax, recon_plot)
        _style_3d_ax(ax)
        ax.view_init(elev=30, azim=45)

        if ref_pts is not None:
            # 2) Overlay: ref + recon (aligned)
            ax = fig.add_subplot(3, nplot, 1 + nplot + col, projection="3d")
            ax.scatter(
                ref_pts[:, 0],
                ref_pts[:, 1],
                ref_pts[:, 2],
                s=point_size,
                c=_mpl_color(LIGHTRED),
            )
            ax.scatter(
                recon_plot[:, 0],
                recon_plot[:, 1],
                recon_plot[:, 2],
                s=point_size,
                c=_mpl_color("lightgray"),
            )
            _set_3d_equal(ax, np.vstack([ref_pts, recon_plot]))
            _style_3d_ax(ax)
            ax.view_init(elev=30, azim=45)

            # 3) Ref only
            ax = fig.add_subplot(3, nplot, 1 + 2 * nplot + col, projection="3d")
            ax.scatter(
                ref_pts[:, 0],
                ref_pts[:, 1],
                ref_pts[:, 2],
                s=point_size,
                c=_mpl_color(LIGHTRED),
            )
            _set_3d_equal(ax, ref_pts)
            _style_3d_ax(ax)
            ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return fig

# def plot_recon_3d(
#     recon_pcs,
#     ref_pcs=None,
#     num_pc=5,
#     offset=0,
#     filename=None,
#     point_size=2,
#     align_to_ref=True,   # <-- new: apply Kabsch if ref is provided
# ):
#     """
#     If ref_pcs is provided and align_to_ref=True:
#       - Kabsch-align recon to ref per protein before plotting
#       - Recon-only panel shows aligned recon (so it's in ref frame)
#       - Overlay panel shows ref + aligned recon
#     """
#     recon_pcs = _to_numpy(recon_pcs)
#     ref_pcs = _to_numpy(ref_pcs) if ref_pcs is not None else None

#     fig = plt.figure(figsize=(2 * num_pc, 6))

#     for col in range(num_pc):
#         recon_pts = recon_pcs[col + offset].reshape(-1, 3)

#         ref_pts = None
#         recon_plot = recon_pts

#         if ref_pcs is not None:
#             ref_pts = ref_pcs[col + offset].reshape(-1, 3)

#             # Ensure same number of points for Kabsch (trim to common length)
#             n = min(recon_pts.shape[0], ref_pts.shape[0])
#             recon_trim = recon_pts[:n]
#             ref_trim   = ref_pts[:n]

#             if align_to_ref:
#                 recon_aligned = kabsch_align(recon_trim, ref_trim)
#                 # If recon/ref had unequal lengths, keep unmatched tail unaligned (rare in your usage)
#                 if recon_pts.shape[0] == n:
#                     recon_plot = recon_aligned
#                 else:
#                     recon_plot = np.concatenate([recon_aligned, recon_pts[n:]], axis=0)

#         # 1) Recon (aligned if ref provided)
#         ax = fig.add_subplot(3, num_pc, 1 + col, projection="3d")
#         ax.scatter(
#             recon_plot[:, 0],
#             recon_plot[:, 1],
#             recon_plot[:, 2],
#             s=point_size,
#             c=_mpl_color("lightgray"),
#         )
#         _set_3d_equal(ax, recon_plot)
#         _style_3d_ax(ax)
#         ax.view_init(elev=30, azim=45)

#         if ref_pts is not None:
#             # 2) Overlay: ref + recon (aligned)
#             ax = fig.add_subplot(3, num_pc, 1 + num_pc + col, projection="3d")
#             ax.scatter(
#                 ref_pts[:, 0],
#                 ref_pts[:, 1],
#                 ref_pts[:, 2],
#                 s=point_size,
#                 c=_mpl_color(LIGHTRED),
#             )
#             ax.scatter(
#                 recon_plot[:, 0],
#                 recon_plot[:, 1],
#                 recon_plot[:, 2],
#                 s=point_size,
#                 c=_mpl_color("lightgray"),
#             )
#             _set_3d_equal(ax, np.vstack([ref_pts, recon_plot]))
#             _style_3d_ax(ax)
#             ax.view_init(elev=30, azim=45)

#             # 3) Ref only
#             ax = fig.add_subplot(3, num_pc, 1 + 2 * num_pc + col, projection="3d")
#             ax.scatter(
#                 ref_pts[:, 0],
#                 ref_pts[:, 1],
#                 ref_pts[:, 2],
#                 s=point_size,
#                 c=_mpl_color(LIGHTRED),
#             )
#             _set_3d_equal(ax, ref_pts)
#             _style_3d_ax(ax)
#             ax.view_init(elev=30, azim=45)

#     plt.tight_layout()
#     if filename is not None:
#         plt.savefig(filename, dpi=200)
#         plt.close(fig)
#     else:
#         plt.show()

#     return fig
