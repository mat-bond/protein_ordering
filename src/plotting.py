import matplotlib
matplotlib.use("Agg")  # safe on headless nodes

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


DEVICE = "cuda:0"
ECT_PLOT_CONFIG = {"cmap": "bone", "vmin": -0.5, "vmax": 1.5}
PC_PLOT_CONFIG = {"s": 5, "c": ".5"}
LIGHTRED = [255, 100, 100]


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def _to_numpy(x):
    if type(x) == Tensor:
        return x.cpu().detach().numpy()
    return x


def _mpl_color(color):
    # Accepts e.g. "lightgray", "red", or [255,100,100]
    if isinstance(color, (list, tuple, np.ndarray)) and len(color) == 3:
        c = np.asarray(color, dtype=float)
        if c.max() > 1.0:
            c = c / 255.0
        return tuple(c.tolist())
    return color


def _set_3d_equal(ax, pts):
    # Best-effort equal aspect in 3D
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


def plot_recon_2d(recon_pcs, ref_pcs, jnt_pcs=None, num_pc=5):

    if jnt_pcs is None:
        jnt_pcs = np.hstack([ref_pcs, recon_pcs])
        colors = np.array(ref_pcs.shape[1] * ["blue"] + recon_pcs.shape[1] * ["red"])
    else:
        colors = np.array(jnt_pcs.shape[1] * [0.5])

    fig, axes = plt.subplots(nrows=3, ncols=num_pc, figsize=(num_pc * 2, 3 * 2))

    for recon_pc, ref_pc, jnt_pc, axis in zip(recon_pcs, ref_pcs, jnt_pcs, axes.T):
        recon_pc = rotate(recon_pc.reshape(-1, 2), degrees=-90)
        ref_pc = rotate(ref_pc.reshape(-1, 2), degrees=-90)
        jnt_pc = rotate(jnt_pc.reshape(-1, 2), degrees=-90)

        ax = axis[0]
        ax.scatter(recon_pc[:, 0], recon_pc[:, 1], **PC_PLOT_CONFIG)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect(1)
        ax.axis("off")

        ax = axis[1]
        ax.scatter(ref_pc[:, 0], ref_pc[:, 1], **PC_PLOT_CONFIG)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect(1)
        ax.axis("off")

        ax = axis[2]
        ax.scatter(jnt_pc[:, 0], jnt_pc[:, 1], s=5, c=colors)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect(1)
        ax.axis("off")
    return fig


def plot_recon_3d(
    recon_pcs,
    ref_pcs=None,
    num_pc=5,
    offset=0,
    filename=None,
    point_size=2,
):

    recon_pcs = _to_numpy(recon_pcs)
    ref_pcs = _to_numpy(ref_pcs) if ref_pcs is not None else None

    fig = plt.figure(figsize=(2 * num_pc, 6))

    for col in range(num_pc):
        recon_pts = recon_pcs[col + offset].reshape(-1, 3)

        # First plot: recon
        ax = fig.add_subplot(3, num_pc, 1 + col, projection="3d")
        ax.scatter(
            recon_pts[:, 0],
            recon_pts[:, 1],
            recon_pts[:, 2],
            s=point_size,
            c=_mpl_color("lightgray"),
        )
        _set_3d_equal(ax, recon_pts)
        _style_3d_ax(ax)
        ax.view_init(elev=30, azim=45)

        if ref_pcs is not None:
            ref_pts = ref_pcs[col + offset].reshape(-1, 3)

            # Second plot: ref + recon overlay (as in original row 1)
            ax = fig.add_subplot(3, num_pc, 1 + num_pc + col, projection="3d")
            ax.scatter(
                ref_pts[:, 0],
                ref_pts[:, 1],
                ref_pts[:, 2],
                s=point_size,
                c=_mpl_color(LIGHTRED),
            )
            ax.scatter(
                recon_pts[:, 0],
                recon_pts[:, 1],
                recon_pts[:, 2],
                s=point_size,
                c=_mpl_color("lightgray"),
            )
            _set_3d_equal(ax, np.vstack([ref_pts, recon_pts]))
            _style_3d_ax(ax)
            ax.view_init(elev=30, azim=45)

            # Third plot: ref only (as in original row 2)
            ax = fig.add_subplot(3, num_pc, 1 + 2 * num_pc + col, projection="3d")
            ax.scatter(
                ref_pts[:, 0],
                ref_pts[:, 1],
                ref_pts[:, 2],
                s=2,
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


def plot_grid(pcs, num_pc=5):
    fig = plt.figure(figsize=(2 * num_pc, 2 * num_pc))

    for col in range(num_pc):
        for row in range(num_pc):
            ax = fig.add_subplot(num_pc, num_pc, row * num_pc + col + 1, projection="3d")
            pts = pcs[col + num_pc * row].reshape(-1, 3)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=2,
                c=_mpl_color("lightgray"),
            )
            _set_3d_equal(ax, pts)
            _style_3d_ax(ax)
            ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def plot_graph(x, edge_index, edge_weigths=None, ax=None):

    nodes = [i for i in range(len(x))]
    pos_dict = {i: p for i, p in zip(nodes, x)}

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_index)
    nx.draw_networkx_nodes(nodes, pos=pos_dict, node_size=100, ax=ax)
    for idx, edge in enumerate(edge_index):
        if edge_weigths is not None:
            if edge_weigths[idx] > 0.01:
                nx.draw_networkx_edges(
                    G,
                    pos_dict,
                    [edge],
                    alpha=edge_weigths[idx],
                    width=2,
                    edge_color="b",
                    ax=ax,
                )
        else:
            nx.draw_networkx_edges(
                G,
                pos_dict,
                [edge],
                width=2,
                edge_color="b",
                ax=ax,
            )
    nx.draw_networkx_labels(G, pos_dict, ax=ax)
    ax.set_aspect(1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    return ax


def plot_ect(ect_gt, ect_pred, num_ects=5, filename=None):

    fig, axes = plt.subplots(nrows=2, ncols=num_ects, figsize=(3 * num_ects, 6))
    for ax, gt, pred in zip(axes.T, ect_gt, ect_pred):

        ax[0].imshow(pred.cpu().detach().squeeze().numpy())
        ax[0].axis("off")

        ax[1].imshow(gt.cpu().squeeze().numpy())
        ax[1].axis("off")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def plot_epoch_ect(x, x_gt, layer_truth, ect_pred, ect_truth):

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))

    # Plot predicted graph
    ax = axes[0][0]
    ax.set_title("Prediction")
    x_pred = np.round(F.tanh(layer_pred.x).detach().numpy(), decimals=2)
    edge_index_pred = layer_pred.ei.T.detach().numpy()
    edge_weights_pred = nn.functional.sigmoid(layer_pred.ew.detach()).numpy()
    plot_graph(x_pred, edge_index_pred, edge_weights_pred, ax)

    # Plot ground truth graph
    ax = axes[0][1]
    ax.set_title("Ground Truth")

    x_gt = np.round(layer_truth.x.detach().numpy(), decimals=2)
    edge_index_gt = layer_truth.ei.T.detach().numpy()
    plot_graph(x_gt, edge_index_gt, None, ax)

    ax = axes[1][0]
    ax.imshow(ect_pred[0].detach().squeeze().numpy())
    ax.axis("off")
    ax.set_title("Points")

    ax = axes[1][1]
    ax.imshow(ect_truth[0].squeeze().numpy())
    ax.axis("off")
    ax.set_title("Points")

    ax = axes[2][0]
    ax.imshow(ect_pred[1].detach().squeeze().numpy())
    ax.axis("off")
    ax.set_title("Edges")

    ax = axes[2][1]
    ax.imshow(ect_truth[1].squeeze().numpy())
    ax.axis("off")
    ax.set_title("Edges")

    plt.tight_layout()
    plt.savefig(f"./anim/{epoch//10}.png")


def plot_epoch(x, x_gt, epoch):
    fig = plt.figure(figsize=(12, 4))

    # subplot 0: colored by z
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    points = x_gt.detach().cpu().view(-1, 3).numpy()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c=points[:, 2])
    _set_3d_equal(ax, points)
    _style_3d_ax(ax)
    ax.view_init(elev=20, azim=145)

    # subplot 1: gt (lightblue) + pred (red)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    gt = x_gt.detach().cpu().view(-1, 3).numpy()
    pred = x.reshape(-1, 3).detach().cpu().numpy()
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=5, c=_mpl_color("lightblue"))
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=5, c=_mpl_color("red"))
    _set_3d_equal(ax, np.vstack([gt, pred]))
    _style_3d_ax(ax)
    ax.view_init(elev=20, azim=145)

    # subplot 2: pred colored by z
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    points = x.reshape(-1, 3).detach().cpu().numpy()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c=points[:, 2])
    _set_3d_equal(ax, points)
    _style_3d_ax(ax)
    ax.view_init(elev=20, azim=145)

    plt.tight_layout()
    plt.savefig(f"./img/{epoch}.png", dpi=200)
    plt.close(fig)
