import torch
from benchmarks.hamiltonian_path import hamiltonian_path
from metrics.loss import masked_kabsch_mse_bidirectional

def permutation_accuracy(pred, true):
    # Backbone direction is ambiguous, so accept forward or reverse
    acc_fwd = (pred == true).float().mean()
    acc_rev = (pred.flip(0) == true).float().mean()

    return max(acc_fwd.item(), acc_rev.item())


def edge_accuracy(pred, true):
    # Treat backbone edges as undirected so reversal does not matter
    pred_edges = {
        tuple(sorted((int(a), int(b))))
        for a, b in zip(pred[:-1], pred[1:])
    }

    true_edges = {
        tuple(sorted((int(a), int(b))))
        for a, b in zip(true[:-1], true[1:])
    }

    return len(pred_edges & true_edges) / max(len(true_edges), 1)

def run_hamiltonian_benchmark(
    pcs_gt: torch.Tensor,
    target_col: torch.Tensor,
    cloud_in: torch.Tensor,
    mask: torch.Tensor,
    n_starts: int,
):
    short_perm_scores = []
    bond_perm_scores = []

    short_edge_scores = []
    bond_edge_scores = []

    xyz_shortest = torch.zeros_like(cloud_in)
    xyz_bond = torch.zeros_like(cloud_in)

    for b in range(cloud_in.shape[0]):

        pred_shortest, pred_bond = hamiltonian_path(
            b=b,
            cloud_in=cloud_in,
            mask=mask,
            n_starts=n_starts,
        )

        valid_idx = mask[b].nonzero(as_tuple=True)[0]
        true_cols = target_col[b, valid_idx]

        short_perm_scores.append(
            permutation_accuracy(pred_shortest, true_cols)
        )
        bond_perm_scores.append(
            permutation_accuracy(pred_bond, true_cols)
        )

        short_edge_scores.append(
            edge_accuracy(pred_shortest, true_cols)
        )
        bond_edge_scores.append(
            edge_accuracy(pred_bond, true_cols)
        )

        # Construct predicted ordered coordinate sequences
        xyz_shortest[b, valid_idx] = cloud_in[b, pred_shortest]
        xyz_bond[b, valid_idx] = cloud_in[b, pred_bond]

    # Same RMSD calculation used by the neural model
    _, short_mse, short_count = masked_kabsch_mse_bidirectional(
        xyz_shortest,
        pcs_gt,
        mask,
    )

    _, bond_mse, bond_count = masked_kabsch_mse_bidirectional(
        xyz_bond,
        pcs_gt,
        mask,
    )

    short_rmsd = (
        100.0
        * torch.sqrt(short_mse[short_count > 0].clamp_min(1e-12))
    )

    bond_rmsd = (
        100.0
        * torch.sqrt(bond_mse[bond_count > 0].clamp_min(1e-12))
    )

    return (
        short_perm_scores,
        bond_perm_scores,
        short_edge_scores,
        bond_edge_scores,
        short_rmsd.detach().cpu().tolist(),
        bond_rmsd.detach().cpu().tolist(),
    )

def run_hamiltonian_ect_benchmark(
    pcs_gt: torch.Tensor,
    cloud_in: torch.Tensor,
    mask: torch.Tensor,
    n_starts: int,
):

    xyz_shortest = torch.zeros_like(cloud_in)
    xyz_bond = torch.zeros_like(cloud_in)

    for b in range(cloud_in.shape[0]):

        pred_shortest, pred_bond = hamiltonian_path(
            b=b,
            cloud_in=cloud_in,
            mask=mask,
            n_starts=n_starts,
        )

        valid_idx = mask[b].nonzero(as_tuple=True)[0]

        # Construct predicted ordered coordinate sequences
        xyz_bond[b, valid_idx] = cloud_in[b, pred_bond]

    # Same RMSD calculation used by the neural model

    _, bond_mse, bond_count = masked_kabsch_mse_bidirectional(
        xyz_bond,
        pcs_gt,
        mask,
    )

    bond_rmsd = (
        100.0
        * torch.sqrt(bond_mse[bond_count > 0].clamp_min(1e-12))
    )

    return bond_rmsd.detach().cpu().tolist()


def correct_up_to_reverse(pred, true):
    return (
        torch.equal(pred.cpu(), true.cpu())
        or torch.equal(pred.cpu(), true.flip(0).cpu())
    )

def main():

    # Coordinates are scaled by 100, so 3.8 Å -> 0.038
    pts_list = [torch.tensor([
        [0.000, 0.0, 0.0],
        [0.038, 0.0, 0.0],
        [0.076, 0.0, 0.0],
        [0.114, 0.0, 0.0],
        [0.152, 0.0, 0.0],
    ]),
    torch.tensor([
        [0.000, 0.0, 0.0],
        [0.037, 0.0, 0.0],
        [0.076, 0.0, 0.0],
        [0.115, 0.0, 0.0],
        [0.153, 0.0, 0.0],
    ])
    ]

    for i,pts in enumerate(pts_list):
        perm = torch.tensor([0, 4, 2, 1, 3])
        cloud_in = pts[perm].unsqueeze(0)

        mask = torch.ones(1,5,dtype=torch.bool)
        true_order = torch.tensor([0, 3, 2, 4, 1])

        pred_shortest, pred_bond = hamiltonian_path(b=0,cloud_in=cloud_in,mask=mask,n_starts=16)

        print("Example: ", i)
        print("shortest correct:", correct_up_to_reverse(pred_shortest, true_order))
        print("shortest: ", pred_shortest)
        print("bond correct:", correct_up_to_reverse(pred_bond, true_order))
        print("bond: ", pred_bond)

if __name__ == "__main__":
    main()