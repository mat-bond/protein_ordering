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