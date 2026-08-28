import torch

def greedy_path(cost,start):
    """
    cost: [N, N] symmetric cost matrix
    returns: tensor of N point indices
    """

    N = cost.shape[0]

    # Track which points have already been visited
    used = torch.zeros(N,dtype=torch.bool,device=cost.device)

    path = []
    current = int(start)

    for _ in range(N):
        path.append(current)
        used[current] = True

        if len(path) == N:
            break

        # Costs from current point to all possible next points
        candidates = cost[current].clone()

        # Do not revisit points already in the path
        candidates[used] = float("inf")

        # Greedily choose the cheapest unused next point
        current = int(candidates.argmin())

    return torch.tensor(path, device=cost.device)

def path_cost(path,cost):
    # Sum costs of all consecutive edges in the path
    return float(cost[path[:-1],path[1:]].sum())


def multi_start_greedy(cost, n_starts=16):
    N = cost.shape[0]

    # Try every start for small graphs, otherwise sample several starts
    if N <= n_starts:
        starts = torch.arange(N, device=cost.device)
    else:
        starts = torch.linspace(
            0, N - 1, n_starts,
            device=cost.device
        ).long()

    best_path = None
    best_cost = float("inf")

    # Keep the lowest-cost greedy path across different starting points
    for start in starts:
        path = greedy_path(cost, int(start))
        score = float(path_cost(path, cost))

        # ## DEBUG
        # print("start:", int(start), "path:", path, "score:", score)

        if score < best_cost:
            best_cost = score
            best_path = path

    return best_path, best_cost


@torch.no_grad()
def hamiltonian_path(b: int, cloud_in: torch.Tensor, mask: torch.Tensor, scale: float = 100, n_starts: int =16):

    # Extract valid points and remember their original padded indices
    valid_idx = mask[b].nonzero(as_tuple=True)[0]
    pts = cloud_in[b, valid_idx]

    # Pairwise C-alpha distances
    d = torch.cdist(pts,pts)
    d_angst = scale*d       # Scale to Å

    # Baseline 1: prefer the shortest possible next edge
    cost_shortest = d_angst.clone()
    cost_shortest.fill_diagonal_(float("inf"))

    # Baseline 2: prefer edges close to the expected ~3.8 Å C-alpha spacing
    cost_bond_length = (d_angst-3.8)**2
    cost_bond_length.fill_diagonal_(float("inf"))

    shortest_path, shortest_cost = multi_start_greedy(cost_shortest,n_starts)
    bond_path, bond_cost = multi_start_greedy(cost_bond_length,n_starts)

    # Convert local point indices back to original cloud columns
    pred_shortest = valid_idx[shortest_path]
    pred_bond = valid_idx[bond_path]

    # ## DEBUG
    # print("valid_idx:", valid_idx)
    # print("pts:", pts)
    # print("distance matrix:\n", d_angst)

    return pred_shortest, pred_bond

