import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_paired_results(path: Path):
    benchmark = []
    fiedler = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "3.8A_hamiltonian_rmsd_A",
            "fiedler_edgece_mse_rmsd_A",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} has columns {reader.fieldnames}, "
                f"expected at least {required}"
            )

        for row in reader:
            benchmark.append(float(row["3.8A_hamiltonian_rmsd_A"]))
            fiedler.append(float(row["fiedler_edgece_mse_rmsd_A"]))

    return (
        np.asarray(benchmark, dtype=np.float64),
        np.asarray(fiedler, dtype=np.float64),
    )


def rmsd_category(values):
    """
    0: near-exact       RMSD < 0.1 Å
    1: intermediate     0.1 Å <= RMSD <= 10 Å
    2: failure          RMSD > 10 Å
    """
    categories = np.ones(len(values), dtype=int)

    categories[values < 0.1] = 0
    categories[values > 10.0] = 2

    return categories


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/figures/all_methods_per_protein.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/figures/"
            "fiedler_vs_3.8A_transition_matrix.png"
        ),
    )

    args = parser.parse_args()

    benchmark, fiedler = load_paired_results(args.input)

    if len(benchmark) != len(fiedler):
        raise RuntimeError("Paired arrays have different lengths")

    n = len(benchmark)

    benchmark_cat = rmsd_category(benchmark)
    fiedler_cat = rmsd_category(fiedler)

    # Rows = 3.8 Å baseline
    # Columns = Fiedler + MSE
    matrix = np.zeros((3, 3), dtype=int)

    for before, after in zip(benchmark_cat, fiedler_cat):
        matrix[before, after] += 1

    labels = [
        "Near-exact\n< 0.1 Å",
        "Intermediate\n0.1–10 Å",
        "Failure\n> 10 Å",
    ]

    print(f"Number of proteins: {n}")
    print()

    print("Transition matrix:")
    print("Rows:    3.8 Å Hamiltonian")
    print("Columns: Fiedler EdgeCE + MSE")
    print()
    print(matrix)
    print()

    # --------------------------------------------------------------
    # Scientifically interesting transitions
    # --------------------------------------------------------------

    baseline_failure = benchmark > 10.0
    fiedler_failure = fiedler > 10.0

    baseline_exact = benchmark < 0.1
    fiedler_exact = fiedler < 0.1

    rescued_failure = baseline_failure & ~fiedler_failure

    fully_rescued = baseline_failure & fiedler_exact

    newly_failed = ~baseline_failure & fiedler_failure

    exact_broken = baseline_exact & ~fiedler_exact

    exact_gained = ~baseline_exact & fiedler_exact

    both_fail = baseline_failure & fiedler_failure

    print("Important transitions:")
    print(
        f"Baseline >10 Å -> Fiedler <=10 Å: "
        f"{rescued_failure.sum()} "
        f"({100 * rescued_failure.mean():.2f}%)"
    )
    print(
        f"Baseline >10 Å -> Fiedler <0.1 Å: "
        f"{fully_rescued.sum()} "
        f"({100 * fully_rescued.mean():.2f}%)"
    )
    print(
        f"Baseline <=10 Å -> Fiedler >10 Å: "
        f"{newly_failed.sum()} "
        f"({100 * newly_failed.mean():.2f}%)"
    )
    print(
        f"Baseline <0.1 Å -> Fiedler >=0.1 Å: "
        f"{exact_broken.sum()} "
        f"({100 * exact_broken.mean():.2f}%)"
    )
    print(
        f"Baseline >=0.1 Å -> Fiedler <0.1 Å: "
        f"{exact_gained.sum()} "
        f"({100 * exact_gained.mean():.2f}%)"
    )
    print(
        f"Both >10 Å: "
        f"{both_fail.sum()} "
        f"({100 * both_fail.mean():.2f}%)"
    )

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(matrix)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Fiedler EdgeCE + MSE")
    ax.set_ylabel("3.8 Å Hamiltonian")

    ax.set_title(
        "Per-protein RMSD regime transitions\n"
        "3.8 Å baseline → Fiedler + MSE"
    )

    # Put counts + percentage of entire test set in each cell.
    for i in range(3):
        for j in range(3):
            count = matrix[i, j]
            percentage = 100.0 * count / n

            ax.text(
                j,
                i,
                f"{count}\n({percentage:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Number of proteins")

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        args.output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()