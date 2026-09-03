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
            "results/figures/fiedler_finetuned_vs_3.8A_scatter.png"
        ),
    )

    parser.add_argument(
        "--max-rmsd",
        type=float,
        default=40.0,
    )

    args = parser.parse_args()

    benchmark, fiedler = load_paired_results(args.input)

    if len(benchmark) != len(fiedler):
        raise RuntimeError("Paired arrays have different lengths")

    n = len(benchmark)

    # --------------------------------------------------------------
    # Paired statistics
    # --------------------------------------------------------------

    # Lower RMSD is better.
    fiedler_wins = fiedler < benchmark
    benchmark_wins = benchmark < fiedler
    ties = np.isclose(fiedler, benchmark, atol=1e-8)

    delta = fiedler - benchmark

    print(f"Number of proteins: {n}")
    print()

    print(
        "Fiedler better:      "
        f"{np.sum(fiedler_wins)} / {n} "
        f"({100 * np.mean(fiedler_wins):.2f}%)"
    )

    print(
        "3.8 Å better:        "
        f"{np.sum(benchmark_wins)} / {n} "
        f"({100 * np.mean(benchmark_wins):.2f}%)"
    )

    print(
        "Approximately tied:  "
        f"{np.sum(ties)} / {n} "
        f"({100 * np.mean(ties):.2f}%)"
    )

    print()
    print(
        f"Mean paired Δ RMSD "
        f"(Fiedler - 3.8 Å): {np.mean(delta):.4f} Å"
    )
    print(
        f"Median paired Δ RMSD "
        f"(Fiedler - 3.8 Å): {np.median(delta):.4f} Å"
    )

    # Useful failure transitions.
    benchmark_failure = benchmark > 10.0
    fiedler_failure = fiedler > 10.0

    fixed_by_fiedler = benchmark_failure & ~fiedler_failure
    broken_by_fiedler = ~benchmark_failure & fiedler_failure

    print()
    print(
        "3.8 Å >10 Å, Fiedler <=10 Å: "
        f"{np.sum(fixed_by_fiedler)} "
        f"({100 * np.mean(fixed_by_fiedler):.2f}%)"
    )

    print(
        "3.8 Å <=10 Å, Fiedler >10 Å: "
        f"{np.sum(broken_by_fiedler)} "
        f"({100 * np.mean(broken_by_fiedler):.2f}%)"
    )

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.scatter(
        benchmark,
        fiedler,
        alpha=0.45,
        s=24,
    )

    # y = x: equal performance
    limit = args.max_rmsd

    ax.plot(
        [0, limit],
        [0, limit],
        linestyle="--",
        linewidth=1.5,
        label="Equal RMSD",
    )

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("3.8 Å Hamiltonian RMSD (Å)")
    ax.set_ylabel("Fiedler EdgeCE + MSE RMSD (Å)")

    ax.set_title(
        "Per-protein RMSD: Fiedler + MSE vs. 3.8 Å baseline"
    )

    ax.grid(alpha=0.25)
    ax.legend()

    annotation = (
        f"Fiedler lower RMSD: "
        f"{100 * np.mean(fiedler_wins):.1f}%\n"
        f"3.8 Å lower RMSD: "
        f"{100 * np.mean(benchmark_wins):.1f}%\n"
        f"Mean Δ: {np.mean(delta):+.2f} Å"
    )

    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round",
            "alpha": 0.8,
        },
    )

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