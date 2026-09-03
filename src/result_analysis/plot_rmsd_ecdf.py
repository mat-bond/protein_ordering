import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_model_rmsd(path: Path) -> dict[str, float]:
    """Load a test_per_protein.csv with columns sample_id,rmsd_A."""
    data = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = {"sample_id", "rmsd_A"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} has columns {reader.fieldnames}, expected {required}"
            )

        for row in reader:
            sample_id = row["sample_id"]

            if sample_id in data:
                raise ValueError(f"Duplicate sample_id={sample_id} in {path}")

            data[sample_id] = float(row["rmsd_A"])

    return data


def load_benchmarks(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """
    Load benchmark_per_protein.csv.

    Returns:
        nearest-neighbor RMSD by sample_id
        3.8 Å Hamiltonian RMSD by sample_id
    """
    nearest = {}
    bond = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "sample_id",
            "nearest_rmsd_A",
            "3.8A_rmsd_A",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} has columns {reader.fieldnames}, expected {required}"
            )

        for row in reader:
            sample_id = row["sample_id"]

            if sample_id in nearest:
                raise ValueError(f"Duplicate sample_id={sample_id} in {path}")

            nearest[sample_id] = float(row["nearest_rmsd_A"])
            bond[sample_id] = float(row["3.8A_rmsd_A"])

    return nearest, bond


def sample_sort_key(sample_id: str):
    """Sort numeric sample IDs numerically when possible."""
    try:
        return (0, int(sample_id))
    except ValueError:
        return (1, sample_id)


def print_summary(name: str, values: np.ndarray):
    print(name)
    print(f"  n:                  {len(values)}")
    print(f"  mean RMSD:          {np.mean(values):.6f} Å")
    print(f"  median RMSD:        {np.median(values):.6f} Å")
    print(f"  p90 RMSD:           {np.percentile(values, 90):.6f} Å")
    print(f"  p95 RMSD:           {np.percentile(values, 95):.6f} Å")
    print(f"  RMSD < 0.1 Å:       {100 * np.mean(values < 0.1):.2f}%")
    print(f"  RMSD > 10 Å:        {100 * np.mean(values > 10.0):.2f}%")
    print()


def plot_ecdf(
    methods: dict[str, np.ndarray],
    filename: Path,
    log_x: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for name, values in methods.items():
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)

        ax.step(
            x,
            y,
            where="post",
            label=name,
            linewidth=2,
        )

    # Thresholds used in the report.
    ax.axvline(
        0.1,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )
    ax.axvline(
        10.0,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel("Per-protein RMSD (Å)")
    ax.set_ylabel("Fraction of proteins with RMSD ≤ x")
    ax.set_ylim(0.0, 1.01)

    title = "Test-set per-protein RMSD ECDF"
    if log_x:
        title += " (log scale)"
    ax.set_title(title)

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    args = parser.parse_args()

    results = args.results_dir

    # ------------------------------------------------------------------
    # Exact result files produced by the current training/evaluation code.
    # ------------------------------------------------------------------

    fiedler_edge_path = (
        results
        / "fiedler_train_ece_fiedler_e2e_test"
        / "test_per_protein.csv"
    )

    fiedler_finetune_path = (
        results
        / "fiedler_train_ece_fiedler_e2e_finetune_test"
        / "test_per_protein.csv"
    )

    rmsd_path = (
        results
        / "rmsd_train_perm_gt_only_curriculum"
        / "test_per_protein.csv"
    )

    benchmark_path = (
        results
        / "rmsd_train_perm_gt_only_curriculum"
        / "benchmark_per_protein.csv"
    )

    for path in [
        fiedler_edge_path,
        fiedler_finetune_path,
        rmsd_path,
        benchmark_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    # ------------------------------------------------------------------
    # Load per-protein results.
    # ------------------------------------------------------------------

    fiedler_edge = load_model_rmsd(fiedler_edge_path)
    fiedler_finetune = load_model_rmsd(fiedler_finetune_path)
    rmsd_decoder = load_model_rmsd(rmsd_path)
    nearest, bond_38 = load_benchmarks(benchmark_path)

    methods = {
        "Nearest neighbor": nearest,
        "3.8 Å Hamiltonian": bond_38,
        "Fiedler EdgeCE": fiedler_edge,
        "Fiedler EdgeCE + MSE": fiedler_finetune,
        "RMSD decoder PermCE": rmsd_decoder,
    }

    # ------------------------------------------------------------------
    # Verify that every file contains exactly the same proteins.
    # ------------------------------------------------------------------

    reference_name = "Nearest neighbor"
    reference_ids = set(methods[reference_name])

    print("Per-file protein counts:")
    for name, values in methods.items():
        ids = set(values)

        missing = reference_ids - ids
        extra = ids - reference_ids

        print(
            f"  {name:24s}: {len(ids):5d} "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

        if ids != reference_ids:
            raise RuntimeError(
                f"{name} does not contain the same sample IDs as "
                f"{reference_name}.\n"
                f"Missing examples: {sorted(missing)[:10]}\n"
                f"Extra examples: {sorted(extra)[:10]}"
            )

    print(f"\nAll methods match on {len(reference_ids)} proteins.\n")

    # ------------------------------------------------------------------
    # Put every method into exactly the same sample order.
    # ------------------------------------------------------------------

    sample_ids = sorted(reference_ids, key=sample_sort_key)

    arrays = {
        name: np.asarray(
            [values[sample_id] for sample_id in sample_ids],
            dtype=np.float64,
        )
        for name, values in methods.items()
    }

    # ------------------------------------------------------------------
    # Sanity-check summary statistics.
    # These should reproduce the numbers in your README table.
    # ------------------------------------------------------------------

    print("Summary statistics:\n")

    for name, values in arrays.items():
        print_summary(name, values)

    # ------------------------------------------------------------------
    # Save one merged file for subsequent paired analyses.
    # ------------------------------------------------------------------

    figures_dir = results / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    merged_path = figures_dir / "all_methods_per_protein.csv"

    with merged_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "sample_id",
                "nearest_neighbor_rmsd_A",
                "3.8A_hamiltonian_rmsd_A",
                "fiedler_edgece_rmsd_A",
                "fiedler_edgece_mse_rmsd_A",
                "rmsd_decoder_permce_rmsd_A",
            ]
        )

        for i, sample_id in enumerate(sample_ids):
            writer.writerow(
                [
                    sample_id,
                    arrays["Nearest neighbor"][i],
                    arrays["3.8 Å Hamiltonian"][i],
                    arrays["Fiedler EdgeCE"][i],
                    arrays["Fiedler EdgeCE + MSE"][i],
                    arrays["RMSD decoder PermCE"][i],
                ]
            )

    # ------------------------------------------------------------------
    # ECDF figures.
    # ------------------------------------------------------------------

    linear_path = figures_dir / "rmsd_ecdf.png"
    log_path = figures_dir / "rmsd_ecdf_logx.png"

    plot_ecdf(
        arrays,
        filename=linear_path,
        log_x=False,
    )

    plot_ecdf(
        arrays,
        filename=log_path,
        log_x=True,
    )

    print("Saved:")
    print(f"  {linear_path}")
    print(f"  {log_path}")
    print(f"  {merged_path}")


if __name__ == "__main__":
    main()