import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_results(path: Path):
    edgece = []
    fiedler_mse = []
    benchmark_38 = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "fiedler_edgece_rmsd_A",
            "fiedler_edgece_mse_rmsd_A",
            "3.8A_hamiltonian_rmsd_A",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} has columns {reader.fieldnames}, "
                f"expected at least {required}"
            )

        for row in reader:
            edgece.append(float(row["fiedler_edgece_rmsd_A"]))
            fiedler_mse.append(
                float(row["fiedler_edgece_mse_rmsd_A"])
            )
            benchmark_38.append(
                float(row["3.8A_hamiltonian_rmsd_A"])
            )

    return (
        np.asarray(edgece),
        np.asarray(fiedler_mse),
        np.asarray(benchmark_38),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/figures/all_methods_per_protein.csv"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/figures/failure_overlap.json"
        ),
    )

    args = parser.parse_args()

    edgece, fiedler_mse, benchmark_38 = load_results(args.input)

    threshold = args.threshold
    n = len(edgece)

    # "Failure" means strictly RMSD > 10 Å,
    # matching the metrics used elsewhere in the project.
    edgece_fail = edgece > threshold
    mse_fail = fiedler_mse > threshold
    benchmark_fail = benchmark_38 > threshold

    # --------------------------------------------------------------
    # Requested overlap quantities
    # --------------------------------------------------------------

    edgece_fails_mse_fixes = edgece_fail & ~mse_fail
    both_fiedler_fail = edgece_fail & mse_fail

    benchmark_fails_fiedler_succeeds = benchmark_fail & ~mse_fail
    fiedler_fails_benchmark_succeeds = mse_fail & ~benchmark_fail

    results = {
        "n_proteins": n,
        "failure_threshold_A": threshold,
        "edgece_fails_mse_fixes": int(
            edgece_fails_mse_fixes.sum()
        ),
        "both_fiedler_variants_fail": int(
            both_fiedler_fail.sum()
        ),
        "3.8A_fails_fiedler_mse_succeeds": int(
            benchmark_fails_fiedler_succeeds.sum()
        ),
        "fiedler_mse_fails_3.8A_succeeds": int(
            fiedler_fails_benchmark_succeeds.sum()
        ),
    }

    # --------------------------------------------------------------
    # Print useful sanity checks
    # --------------------------------------------------------------

    print(f"Number of proteins: {n}")
    print(f"Failure threshold: RMSD > {threshold:g} Å")
    print()

    print("Individual failure counts:")
    print(
        f"  Fiedler EdgeCE:       "
        f"{edgece_fail.sum()} "
        f"({100 * edgece_fail.mean():.2f}%)"
    )
    print(
        f"  Fiedler EdgeCE + MSE: "
        f"{mse_fail.sum()} "
        f"({100 * mse_fail.mean():.2f}%)"
    )
    print(
        f"  3.8 Å Hamiltonian:    "
        f"{benchmark_fail.sum()} "
        f"({100 * benchmark_fail.mean():.2f}%)"
    )

    print()
    print("Failure overlap:")
    print(
        "  EdgeCE fails, MSE fine-tuning fixes: "
        f"{edgece_fails_mse_fixes.sum()}"
    )
    print(
        "  Both Fiedler variants fail:          "
        f"{both_fiedler_fail.sum()}"
    )
    print(
        "  3.8 Å fails, Fiedler+MSE succeeds:   "
        f"{benchmark_fails_fiedler_succeeds.sum()}"
    )
    print(
        "  Fiedler+MSE fails, 3.8 Å succeeds:   "
        f"{fiedler_fails_benchmark_succeeds.sum()}"
    )

    # --------------------------------------------------------------
    # Print copy/paste-ready Markdown
    # --------------------------------------------------------------

    print()
    print("Markdown table:")
    print()
    print("| Outcome | Number of proteins |")
    print("|---|---:|")
    print(
        "| EdgeCE fails, MSE fine-tuning fixes | "
        f"{edgece_fails_mse_fixes.sum()} |"
    )
    print(
        "| Both Fiedler variants fail | "
        f"{both_fiedler_fail.sum()} |"
    )
    print(
        "| 3.8 Å fails, Fiedler+MSE succeeds | "
        f"{benchmark_fails_fiedler_succeeds.sum()} |"
    )
    print(
        "| Fiedler+MSE fails, 3.8 Å succeeds | "
        f"{fiedler_fails_benchmark_succeeds.sum()} |"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()