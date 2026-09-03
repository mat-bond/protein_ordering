import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_paired_results(path: Path):
    edgece = []
    edgece_mse = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "fiedler_edgece_rmsd_A",
            "fiedler_edgece_mse_rmsd_A",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} has columns {reader.fieldnames}, "
                f"expected at least {required}"
            )

        for row in reader:
            edgece.append(float(row["fiedler_edgece_rmsd_A"]))
            edgece_mse.append(
                float(row["fiedler_edgece_mse_rmsd_A"])
            )

    return (
        np.asarray(edgece, dtype=np.float64),
        np.asarray(edgece_mse, dtype=np.float64),
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
        "--output",
        type=Path,
        default=Path(
            "results/figures/"
            "fiedler_edgece_vs_edgece_mse_scatter.png"
        ),
    )

    args = parser.parse_args()

    before, after = load_paired_results(args.input)

    if len(before) != len(after):
        raise RuntimeError("Paired arrays have different lengths")

    n = len(before)
    delta = after - before

    # Any numerical improvement.
    improved = after < before
    worsened = after > before

    # More practically meaningful changes.
    substantially_improved = delta < -0.1
    substantially_worsened = delta > 0.1
    essentially_same = np.abs(delta) <= 0.1

    print(f"Number of proteins: {n}")
    print()

    print(f"Before mean RMSD: {np.mean(before):.4f} Å")
    print(f"After mean RMSD:  {np.mean(after):.4f} Å")
    print(
        f"Mean paired Δ RMSD (after - before): "
        f"{np.mean(delta):+.4f} Å"
    )
    print(
        f"Median paired Δ RMSD: "
        f"{np.median(delta):+.4f} Å"
    )

    print()
    print(
        f"Lower RMSD after fine-tuning: "
        f"{improved.sum()} / {n} "
        f"({100 * improved.mean():.2f}%)"
    )
    print(
        f"Higher RMSD after fine-tuning: "
        f"{worsened.sum()} / {n} "
        f"({100 * worsened.mean():.2f}%)"
    )

    print()
    print("Using a ±0.1 Å meaningful-change threshold:")
    print(
        f"Improved by >0.1 Å: "
        f"{substantially_improved.sum()} "
        f"({100 * substantially_improved.mean():.2f}%)"
    )
    print(
        f"Changed by ≤0.1 Å: "
        f"{essentially_same.sum()} "
        f"({100 * essentially_same.mean():.2f}%)"
    )
    print(
        f"Worsened by >0.1 Å: "
        f"{substantially_worsened.sum()} "
        f"({100 * substantially_worsened.mean():.2f}%)"
    )

    # --------------------------------------------------------------
    # Catastrophic-failure transitions
    # --------------------------------------------------------------

    before_failure = before > 10.0
    after_failure = after > 10.0

    rescued = before_failure & ~after_failure
    new_failures = ~before_failure & after_failure
    both_fail = before_failure & after_failure

    fully_rescued = before_failure & (after < 0.1)

    print()
    print("Failure transitions:")
    print(
        f">10 Å before -> <=10 Å after: "
        f"{rescued.sum()} "
        f"({100 * rescued.mean():.2f}%)"
    )
    print(
        f">10 Å before -> <0.1 Å after: "
        f"{fully_rescued.sum()} "
        f"({100 * fully_rescued.mean():.2f}%)"
    )
    print(
        f"<=10 Å before -> >10 Å after: "
        f"{new_failures.sum()} "
        f"({100 * new_failures.mean():.2f}%)"
    )
    print(
        f">10 Å under both: "
        f"{both_fail.sum()} "
        f"({100 * both_fail.mean():.2f}%)"
    )

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        before,
        after,
        s=24,
        alpha=0.45,
    )

    # RMSDs have a numerical floor around 1e-4 Å in the evaluation.
    min_limit = 1e-4

    max_value = max(
        np.max(before),
        np.max(after),
    )
    max_limit = 10 ** np.ceil(np.log10(max_value))

    ax.plot(
        [min_limit, max_limit],
        [min_limit, max_limit],
        linestyle="--",
        linewidth=1.5,
        label="No change",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(min_limit, max_limit)
    ax.set_ylim(min_limit, max_limit)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Fiedler EdgeCE RMSD before fine-tuning (Å)")
    ax.set_ylabel("Fiedler EdgeCE + MSE RMSD (Å)")

    ax.set_title(
        "Effect of structural MSE fine-tuning"
    )

    ax.grid(alpha=0.25)
    ax.legend()

    annotation = (
        f"Mean: {np.mean(before):.2f} → "
        f"{np.mean(after):.2f} Å\n"
        f"Mean Δ: {np.mean(delta):+.2f} Å\n"
        f"Improved >0.1 Å: "
        f"{100 * substantially_improved.mean():.1f}%\n"
        f"Worsened >0.1 Å: "
        f"{100 * substantially_worsened.mean():.1f}%"
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

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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