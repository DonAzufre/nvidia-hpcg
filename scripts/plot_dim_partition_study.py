#!/usr/bin/env python3
"""
Generate a bar chart from the dimension-partitioning study CSV.

Usage:
    python3 scripts/plot_dim_partition_study.py <csv_path> [output_png]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_path> [output_png]", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_png = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.with_suffix(".png")

    df = pd.read_csv(csv_path)

    # Summarize symmetric and asymmetric performance by dimension
    fig, ax = plt.subplots(figsize=(10, 6))

    dims = ["X", "Y", "Z"]
    equal_vals = []
    asym_vals = []
    for dim in dims:
        equal_row = df[(df["split_type"] == "equal") & (df["dim"] == dim)]
        asym_row = df[(df["split_type"] == "asymmetric") & (df["dim"] == dim)]
        equal_vals.append(float(equal_row["gflops_rating"].values[0]) if not equal_row.empty else 0)
        asym_vals.append(float(asym_row["gflops_rating"].values[0]) if not asym_row.empty else 0)

    x = range(len(dims))
    width = 0.35
    bars1 = ax.bar([i - width / 2 for i in x], equal_vals, width, label="Equal split", color="steelblue")
    bars2 = ax.bar([i + width / 2 for i in x], asym_vals, width, label="Asymmetric (ratio 1.6)", color="coral")

    ax.set_xlabel("Partition dimension")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Dual-GPU HPCG dimension-partitioning study (rt=30, local 128³ base)")
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"Saved {output_png}")


if __name__ == "__main__":
    main()
