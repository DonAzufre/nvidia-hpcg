#!/usr/bin/env python3
"""Generate HPCG benchmark figures for the project report."""

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

CONFIG_ORDER = ["RTX4090", "RTX5080", "DUAL_EQUAL", "DUAL_EQUAL_128", "DUAL_HET"]
CONFIG_LABELS = {
    "RTX4090": "RTX 4090\n(128³)",
    "RTX5080": "RTX 5080\n(128³)",
    "DUAL_EQUAL": "DUAL_EQUAL\n(128³×2)",
    "DUAL_EQUAL_128": "DUAL_EQUAL_128\n(64³×2)",
    "DUAL_HET": "DUAL_HET\n(160/96)",
}

DUAL_CONFIGS = ["DUAL_EQUAL", "DUAL_EQUAL_128", "DUAL_HET"]

DATA_DIR = Path(__file__).resolve().parents[2] / "benchmark_data" / "2026-06-17_00-54-42"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def load_metrics():
    data = {}
    for cfg in CONFIG_ORDER:
        path = DATA_DIR / cfg / "metrics.json"
        with open(path, "r") as f:
            data[cfg] = json.load(f)
    return data


def active_gpu_metrics(cfg_data):
    """For single-GPU configs return the active GPU; for dual configs return overall."""
    per_gpu = cfg_data["nsys"]["gpu_metrics"]["per_gpu"]
    overall = cfg_data["nsys"]["gpu_metrics"]["overall"]
    if len(per_gpu) == 1:
        # Single GPU: only one entry exists.
        return list(per_gpu.values())[0]
    # Dual GPU: pick the GPU with the higher SMs Active as the representative active view,
    # but for dual configs the relevant comparison is the overall average.
    if cfg_data["config"].startswith("DUAL"):
        return overall
    # Fallback for single GPU if both entries present: pick active one.
    return max(per_gpu.values(), key=lambda g: g.get("SMs Active [Throughput %]", 0))


def make_gflops_comparison(data):
    gflops = [data[c]["hpcg"]["gflops_rating"] for c in CONFIG_ORDER]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.bar([CONFIG_LABELS[c] for c in CONFIG_ORDER], gflops, color=colors, edgecolor="black")
    ax.set_ylabel("GFLOP/s", fontsize=12)
    ax.set_title("HPCG Performance Comparison (Five Configurations)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(gflops) * 1.15)
    for bar, val in zip(bars, gflops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gflops_comparison.png", dpi=300)
    plt.close(fig)


def make_ddot_imbalance(data):
    max_vals = [data[c]["hpcg"]["ddot_max"] for c in DUAL_CONFIGS]
    min_vals = [data[c]["hpcg"]["ddot_min"] for c in DUAL_CONFIGS]
    labels = [CONFIG_LABELS[c] for c in DUAL_CONFIGS]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bars1 = ax.bar(x - width / 2, max_vals, width, label="DDOT Allreduce Max (s)", color="#e41a1c", edgecolor="black")
    bars2 = ax.bar(x + width / 2, min_vals, width, label="DDOT Allreduce Min (s)", color="#377eb8", edgecolor="black")

    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("DDOT MPI_Allreduce Imbalance (Dual-GPU Configurations)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ddot_imbalance.png", dpi=300)
    plt.close(fig)


def make_nsys_breakdown(data):
    configs = CONFIG_ORDER
    labels = [CONFIG_LABELS[c] for c in configs]

    kernel_times = []
    memcpy_times = []
    memset_times = []
    for c in configs:
        nsys = data[c]["nsys"]
        total = nsys["total_kernel_and_mem_time_ns"]
        memcpy = nsys["memcpy_time_ns"]
        memset = nsys["memset_time_ns"]
        kernel = max(total - memcpy - memset, 0)
        kernel_times.append(kernel / 1e6)
        memcpy_times.append(memcpy / 1e6)
        memset_times.append(memset / 1e6)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(labels, kernel_times, label="CUDA Kernel", color="#1f77b4", edgecolor="black")
    ax.bar(labels, memcpy_times, bottom=kernel_times, label="Memcpy", color="#ff7f0e", edgecolor="black")
    ax.bar(labels, memset_times, bottom=np.array(kernel_times) + np.array(memcpy_times), label="Memset", color="#2ca02c", edgecolor="black")

    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Nsight Systems Breakdown: Kernel vs Memcpy vs Memset", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "nsys_breakdown.png", dpi=300)
    plt.close(fig)


def make_gpu_metrics_bar(data):
    configs = CONFIG_ORDER
    labels = [CONFIG_LABELS[c] for c in configs]
    metrics = [
        "SMs Active [Throughput %]",
        "SM Issue [Throughput %]",
        "GR Active [Throughput %]",
        "Compute Warps in Flight [Throughput %]",
        "DRAM Read Bandwidth [Throughput %]",
        "DRAM Write Bandwidth [Throughput %]",
    ]
    short_names = ["SMs Active", "SM Issue", "GR Active", "Compute Warps", "DRAM Read", "DRAM Write"]

    values = {m: [] for m in metrics}
    for c in configs:
        g = active_gpu_metrics(data[c])
        for m in metrics:
            values[m].append(g.get(m, 0))

    x = np.arange(len(labels))
    width = 0.12
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    for i, (m, name, color) in enumerate(zip(metrics, short_names, colors)):
        ax.bar(x + i * width, values[m], width, label=name, color=color, edgecolor="black")

    ax.set_ylabel("Throughput (%)", fontsize=12)
    ax.set_title("GPU Hardware Metrics by Configuration", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gpu_metrics_bar.png", dpi=300)
    plt.close(fig)


def make_speedup_comparison(data):
    base_4090 = data["RTX4090"]["hpcg"]["gflops_rating"]
    base_5080 = data["RTX5080"]["hpcg"]["gflops_rating"]

    speedup_4090 = [data[c]["hpcg"]["gflops_rating"] / base_4090 for c in CONFIG_ORDER]
    speedup_5080 = [data[c]["hpcg"]["gflops_rating"] / base_5080 for c in CONFIG_ORDER]

    labels = [CONFIG_LABELS[c] for c in CONFIG_ORDER]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars1 = ax.bar(x - width / 2, speedup_4090, width, label="Speedup vs RTX 4090 single", color="#1f77b4", edgecolor="black")
    bars2 = ax.bar(x + width / 2, speedup_5080, width, label="Speedup vs RTX 5080 single", color="#ff7f0e", edgecolor="black")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_title("Speedup Relative to Single-GPU Baselines", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}×", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}×", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "speedup_comparison.png", dpi=300)
    plt.close(fig)


def write_summary(data):
    summary = {
        "configurations": {},
        "speedups": {},
        "notes": "Auto-generated from benchmark_data/2026-06-17_00-54-42/*/metrics.json",
    }

    base_4090 = data["RTX4090"]["hpcg"]["gflops_rating"]
    base_5080 = data["RTX5080"]["hpcg"]["gflops_rating"]

    for c in CONFIG_ORDER:
        h = data[c]["hpcg"]
        n = data[c]["nsys"]
        g = active_gpu_metrics(data[c])
        total = n["total_kernel_and_mem_time_ns"]
        memcpy = n["memcpy_time_ns"]
        memset = n["memset_time_ns"]
        kernel = max(total - memcpy - memset, 0)

        summary["configurations"][c] = {
            "gflops_rating": h["gflops_rating"],
            "hist_rating": h["hist_rating"],
            "exec_time": h["exec_time"],
            "global_size": [h["global_nx"], h["global_ny"], h["global_nz"]],
            "local_size": [h["local_nx"], h["local_ny"], h["local_nz"]],
            "iterations": h["total_iterations"],
            "ddot_max": h.get("ddot_max"),
            "ddot_min": h.get("ddot_min"),
            "nsys_kernel_ms": kernel / 1e6,
            "nsys_memcpy_ms": memcpy / 1e6,
            "nsys_memset_ms": memset / 1e6,
            "sm_active_pct": g.get("SMs Active [Throughput %]"),
            "sm_issue_pct": g.get("SM Issue [Throughput %]"),
            "gr_active_pct": g.get("GR Active [Throughput %]"),
            "compute_warps_pct": g.get("Compute Warps in Flight [Throughput %]"),
            "dram_read_pct": g.get("DRAM Read Bandwidth [Throughput %]"),
            "dram_write_pct": g.get("DRAM Write Bandwidth [Throughput %]"),
            "gpc_clock_mhz": g.get("GPC Clock Frequency [MHz]"),
        }

    for c in CONFIG_ORDER:
        summary["speedups"][c] = {
            "vs_rtx4090_single": data[c]["hpcg"]["gflops_rating"] / base_4090,
            "vs_rtx5080_single": data[c]["hpcg"]["gflops_rating"] / base_5080,
        }

    with open(FIG_DIR / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_metrics()

    make_gflops_comparison(data)
    make_ddot_imbalance(data)
    make_nsys_breakdown(data)
    make_gpu_metrics_bar(data)
    make_speedup_comparison(data)
    write_summary(data)

    print("Figures saved to:", FIG_DIR)
    for name in [
        "gflops_comparison.png",
        "ddot_imbalance.png",
        "nsys_breakdown.png",
        "gpu_metrics_bar.png",
        "speedup_comparison.png",
        "data_summary.json",
    ]:
        print("  -", name)


if __name__ == "__main__":
    main()
