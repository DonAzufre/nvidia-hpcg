#!/usr/bin/env python3
"""
Generate figures and a JSON summary from HPCG benchmark metrics.

Usage:
    python report/scripts/generate_figures.py \
        benchmark_data/2026-06-17_00-54-42 \
        report/figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CONFIG_ORDER = ["RTX4090", "RTX5080", "DUAL_EQUAL", "DUAL_EQUAL_128", "DUAL_HET"]
CONFIG_LABELS = {
    "RTX4090": "RTX 4090\n(128³)",
    "RTX5080": "RTX 5080\n(128³)",
    "DUAL_EQUAL": "DUAL_EQUAL\n(128³/rank)",
    "DUAL_EQUAL_128": "DUAL_EQUAL_128\n(64³/rank)",
    "DUAL_HET": "DUAL_HET\n(160+96)",
}


def load_metrics(benchmark_dir: Path) -> dict[str, dict]:
    data: dict[str, dict] = {}
    for cfg in CONFIG_ORDER:
        path = benchmark_dir / cfg / "metrics.json"
        if path.exists():
            with open(path) as f:
                data[cfg] = json.load(f)
        else:
            raise FileNotFoundError(path)
    return data


def build_summary(data: dict[str, dict]) -> dict:
    summary = {}
    for cfg in CONFIG_ORDER:
        hpcg = data[cfg]["hpcg"]
        nsys = data[cfg].get("nsys", {})
        gpu = nsys.get("gpu_metrics", {}).get("overall", {})
        summary[cfg] = {
            "gflops_rating": hpcg.get("gflops_rating"),
            "hist_rating": hpcg.get("hist_rating"),
            "exec_time": hpcg.get("exec_time"),
            "ddot_max": hpcg.get("ddot_max"),
            "ddot_min": hpcg.get("ddot_min"),
            "global_size": [hpcg.get("global_nx"), hpcg.get("global_ny"), hpcg.get("global_nz")],
            "local_size": [hpcg.get("local_nx"), hpcg.get("local_ny"), hpcg.get("local_nz")],
            "total_iterations": hpcg.get("total_iterations"),
            "total_kernel_and_mem_time_ms": nsys.get("total_kernel_and_mem_time_ns", 0) / 1e6,
            "memcpy_time_ms": nsys.get("memcpy_time_ns", 0) / 1e6,
            "memset_time_ms": nsys.get("memset_time_ns", 0) / 1e6,
            "sm_active_pct": gpu.get("SMs Active [Throughput %]"),
            "sm_issue_pct": gpu.get("SM Issue [Throughput %]"),
            "gr_active_pct": gpu.get("GR Active [Throughput %]"),
            "compute_warps_pct": gpu.get("Compute Warps in Flight [Throughput %]"),
            "dram_read_pct": gpu.get("DRAM Read Bandwidth [Throughput %]"),
            "dram_write_pct": gpu.get("DRAM Write Bandwidth [Throughput %]"),
            "gpc_clock_mhz": gpu.get("GPC Clock Frequency [MHz]"),
        }
    return summary


def plot_gflops_comparison(summary: dict, output_dir: Path) -> None:
    labels = [CONFIG_LABELS[c] for c in CONFIG_ORDER]
    values = [summary[c]["gflops_rating"] for c in CONFIG_ORDER]
    colors = sns.color_palette("husl", n_colors=len(CONFIG_ORDER))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("GFLOP/s")
    ax.set_title("HPCG 性能对比：单卡与双卡配置")
    ax.set_ylim(0, max(values) * 1.15)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "gflops_comparison.png", dpi=300)
    plt.close(fig)


def plot_ddot_imbalance(summary: dict, output_dir: Path) -> None:
    dual_configs = ["DUAL_EQUAL", "DUAL_EQUAL_128", "DUAL_HET"]
    labels = [c.replace("_", "\n") for c in dual_configs]
    max_vals = [summary[c]["ddot_max"] for c in dual_configs]
    min_vals = [summary[c]["ddot_min"] for c in dual_configs]

    x = np.arange(len(dual_configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width / 2, max_vals, width, label="Max DDOT Allreduce (s)", color="coral")
    bars2 = ax.bar(x + width / 2, min_vals, width, label="Min DDOT Allreduce (s)", color="skyblue")
    ax.set_ylabel("Time (s)")
    ax.set_title("双卡配置 DDOT MPI_Allreduce 负载不均衡度")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height * 1.15,
                f"{height:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "ddot_imbalance.png", dpi=300)
    plt.close(fig)


def plot_nsys_breakdown(summary: dict, output_dir: Path) -> None:
    labels = [CONFIG_LABELS[c] for c in CONFIG_ORDER]
    kernel = [
        summary[c]["total_kernel_and_mem_time_ms"]
        - summary[c]["memcpy_time_ms"]
        - summary[c]["memset_time_ms"]
        for c in CONFIG_ORDER
    ]
    memcpy = [summary[c]["memcpy_time_ms"] for c in CONFIG_ORDER]
    memset = [summary[c]["memset_time_ms"] for c in CONFIG_ORDER]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, kernel, label="CUDA kernel", color="steelblue")
    ax.bar(labels, memcpy, bottom=kernel, label="Memcpy", color="darkorange")
    ax.bar(labels, memset, bottom=np.array(kernel) + np.array(memcpy), label="Memset", color="mediumseagreen")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Nsight Systems 时间分解：Kernel / Memcpy / Memset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "nsys_breakdown.png", dpi=300)
    plt.close(fig)


def plot_gpu_metrics_bar(summary: dict, output_dir: Path) -> None:
    metrics = [
        ("SMs Active [Throughput %]", "sm_active_pct", "SMs Active (%)"),
        ("SM Issue [Throughput %]", "sm_issue_pct", "SM Issue (%)"),
        ("GR Active [Throughput %]", "gr_active_pct", "GR Active (%)"),
        ("Compute Warps in Flight [Throughput %]", "compute_warps_pct", "Compute Warps (%)"),
        ("DRAM Read Bandwidth [Throughput %]", "dram_read_pct", "DRAM Read (%)"),
        ("DRAM Write Bandwidth [Throughput %]", "dram_write_pct", "DRAM Write (%)"),
    ]
    short_labels = [m[2] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = sns.color_palette("tab10", n_colors=len(CONFIG_ORDER))
    for i, cfg in enumerate(CONFIG_ORDER):
        vals = [summary[cfg][m[1]] for m in metrics]
        offset = width * (i - len(CONFIG_ORDER) / 2 + 0.5)
        ax.bar(x + offset, vals, width, label=CONFIG_LABELS[cfg], color=colors[i])

    ax.set_ylabel("Throughput / Utilization (%)")
    ax.set_title("GPU 硬件指标对比（Nsight Systems 采样均值）")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=15, ha="right")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "gpu_metrics_bar.png", dpi=300)
    plt.close(fig)


def plot_speedup_comparison(summary: dict, output_dir: Path) -> None:
    base_4090 = summary["RTX4090"]["gflops_rating"]
    base_5080 = summary["RTX5080"]["gflops_rating"]
    labels = [CONFIG_LABELS[c] for c in CONFIG_ORDER]
    speedup_4090 = [summary[c]["gflops_rating"] / base_4090 for c in CONFIG_ORDER]
    speedup_5080 = [summary[c]["gflops_rating"] / base_5080 for c in CONFIG_ORDER]

    x = np.arange(len(CONFIG_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, speedup_4090, width, label="相对 RTX 4090 单卡", color="royalblue")
    ax.bar(x + width / 2, speedup_5080, width, label="相对 RTX 5080 单卡", color="seagreen")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="基线 1.0×")
    ax.set_ylabel("Speedup")
    ax.set_title("加速比对比")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for i, (v1, v2) in enumerate(zip(speedup_4090, speedup_5080)):
        ax.text(i - width / 2, v1 + 0.03, f"{v1:.2f}×", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, v2 + 0.03, f"{v2:.2f}×", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "speedup_comparison.png", dpi=300)
    plt.close(fig)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <benchmark_data_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)

    benchmark_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_metrics(benchmark_dir)
    summary = build_summary(data)

    with open(output_dir / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {output_dir / 'data_summary.json'}")

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10

    # Prefer a CJK-capable font so Chinese titles/labels render correctly.
    # Matplotlib exposes the Noto CJK TTC fonts under the JP variant name.
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for cjk_font in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC", "Noto Serif CJK JP"):
        if cjk_font in available:
            plt.rcParams["font.family"] = cjk_font
            break
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plot_gflops_comparison(summary, output_dir)
    plot_ddot_imbalance(summary, output_dir)
    plot_nsys_breakdown(summary, output_dir)
    plot_gpu_metrics_bar(summary, output_dir)
    plot_speedup_comparison(summary, output_dir)

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
