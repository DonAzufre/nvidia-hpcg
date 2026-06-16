#!/usr/bin/env python3
"""
Generate a summary markdown report from per-configuration metrics.json files.

Usage:
    python3 scripts/generate_summary.py <benchmark_data_dir>
"""

import json
import sys
from pathlib import Path


CONFIGS = ["RTX4090", "RTX5080", "DUAL_EQUAL", "DUAL_EQUAL_128"]
CONFIG_LABELS = {
    "RTX4090": "RTX 4090 single (128³)",
    "RTX5080": "RTX 5080 single (128³)",
    "DUAL_EQUAL": "Dual equal (global 128×128×256, rank 128³)",
    "DUAL_EQUAL_128": "Dual equal (global 128³, rank 64³)",
}


def load_metrics(output_dir, config):
    path = Path(output_dir) / config / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(value, unit=""):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}{unit}"
    return f"{value}{unit}"


def ns_to_ms(ns):
    if ns is None:
        return None
    return ns / 1e6


def gpu_metric(m, name):
    """Return overall nsys GPU metric value by name."""
    if not m:
        return None
    nsys = m.get("nsys", {})
    gpu = nsys.get("gpu_metrics", {})
    overall = gpu.get("overall", {})
    return overall.get(name)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <benchmark_data_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = sys.argv[1]
    all_metrics = {cfg: load_metrics(output_dir, cfg) for cfg in CONFIGS}

    lines = []
    lines.append("# HPCG Benchmark Summary")
    lines.append("")
    lines.append(f"Output directory: `{output_dir}`")
    lines.append("")

    # HPCG performance table
    lines.append("## HPCG Performance")
    lines.append("")
    lines.append("| Configuration | GFLOP/s | Historical GFLOP/s | Exec Time (s) | Global Size | Local Size | Iterations | DDOT Max (s) | DDOT Min (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for cfg in CONFIGS:
        m = all_metrics.get(cfg)
        h = m.get("hpcg", {}) if m else {}
        global_size = "×".join(str(h.get(k, "?")) for k in ["global_nx", "global_ny", "global_nz"])
        local_size = "×".join(str(h.get(k, "?")) for k in ["local_nx", "local_ny", "local_nz"])
        lines.append(
            f"| {CONFIG_LABELS[cfg]} | "
            f"{fmt(h.get('gflops_rating'))} | "
            f"{fmt(h.get('hist_rating'))} | "
            f"{fmt(h.get('exec_time'))} | "
            f"{global_size} | "
            f"{local_size} | "
            f"{fmt(h.get('total_iterations'))} | "
            f"{fmt(h.get('ddot_max'))} | "
            f"{fmt(h.get('ddot_min'))} |"
        )

    lines.append("")

    # Speedup analysis
    lines.append("## Speedup Analysis")
    lines.append("")
    m4090 = all_metrics.get("RTX4090", {}).get("hpcg", {})
    m5080 = all_metrics.get("RTX5080", {}).get("hpcg", {})
    m_dual = all_metrics.get("DUAL_EQUAL", {}).get("hpcg", {})
    m_dual_128 = all_metrics.get("DUAL_EQUAL_128", {}).get("hpcg", {})

    g4090 = m4090.get("gflops_rating")
    g5080 = m5080.get("gflops_rating")
    g_dual = m_dual.get("gflops_rating")
    g_dual_128 = m_dual_128.get("gflops_rating")

    if g4090 and g_dual:
        lines.append(f"- Dual equal (rank 128³) vs RTX 4090 single: **{g_dual / g4090:.2f}×**")
    if g4090 and g_dual_128:
        lines.append(f"- Dual equal (global 128³) vs RTX 4090 single: **{g_dual_128 / g4090:.2f}×**")
    if g5080 and g_dual:
        lines.append(f"- Dual equal (rank 128³) vs RTX 5080 single: **{g_dual / g5080:.2f}×**")
    lines.append("")

    # NSYS summary
    lines.append("## Nsight Systems Summary")
    lines.append("")
    lines.append("| Configuration | Total CUDA Time (ms) | Memcpy Time (ms) | Memset Time (ms) | Top Operation |")
    lines.append("|---|---|---|---|---|")

    for cfg in CONFIGS:
        m = all_metrics.get(cfg)
        n = m.get("nsys", {}) if m else {}
        top = n.get("top_operations", [{}])[0]
        op_name = top.get("operation", "N/A")
        # Truncate very long kernel names
        if len(op_name) > 80:
            op_name = op_name[:77] + "..."
        lines.append(
            f"| {CONFIG_LABELS[cfg]} | "
            f"{fmt(ns_to_ms(n.get('total_kernel_and_mem_time_ns')))} | "
            f"{fmt(ns_to_ms(n.get('memcpy_time_ns')))} | "
            f"{fmt(ns_to_ms(n.get('memset_time_ns')))} | "
            f"{op_name} ({top.get('percent', 0):.1f}%) |"
        )

    lines.append("")

    # GPU hardware metrics from nsys
    lines.append("## GPU Hardware Metrics (from Nsight Systems sampling)")
    lines.append("")
    lines.append("| Configuration | SMs Active (%) | SM Issue (%) | GR Active (%) | Compute Warps (%) | DRAM Read (%) | DRAM Write (%) | GPC Clock (MHz) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for cfg in CONFIGS:
        m = all_metrics.get(cfg)
        lines.append(
            f"| {CONFIG_LABELS[cfg]} | "
            f"{fmt(gpu_metric(m, 'SMs Active [Throughput %]'))} | "
            f"{fmt(gpu_metric(m, 'SM Issue [Throughput %]'))} | "
            f"{fmt(gpu_metric(m, 'GR Active [Throughput %]'))} | "
            f"{fmt(gpu_metric(m, 'Compute Warps in Flight [Throughput %]'))} | "
            f"{fmt(gpu_metric(m, 'DRAM Read Bandwidth [Throughput %]'))} | "
            f"{fmt(gpu_metric(m, 'DRAM Write Bandwidth [Throughput %]'))} | "
            f"{fmt(gpu_metric(m, 'GPC Clock Frequency [MHz]'))} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- CUDA device enumeration on this node is reversed relative to `nvidia-smi`:")
    lines.append("  CUDA 0 = RTX 4090, CUDA 1 = RTX 5080.")
    lines.append("- DDOT Max/Min times are only meaningful for dual-GPU configurations and reflect")
    lines.append("  load imbalance between the faster and slower GPU at `MPI_Allreduce` barriers.")
    lines.append("- GPU hardware metrics are sampled by Nsight Systems; values are averages over the")
    lines.append("  profiled window and may differ from peak theoretical values.")
    lines.append("")

    summary_path = Path(output_dir) / "summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
