#!/usr/bin/env python3
"""
Analyze a dual-GPU monitoring CSV produced by monitor_dual_gpu.py and write
a short markdown report.

Usage:
    python3 scripts/analyze_gpu_monitor.py <monitor.csv> [hpcg.log] > summary.md
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean


THROTTLE_BITS = {
    0: "GpuIdle",
    1: "ApplicationsClocksSetting",
    2: "SwPowerCap",
    3: "HwSlowdown",
    4: "HwThermalSlowdown",
    5: "HwPowerBrakeSlowdown",
    6: "SwThermalSlowdown",
    7: "SyncBoost",
}


def _int(value):
    if value is None or value in ("", "N/A", "-", "[Not Supported]", "[Unknown Error]"):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def decode_throttle(hex_str):
    """Decode NVML throttle bitmask to human-readable list."""
    if not hex_str or hex_str.startswith("0x") is False:
        return []
    try:
        value = int(hex_str, 16)
    except ValueError:
        return []
    reasons = []
    for bit, name in THROTTLE_BITS.items():
        if value & (1 << bit):
            reasons.append(name)
    return reasons if reasons else ["None"]


def parse_hpcg_log(log_path):
    """Extract key HPCG metrics from the log, if provided."""
    if not log_path or not Path(log_path).exists():
        return {}
    text = Path(log_path).read_text(errors="replace")

    def find(pattern, cast=float):
        m = re.search(pattern, text)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except Exception:
            return None

    valid = None
    if re.search(r"HPCG result is VALID", text):
        valid = True
    elif re.search(r"HPCG result is INVALID", text):
        valid = False

    return {
        "valid": valid,
        "gflops": find(r"HPCG result is VALID with a GFLOP/s rating of=([0-9.]+)"),
        "exec_time": find(r"Results are valid but execution time \(sec\) is=([0-9.]+)"),
        "ddot_max": find(r"Max DDOT MPI_Allreduce time=([0-9.]+)"),
        "ddot_min": find(r"Min DDOT MPI_Allreduce time=([0-9.]+)"),
        "total_iterations": find(r"Total number of optimized iterations=(\d+)", int),
    }


def summarize(rows):
    by_gpu = {}
    for r in rows:
        idx = r["gpu_index"]
        by_gpu.setdefault(idx, []).append(r)

    summary = {}
    for idx, samples in by_gpu.items():
        nums = {
            k: [float(r[k]) for r in samples if r[k] not in (None, "", "N/A")]
            for k in ("gpu_utilization_pct", "memory_utilization_pct",
                      "temperature_c", "power_w", "sm_clock_mhz",
                      "memory_clock_mhz", "pcie_rx_mbps", "pcie_tx_mbps")
        }
        stats = {}
        for k, vals in nums.items():
            if vals:
                stats[f"{k}_mean"] = mean(vals)
                stats[f"{k}_max"] = max(vals)
                stats[f"{k}_min"] = min(vals)
            else:
                stats[f"{k}_mean"] = None
                stats[f"{k}_max"] = None
                stats[f"{k}_min"] = None

        # Most-common categorical values
        names = Counter(r["gpu_name"] for r in samples)
        stats["gpu_name"] = names.most_common(1)[0][0]
        gens = Counter(_int(r["pcie_gen_current"]) for r in samples if r["pcie_gen_current"])
        widths = Counter(_int(r["pcie_width_current"]) for r in samples if r["pcie_width_current"])
        stats["pcie_gen_mode"] = gens.most_common(1)[0][0] if gens else None
        stats["pcie_width_mode"] = widths.most_common(1)[0][0] if widths else None

        # Throttle reasons seen across samples
        all_reasons = set()
        for r in samples:
            all_reasons.update(decode_throttle(r.get("throttle_reasons", "")))
        stats["throttle_reasons"] = sorted(all_reasons) if all_reasons else ["None"]

        stats["sample_count"] = len(samples)
        summary[idx] = stats

    return summary


def fmt(value, unit=""):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}{unit}"
    return f"{value}{unit}"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <monitor.csv> [hpcg.log]", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    log_path = sys.argv[2] if len(sys.argv) > 2 else None

    rows = list(csv.DictReader(open(csv_path, newline="")))
    stats = summarize(rows)
    hpcg = parse_hpcg_log(log_path)

    # Convenience labels (this node enumerates nvidia-smi opposite to CUDA)
    label = {}
    for idx, s in stats.items():
        name = s.get("gpu_name", "")
        if "4090" in name:
            label[idx] = "RTX 4090 (fast, nvidia-smi index 1, CUDA 0)"
        elif "5080" in name:
            label[idx] = "RTX 5080 (slow, nvidia-smi index 0, CUDA 1)"
        else:
            label[idx] = name

    lines = []
    lines.append("# Dual-GPU Monitoring Report")
    lines.append("")
    lines.append(f"- CSV: `{csv_path}`")
    lines.append(f"- Samples per GPU: {next(iter(stats.values()))['sample_count'] if stats else 0}")
    if hpcg:
        lines.append(f"- HPCG log: `{log_path}`")
        lines.append(f"- HPCG valid: {'YES' if hpcg.get('valid') else 'NO' if hpcg.get('valid') is False else 'unknown'}")
        lines.append(f"- HPCG GFLOP/s: {fmt(hpcg.get('gflops'))}")
        lines.append(f"- HPCG exec time: {fmt(hpcg.get('exec_time'), 's')}")
        lines.append(f"- DDOT max/min: {fmt(hpcg.get('ddot_max'), 's')} / {fmt(hpcg.get('ddot_min'), 's')}")
    lines.append("")

    lines.append("## Per-GPU summary")
    lines.append("")
    lines.append("| GPU | Util (%) | Mem Util (%) | Temp (C) | Power (W) | SM clock (MHz) | Mem clock (MHz) | PCIe gen | PCIe width | PCIe RX (MB/s) | PCIe TX (MB/s) | Throttle reasons |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for idx in sorted(stats.keys(), key=lambda x: int(x)):
        s = stats[idx]
        lines.append(
            f"| {label.get(idx, idx)} | "
            f"{fmt(s['gpu_utilization_pct_mean'])} | "
            f"{fmt(s['memory_utilization_pct_mean'])} | "
            f"{fmt(s['temperature_c_mean'])} | "
            f"{fmt(s['power_w_mean'])} | "
            f"{fmt(s['sm_clock_mhz_mean'])} | "
            f"{fmt(s['memory_clock_mhz_mean'])} | "
            f"{fmt(s['pcie_gen_mode'])} | "
            f"{fmt(s['pcie_width_mode'])} | "
            f"{fmt(s['pcie_rx_mbps_mean'])} (max {fmt(s['pcie_rx_mbps_max'])}) | "
            f"{fmt(s['pcie_tx_mbps_mean'])} (max {fmt(s['pcie_tx_mbps_max'])}) | "
            f"{', '.join(s['throttle_reasons'])} |"
        )
    lines.append("")

    # Bottleneck evidence
    lines.append("## PCIe bottleneck evidence")
    lines.append("")
    width_5080 = stats.get("0", {}).get("pcie_width_mode")
    width_4090 = stats.get("1", {}).get("pcie_width_mode")
    rx_5080 = stats.get("0", {}).get("pcie_rx_mbps_mean") or 0
    tx_5080 = stats.get("0", {}).get("pcie_tx_mbps_mean") or 0
    rx_4090 = stats.get("1", {}).get("pcie_rx_mbps_mean") or 0
    tx_4090 = stats.get("1", {}).get("pcie_tx_mbps_mean") or 0

    if width_5080 == 1:
        lines.append(f"- **RTX 5080 PCIe width stayed at x1 throughout the run.** This is far below its x16 capability and matches the pre-run topology observation.")
    else:
        lines.append(f"- RTX 5080 PCIe width mode was x{width_5080}; it did **not** stay at x1 under load.")

    if width_4090 == 16:
        lines.append(f"- RTX 4090 PCIe width mode was x16, as expected for the primary slot.")
    else:
        lines.append(f"- RTX 4090 PCIe width mode was x{width_4090}.")

    # Compare throughput
    if rx_5080 or tx_5080 or rx_4090 or tx_4090:
        lines.append(f"- Mean PCIe throughput: RTX 5080 RX={fmt(rx_5080)} TX={fmt(tx_5080)}; RTX 4090 RX={fmt(rx_4090)} TX={fmt(tx_4090)}.")
        if max(rx_5080, tx_5080) > max(rx_4090, tx_4090, 0) * 1.5:
            lines.append("- The x1-limited RTX 5080 is pushing more PCIe traffic per second, consistent with host-side MPI halo data being squeezed through a narrow link.")
        elif max(rx_4090, tx_4090) > max(rx_5080, tx_5080, 0) * 1.5:
            lines.append("- The RTX 4090 is moving more PCIe traffic, likely because it owns the larger domain slice.")
        else:
            lines.append("- PCIe throughput is comparable between the two GPUs.")
    lines.append("")

    # Compare utilization / temp / power
    lines.append("## Utilization / temperature / power comparison")
    lines.append("")
    util_5080 = stats.get("0", {}).get("gpu_utilization_pct_mean") or 0
    util_4090 = stats.get("1", {}).get("gpu_utilization_pct_mean") or 0
    temp_5080 = stats.get("0", {}).get("temperature_c_mean") or 0
    temp_4090 = stats.get("1", {}).get("temperature_c_mean") or 0
    pwr_5080 = stats.get("0", {}).get("power_w_mean") or 0
    pwr_4090 = stats.get("1", {}).get("power_w_mean") or 0
    lines.append(f"- GPU utilization: RTX 5080 {fmt(util_5080)}%, RTX 4090 {fmt(util_4090)}%.")
    lines.append(f"- Temperature: RTX 5080 {fmt(temp_5080)}C, RTX 4090 {fmt(temp_4090)}C.")
    lines.append(f"- Power: RTX 5080 {fmt(pwr_5080)}W, RTX 4090 {fmt(pwr_4090)}W.")
    if util_4090 > util_5080 * 1.2:
        lines.append("- The RTX 4090 shows higher utilization, consistent with it carrying the larger/faster workload.")
    elif util_5080 > util_4090 * 1.2:
        lines.append("- The RTX 5080 shows higher utilization despite being the slower card; this can indicate it is struggling to keep up (e.g., PCIe or kernel launch overhead).")
    else:
        lines.append("- Utilization is similar between the two GPUs.")
    lines.append("")

    # Raw notes
    lines.append("## Notes")
    lines.append("")
    lines.append("- `nvidia-smi` enumerates the RTX 5080 as index 0 and the RTX 4090 as index 1 on this node (opposite to CUDA device order).")
    lines.append("- PCIe link width is reported by the current driver; a value of x1 under load strongly suggests a physical or motherboard slot negotiation problem for the RTX 5080.")
    lines.append("- Values shown are averages over the monitored window; peak PCIe throughput may briefly exceed the mean.")
    lines.append("")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
