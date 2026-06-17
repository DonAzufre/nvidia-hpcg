#!/usr/bin/env python3
"""
Periodically sample both GPUs with nvidia-smi while an HPCG run is in progress.

Records a CSV with:
    timestamp, gpu_index, gpu_name,
    gpu_utilization_pct, memory_utilization_pct,
    temperature_c, power_w,
    sm_clock_mhz, memory_clock_mhz,
    pcie_gen_current, pcie_width_current,
    pcie_rx_mbps, pcie_tx_mbps,
    throttle_reasons

Usage example:
    python3 scripts/monitor_dual_gpu.py --output monitor.csv --interval 1 --duration 300
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_GPUS = ["0", "1"]
QUERY_FIELDS = [
    "timestamp",
    "index",
    "gpu_name",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "clocks.current.sm",
    "clocks.current.memory",
    "pcie.link.gen.gpucurrent",
    "pcie.link.width.current",
    "clocks_event_reasons.active",
]

# Global flag used to stop the loop on SIGINT/SIGTERM.
stop_requested = False


def signal_handler(signum, frame):
    global stop_requested
    stop_requested = True


def run_query(gpus):
    """Return list of dicts with per-GPU static/counter metrics."""
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
        "-i", ",".join(gpus),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10)
    except subprocess.CalledProcessError as e:
        print(f"[monitor] nvidia-smi query failed: {e.output}", file=sys.stderr)
        return []
    except subprocess.TimeoutExpired:
        print("[monitor] nvidia-smi query timed out", file=sys.stderr)
        return []

    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            continue
        rows.append({
            "timestamp": parts[0],
            "gpu_index": parts[1],
            "gpu_name": parts[2],
            "gpu_utilization_pct": _float(parts[3]),
            "memory_utilization_pct": _float(parts[4]),
            "temperature_c": _float(parts[5]),
            "power_w": _float(parts[6]),
            "sm_clock_mhz": _float(parts[7]),
            "memory_clock_mhz": _float(parts[8]),
            "pcie_gen_current": _int(parts[9]),
            "pcie_width_current": _int(parts[10]),
            "throttle_reasons": parts[11],
        })
    return rows


def run_dmon(gpus):
    """Return dict gpu_index -> (rxpci_MBps, txpci_MBps)."""
    cmd = [
        "nvidia-smi", "dmon",
        "-s", "puct",
        "-i", ",".join(gpus),
        "-c", "1",
        "-o", "T",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10)
    except subprocess.CalledProcessError as e:
        print(f"[monitor] nvidia-smi dmon failed: {e.output}", file=sys.stderr)
        return {}
    except subprocess.TimeoutExpired:
        print("[monitor] nvidia-smi dmon timed out", file=sys.stderr)
        return {}

    result = {}
    for line in out.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split()
        if len(cols) < 15:
            continue
        gpu_idx = cols[1]
        try:
            rx = float(cols[13]) if cols[13] not in ("-", "N/A") else None
            tx = float(cols[14]) if cols[14] not in ("-", "N/A") else None
        except ValueError:
            rx = tx = None
        result[gpu_idx] = (rx, tx)
    return result


def _float(value):
    if value is None or value in ("", "N/A", "-", "[Not Supported]", "[Unknown Error]"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int(value):
    if value is None or value in ("", "N/A", "-", "[Not Supported]", "[Unknown Error]"):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Sample dual-GPU metrics during an HPCG run.")
    parser.add_argument("--output", required=True, help="Output CSV file path.")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1).")
    parser.add_argument("--duration", type=float, default=None, help="Maximum monitoring duration in seconds (default: unlimited).")
    parser.add_argument("--gpus", default=",".join(DEFAULT_GPUS), help="Comma-separated GPU indices (default: 0,1).")
    args = parser.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    fieldnames = [
        "timestamp", "gpu_index", "gpu_name",
        "gpu_utilization_pct", "memory_utilization_pct",
        "temperature_c", "power_w",
        "sm_clock_mhz", "memory_clock_mhz",
        "pcie_gen_current", "pcie_width_current",
        "pcie_rx_mbps", "pcie_tx_mbps",
        "throttle_reasons",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"[monitor] Started sampling GPUs {gpus} every {args.interval}s -> {out_path}", file=sys.stderr)
        start = time.monotonic()
        sample_count = 0

        while not stop_requested:
            loop_start = time.monotonic()
            elapsed = loop_start - start
            if args.duration is not None and elapsed >= args.duration:
                break

            query_rows = run_query(gpus)
            dmon_rows = run_dmon(gpus)

            for row in query_rows:
                gpu_idx = row["gpu_index"]
                rx, tx = dmon_rows.get(gpu_idx, (None, None))
                row["pcie_rx_mbps"] = rx
                row["pcie_tx_mbps"] = tx
                writer.writerow(row)
            f.flush()
            sample_count += len(query_rows)

            # Sleep until next interval, accounting for sampling overhead.
            next_time = loop_start + args.interval
            sleep_time = next_time - time.monotonic()
            while sleep_time > 0 and not stop_requested:
                time.sleep(min(sleep_time, 0.2))
                sleep_time = next_time - time.monotonic()

    print(f"[monitor] Finished. Wrote {sample_count} samples to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
