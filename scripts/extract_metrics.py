#!/usr/bin/env python3
"""
Extract metrics from HPCG and nsys outputs for each benchmark configuration.

Usage:
    python3 scripts/extract_metrics.py <benchmark_data_dir>
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path


def parse_hpcg_log(log_path):
    """Parse key HPCG summary metrics from hpcg.log."""
    text = Path(log_path).read_text(errors="replace")

    def find(pattern, cast=float):
        m = re.search(pattern, text)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except Exception:
            return None

    # Local domain dimensions may appear as a multi-line block; try both forms.
    local_nz = find(r"Local Domain Dimensions::nz=(\d+)", int)
    if local_nz is None:
        m = re.search(r"Local Domain: (\d+)x(\d+)x(\d+)", text)
        if m:
            local_nz = int(m.group(3))

    return {
        "gflops_rating": find(r"Final Summary::HPCG result is VALID with a GFLOP/s rating of=([\d.]+)"),
        "hist_rating": find(r"Final Summary::HPCG 2\.4 rating for historical reasons is=([\d.]+)"),
        "exec_time": find(r"Final Summary::Results are valid but execution time \(sec\) is=([\d.]+)"),
        "ddot_max": find(r"DDOT Timing Variations::Max DDOT MPI_Allreduce time=([\d.]+)"),
        "ddot_min": find(r"DDOT Timing Variations::Min DDOT MPI_Allreduce time=([\d.]+)"),
        "global_nx": find(r"Global Problem Dimensions::Global nx=(\d+)", int),
        "global_ny": find(r"Global Problem Dimensions::Global ny=(\d+)", int),
        "global_nz": find(r"Global Problem Dimensions::Global nz=(\d+)", int),
        "local_nx": find(r"Local Domain Dimensions::nx=(\d+)", int),
        "local_ny": find(r"Local Domain Dimensions::ny=(\d+)", int),
        "local_nz": local_nz,
        "total_iterations": find(r"Iteration Count Information::Total number of optimized iterations=(\d+)", int),
    }


def parse_nsys_gpu_sum(nsys_rep_path):
    """Run nsys stats and parse CUDA GPU summary."""
    nsys_rep = Path(nsys_rep_path)
    if not nsys_rep.exists():
        return {"error": "nsys report not found"}

    result = {}
    try:
        out = subprocess.check_output(
            ["nsys", "stats", "--report", "cuda_gpu_sum", str(nsys_rep)],
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        return {"error": str(e), "output": e.output}

    lines = out.splitlines()
    data_started = False
    rows = []
    for line in lines:
        if line.strip().startswith("Time (%)"):
            data_started = True
            continue
        if data_started and line.strip() and not line.strip().startswith("-----"):
            rows.append(line)
        elif data_started and not line.strip():
            break

    total_time_ns = 0
    top_operations = []
    memcpy_time_ns = 0
    memset_time_ns = 0

    for line in rows[:20]:
        # The line layout is fixed-width.  Split on two or more spaces to
        # separate the numeric columns from the operation name.
        parts = re.split(r"\s{2,}", line.strip(), maxsplit=5)
        if len(parts) < 6:
            continue
        try:
            pct = float(parts[0])
            total_ns = float(parts[1])
            operation = parts[5].strip()
        except Exception:
            continue
        total_time_ns += total_ns
        # Strip the leading numeric index that nsys prints for each row.
        operation = re.sub(r"^\d+\s*", "", operation)
        entry = {"percent": pct, "time_ns": total_ns, "operation": operation}
        if "memcpy" in operation.lower():
            memcpy_time_ns += total_ns
        if "memset" in operation.lower():
            memset_time_ns += total_ns
        top_operations.append(entry)

    result["total_kernel_and_mem_time_ns"] = total_time_ns
    result["top_operations"] = top_operations[:10]
    result["memcpy_time_ns"] = memcpy_time_ns
    result["memset_time_ns"] = memset_time_ns
    return result


def parse_nsys_gpu_metrics(nsys_rep_path):
    """
    Read GPU metrics samples from the nsys SQLite export.

    Returns average values for selected hardware counters.  For dual-GPU
    configurations, metrics are returned both per GPU and as an overall average.
    """
    nsys_rep = Path(nsys_rep_path)
    sqlite_path = nsys_rep.with_suffix(".sqlite")

    if not sqlite_path.exists():
        # Try to force export
        try:
            subprocess.run(
                ["nsys", "stats", "--report", "cuda_gpu_sum", str(nsys_rep)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    if not sqlite_path.exists():
        return {"error": "nsys sqlite export not found"}

    try:
        conn = sqlite3.connect(str(sqlite_path))
        cur = conn.cursor()

        # Map (typeId, metricId) -> metricName
        cur.execute(
            "SELECT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS"
        )
        metric_names = {
            (type_id, metric_id): name
            for type_id, metric_id, name in cur.fetchall()
        }

        # Selected metrics of interest
        wanted = {
            "SMs Active [Throughput %]",
            "SM Issue [Throughput %]",
            "GR Active [Throughput %]",
            "Compute Warps in Flight [Throughput %]",
            "DRAM Read Bandwidth [Throughput %]",
            "DRAM Write Bandwidth [Throughput %]",
            "GPC Clock Frequency [MHz]",
        }
        wanted_ids = {
            k for k, name in metric_names.items() if name in wanted
        }

        cur.execute(
            "SELECT typeId, metricId, value FROM GPU_METRICS "
            "WHERE (typeId, metricId) IN (VALUES "
            + ",".join(["(?, ?)"] * len(wanted_ids))
            + ")",
            [x for pair in wanted_ids for x in pair],
        )

        # Accumulate values per typeId and metricName.  Nsight stores some
        # counters as unsigned integers in signed columns; reinterpret those
        # values as unsigned to avoid negative frequencies and bandwidths.
        accum = {}
        for type_id, metric_id, value in cur.fetchall():
            name = metric_names.get((type_id, metric_id))
            if name is None:
                continue
            if value < 0:
                value &= 0xFFFFFFFFFFFFFFFF
            key = (type_id, name)
            accum.setdefault(key, []).append(value)

        conn.close()

        if not accum:
            return {"error": "no GPU metric samples found"}

        # Average per GPU, then overall average across GPUs
        per_gpu = {}
        overall = {}
        metric_values = {name: [] for _, name in accum.keys()}

        for (type_id, name), vals in accum.items():
            avg = sum(vals) / len(vals)
            per_gpu.setdefault(type_id, {})[name] = avg
            metric_values[name].append(avg)

        for name, vals in metric_values.items():
            overall[name] = sum(vals) / len(vals) if vals else None

        return {
            "per_gpu": per_gpu,
            "overall": overall,
        }
    except Exception as e:
        return {"error": str(e)}


def process_config(output_dir, config_name):
    cfg_dir = Path(output_dir) / config_name
    metrics = {"config": config_name}

    hpcg_log = cfg_dir / "hpcg.log"
    if hpcg_log.exists():
        metrics["hpcg"] = parse_hpcg_log(hpcg_log)
    else:
        metrics["hpcg"] = {"error": "hpcg.log not found"}

    nsys_rep = cfg_dir / "nsys.nsys-rep"
    nsys_metrics = parse_nsys_gpu_sum(nsys_rep)
    gpu_metrics = parse_nsys_gpu_metrics(nsys_rep)
    nsys_metrics["gpu_metrics"] = gpu_metrics
    metrics["nsys"] = nsys_metrics

    out_path = cfg_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_path}")
    return metrics


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <benchmark_data_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = sys.argv[1]
    configs = ["RTX4090", "RTX5080", "DUAL_EQUAL", "DUAL_EQUAL_128"]

    for cfg in configs:
        process_config(output_dir, cfg)


if __name__ == "__main__":
    main()
