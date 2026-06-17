#!/usr/bin/env python3
"""
Adaptive heterogeneous split tuner for nvidia-hpcg.

This is a lightweight "dynamic load balance" prototype.  Instead of modifying
HPCG's core kernels, it treats the existing `--het-ratio` parameter as a knob
and searches for the value that maximises dual-GPU throughput for the current
machine state (GPU clocks, thermals, PCIe width, etc.).

Usage:
    python3 scripts/dynamic_load_balance.py
    python3 scripts/dynamic_load_balance.py --rt-search 20 --rt-final 60
    python3 scripts/dynamic_load_balance.py --ratios 1.0 1.3 1.6 1.9 2.2

The script:
  1. Runs a short HPCG benchmark for each candidate ratio.
  2. Parses GFLOP/s and DDOT max-min imbalance from each log.
  3. Picks the ratio with the best GFLOP/s (tie-break: smallest imbalance).
  4. Runs a final benchmark with the chosen ratio and a longer runtime.
  5. Writes a CSV summary and the final HPCG log to
     `benchmark_data/dynamic_load_balance/<timestamp>/`.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
HPCG_SH = REPO_ROOT / "bin" / "hpcg.sh"
XHPCG = REPO_ROOT / "bin" / "xhpcg"
MPIRUN = "/usr/bin/mpirun.openmpi"


def parse_log(log_path: Path):
    """Parse key metrics from an HPCG log file."""
    text = log_path.read_text(errors="replace")

    def find(pattern, cast=float):
        m = re.search(pattern, text)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except Exception:
            return None

    # Final summary
    gflops = find(r"Final Summary::HPCG result is VALID with a GFLOP/s rating of=([\d.]+)")
    exec_time = find(r"Final Summary::Results are valid but execution time \(sec\) is=([\d.]+)")

    # Load-balance proxy: DDOT allreduce times are MPI-reduced max/min across ranks
    ddot_max = find(r"DDOT Timing Variations::Max DDOT MPI_Allreduce time=([\d.]+)")
    ddot_min = find(r"DDOT Timing Variations::Min DDOT MPI_Allreduce time=([\d.]+)")
    ddot_avg = find(r"DDOT Timing Variations::Avg DDOT MPI_Allreduce time=([\d.]+)")

    # Local domain reported by rank 0 (fast GPU in het mode)
    local_nx = local_ny = local_nz = None
    m = re.search(r"Local Domain: (\d+)x(\d+)x(\d+)", text)
    if m:
        local_nx, local_ny, local_nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        local_nx = find(r"Local Domain Dimensions::nx=(\d+)", int)
        local_ny = find(r"Local Domain Dimensions::ny=(\d+)", int)
        local_nz = find(r"Local Domain Dimensions::nz=(\d+)", int)

    valid = "HPCG result is VALID" in text and gflops is not None
    imbalance = (ddot_max - ddot_min) if (ddot_max is not None and ddot_min is not None) else None

    return {
        "valid": valid,
        "gflops": gflops,
        "exec_time": exec_time,
        "ddot_max": ddot_max,
        "ddot_min": ddot_min,
        "ddot_avg": ddot_avg,
        "imbalance": imbalance,
        "local_nx": local_nx,
        "local_ny": local_ny,
        "local_nz": local_nz,
    }


def run_hpcg(
    output_dir: Path,
    name: str,
    nx: int,
    ny: int,
    nz: int,
    rt: int,
    npx: int,
    npy: int,
    npz: int,
    het_dim: int,
    het_ratio: float,
    gpu_affinity: str,
    cpu_affinity: str,
):
    """Run one HPCG configuration and return parsed metrics."""
    cfg_dir = output_dir / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg_dir / "hpcg.log"

    cmd = [
        MPIRUN,
        "--allow-run-as-root",
        "--oversubscribe",
        "--bind-to",
        "none",
        "-np",
        str(npx * npy * npz),
        str(HPCG_SH),
        "--exec-name",
        str(XHPCG),
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--nz",
        str(nz),
        "--rt",
        str(rt),
        "--gpu-affinity",
        gpu_affinity,
        "--cpu-affinity",
        cpu_affinity,
        "--p2p",
        "0",
        "--b",
        "1",
        "--npx",
        str(npx),
        "--npy",
        str(npy),
        "--npz",
        str(npz),
        "--het-split",
        "1",
        "--het-dim",
        str(het_dim),
        "--het-ratio",
        str(het_ratio),
    ]

    env = os.environ.copy()
    env["HWLOC_COMPONENTS"] = "-gl"

    print(f"\n[SEARCH] Running {name} with ratio={het_ratio}, rt={rt} ...")
    start = time.time()
    with open(log_path, "w") as fout:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=fout,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start
    print(f"[SEARCH] {name} finished in {elapsed:.1f}s (exit={result.returncode})")

    metrics = parse_log(log_path)
    metrics["name"] = name
    metrics["ratio"] = het_ratio
    metrics["rt"] = rt
    return metrics


def select_best(results):
    """Pick the candidate with the highest GFLOP/s; tie-break on imbalance."""
    valid = [r for r in results if r["valid"] and r["gflops"] is not None]
    if not valid:
        return None
    # Sort descending by gflops, then ascending by imbalance
    valid.sort(key=lambda r: (-r["gflops"], r["imbalance"] if r["imbalance"] is not None else float("inf")))
    return valid[0]


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive het-ratio tuner for dual-GPU HPCG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nx", type=int, default=128, help="Fast-GPU local X dimension")
    parser.add_argument("--ny", type=int, default=128, help="Fast-GPU local Y dimension")
    parser.add_argument("--nz", type=int, default=160, help="Fast-GPU local Z dimension")
    parser.add_argument("--npx", type=int, default=1)
    parser.add_argument("--npy", type=int, default=1)
    parser.add_argument("--npz", type=int, default=2)
    parser.add_argument("--het-dim", type=int, default=3, choices=[1, 2, 3], help="1=X, 2=Y, 3=Z")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 1.3, 1.6, 1.9, 2.2])
    parser.add_argument("--rt-search", type=int, default=20, help="Runtime (sec) for each search run")
    parser.add_argument("--rt-final", type=int, default=60, help="Runtime (sec) for the final tuned run")
    parser.add_argument("--gpu-affinity", type=str, default="0:1")
    parser.add_argument("--cpu-affinity", type=str, default="0-7:8-15")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    if not HPCG_SH.exists():
        print(f"ERROR: {HPCG_SH} not found.", file=sys.stderr)
        sys.exit(1)
    if not XHPCG.exists():
        print(f"ERROR: {XHPCG} not found; build HPCG first.", file=sys.stderr)
        sys.exit(1)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / "benchmark_data" / "dynamic_load_balance" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ---------- Search phase ----------
    results = []
    for ratio in args.ratios:
        name = f"ratio_{ratio:.2f}"
        metrics = run_hpcg(
            output_dir=output_dir,
            name=name,
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            rt=args.rt_search,
            npx=args.npx,
            npy=args.npy,
            npz=args.npz,
            het_dim=args.het_dim,
            het_ratio=ratio,
            gpu_affinity=args.gpu_affinity,
            cpu_affinity=args.cpu_affinity,
        )
        gflops_str = f"{metrics['gflops']:.2f}" if metrics['gflops'] is not None else "N/A"
        imbalance_str = f"{metrics['imbalance']:.3f}" if metrics['imbalance'] is not None else "N/A"
        print(
            f"  -> valid={metrics['valid']} gflops={gflops_str} "
            f"imbalance={imbalance_str}s"
        )
        results.append(metrics)

    # Save search results CSV
    csv_path = output_dir / "search_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "ratio",
                "rt",
                "valid",
                "gflops",
                "exec_time",
                "ddot_max",
                "ddot_min",
                "ddot_avg",
                "imbalance",
                "local_nx",
                "local_ny",
                "local_nz",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    print(f"\nSearch results written to {csv_path}")

    best = select_best(results)
    if best is None:
        print("ERROR: no valid runs; aborting final benchmark.", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nBest search candidate: ratio={best['ratio']:.2f} -> "
        f"GFLOP/s={best['gflops']:.2f}, imbalance={best['imbalance']:.3f}s"
    )

    # ---------- Final tuned phase ----------
    final = run_hpcg(
        output_dir=output_dir,
        name="final_tuned",
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        rt=args.rt_final,
        npx=args.npx,
        npy=args.npy,
        npz=args.npz,
        het_dim=args.het_dim,
        het_ratio=best["ratio"],
        gpu_affinity=args.gpu_affinity,
        cpu_affinity=args.cpu_affinity,
    )
    print(
        f"\nFinal run: ratio={final['ratio']:.2f}, "
        f"GFLOP/s={final['gflops']:.2f}, imbalance={final['imbalance']:.3f}s, "
        f"exec_time={final['exec_time']:.2f}s"
    )

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Adaptive het-ratio tuning summary\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Problem: local fast-GPU domain {args.nx}x{args.ny}x{args.nz}, "
                f"grid {args.npx}x{args.npy}x{args.npz}, dim={args.het_dim}\n")
        f.write(f"Search ratios: {args.ratios}\n")
        f.write(f"Search rt: {args.rt_search}s, final rt: {args.rt_final}s\n")
        f.write(f"Best ratio: {best['ratio']:.2f}\n")
        f.write(f"Best search GFLOP/s: {best['gflops']:.2f}\n")
        f.write(f"Final tuned GFLOP/s: {final['gflops']:.2f}\n")
        f.write(f"Final tuned DDOT imbalance (max-min): {final['imbalance']:.3f}s\n")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
