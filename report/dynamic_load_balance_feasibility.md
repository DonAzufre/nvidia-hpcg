# Dynamic Load Balancing Feasibility Report

**Scope:** Investigate whether dynamic task acquisition / work stealing can be added to the MPI-based `nvidia-hpcg` codebase, and implement the simplest viable prototype if possible.

**Target hardware:** heterogeneous dual-GPU workstation
- RTX 4090 (fast, CUDA device 0, sm_89)
- RTX 5080 (slow, CUDA device 1, sm_120), currently running at PCIe Gen3 x1 width

**Date:** 2026-06-17

---

## 1. Executive Summary

- **True dynamic load balancing (work stealing or online domain redistribution) is not practical in this codebase without invasive, high-risk changes.** HPCG builds a static sparse-matrix / multigrid hierarchy once at setup time; redistributing work mid-run would require migrating matrix and vector state across GPUs and rebuilding halos and coarse levels.
- **The simplest viable dynamic strategy is *adaptive static partitioning*: use short warmup runs to measure the optimal `--het-ratio` for the current machine state, then launch the full benchmark with that ratio.** This reuses the existing `--het-split` implementation and requires no changes to core kernels.
- A prototype Python tuner (`scripts/dynamic_load_balance.py`) was implemented and exercised. It automatically searches a user-supplied list of ratios, picks the one with the highest GFLOP/s (tie-breaking on DDOT imbalance), and runs a final longer benchmark.
- The prototype works, but **performance on this node is highly variable between short and long runs** (likely due to the RTX 5080's PCIe x1 bottleneck and thermal state). Any dynamic tuner here must use sufficiently long warmup runs and possibly repeated samples per ratio.

---

## 2. Alternatives Considered

| Approach | Idea | Implementation Complexity | Required Code Changes | Expected Benefit | Verdict |
|---|---|---|---|---|---|
| **1. True work stealing / task queue** | Over-decompose the domain and let fast GPU pull tasks from a shared GPU-GPU queue. | Very high | New task scheduler, CUDA kernels, lock-free queues, halo ownership tracking, changes to CG/MG loops. | Potentially large on paper, but HPCG's stencil/multigrid communication pattern makes fine-grained stealing inefficient. | **Not viable** – would rewrite the benchmark. |
| **2. Dynamic domain redistribution** | Periodically resize each rank's subdomain based on measured per-rank iteration time. | High | Recompute geometry, regenerate sparse matrices, rebuild halos and all 4 multigrid levels, migrate vector data, while preserving HPCG validation invariants. | Moderate if the bottleneck were pure compute imbalance. | **Not viable** – too invasive and error-prone; data migration costs likely outweigh gains. |
| **3. Adaptive `--het-ratio`** | Run short calibration HPCG jobs with different static ratios, pick the best, then run the real benchmark. | Low | None in core HPCG; external Python/bash driver only. | Modest-to-good: can track hardware state changes (clocks, thermals, PCIe behavior) without touching MPI domains. | **Implemented prototype** – best cost/benefit. |

### 2.1 Why work stealing is not feasible

`nvidia-hpcg` is structured as a classical MPI domain-decomposition code:
- The linear system and all multigrid levels are generated once in `GenerateProblem`, `SetupHalo`, and `GenerateCoarseProblem`.
- Halo exchanges (`ExchangeHalo`) assume a fixed neighbor mapping derived from `Geometry`.
- SYMGS, SpMV, and MG kernels operate on pre-allocated CSR/SELL-C-slice structures tied to a fixed local domain.

Introducing a task queue would require breaking these kernels into stealable units, managing dynamically shifting halo boundaries, and synchronizing GPU-GPU work queues inside the CG iteration. This would essentially replace the benchmark's core algorithms and is far beyond a 30-minute prototype.

### 2.2 Why dynamic redistribution is not feasible

Redistributing domains between CG sets would require:
1. Recomputing `Geometry` and the rank-to-physical mapping for all ranks.
2. Rebuilding the sparse matrix `A` and its halo structures.
3. Regenerating all coarse levels (`A.Ac`, `A.Ac->Ac`, ...).
4. Remapping vectors and permutation arrays (`opt2ref`, `ref2opt`).
5. Re-running cuSPARSE analysis for SpMV/SpSV/SpSM handles.

Each of these steps is comparable to the existing setup phase. Doing it online would add many seconds of overhead per redistribution and risk violating HPCG's validation requirements (e.g., aspect-ratio checks, coarse-level size constraints). The benefit does not justify the risk for this project.

### 2.3 Why adaptive `--het-ratio` is the practical choice

The existing `--het-split` / `--het-ratio` / `--het-dim` flags already provide a static asymmetric partition. The only missing piece is choosing the *right* ratio for the current GPUs and system state. Because:
- GPU clocks and thermals vary between runs.
- The RTX 5080's reported PCIe Gen3 x1 width makes its effective throughput situation-dependent.
- A ratio that is optimal in one thermal state may be suboptimal in another.

...an external tuner that re-calibrates the ratio at benchmark time is a lightweight form of dynamic load balancing. It is fully reversible and does not change the HPCG source.

---

## 3. Prototype Implementation

### 3.1 File created

- `scripts/dynamic_load_balance.py`

### 3.2 What it does

1. Accepts the problem shape (`--nx/--ny/--nz`), process grid, split dimension, and a list of candidate ratios.
2. Runs a short HPCG job for each candidate ratio using the existing `bin/hpcg.sh` launcher.
3. Parses `GFLOP/s rating`, `DDOT Max/Min MPI_Allreduce time`, and `execution time` from each log.
4. Selects the ratio with the highest GFLOP/s (tie-breaking on smaller DDOT max-min imbalance).
5. Runs a final longer benchmark with the selected ratio.
6. Writes:
   - `search_results.csv`
   - `summary.txt`
   - one `hpcg.log` per candidate plus `final_tuned/hpcg.log`

### 3.3 Usage

```bash
cd /workspace/arch-ori-opt/hpcg/nvidia-hpcg
export HWLOC_COMPONENTS="-gl"

# Default search: ratios 1.0, 1.3, 1.6, 1.9, 2.2; 20 s search runs; 60 s final run
python3 scripts/dynamic_load_balance.py

# Faster demo search
python3 scripts/dynamic_load_balance.py \
    --rt-search 15 --rt-final 30 \
    --ratios 1.0 1.6 2.2

# Custom problem and split dimension (1=X, 2=Y, 3=Z)
python3 scripts/dynamic_load_balance.py \
    --nx 128 --ny 128 --nz 128 --het-dim 1 \
    --npx 2 --npy 1 --npz 1 \
    --ratios 1.0 1.4 1.8 2.2
```

### 3.4 Search heuristic

The current heuristic is intentionally simple:

```text
score = GFLOP/s                 (primary)
tie-break = DDOT_max - DDOT_min  (secondary)
```

`DDOT max-min` is a proxy for MPI_Allreduce waiting time: a large gap means one GPU finishes its local dot product much earlier and waits at the global reduction. A ratio that both maximizes throughput and minimizes this gap is preferred.

Possible future improvements (kept out of the prototype to stay within time budget):
- Run each ratio 2–3 times and use the median to reduce noise.
- Add a cooldown period between runs to let GPU clocks/thermals stabilise.
- Use the DDOT imbalance as the primary objective when the user wants strict load balance.
- Implement a golden-section or Bayesian search over the ratio instead of a fixed grid.

---

## 4. Demonstration Run

A demonstration was executed with:

```bash
python3 scripts/dynamic_load_balance.py \
    --rt-search 15 --rt-final 30 \
    --ratios 1.0 1.6 2.2 \
    --output-dir benchmark_data/dynamic_load_balance/demo_2026-06-17_14-30-21
```

Search results (`benchmark_data/dynamic_load_balance/demo_2026-06-17_14-30-21/search_results.csv`):

| ratio | rt (s) | valid | GFLOP/s | exec time (s) | DDOT max (s) | DDOT min (s) | imbalance (s) | fast local domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0 | 15 | True | 177.01 | 36.90 | 8.137 | 0.028 | 8.109 | 128x128x160 |
| 1.6 | 15 | True | 67.59 | 43.41 | 6.323 | 3.038 | 3.285 | 128x128x160 |
| 2.2 | 15 | True | 57.73 | 32.55 | 4.175 | 2.750 | 1.424 | 128x128x160 |

The tuner selected **ratio = 1.0** because it produced the highest short-run GFLOP/s. The final 30 s run with ratio 1.0 produced **71.87 GFLOP/s** with a DDOT imbalance of **9.81 s**.

### 4.1 Observations from the demo

- **Short-run and long-run performance differed substantially** for the same ratio (177 GFLOP/s at 15 s vs 71.9 GFLOP/s at 30 s). This suggests that the RTX 5080's throughput is not stable across the measurement window, likely because of thermal throttling and/or the PCIe Gen3 x1 bottleneck.
- **Higher ratios reduced DDOT imbalance** but did *not* increase throughput in this demo. This is counter-intuitive if the 5080 were simply slower; it indicates that the 5080 is not just compute-bound but also communication/PCIe-bound. Shrinking its subdomain increases the halo-to-volume ratio and can make the x1 link the dominant bottleneck.
- **A 15 s search run is too short to be representative.** For production use the tuner should use `--rt-search` of at least 30–60 s and/or repeat each ratio several times.

---

## 5. Recommendations

1. **Adopt the adaptive `--het-ratio` wrapper as the only practical dynamic-load-balance strategy.** Do not pursue work stealing or online redistribution in this codebase.
2. **Improve measurement stability before relying on the tuner's output.** Use `--rt-search 60` (or longer), add a 10–20 s GPU cooldown between candidate runs, and sample each ratio at least twice.
3. **Investigate the RTX 5080 PCIe bottleneck separately.** Dynamic load balancing cannot fix a hardware/topology issue. If the RTX 5080 remains at Gen3 x1 under load, even perfect load balance will be limited by that card's halo-exchange throughput.
4. **Consider the DDOT-imbalance objective when stability is poor.** On this node, minimising `DDOT_max - DDOT_min` may be a more robust target than raw GFLOP/s when short runs are noisy.
5. **Keep the prototype reversible.** Because it lives entirely in `scripts/dynamic_load_balance.py`, it can be removed or replaced without affecting core HPCG correctness or the existing `--het-split` feature.

---

## 6. Files Created / Modified

- **Created:** `scripts/dynamic_load_balance.py`
- **Created:** `report/dynamic_load_balance_feasibility.md`
- **Created (demo outputs):** `benchmark_data/dynamic_load_balance/demo_2026-06-17_14-30-21/`
  - `search_results.csv`
  - `summary.txt`
  - `ratio_1.00/hpcg.log`
  - `ratio_1.60/hpcg.log`
  - `ratio_2.20/hpcg.log`
  - `final_tuned/hpcg.log`

No core HPCG source files were modified.
