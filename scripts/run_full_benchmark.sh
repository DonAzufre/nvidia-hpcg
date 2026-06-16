#!/usr/bin/env bash
#
# Run a full HPCG benchmark suite and collect HPCG, nsys, and ncu data.
#
# Configurations:
#   1. RTX4090      - single GPU, local 128^3
#   2. RTX5080      - single GPU, local 128^3
#   3. DUAL_EQUAL   - 2 GPUs, global 128x128x256, each rank 128^3
#   4. DUAL_EQUAL_128 - 2 GPUs, global 128x128x128, each rank 64^3
#   5. DUAL_HET     - 2 GPUs, global 128x128x256, ranks 128x128x160 and 128x128x96
#
# Output is written to benchmark_data/<timestamp>/ and is gitignored.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

# Environment workarounds for this heterogeneous node
export HWLOC_COMPONENTS="-gl"

# Ensure ~/.ssh/config is set up for port 51111
if ! ssh -G localhost 2>/dev/null | grep -qiE "^port 51111"; then
    echo "WARNING: ~/.ssh/config does not appear to map localhost to port 51111."
    echo "         OpenMPI may fail to spawn ranks. See docs/DUAL_GPU_EQUAL.md."
fi

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR="benchmark_data/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "Benchmark output directory: ${OUTPUT_DIR}"

# Runtimes for profiling
# Nsight Systems with --gpu-metrics-devices replaces NCU for hardware counters.
RT_BASELINE=60
RT_NSYS=30

# Run a single configuration
# usage: run_config <name> <np> <nx> <ny> <nz> <gpu_affinity> <cpu_affinity> [extra_hpcg_args...]
run_config() {
    local name="$1"
    local np="$2"
    local nx="$3"
    local ny="$4"
    local nz="$5"
    local gpu_aff="$6"
    local cpu_aff="$7"
    shift 7
    local extra_args=("$@")
    local cfg_dir="${OUTPUT_DIR}/${name}"
    mkdir -p "${cfg_dir}"

    echo ""
    echo "========================================"
    echo "Configuration: ${name}"
    echo "========================================"

    # 1. Baseline HPCG run
    echo "[${name}] Running baseline HPCG (rt=${RT_BASELINE})..."
    /usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none \
        -np "${np}" ./bin/hpcg.sh --exec-name ./bin/xhpcg \
        --nx "${nx}" --ny "${ny}" --nz "${nz}" --rt "${RT_BASELINE}" \
        --gpu-affinity "${gpu_aff}" --cpu-affinity "${cpu_aff}" \
        --p2p 0 --b 1 "${extra_args[@]}" \
        > "${cfg_dir}/hpcg.log" 2>&1
    echo "[${name}] Baseline finished."

    # 2. Nsight Systems profile (includes GPU metrics)
    echo "[${name}] Running Nsight Systems profile with GPU metrics (rt=${RT_NSYS})..."
    nsys profile \
        --trace cuda,mpi \
        --gpu-metrics-devices all \
        -o "${cfg_dir}/nsys" --force-overwrite true \
        /usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none \
        -np "${np}" ./bin/hpcg.sh --exec-name ./bin/xhpcg \
        --nx "${nx}" --ny "${ny}" --nz "${nz}" --rt "${RT_NSYS}" \
        --gpu-affinity "${gpu_aff}" --cpu-affinity "${cpu_aff}" \
        --p2p 0 --b 1 "${extra_args[@]}" \
        > "${cfg_dir}/nsys_run.log" 2>&1
    echo "[${name}] Nsight Systems finished."

    # Export nsys text summary (kernel/memop breakdown)
    nsys stats --report cuda_gpu_sum "${cfg_dir}/nsys.nsys-rep" \
        > "${cfg_dir}/nsys_cuda_gpu_sum.txt" 2>&1 || true
}

# Configuration 1: RTX 4090 only
run_config "RTX4090" 1 128 128 128 "0" "0-7"

# Configuration 2: RTX 5080 only
run_config "RTX5080" 1 128 128 128 "1" "8-15"

# Configuration 3: Dual GPU equal, global 128x128x256 (each rank 128^3)
run_config "DUAL_EQUAL" 2 128 128 128 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2

# Configuration 4: Dual GPU equal, total scale 128^3 (each rank 64^3)
run_config "DUAL_EQUAL_128" 2 128 128 64 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2

# Configuration 5: Dual GPU heterogeneous, global 128x128x256.
# Fast rank (RTX 4090) owns 128x128x160; slow rank (RTX 5080) owns 128x128x96.
# --het-ratio=1.6 rounds the slow size to a multiple of 16, giving an actual
# work ratio of 160/96 = 1.667.
run_config "DUAL_HET" 2 128 128 160 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2 --het-split 1 --het-ratio 1.6

# Extract metrics and generate summary
python3 "${SCRIPT_DIR}/extract_metrics.py" "${OUTPUT_DIR}"
python3 "${SCRIPT_DIR}/generate_summary.py" "${OUTPUT_DIR}"

echo ""
echo "========================================"
echo "Benchmark suite complete."
echo "Output: ${OUTPUT_DIR}"
echo "Summary: ${OUTPUT_DIR}/summary.md"
echo "========================================"
