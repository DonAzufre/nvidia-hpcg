#!/usr/bin/env bash
#
# Run a dual-GPU HPCG configuration while monitoring both GPUs.
#
# Usage:
#   scripts/run_with_gpu_monitor.sh <config_name> <rt> [extra_hpcg_args...]
#
# Example:
#   scripts/run_with_gpu_monitor.sh DUAL_EQUAL_Z 60 --npx 1 --npy 1 --npz 2

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

export HWLOC_COMPONENTS="-gl"

CONFIG_NAME="${1:-DUAL_EQUAL_Z}"
RT="${2:-60}"
shift 2 || true
EXTRA_ARGS=("$@")

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
MON_DIR="benchmark_data/gpu_monitoring/${TIMESTAMP}_${CONFIG_NAME}"
mkdir -p "${MON_DIR}"
RUN_DIR="benchmark_data/gpu_monitoring_runs/${TIMESTAMP}_${CONFIG_NAME}"
mkdir -p "${RUN_DIR}"

LOG="${RUN_DIR}/hpcg.log"
DURATION=$((RT + 60))

echo "[monitor] Starting GPU monitor for ${DURATION}s -> ${MON_DIR}"
python3 "${SCRIPT_DIR}/monitor_dual_gpu.py" \
    --output-dir "${MON_DIR}" \
    --duration "${DURATION}" \
    --interval 1 \
    > "${RUN_DIR}/monitor.log" 2>&1 &
MON_PID=$!

# Wait for monitor to initialize
sleep 2

echo "[hpcg] Running ${CONFIG_NAME} rt=${RT} extra=${EXTRA_ARGS[*]}"
/usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none \
    -np 2 ./bin/hpcg.sh --exec-name ./bin/xhpcg \
    --nx 128 --ny 128 --nz 128 --rt "${RT}" \
    --gpu-affinity 0:1 --cpu-affinity 0-7:8-15 \
    --p2p 0 --b 1 \
    "${EXTRA_ARGS[@]}" \
    > "${LOG}" 2>&1 || true

HPCG_EXIT=$?
echo "[hpcg] Finished with exit code ${HPCG_EXIT}"

# Wait for monitor to finish its duration
wait "${MON_PID}" || true

echo "[monitor] GPU monitor finished. Output: ${MON_DIR}"
echo "[monitor] analysis:"
cat "${MON_DIR}/analysis.txt"

# Extract key HPCG metrics
VALID="UNKNOWN"
GFLOPS="N/A"
EXEC_TIME="N/A"
if [[ -r "${LOG}" ]]; then
    if grep -q "HPCG result is VALID" "${LOG}"; then
        VALID="VALID"
    elif grep -qE "HPCG result is INVALID|ERROR|failure|Segmentation fault" "${LOG}"; then
        VALID="INVALID"
    fi
    GFLOPS=$(grep -oP 'HPCG result is VALID with a GFLOP/s rating of=\K[0-9.]+' "${LOG}" | tail -1 || echo "N/A")
    EXEC_TIME=$(grep -oP 'Results are valid but execution time \(sec\) is=\K[0-9.]+' "${LOG}" | tail -1 || echo "N/A")
fi

echo "[hpcg] ${CONFIG_NAME} rt=${RT}: valid=${VALID} GFLOP/s=${GFLOPS} exec_time=${EXEC_TIME}s"
