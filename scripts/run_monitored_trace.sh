#!/usr/bin/env bash
# Run one dual-GPU HPCG configuration while sampling GPUs.
# Usage: bash scripts/run_monitored_trace.sh [DUAL_EQUAL|DUAL_HET] [rt]
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

CONFIG="${1:-DUAL_HET}"
RT="${2:-60}"

export HWLOC_COMPONENTS="-gl"

TS=$(date +%Y-%m-%d_%H-%M-%S)
OUT="benchmark_data/gpu_monitoring/${TS}_${CONFIG}_rt${RT}"
mkdir -p "${OUT}"

echo "[trace] output directory: ${OUT}"

# Start monitor
python3 "${SCRIPT_DIR}/monitor_dual_gpu.py" \
    --output "${OUT}/gpu_metrics.csv" \
    --interval 1 \
    > "${OUT}/monitor.log" 2>&1 &
MON=$!
echo "[trace] monitor PID=${MON}"

# Wait for monitor to initialize
sleep 2

# Build HPCG command
GPU_AFFINITY="0:1"
CPU_AFFINITY="0-7:8-15"
if [[ "${CONFIG}" == "DUAL_EQUAL" ]]; then
    NX=128; NY=128; NZ=128
    NPX=1; NPY=1; NPZ=2
    EXTRA=""
elif [[ "${CONFIG}" == "DUAL_HET" ]]; then
    NX=128; NY=128; NZ=160
    NPX=1; NPY=1; NPZ=2
    EXTRA="--het-split 1 --het-ratio 1.6 --het-dim 3"
else
    echo "Unknown config ${CONFIG}"
    kill "${MON}" || true
    exit 1
fi

echo "[trace] Running HPCG ${CONFIG} rt=${RT} ..."
timeout $((RT + 240)) /usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none \
    -np 2 ./bin/hpcg.sh --exec-name ./bin/xhpcg \
    --nx "${NX}" --ny "${NY}" --nz "${NZ}" --rt "${RT}" \
    --gpu-affinity "${GPU_AFFINITY}" --cpu-affinity "${CPU_AFFINITY}" \
    --p2p 0 --b 1 \
    --npx "${NPX}" --npy "${NPY}" --npz "${NPZ}" \
    ${EXTRA} \
    > "${OUT}/hpcg.log" 2>&1

echo "[trace] HPCG finished"
kill "${MON}" || true
wait "${MON}" 2>/dev/null || true

# Generate report
python3 "${SCRIPT_DIR}/analyze_gpu_monitor.py" \
    "${OUT}/gpu_metrics.csv" "${OUT}/hpcg.log" \
    > "${OUT}/summary.md"

echo "[trace] Done: ${OUT}"
echo "[trace] CSV:      ${OUT}/gpu_metrics.csv"
echo "[trace] HPCG log: ${OUT}/hpcg.log"
echo "[trace] Summary:  ${OUT}/summary.md"
