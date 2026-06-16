#!/usr/bin/env bash
#
# Run HPCG on a single GPU.
#
# Usage:
#   bash scripts/run_single_gpu.sh <gpu_affinity> <output_dir> [rt]
#
# Arguments:
#   gpu_affinity   CUDA visible device index (0 = RTX 4090, 1 = RTX 5080)
#   output_dir     Directory where hpcg.log will be written
#   rt             Runtime in seconds (default: 60)

set -euo pipefail

GPU_AFFINITY="${1}"
OUTPUT_DIR="${2}"
RT="${3:-60}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

mkdir -p "${OUTPUT_DIR}"

export HWLOC_COMPONENTS="-gl"

/usr/bin/mpirun.openmpi --allow-run-as-root \
    --oversubscribe \
    --bind-to none \
    -np 1 \
    ./bin/hpcg.sh \
        --exec-name ./bin/xhpcg \
        --nx 128 --ny 128 --nz 128 --rt "${RT}" \
        --gpu-affinity "${GPU_AFFINITY}" \
        --cpu-affinity 0 \
        --p2p 0 --b 1 \
        > "${OUTPUT_DIR}/hpcg.log" 2>&1

echo "Single-GPU run finished: ${OUTPUT_DIR}/hpcg.log"
