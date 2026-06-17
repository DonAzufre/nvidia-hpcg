#!/usr/bin/env bash
#
# Tune --gpu-slice-size for single-GPU HPCG on RTX 4090.
# Tests a range of slice sizes and records GFLOP/s and VALID status.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

export HWLOC_COMPONENTS="-gl"

GPU=0
CPU="0-7"
NX=128
NY=128
NZ=128
RT=30

OUT_DIR="benchmark_data/slice_size_tuning/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${OUT_DIR}"

SUMMARY="${OUT_DIR}/summary.csv"
echo "slice_size,gflops,valid,exec_time,total_iters" > "${SUMMARY}"

for SLICE in 512 1024 2048 4096 8192 16384; do
    echo "=== slice_size=${SLICE} ==="
    LOG="${OUT_DIR}/slice_${SLICE}.log"
    /usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none -np 1 \
        ./bin/hpcg.sh --exec-name ./bin/xhpcg \
        --nx "${NX}" --ny "${NY}" --nz "${NZ}" --rt "${RT}" \
        --gpu-affinity "${GPU}" --cpu-affinity "${CPU}" \
        --p2p 0 --b 1 --gss "${SLICE}" \
        > "${LOG}" 2>&1 || true

    GFLOPS=$(grep "HPCG result is VALID with a GFLOP/s rating of=" "${LOG}" | sed 's/.*=//' | tr -d ' ' || echo "NA")
    VALID=$(grep -c "HPCG result is VALID" "${LOG}" || echo 0)
    EXEC_TIME=$(grep "Results are valid but execution time (sec) is=" "${LOG}" | sed 's/.*=//' | tr -d ' ' || echo "NA")
    ITERS=$(grep "Total number of optimized iterations=" "${LOG}" | sed 's/.*=//' | tr -d ' ' || echo "NA")
    echo "${SLICE},${GFLOPS},${VALID},${EXEC_TIME},${ITERS}" >> "${SUMMARY}"
    echo "  GFLOP/s=${GFLOPS}, VALID=${VALID}"
done

echo ""
echo "Summary written to: ${SUMMARY}"
cat "${SUMMARY}"
