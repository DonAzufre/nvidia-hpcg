#!/usr/bin/env bash
#
# Run a single dimension-partitioning case and append to CSV.
#
# Usage:
#   bash scripts/run_dim_partition_case.sh <config_name> <rt> <nx> <ny> <nz> <npx> <npy> <npz> [het_dim] [het_ratio]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

export HWLOC_COMPONENTS="-gl"

CONFIG="$1"
RT="$2"
NX="$3"; NY="$4"; NZ="$5"
NPX="$6"; NPY="$7"; NPZ="$8"
HET_DIM="${9:-}"
HET_RATIO="${10:-}"

OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/dim_partition_study/current}"
mkdir -p "${OUTPUT_DIR}"
CSV="${OUTPUT_DIR}/dim_partition_study.csv"
LOG="${OUTPUT_DIR}/study.log"

if [[ ! -f "${CSV}" ]]; then
    echo "config,split_type,dim,nx,ny,nz,npx,npy,npz,het_split,het_dim,het_ratio,fast_size,slow_size,global_nx,global_ny,global_nz,rt,valid,gflops_rating,hist_rating,exec_time,ddot_min,ddot_max,ddot_avg,wall_time,status" > "${CSV}"
fi

SPLIT_TYPE="equal"
HET_SPLIT="0"
DIM="Z"
case "${CONFIG}" in
    *X*) DIM="X" ;;
    *Y*) DIM="Y" ;;
    *Z*) DIM="Z" ;;
esac
if [[ "${CONFIG}" == ASYMM_* ]]; then SPLIT_TYPE="asymmetric"; HET_SPLIT="1"; fi

EXTRA_ARGS=()
if [[ "${HET_SPLIT}" == "1" ]]; then
    EXTRA_ARGS=(--het-split 1 --het-dim "${HET_DIM}" --het-ratio "${HET_RATIO}")
fi

CFG_DIR="${OUTPUT_DIR}/${CONFIG}"
mkdir -p "${CFG_DIR}"

{
echo ""
echo "[START] ${CONFIG}  local=${NX}x${NY}x${NZ} grid=${NPX}x${NPY}x${NPZ} rt=${RT} ${EXTRA_ARGS[*]}"
} >> "${LOG}"

start=$(date +%s.%N)

# Per-config wall-clock timeout of 240s
set +e
timeout 240 /usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none \
    -np 2 ./bin/hpcg.sh --exec-name ./bin/xhpcg \
    --nx "${NX}" --ny "${NY}" --nz "${NZ}" --rt "${RT}" \
    --gpu-affinity 0:1 --cpu-affinity 0-7:8-15 \
    --p2p 0 --b 1 \
    --npx "${NPX}" --npy "${NPY}" --npz "${NPZ}" \
    "${EXTRA_ARGS[@]}" \
    > "${CFG_DIR}/hpcg.log" 2>&1
RUN_EXIT=$?
set -e

end=$(date +%s.%N)
elapsed=$(awk "BEGIN {printf \"%.2f\", ${end} - ${start}}")

status="OK"
if [[ ${RUN_EXIT} -eq 124 ]]; then
    status="TIMEOUT"
elif [[ ${RUN_EXIT} -ne 0 ]]; then
    status="ERROR_${RUN_EXIT}"
fi

valid="UNKNOWN"
gflops=""
hist=""
exec_time=""
ddot_min=""
ddot_max=""
ddot_avg=""
LOG_PATH="${CFG_DIR}/hpcg.log"

if [[ -r "${LOG_PATH}" ]]; then
    if grep -q "HPCG result is VALID" "${LOG_PATH}"; then
        valid="VALID"
    elif grep -qE "HPCG result is INVALID|ERROR|failure|Segmentation fault|MPI_Abort" "${LOG_PATH}"; then
        valid="INVALID"
    fi
    gflops=$(grep -oP 'HPCG result is VALID with a GFLOP/s rating of=\K[0-9.]+' "${LOG_PATH}" | tail -1 || true)
    hist=$(grep -oP 'HPCG 2\.4 rating for historical reasons is=\K[0-9.]+' "${LOG_PATH}" | tail -1 || true)
    exec_time=$(grep -oP 'Results are valid but execution time \(sec\) is=\K[0-9.]+' "${LOG_PATH}" | tail -1 || true)
    ddot_min=$(grep -oP 'DDOT Timing Variations::Min DDOT MPI_Allreduce time=\K[0-9.]+' "${LOG_PATH}" | tail -1 || true)
    ddot_max=$(grep -oP 'DDOT Timing Variations::Max DDOT MPI_Allreduce time=\K[0-9.]+' "${LOG_PATH}" | tail -1 || true)
    ddot_avg=$(grep -oP 'DDOT Timing Variations::Avg DDOT MPI_Allreduce time=\K[0-9.]+' "${LOG_PATH}" | tail -1 || true)
fi

fast_size=""
slow_size=""
if [[ -r "${LOG_PATH}" ]]; then
    sizes=$(grep -oP 'Local Domain: \K\d+x\d+x\d+' "${LOG_PATH}" | sort -u | tr '\n' ';' || true)
    fast_size=$(echo "${sizes}" | cut -d';' -f1)
    slow_size=$(echo "${sizes}" | cut -d';' -f2)
fi

global_nx=$((NPX * NX))
global_ny=$((NPY * NY))
global_nz=$((NPZ * NZ))
if [[ "${HET_SPLIT}" == "1" && -r "${LOG_PATH}" ]]; then
    global_nx=$(grep -oP 'Global Problem Dimensions::Global nx=\K\d+' "${LOG_PATH}" | tail -1 || echo "${global_nx}")
    global_ny=$(grep -oP 'Global Problem Dimensions::Global ny=\K\d+' "${LOG_PATH}" | tail -1 || echo "${global_ny}")
    global_nz=$(grep -oP 'Global Problem Dimensions::Global nz=\K\d+' "${LOG_PATH}" | tail -1 || echo "${global_nz}")
fi

echo "${CONFIG},${SPLIT_TYPE},${DIM},${NX},${NY},${NZ},${NPX},${NPY},${NPZ},${HET_SPLIT},${HET_DIM},${HET_RATIO},\"${fast_size}\",\"${slow_size}\",${global_nx},${global_ny},${global_nz},${RT},${valid},${gflops},${hist},${exec_time},${ddot_min},${ddot_max},${ddot_avg},${elapsed},${status}" >> "${CSV}"

{
echo "[END]   ${CONFIG} valid=${valid} gflops=${gflops} exec_time=${exec_time}s wall_time=${elapsed}s status=${status}"
} >> "${LOG}"
