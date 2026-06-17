#!/usr/bin/env bash
#
# Systematic dual-GPU dimension partitioning study for nvidia-hpcg.
#
# Tests symmetric (equal) and asymmetric (het-split) partitioning along
# X, Y, and Z dimensions at a fixed local base size of 128^3.
#
# Usage:
#   bash scripts/run_dim_partition_study.sh [rt]
#
# Output:
#   benchmark_data/dim_partition_study/<timestamp>/<config>/hpcg.log
#   benchmark_data/dim_partition_study/<timestamp>/dim_partition_study.csv

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

export HWLOC_COMPONENTS="-gl"

RT="${1:-30}"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR="benchmark_data/dim_partition_study/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

CSV="${OUTPUT_DIR}/dim_partition_study.csv"

log() {
    echo "$@" | tee -a "${OUTPUT_DIR}/study.log"
}

# CSV header
echo "config,split_type,dim,nx,ny,nz,npx,npy,npz,het_split,het_dim,het_ratio,fast_size,slow_size,global_nx,global_ny,global_nz,rt,valid,gflops_rating,hist_rating,exec_time,ddot_min,ddot_max,ddot_avg,wall_time" > "${CSV}"

run_case() {
    local config="$1"
    local split_type="$2"   # equal or asymmetric
    local dim="$3"          # X, Y, or Z
    local nx="$4"
    local ny="$5"
    local nz="$6"
    local npx="$7"
    local npy="$8"
    local npz="$9"
    local het_split="${10}"
    local het_dim="${11}"
    local het_ratio="${12}"

    local cfg_dir="${OUTPUT_DIR}/${config}"
    mkdir -p "${cfg_dir}"

    local extra_args=()
    if [[ "${het_split}" == "1" ]]; then
        extra_args=(--het-split 1 --het-dim "${het_dim}" --het-ratio "${het_ratio}")
    fi

    log ""
    log "[START] ${config}  local=${nx}x${ny}x${nz} grid=${npx}x${npy}x${npz} rt=${RT} ${extra_args[*]}"
    local start end elapsed
    start=$(date +%s.%N)

    /usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none \
        -np 2 ./bin/hpcg.sh --exec-name ./bin/xhpcg \
        --nx "${nx}" --ny "${ny}" --nz "${nz}" --rt "${RT}" \
        --gpu-affinity 0:1 --cpu-affinity 0-7:8-15 \
        --p2p 0 --b 1 \
        --npx "${npx}" --npy "${npy}" --npz "${npz}" \
        "${extra_args[@]}" \
        > "${cfg_dir}/hpcg.log" 2>&1 || true

    end=$(date +%s.%N)
    elapsed=$(awk "BEGIN {printf \"%.2f\", ${end} - ${start}}")

    local valid="UNKNOWN"
    local gflops=""
    local hist=""
    local exec_time=""
    local ddot_min=""
    local ddot_max=""
    local ddot_avg=""
    local log_path="${cfg_dir}/hpcg.log"

    if [[ -r "${log_path}" ]]; then
        if grep -q "HPCG result is VALID" "${log_path}"; then
            valid="VALID"
        elif grep -qE "HPCG result is INVALID|ERROR|failure|Segmentation fault|MPI_Abort" "${log_path}"; then
            valid="INVALID"
        fi
        gflops=$(grep -oP 'HPCG result is VALID with a GFLOP/s rating of=\K[0-9.]+' "${log_path}" | tail -1 || true)
        hist=$(grep -oP 'HPCG 2\.4 rating for historical reasons is=\K[0-9.]+' "${log_path}" | tail -1 || true)
        exec_time=$(grep -oP 'Results are valid but execution time \(sec\) is=\K[0-9.]+' "${log_path}" | tail -1 || true)
        ddot_min=$(grep -oP 'DDOT Timing Variations::Min DDOT MPI_Allreduce time=\K[0-9.]+' "${log_path}" | tail -1 || true)
        ddot_max=$(grep -oP 'DDOT Timing Variations::Max DDOT MPI_Allreduce time=\K[0-9.]+' "${log_path}" | tail -1 || true)
        ddot_avg=$(grep -oP 'DDOT Timing Variations::Avg DDOT MPI_Allreduce time=\K[0-9.]+' "${log_path}" | tail -1 || true)
    fi

    # Determine actual fast/slow sizes from the log
    local fast_size=""
    local slow_size=""
    if [[ -r "${log_path}" ]]; then
        local sizes
        sizes=$(grep -oP 'Local Domain: \K\d+x\d+x\d+' "${log_path}" | sort -u | tr '\n' ';' || true)
        fast_size=$(echo "${sizes}" | cut -d';' -f1)
        slow_size=$(echo "${sizes}" | cut -d';' -f2)
    fi

    # Global sizes
    local global_nx=$((npx * nx))
    local global_ny=$((npy * ny))
    local global_nz=$((npz * nz))
    # For asymmetric, global size changes; compute from log if available
    if [[ "${het_split}" == "1" && -r "${log_path}" ]]; then
        global_nx=$(grep -oP 'Global Problem Dimensions::Global nx=\K\d+' "${log_path}" | tail -1 || echo "${global_nx}")
        global_ny=$(grep -oP 'Global Problem Dimensions::Global ny=\K\d+' "${log_path}" | tail -1 || echo "${global_ny}")
        global_nz=$(grep -oP 'Global Problem Dimensions::Global nz=\K\d+' "${log_path}" | tail -1 || echo "${global_nz}")
    fi

    echo "${config},${split_type},${dim},${nx},${ny},${nz},${npx},${npy},${npz},${het_split},${het_dim},${het_ratio},\"${fast_size}\",\"${slow_size}\",${global_nx},${global_ny},${global_nz},${RT},${valid},${gflops},${hist},${exec_time},${ddot_min},${ddot_max},${ddot_avg},${elapsed}" >> "${CSV}"
    log "[END]   ${config} valid=${valid} gflops=${gflops} exec_time=${exec_time}s wall_time=${elapsed}s"
}

log "================================================================================"
log "Dimension partitioning study started at $(date -Iseconds)"
log "Output directory: ${OUTPUT_DIR}"
log "Target runtime per config: rt=${RT}"
log "================================================================================"

overall_start=$(date +%s.%N)

# Symmetric (equal) splits along each dimension
# Base local size 128^3; process grid places the 2 ranks along the chosen dimension.
run_case "EQUAL_X"  "equal"        "X" 128 128 128 2 1 1 0 "" ""
run_case "EQUAL_Y"  "equal"        "Y" 128 128 128 1 2 1 0 "" ""
run_case "EQUAL_Z"  "equal"        "Z" 128 128 128 1 1 2 0 "" ""

# Asymmetric splits along each dimension
# het-dim: 1=X, 2=Y, 3=Z
run_case "ASYMM_X"  "asymmetric"   "X" 128 128 128 2 1 1 1 1 1.6
run_case "ASYMM_Y"  "asymmetric"   "Y" 128 128 128 1 2 1 1 2 1.6
run_case "ASYMM_Z"  "asymmetric"   "Z" 128 128 128 1 1 2 1 3 1.6

overall_end=$(date +%s.%N)
total_elapsed=$(awk "BEGIN {printf \"%.2f\", ${overall_end} - ${overall_start}}")

log ""
log "================================================================================"
log "Dimension partitioning study finished at $(date -Iseconds)"
log "Total wall-clock time: ${total_elapsed}s"
log "CSV: ${CSV}"
log "================================================================================"

# Also copy the CSV to the canonical location expected by the report pipeline
mkdir -p "benchmark_data"
cp "${CSV}" "benchmark_data/dim_partition_study.csv"
log "Copied summary CSV to benchmark_data/dim_partition_study.csv"
