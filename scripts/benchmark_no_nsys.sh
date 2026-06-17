#!/usr/bin/env bash
#
# Run the five standard HPCG configurations without nsys profiling and record
# wall-clock timings.  Optionally run additional scaling experiments.
#
# Usage:
#   scripts/benchmark_no_nsys.sh                  # five standard configs only
#   scripts/benchmark_no_nsys.sh --scaling          # five configs + scaling experiments
#   scripts/benchmark_no_nsys.sh --scaling-only     # scaling experiments only
#   scripts/benchmark_no_nsys.sh --config <name>    # run one config by name
#
# Supported config names: RTX4090, RTX5080, DUAL_EQUAL, DUAL_EQUAL_128, DUAL_HET,
# DUAL_EQUAL_160, DUAL_EQUAL_192, DUAL_HET_192.
#
# Output:
#   benchmark_data/scaling_experiments.log (timings, validity, GFLOP/s)
#   benchmark_data/no_nsys_<timestamp>/<config>/hpcg.log
#

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

export HWLOC_COMPONENTS="-gl"

LOG_FILE="benchmark_data/scaling_experiments.log"
mkdir -p "$(dirname "$LOG_FILE")"

RT=60
OUTPUT_DIR="benchmark_data/no_nsys_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${OUTPUT_DIR}"

SCALING=0
SCALING_ONLY=0
SELECTED_CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scaling) SCALING=1; shift ;;
        --scaling-only) SCALING_ONLY=1; shift ;;
        --config)
            if [[ -n "${2:-}" ]]; then
                SELECTED_CONFIG="$2"
                shift 2
            else
                echo "Missing argument for --config"; exit 1
            fi
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

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

    log ""
    log "[$(date -Iseconds)] START ${name}  np=${np} local=${nx}x${ny}x${nz} gpu=${gpu_aff} cpu=${cpu_aff} rt=${RT}"
    local start end elapsed
    start=$(date +%s.%N)

    # Do not stop the suite if an individual scaling experiment fails.
    /usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none \
        -np "${np}" ./bin/hpcg.sh --exec-name ./bin/xhpcg \
        --nx "${nx}" --ny "${ny}" --nz "${nz}" --rt "${RT}" \
        --gpu-affinity "${gpu_aff}" --cpu-affinity "${cpu_aff}" \
        --p2p 0 --b 1 "${extra_args[@]}" \
        > "${cfg_dir}/hpcg.log" 2>&1 || true

    end=$(date +%s.%N)
    elapsed=$(awk "BEGIN {printf \"%.2f\", ${end} - ${start}}")

    log "[$(date -Iseconds)] END   ${name}  wall_time=${elapsed}s"

    local valid="UNKNOWN"
    local gflops="N/A"
    local exec_time="N/A"
    if [[ -r "${cfg_dir}/hpcg.log" ]]; then
        if grep -q "HPCG result is VALID" "${cfg_dir}/hpcg.log"; then
            valid="VALID"
        elif grep -qE "HPCG result is INVALID|ERROR|failure|Segmentation fault" "${cfg_dir}/hpcg.log"; then
            valid="INVALID"
        fi
        gflops=$(grep -oP 'HPCG result is VALID with a GFLOP/s rating of=\K[0-9.]+' "${cfg_dir}/hpcg.log" | tail -1 || echo "N/A")
        exec_time=$(grep -oP 'Results are valid but execution time \(sec\) is=\K[0-9.]+' "${cfg_dir}/hpcg.log" | tail -1 || echo "N/A")
    fi

    log "  result=${valid} gflops=${gflops} exec_time=${exec_time}s log=${cfg_dir}/hpcg.log"
}

log "================================================================================"
log "HPCG no-nsys timing run started at $(date -Iseconds)"
log "Output directory: ${OUTPUT_DIR}"
log "================================================================================"

overall_start=$(date +%s.%N)

run_baseline_configs() {
    log ""
    log "--- Phase 1: five standard configurations ---"

    run_config "RTX4090"       1 128 128 128 "0"   "0-7"
    run_config "RTX5080"       1 128 128 128 "1"   "8-15"
    run_config "DUAL_EQUAL"    2 128 128 128 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2
    run_config "DUAL_EQUAL_128" 2 128 128 64  "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2
    run_config "DUAL_HET"      2 128 128 160 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2 --het-split 1 --het-ratio 1.6
}

run_scaling_configs() {
    log ""
    log "--- Phase 2: scaling experiments ---"

    # Local 160^3 per rank, global 160x160x320, equal split.
    run_config "DUAL_EQUAL_160" 2 160 160 160 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2

    # Local 192^3 per rank, global 192x192x384, equal split.
    run_config "DUAL_EQUAL_192" 2 192 192 192 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2

    # Heterogeneous split: fast GPU (RTX 4090) gets 192x192x192,
    # slow GPU (RTX 5080) gets 192x192x128 (rounded to multiple of 16).
    run_config "DUAL_HET_192" 2 192 192 192 "0:1" "0-7:8-15" \
        --npx 1 --npy 1 --npz 2 --het-split 1 --het-ratio 1.6
}

if [[ -n "${SELECTED_CONFIG}" ]]; then
    log ""
    log "--- Single config: ${SELECTED_CONFIG} ---"
    case "${SELECTED_CONFIG}" in
        RTX4090)        run_config "RTX4090"       1 128 128 128 "0"   "0-7" ;;
        RTX5080)        run_config "RTX5080"       1 128 128 128 "1"   "8-15" ;;
        DUAL_EQUAL)     run_config "DUAL_EQUAL"    2 128 128 128 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2 ;;
        DUAL_EQUAL_128) run_config "DUAL_EQUAL_128" 2 128 128 64  "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2 ;;
        DUAL_HET)       run_config "DUAL_HET"      2 128 128 160 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2 --het-split 1 --het-ratio 1.6 ;;
        DUAL_EQUAL_160) run_config "DUAL_EQUAL_160" 2 160 160 160 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2 ;;
        DUAL_EQUAL_192) run_config "DUAL_EQUAL_192" 2 192 192 192 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2 ;;
        DUAL_HET_192)   run_config "DUAL_HET_192" 2 192 192 192 "0:1" "0-7:8-15" --npx 1 --npy 1 --npz 2 --het-split 1 --het-ratio 1.6 ;;
        *) echo "Unknown config: ${SELECTED_CONFIG}"; exit 1 ;;
    esac
else
    if [[ ${SCALING_ONLY} -eq 0 ]]; then
        run_baseline_configs
    fi
    if [[ ${SCALING} -eq 1 || ${SCALING_ONLY} -eq 1 ]]; then
        run_scaling_configs
    fi
fi

overall_end=$(date +%s.%N)
total_elapsed=$(awk "BEGIN {printf \"%.2f\", ${overall_end} - ${overall_start}}")

log ""
log "================================================================================"
log "Total wall-clock time for this run: ${total_elapsed}s"
log "HPCG no-nsys timing run ended at $(date -Iseconds)"
log "================================================================================"
log ""
