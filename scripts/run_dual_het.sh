#!/usr/bin/env bash
#
# Run HPCG on two local GPUs with an asymmetric (heterogeneous) workload split.
#
# The faster GPU (RTX 4090, CUDA device 0) is given more work than the slower
# GPU (RTX 5080, CUDA device 1) based on the --het-ratio parameter.  The split
# is applied along the Z dimension using the new --het-split option.
#
# Environment assumptions:
#   - OpenMPI is installed and mpirun is available.
#   - Passwordless SSH on port 51111 is configured (required by OpenMPI).
#     This can be done either via ~/.ssh/config or via an SSH wrapper script
#     pointed to by OMPI_MCA_plm_rsh_agent.  This script assumes ~/.ssh/config
#     contains an entry for localhost/127.0.0.1 using port 51111.
#   - HWLOC_PCI_DOMAIN environment quirks are worked around.
#   - The executable ./bin/xhpcg has been built with sm_89/sm_120 support.
#
# CUDA device note:
#   nvidia-smi and CUDA enumerate devices in opposite order on this node.
#   --gpu-affinity 0 -> CUDA device 0 -> RTX 4090 (Ada, sm_89)
#   --gpu-affinity 1 -> CUDA device 1 -> RTX 5080 (Blackwell, sm_120)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

# OpenMPI uses SSH to spawn ranks even on the local node.  If the local SSH
# server listens on a non-standard port (51111 in this environment), OpenMPI
# must be told about it.  The simplest robust way is to configure ssh itself
# via ~/.ssh/config instead of maintaining a transient wrapper script.
#
# Example ~/.ssh/config entry:
#   Host localhost 127.0.0.1 dell-server
#       HostName 127.0.0.1
#       Port 51111
#       StrictHostKeyChecking no
#       BatchMode yes
#       PasswordAuthentication no
#       ForwardX11 no
#
# Alternatively, set OMPI_MCA_plm_rsh_agent=/path/to/ssh_wrapper where the
# wrapper calls: ssh -p 51111 -o StrictHostKeyChecking=no -o BatchMode=yes ...
export HWLOC_COMPONENTS="-gl"

# Problem: 128 x 128 x 256 global, split 1x1x2 with het-split.
# --nz is the fast-GPU local Z size; the slow GPU gets a reduced size based
# on --het-ratio.  With nz=160 and ratio=1.6 the slow size rounds to 96,
# giving a total global Z of 160 + 96 = 256 and an actual work ratio of
# 160/96 = 1.667.
NX=128
NY=128
NZ=160
RT=60
NPX=1
NPY=1
NPZ=2
HET_SPLIT=1
HET_RATIO=1.6
GPU_AFFINITY="0:1"
CPU_AFFINITY="0:24"
P2P=0

/usr/bin/mpirun.openmpi --allow-run-as-root \
    --oversubscribe \
    --bind-to none \
    -np 2 \
    ./bin/hpcg.sh \
        --exec-name ./bin/xhpcg \
        --nx "${NX}" --ny "${NY}" --nz "${NZ}" --rt "${RT}" \
        --gpu-affinity "${GPU_AFFINITY}" \
        --cpu-affinity "${CPU_AFFINITY}" \
        --p2p "${P2P}" --b 1 \
        --npx "${NPX}" --npy "${NPY}" --npz "${NPZ}" \
        --het-split "${HET_SPLIT}" --het-ratio "${HET_RATIO}"
