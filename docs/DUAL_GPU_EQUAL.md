# Dual-GPU HPCG (Equal and Heterogeneous Workloads)

This branch adds build and runtime support needed to run NVIDIA `nvidia-hpcg`
on a heterogeneous dual-GPU workstation (RTX 4090 + RTX 5080). Two workload
modes are supported:

1. **Equal workload** – both MPI ranks own the same local domain size.
2. **Heterogeneous workload** – the faster GPU is given a larger domain to
   reduce `MPI_Allreduce` waiting time.

## Equal workload

The global problem is divided evenly among two MPI ranks. Each rank owns the
same local domain size, so no per-rank load balancing is performed. This is the
simplest way to use both GPUs and corresponds to the existing `GPUONLY`
execution mode (`--exm 0`) of `nvidia-hpcg`.

Example:

```text
Global problem: 128 x 128 x 256
Process grid : 1 x 1 x 2
Rank 0 local : 128 x 128 x 128  -> CUDA device 0 -> RTX 4090
Rank 1 local : 128 x 128 x 128  -> CUDA device 1 -> RTX 5080
```

## Build changes

Only `setup/Make.CUDA_X86` is modified:

1. Add GPU architectures for Ada Lovelace and Blackwell:
   ```makefile
   CUDA_ARCH += -gencode=arch=compute_89,code=sm_89 \
                -gencode=arch=compute_120,code=sm_120 \
                -gencode=arch=compute_120,code=compute_120
   ```
2. Link the OpenMPI C++ bindings library (`libmpi_cxx`) required by the
   system OpenMPI package on Ubuntu 22.04:
   ```makefile
   HPCG_LIBS = -L${MPlib} -lmpi -lmpi_cxx
   ```

## Build instructions

```bash
cd /workspace/arch-ori-opt/hpcg/nvidia-hpcg
export MPI_PATH=/usr/lib/x86_64-linux-gnu/openmpi
export CUDA_PATH=/usr/local/cuda
export MATHLIBS_PATH=/usr/local/cuda
export NCCL_PATH=/usr
export NVPL_SPARSE_PATH=/usr/local/cuda
bash build_sample.sh "" "" "" "" 1 0 1 0
```

The last five arguments enable: `USE_CUDA=1`, `USE_NCCL=1`, no Grace,
no engineering version.

## SSH / MPI launcher setup

OpenMPI uses SSH to spawn ranks even when all processes stay on the local
node. In this environment the SSH server listens on port **51111**, so
OpenMPI (and therefore `mpirun`) must be told how to connect.

There are two common ways:

### Option A: `~/.ssh/config` (recommended)

Create or edit `~/.ssh/config` so that plain `ssh localhost` uses port 51111:

```bash
mkdir -p ~/.ssh
cat > ~/.ssh/config <<'EOF'
Host localhost 127.0.0.1 dell-server
    HostName 127.0.0.1
    Port 51111
    StrictHostKeyChecking no
    BatchMode yes
    PasswordAuthentication no
    ForwardX11 no
EOF
chmod 600 ~/.ssh/config
```

After this, no wrapper script or extra environment variable is needed.

### Option B: SSH wrapper script

Create a wrapper and point OpenMPI to it:

```bash
cat > /tmp/ssh_for_mpi <<'EOF'
#!/bin/bash
exec ssh -p 51111 -o StrictHostKeyChecking=no -o BatchMode=yes \
     -o PasswordAuthentication=no -o ForwardX11=no "$@"
EOF
chmod +x /tmp/ssh_for_mpi
export OMPI_MCA_plm_rsh_agent=/tmp/ssh_for_mpi
```

The wrapper is more fragile because `/tmp` files may be cleaned on reboot.

## Run dual-GPU equal workload

A helper script is provided:

```bash
bash scripts/run_dual_equal.sh
```

Equivalent manual command:

```bash
export HWLOC_COMPONENTS="-gl"

/usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none -np 2 \
  ./bin/hpcg.sh --exec-name ./bin/xhpcg \
  --nx 128 --ny 128 --nz 128 --rt 60 \
  --gpu-affinity 0:1 --cpu-affinity 0:24 \
  --p2p 0 --b 1 \
  --npx 1 --npy 1 --npz 2
```

## Heterogeneous workload

The RTX 4090 is roughly 1.5–1.6× faster than the RTX 5080 in this HPCG build.
When both ranks own the same domain, the faster GPU waits at `MPI_Allreduce`
barriers. The `--het-split` option gives the faster GPU a larger domain along
one dimension (Z by default) to reduce this waiting.

Example:

```text
Global problem: 128 x 128 x 256
Process grid : 1 x 1 x 2
Rank 0 local : 128 x 128 x 160  -> CUDA device 0 -> RTX 4090 (fast)
Rank 1 local : 128 x 128 x 96   -> CUDA device 1 -> RTX 5080 (slow)
Actual work ratio              : 160 / 96 = 1.667
```

The slow rank's dimension is rounded to the nearest multiple of 16 so that the
coarsest multigrid level keeps an even size; otherwise cuSPARSE's SpSV analysis
can crash.

Run it with:

```bash
bash scripts/run_dual_het.sh
```

Equivalent manual command:

```bash
export HWLOC_COMPONENTS="-gl"

/usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none -np 2 \
  ./bin/hpcg.sh --exec-name ./bin/xhpcg \
  --nx 128 --ny 128 --nz 160 --rt 60 \
  --gpu-affinity 0:1 --cpu-affinity 0:24 \
  --p2p 0 --b 1 \
  --npx 1 --npy 1 --npz 2 \
  --het-split 1 --het-ratio 1.6
```

`--het-ratio` is the desired fast/slow work ratio (must be `>= 1.0`). The code
rounds the slow rank's split dimension to a multiple of 16, so the actual ratio
may differ slightly from the requested value.

## Automated benchmark suite

`scripts/run_full_benchmark.sh` runs all reference configurations back-to-back
and produces a `summary.md` report:

1. RTX 4090 single, local `128³`
2. RTX 5080 single, local `128³`
3. Dual equal, global `128×128×256` (rank `128³`)
4. Dual equal, global `128³` (rank `64³`)
5. Dual heterogeneous, global `128×128×256` (ranks `128×128×160` / `128×128×96`)

The script also captures an Nsight Systems profile with GPU hardware metrics for
each configuration. Output is written to `benchmark_data/<timestamp>/`, which is
`gitignore`d.

```bash
bash scripts/run_full_benchmark.sh
# -> benchmark_data/YYYY-MM-DD_HH-MM-SS/summary.md
```

## CUDA device ordering note

On this node `nvidia-smi` and CUDA enumerate GPUs in opposite order:

| `nvidia-smi` index | CUDA device | Card |
|--------------------|-------------|------|
| 0 | 1 | RTX 5080 (Blackwell, sm_120) |
| 1 | 0 | RTX 4090 (Ada, sm_89) |

Therefore `--gpu-affinity 0:1` maps rank 0 to the RTX 4090 and rank 1 to the
RTX 5080. This is important when interpreting performance numbers.

## Expected results

An equal-workload run should produce a line like:

```text
Final Summary::HPCG result is VALID with a GFLOP/s rating of=210.0
```

A heterogeneous run with the settings above should be similar or slightly
higher, with much smaller `DDOT Timing Variations::Max - Min` because the
workload is better matched to the two GPUs' performance.

The exact number depends on GPU clocks and thermal state.

## Single-GPU reference runs

To compare against single-GPU performance, use `scripts/run_single_gpu.sh`:

```bash
# RTX 4090 (CUDA device 0)
bash scripts/run_single_gpu.sh 0 /tmp/rtx4090 60

# RTX 5080 (CUDA device 1)
bash scripts/run_single_gpu.sh 1 /tmp/rtx5080 60
```

Equivalent manual commands:

```bash
# RTX 4090 (CUDA device 0)
mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none -np 1 \
  ./bin/hpcg.sh --exec-name ./bin/xhpcg \
  --nx 128 --ny 128 --nz 128 --rt 60 \
  --gpu-affinity 0 --cpu-affinity 0 --p2p 0 --b 1

# RTX 5080 (CUDA device 1)
mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none -np 1 \
  ./bin/hpcg.sh --exec-name ./bin/xhpcg \
  --nx 128 --ny 128 --nz 128 --rt 60 \
  --gpu-affinity 1 --cpu-affinity 0 --p2p 0 --b 1
```

## Scope of this branch

- Build-time architecture support for sm_89 (RTX 4090) and sm_120 (RTX 5080).
- Convenience runners for equal and heterogeneous dual-GPU execution.
- Heterogeneous split support in HPCG parameter parsing and geometry generation
  (`--het-split`, `--het-dim`, `--het-ratio`).
- A fix for the GPU halo exchange path (`extToLocMap` allocation) when MPI
  neighbors have different local sizes.
