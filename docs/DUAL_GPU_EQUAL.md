# Dual-GPU Equal Workload HPCG

This branch adds the minimal build and runtime support needed to run NVIDIA
`nvidia-hpcg` on a heterogeneous dual-GPU workstation (RTX 4090 + RTX 5080)
with an **equal workload split** across the two GPUs.

## What is "equal workload"?

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

## Run dual-GPU equal workload

A helper script is provided:

```bash
bash scripts/run_dual_equal.sh
```

Equivalent manual command:

```bash
export OMPI_MCA_plm_rsh_agent=/tmp/ssh_for_mpi
export HWLOC_COMPONENTS="-gl"

/usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none -np 2 \
  ./bin/hpcg.sh --exec-name ./bin/xhpcg \
  --nx 128 --ny 128 --nz 256 --rt 60 \
  --gpu-affinity 0:1 --cpu-affinity 0:24 \
  --p2p 0 --b 1 \
  --npx 1 --npy 1 --npz 2
```

## CUDA device ordering note

On this node `nvidia-smi` and CUDA enumerate GPUs in opposite order:

| `nvidia-smi` index | CUDA device | Card |
|--------------------|-------------|------|
| 0 | 1 | RTX 5080 (Blackwell, sm_120) |
| 1 | 0 | RTX 4090 (Ada, sm_89) |

Therefore `--gpu-affinity 0:1` maps rank 0 to the RTX 4090 and rank 1 to the
RTX 5080. This is important when interpreting performance numbers.

## Expected result

The run should produce a line like:

```text
Final Summary::HPCG result is VALID with a GFLOP/s rating of=210.0
```

The exact number depends on GPU clocks and thermal state. Because the two GPUs
have different performance, the faster GPU waits for the slower one at
`MPI_Allreduce` barriers (visible in the `DDOT Timing Variations` section).

## Single-GPU reference runs

To compare against single-GPU performance:

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

- Only build-time architecture support and a convenience runner are added.
- No changes to HPCG core algorithms or geometry generation.
- No asymmetric load-balancing mode is included.
