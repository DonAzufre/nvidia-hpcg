# nvidia-hpcg 异构双 GPU 实验指南

本仓库在 NVIDIA `nvidia-hpcg` 基础上，针对一台搭载 **RTX 4090**（快卡，CUDA 0，sm_89）和 **RTX 5080**（慢卡，CUDA 1，sm_120）的异构双 GPU 工作站进行了扩展，支持等分和异构两种双卡负载模式。

> 说明：CUDA 枚举顺序与 `nvidia-smi` 相反。本环境中 `CUDA 0` 对应 RTX 4090，`CUDA 1` 对应 RTX 5080。

---

## 1. 环境要求

- Linux（已验证 Ubuntu 22.04）
- CUDA Toolkit 12.3+（当前使用 CUDA 13.2）
- OpenMPI 4.1+（当前使用 4.1.2）
- NCCL（用于 GPU 间 P2P）
- `typst`、`uv`（用于生成实验报告）

---

## 2. 构建

进入 `build/` 目录编译：

```bash
cd build
make clean
make USE_CUDA=1 USE_NCCL=1 MPdir=/usr/lib/x86_64-linux-gnu/openmpi MPlib=/usr/lib/x86_64-linux-gnu/openmpi/lib -j$(nproc)
cp bin/xhpcg ../bin/xhpcg
```

编译完成后确认二进制存在：

```bash
ls -lh ../bin/xhpcg
```

---

## 3. 基础运行命令

所有运行前建议设置：

```bash
export HWLOC_COMPONENTS="-gl"
```

### 3.1 单卡运行

```bash
# RTX 4090 单卡
/usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none -np 1 \
  ./bin/hpcg.sh --exec-name ./bin/xhpcg --nx 128 --ny 128 --nz 128 --rt 60 \
  --gpu-affinity 0 --cpu-affinity 0-7 --p2p 0 --b 1

# RTX 5080 单卡
/usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none -np 1 \
  ./bin/hpcg.sh --exec-name ./bin/xhpcg --nx 128 --ny 128 --nz 128 --rt 60 \
  --gpu-affinity 1 --cpu-affinity 8-15 --p2p 0 --b 1
```

### 3.2 等分双卡运行

```bash
/usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none -np 2 \
  ./bin/hpcg.sh --exec-name ./bin/xhpcg --nx 128 --ny 128 --nz 128 --rt 60 \
  --gpu-affinity 0:1 --cpu-affinity 0-7:8-15 --p2p 0 --b 1 \
  --npx 1 --npy 1 --npz 2
```

参数含义：
- `--nx/--ny/--nz`：每个 rank 的本地问题规模。
- `--npx/--npy/--npz`：MPI 进程网格，这里 `1x1x2` 表示沿 Z 轴分成 2 个 rank。
- 全局规模为 `128 x 128 x 256`，两个 rank 各拥有 `128³`。

### 3.3 异构双卡运行

```bash
/usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none -np 2 \
  ./bin/hpcg.sh --exec-name ./bin/xhpcg --nx 128 --ny 128 --nz 160 --rt 60 \
  --gpu-affinity 0:1 --cpu-affinity 0-7:8-15 --p2p 0 --b 1 \
  --npx 1 --npy 1 --npz 2 --het-split 1 --het-ratio 1.6
```

参数含义：
- `--het-split 1`：启用异构分区。
- `--het-dim`：可选，指定分割维度（1=X, 2=Y, 3=Z）；不指定时自动选择可偶数划分的维度。
- `--het-ratio 1.6`：期望的快/慢卡工作比例。代码会将其反比 slow 维度取整到 16 的倍数。
- 上例中 rank 0（RTX 4090）本地为 `128x128x160`，rank 1（RTX 5080）本地为 `128x128x96`，全局 `128x128x256`，实际工作比约 1.667。

---

## 4. 便捷脚本

| 脚本 | 作用 |
|---|---|
| `scripts/run_single_gpu.sh <0\|1>` | 单卡运行 RTX 4090 或 RTX 5080 |
| `scripts/run_dual_equal.sh` | 等分双卡运行 |
| `scripts/run_dual_het.sh` | 异构双卡运行 |
| `scripts/run_full_benchmark.sh` | 自动运行全部 5 个基准配置并生成 `summary.md` |
| `scripts/benchmark_no_nsys.sh` | 不启用 nsys，仅测量 HPCG 执行时间 |

示例：

```bash
bash scripts/run_dual_het.sh
bash scripts/run_full_benchmark.sh
# -> 输出到 benchmark_data/<timestamp>/
```

---

## 5. 关键参数速查

| 参数 | 说明 |
|---|---|
| `--exec-name` | xhpcg 可执行文件路径 |
| `--nx/--ny/--nz` | 本地问题规模（在异构模式下作为快卡尺寸） |
| `--npx/--npy/--npz` | MPI 进程网格 |
| `--rt` | 目标运行时间（秒），HPCG 会自动调整迭代次数 |
| `--gpu-affinity` | 每个 rank 使用的 CUDA 设备，冒号分隔 |
| `--cpu-affinity` | 每个 rank 使用的 CPU 核心，冒号分隔 |
| `--p2p 0` | 使用 MPI Host 进行 halo 通信 |
| `--b 1` | 启用 barrier 同步 |
| `--het-split` | 启用异构分区 |
| `--het-ratio` | 快/慢卡工作比例（必须 ≥ 1.0） |

---

## 6. 数据与报告

### 6.1 指标提取与汇总

基准运行完成后，执行：

```bash
python3 scripts/extract_metrics.py benchmark_data/<timestamp>
python3 scripts/generate_summary.py benchmark_data/<timestamp>
# -> benchmark_data/<timestamp>/summary.md
```

### 6.2 生成 Typst 实验报告

```bash
cd report
uv run python scripts/generate_plots.py   # 重新生成图表（可选）
typst compile main.typ main.pdf
```

报告源文件位于 `report/sections/`，图片位于 `report/figures/`。

---

## 7. 已知注意事项

1. **HWLOC 冲突**：运行前建议设置 `export HWLOC_COMPONENTS="-gl"`，否则 OpenMPI 可能因 hwloc 与 CUDA 冲突而报错。
2. **CPU affinity**：双卡运行时请为每个 rank 分配独立 CPU 核心（如 `0-7:8-15`），避免单核心成为瓶颈。
3. **慢卡尺寸取整**：`--het-ratio` 会按反比计算 slow 维度并取整到 16 的倍数，因此实际比例可能与输入略有差异。
4. **nsys 耗时**：`run_full_benchmark.sh` 默认会采集 GPU metrics，总耗时约 25–35 分钟；若仅需 HPCG 时间，使用 `benchmark_no_nsys.sh`。

---

## 8. 典型结果参考（`benchmark_data/2026-06-17_00-54-42`）

| 配置 | GFLOP/s | DDOT Max (s) | DDOT Min (s) |
|---|---|---|---|
| RTX 4090 single | 189.22 | 0.01 | 0.01 |
| RTX 5080 single | 120.70 | 0.01 | 0.01 |
| DUAL_EQUAL | 178.74 | 31.30 | 0.25 |
| DUAL_EQUAL_128 | 157.49 | 28.02 | 0.21 |
| DUAL_HET | 188.96 | 16.78 | 11.58 |
