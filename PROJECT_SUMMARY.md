# nvidia-hpcg 异构双 GPU 实验项目文档

## 1. 项目简介

本项目基于 NVIDIA 官方 `nvidia-hpcg`，在一台配备 **RTX 4090** 与 **RTX 5080** 的异构双 GPU 工作站上进行 HPCG 基准测试与优化实验。核心目标是在保持 HPCG 题目条件不变（问题规模、迭代次数、收敛判据等）的前提下，提升最终报告的 GFLOP/s 数值。

### 硬件与软件环境

| 组件 | 配置 |
|---|---|
| CPU | 2× Intel Xeon Gold 6248R |
| GPU 0 (CUDA 0) | NVIDIA GeForce RTX 4090 (sm_89, Ada Lovelace) |
| GPU 1 (CUDA 1) | NVIDIA GeForce RTX 5080 (sm_120, Blackwell) |
| 主板 | Dell Precision 7920 Tower (060K5C) |
| CUDA | 13.2 |
| MPI | OpenMPI 4.1.2 |
| NCCL | 已启用 |

> 注意：CUDA 枚举顺序与 `nvidia-smi` 相反。`CUDA 0` 对应 RTX 4090，`CUDA 1` 对应 RTX 5080。

---

## 2. 目前已实现的优化方案

### 2.1 构建与架构支持

- 在 `setup/Make.CUDA_X86` 中补充 `sm_89`（RTX 4090）和 `sm_120`（RTX 5080）的编译目标，使二进制可在两种 GPU 上原生运行。
- 使用 `USE_CUDA=1 USE_NCCL=1` 启用 GPU 执行与 NCCL 通信支持。

### 2.2 等分双卡执行

- 通过 MPI 启动 2 个 rank，分别绑定到 RTX 4090 和 RTX 5080。
- 全局规模 128×128×256，每个 rank 负责 128³。
- 脚本：`scripts/run_dual_equal.sh`。

### 2.3 异构双卡负载划分

- 新增参数 `--het-split`、`--het-dim`、`--het-ratio`。
- `GenerateGeometry.cpp` 按 rank 奇偶分配快/慢 GPU 的本地规模。
- 慢 rank 的分割维度取整到 16 的倍数，避免粗化后最细网格维度为奇数导致 cuSPARSE SpSV 崩溃。
- 脚本：`scripts/run_dual_het.sh`。
- 当前配置：快 rank `128×128×160`，慢 rank `128×128×96`，实际工作比约 1.667。

### 2.4 GPU Halo 交换修复

- 修复 `SetupHalo.cpp` 中 `extToLocMap` 的越界写问题：当不同 rank 本地规模不同时，按全局最大本地行数分配映射表，避免从较大邻居接收的列索引越界。

### 2.5 CPU Affinity 修正

- 将双卡运行时的 CPU 亲和性从 `0:24`（每 rank 单核心）改为 `0-7:8-15`（每 rank 8 核心），消除了 CPU 侧瓶颈，使 DUAL_EQUAL 从约 106 GFLOP/s 提升到约 200 GFLOP/s。

### 2.6 多维度分块研究

- 系统测试了 X、Y、Z 三个方向的对称与非对称分块。
- 脚本：`scripts/run_dim_partition_study.sh`、`scripts/plot_dim_partition_study.py`。
- 主要发现：等分模式下 Z 方向最优；非对称模式下三个方向性能接近，Z 方向同时具有最高吞吐和最佳负载均衡。

### 2.7 GPU 状态监控

- 开发 `scripts/monitor_dual_gpu.py`，在 HPCG 运行期间每秒采样两张显卡的 PCIe 链路宽度、吞吐、利用率、温度、功耗等指标。
- 脚本：`scripts/run_monitored_trace.sh`、`scripts/analyze_gpu_monitor.py`。

---

## 3. 实验数据总结

### 3.1 五种基准配置（`benchmark_data/2026-06-17_00-54-42`）

| 配置 | 全局规模 | 每 rank 规模 | GFLOP/s | DDOT Max (s) | DDOT Min (s) |
|---|---|---|---|---|---|
| RTX 4090 单卡 | 128³ | 128³ | 189.22 | 0.01 | 0.01 |
| RTX 5080 单卡 | 128³ | 128³ | 120.70 | 0.01 | 0.01 |
| DUAL_EQUAL | 128×128×256 | 2×128³ | 178.74 | 31.30 | 0.25 |
| DUAL_EQUAL_128 | 128³ | 2×64³ | 157.49 | 28.02 | 0.21 |
| DUAL_HET | 128×128×256 | 128×128×160 / 128×128×96 | 188.96 | 16.78 | 11.58 |

### 3.2 维度分块对比（rt=30）

| 配置 | GFLOP/s | DDOT Max (s) | DDOT Min (s) |
|---|---|---|---|
| EQUAL_X | 155.74 | 19.77 | 0.03 |
| EQUAL_Y | 204.15 | 11.95 | 0.02 |
| EQUAL_Z | 207.25 | 10.89 | 0.02 |
| ASYMM_X | 189.99 | 3.27 | 1.21 |
| ASYMM_Y | 190.44 | 3.05 | 1.24 |
| ASYMM_Z | 191.49 | 2.92 | 1.14 |

### 3.3 RTX 5080 性能偏低的根因

- `nvidia-smi` 与 `/sys/bus/pci/devices/...` 显示：**RTX 5080 当前 PCIe 链路宽度为 x1**（应为 x16）。
- RTX 4090 在全负载下为 Gen3 x16。
- Dell Precision 7920 Tower 的 Slot 2 是物理 x16、电气 x1 的插槽；RTX 5080 正好位于该插槽。
- 在当前硬件状态下，RTX 5080 的单卡性能和双卡总吞吐均被 PCIe x1 带宽严重限制。

### 3.4 可调参数扫描结论

| 参数/模式 | 测试结果 | 结论 |
|---|---|---|
| `--gss` 512–4096 | 性能几乎无差异 (~189 GFLOP/s) | 默认值 4096 已最优 |
| `--gss` 8192+ | 崩溃或超时 | 不可用 |
| `Use_Hpcg_Mem_Reduction=false` | 187.60 GFLOP/s | 默认 `true` 更优 |
| `p2p=0 MPI_CPU` | DUAL_EQUAL 203.34 GFLOP/s | 当前硬件下最优 |
| `p2p=3 CUDA_AWARE_All2allv` | 段错误 | 未配置 |
| `p2p=4 NCCL` | DUAL_EQUAL 181.35 GFLOP/s | 受 x1 PCIe 限制 |

---

## 4. 可能的优化方向

### 4.1 单卡优化

#### 4.1.1 自定义 SYMGS 核函数

- `nsys` 显示 `cusparse::spsv_sell_single_color_v1_kernel` 占 43–52% GPU 时间，是当前最大热点。
- 方向：针对 27-点 stencil 实现红-黑 Gauss-Seidel CUDA 核，使用 warp 规约、shared memory 缓存 halo、3D thread block 映射。
- 预期收益：20–50%。

#### 4.1.2 内核融合与 CUDA Graph

- 当前 CG 每次迭代由多个独立核函数（SpMV、SYMGS、WAXPBY、DDOT、MG）组成，存在 kernel launch 和同步开销。
- 方向：融合无数据依赖的相邻操作；使用 `cudaGraph` 捕获重复的 CG set 执行序列。
- 预期收益：10–25%。

#### 4.1.3 SpMV 格式优化

- `cusparse::sellmv_v1_2D_kernel` 占 34–42%。
- 方向：评估 CSR/DIA/自定义 stencil 专用格式；分离内部点与边界点以消除分支；预取 halo 到 shared memory。
- 预期收益：10–20%。

#### 4.1.4 单卡目标

- RTX 4090 单卡目标：从当前 189 GFLOP/s 提升至 250–300 GFLOP/s。

### 4.2 多卡优化

#### 4.2.1 硬件层面

- 将 RTX 5080 从 Slot 2 移动到 Slot 4 或 Slot 5（全速 x16 CPU 直连插槽）。
- 这是消除多卡性能瓶颈的最根本手段；在 x1 插槽下，软件优化难以使双卡总吞吐大幅超越单卡 RTX 4090。

#### 4.2.2 通信优化

- 在硬件修复后，重新评估 `p2p=4 NCCL` 与 `p2p=3 CUDA-Aware MPI`。
- 实现 halo 与内部计算的 split-phase 重叠，用独立 CUDA stream 在 halo 传输期间计算内部点。

#### 4.2.3 自适应非对称划分

- 当前 `--het-ratio` 是静态的；不同规模和通信模式下最优比例会变化。
- 方向：通过短预热运行自动搜索最优 `--het-ratio`，选择 GFLOP/s 最高且 DDOT Max/Min 最接近的配置。

#### 4.2.4 NUMA 感知绑定

- 使用 `numactl` 将每个 rank 绑定到对应 GPU 所在的 NUMA 节点，减少 CPU-GPU 数据搬移延迟。

### 4.3 综合路线图

| 阶段 | 任务 | 优先级 |
|---|---|---|
| 1 | 修复 RTX 5080 PCIe 插槽（硬件） | 最高 |
| 2 | 自定义 SYMGS 核函数 | 高 |
| 3 | 内核融合 / CUDA Graph | 中高 |
| 4 | halo 与计算重叠 | 中 |
| 5 | 自适应 het-ratio 搜索 | 中 |
| 6 | NUMA 感知绑定与通信模式再评估 | 低 |

---

## 5. 相关脚本与文档

| 文件 | 说明 |
|---|---|
| `README_EXPERIMENTAL.md` | 实验构建与运行命令速查 |
| `report/main.pdf` | Typst 实验报告（含数据分析、维度分块、PCIe 瓶颈分析） |
| `scripts/run_dual_equal.sh` | 等分双卡运行 |
| `scripts/run_dual_het.sh` | 异构双卡运行 |
| `scripts/run_full_benchmark.sh` | 五种配置自动基准测试 |
| `scripts/tune_slice_size.sh` | slice_size 参数扫描 |
| `scripts/run_dim_partition_study.sh` | 维度分块研究 |
| `scripts/monitor_dual_gpu.py` | GPU 状态监控 |
| `scripts/dynamic_load_balance.py` | 自适应 het-ratio 原型 |

---

## 6. 关键结论

1. RTX 5080 当前性能异常偏低的根本原因是其位于主板 x1 电气插槽；在换到 x16 插槽前，多卡软件优化效果有限。
2. 单卡层面，`--gss` 等易调参数已无提升空间，需要自定义 SYMGS 等大型内核优化。
3. 异构负载划分（`--het-split`）在当前硬件下已接近 RTX 4090 单卡性能，并显著改善双卡负载均衡。
4. 维度分块研究表明：等分用 Z 方向，非对称分块对方向不敏感但 Z 方向仍综合最优。
