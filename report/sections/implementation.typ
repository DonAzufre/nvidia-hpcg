= 实现

== 对称双 GPU 配置

对称双 GPU 配置沿某一维度将全局问题均匀划分为两个子域，每个 MPI rank 负责一个子域，并分别绑定到 RTX 4090 与 RTX 5080。以 Z 轴划分为例，使用 `1x1x2` 的进程网格：

- 全局规模为 $128 times 128 times 256$；
- 每个 rank 的本地规模为 $128 times 128 times 128$；
- rank 0 绑定 CUDA 0（RTX 4090），rank 1 绑定 CUDA 1（RTX 5080）。

对应的运行命令如下：

```bash
/usr/bin/mpirun.openmpi --allow-run-as-root --oversubscribe --bind-to none -np 2 \
  ./bin/hpcg.sh --exec-name ./bin/xhpcg \
  --nx 128 --ny 128 --nz 256 --rt 60 \
  --gpu-affinity 0:1 --cpu-affinity 0-7:8-15 \
  --p2p 0 --b 1 --npx 1 --npy 1 --npz 2
```

其中 `--gpu-affinity 0:1` 表示 rank 0 使用 CUDA 0，rank 1 使用 CUDA 1；`--cpu-affinity 0-7:8-15` 为两个 rank 分配不同的 CPU 核心，减少 CPU 侧争用。

== 异构分区设计

为了让快卡承担更多计算量，我们在 `bin/hpcg.sh` 和 `src/GenerateGeometry.cpp` 中新增了三项参数：

- `--het-split`：是否启用异构分区（`0` 关闭，`1` 开启）。
- `--het-dim`：指定划分维度（`1=X`、`2=Y`、`3=Z`）。
- `--het-ratio`：快卡与慢卡的工作量比例，例如 `1.6` 表示快卡约承担慢卡 1.6 倍的工作。

在 `src/GenerateGeometry.cpp` 中，当 `params.exec_mode == GPUONLY && params.het_split` 为真时，代码首先确定划分维度，然后按 rank 奇偶性区分快卡（偶数 rank）和慢卡（奇数 rank），并重新计算该维度上的本地尺寸：

```cpp
if (params.exec_mode == GPUONLY && params.het_split)
{
    // ... 维度选择逻辑 ...
    bool is_fast_gpu = (rank % 2 == 0);
    int& split_size = (user_diff_dim == X) ? nx : (user_diff_dim == Y) ? ny : nz;
    int fast_size = split_size;
    int slow_size = fast_size;
    if (params.het_ratio > 1.0)
    {
        int raw_slow = int(std::round((double) fast_size / params.het_ratio));
        // 保证粗化后维度仍为偶数，避免 cuSPARSE SpSV 崩溃
        slow_size = ((raw_slow + 8) / 16) * 16;
        if (slow_size < 16) slow_size = 16;
        if (slow_size >= fast_size)
            slow_size = (fast_size / 2 + 8) / 16 * 16;
    }
    split_size = is_fast_gpu ? fast_size : slow_size;
}
```

以 `--nz 160 --het-ratio 1.6` 为例，快卡本地 Z 尺寸保持 160，慢卡尺寸计算为：

$ "round"(160 / 1.6) = 100, quad "向上取整到 16 的倍数" = 96 $

因此全局 Z 尺寸为 $160 + 96 = 256$，实际工作量比例约为 $160 : 96 = 1.667 : 1$。

== 关键 Bug 修复

在实现异构分区的过程中，我们遇到并修复了两个关键问题。

=== 1. `extToLocMap` 越界写入

在 `src/SetupHalo.cpp` 的 GPU halo 交换路径中，原本使用当前 rank 的 `localNumberOfRows` 分配 `extToLocMap`：

```cpp
CHECK_CUDART(cudaMalloc(&extToLocMap, sizeof(local_int_t) * localNumberOfRows));
```

当两个 rank 的本地规模不同时（例如快卡 160、慢卡 96），邻居 rank 发送来的列索引可能大于当前 rank 的 `localNumberOfRows`，导致 `ExtToLocMapCuda` 在写入 `extToLocMap` 时发生越界访问。

修复方式是通过一次 `MPI_Allreduce` 取所有 rank 中的最大本地行数，并据此分配映射表：

```cpp
local_int_t maxLocalRows = localNumberOfRows;
MPI_Allreduce(&localNumberOfRows, &maxLocalRows, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
CHECK_CUDART(cudaMalloc(&extToLocMap, sizeof(local_int_t) * maxLocalRows));
```

同时在每次使用 `extToLocMap` 前，用 `cudaMemsetAsync` 将其清零，确保旧数据不会影响当前邻居的映射。

=== 2. 粗网格奇数维度导致 cuSPARSE SpSV 崩溃

HPCG 的多层网格预处理会对每个维度的尺寸反复二分。当异构分区使慢卡某一维度的本地尺寸为奇数时，最粗网格上可能出现奇数维度，进而触发 cuSPARSE SpSV 分析阶段的段错误。

解决方案是在 `GenerateGeometry.cpp` 中，将慢卡的划分尺寸向上取整到 16 的倍数：

```cpp
slow_size = ((raw_slow + 8) / 16) * 16;
```

这样可以保证在 3～4 层粗化之后，最粗网格的各维度仍然为偶数，从而避免 cuSPARSE SpSV 分析崩溃。

=== 3. CPU 亲和性修正

在双卡调试初期，我们曾为两个 rank 均绑定 CPU 核心 `0:24`，导致 CPU 侧任务严重串行化，双卡总吞吐大幅下降。最终采用 `0-7:8-15` 的分离式 CPU 亲和性设置，使两个 rank 拥有独立的 CPU 资源，MPI 通信与 CUDA 流同步不再相互抢占。

== 执行流程小结

整个双卡异构运行的流程可概括为：

1. `bin/hpcg.sh` 解析 `--het-split`、`--het-dim`、`--het-ratio` 等参数，并将其透传给 `xhpcg`。
2. `src/GenerateGeometry.cpp` 根据快/慢卡身份重新计算本地子域尺寸，并沿划分维度完成逻辑 rank 映射。
3. `src/SetupHalo.cpp` 基于最大本地行数分配 `extToLocMap`，完成非对称 halo 索引交换。
4. GPU 端通过 cuSPARSE/cuBLAS 执行 PCG 迭代，CPU 端通过 MPI 完成 DDOT Allreduce 与 halo 索引交换。

后续章节将基于上述实现，对实验结果进行详细分析。
