= 引言

== HPCG 基准测试概述

HPCG（High Performance Conjugate Gradient）是由美国 Sandia 国家实验室牵头设计的稀疏线性代数基准测试，旨在补充 LINPACK/HPL 对稠密矩阵计算能力的评价，从更接近实际科学与工程应用的角度衡量超级计算机的性能。HPCG 的核心计算模式是预处理共轭梯度法（Preconditioned Conjugate Gradient, PCG），其主要算子包括稀疏矩阵向量乘法（SpMV）、向量点积（DDOT）、向量更新（WAXPBY）以及基于多层网格的几何粗化预处理。这些操作具有低计算密度、强内存依赖和频繁同步的特点，因此对内存带宽、缓存效率、网络延迟以及节点内并行调度都提出了很高的要求。

HPCG 的评分以 *GFLOP/s* 为单位，最终结果是多次运行后得到的稳定浮点性能。由于算法本身的局部性和全局通信需求，HPCG 在多 GPU、多节点场景下往往难以实现理想的线性扩展，负载不均衡、通信开销以及同步等待会成为主要的性能瓶颈。

== nvidia-hpcg 简介

nvidia-hpcg 是 NVIDIA 基于 HPCG 3.x 参考实现开发的 GPU 优化版本。它使用 CUDA、cuSPARSE、cuBLAS 等库将核心算子 offload 到 GPU，并支持通过 MPI 在多个 GPU 之间进行分布式扩展。其默认执行模式 `GPUONLY` 会将所有计算任务放在 GPU 上，CPU 主要负责 MPI 通信控制和进程管理。

在本项目中，我们基于 nvidia-hpcg 源码，针对一台搭载 *RTX 4090*（CUDA 0，快卡）和 *RTX 5080*（CUDA 1，慢卡）的异构双 GPU 工作站进行了适配与扩展。目标是在保留 GPU 计算路径的前提下，实现两块不同性能 GPU 的协同运行，并通过非对称负载划分提升整体吞吐量。

== 项目目标

本项目围绕以下几个核心目标展开：

1. *编译与运行适配*：在 RTX 4090 + RTX 5080 的混合架构上完成 nvidia-hpcg 的编译，确保单卡与双卡模式均能正确运行并得到 VALID 结果。
2. *对称双卡基线*：建立若干对称双 GPU 配置（全局规模相同、每 rank 本地规模相同），作为性能对比的基准。
3. *异构负载均衡*：引入 `--het-split`、`--het-dim`、`--het-ratio` 等参数，使快卡（RTX 4090）承担更多工作、慢卡（RTX 5080）承担较少工作，从而减少 MPI 同步等待，提高总吞吐。
4. *性能分析与可视化*：利用 Nsight Systems 采集 CUDA kernel、memcpy/memset 时间以及 GPU 硬件指标（SM active、SM issue、GR active、DRAM 带宽、GPC clock 等），对单卡与双卡配置进行深入分析。
5. *扩展性探索*：在基准规模之上尝试更大的问题规模，评估运行时间与结果有效性，为后续优化提供数据支撑。

后续章节将详细介绍实现细节、实验配置以及结果分析，相关性能图表将在后续章节中插入。
