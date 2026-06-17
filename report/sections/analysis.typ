= 结果分析

本章对五种配置进行系统对比：RTX 4090 单卡（128³）、RTX 5080 单卡（128³）、DUAL_EQUAL（全局 $128 times 128 times 256$，每 rank 128³）、DUAL_EQUAL_128（全局 128³，每 rank 64³）以及 DUAL_HET（全局 $128 times 128 times 256$，rank 尺寸分别为 160 与 96）。所有数值均来自 `benchmark_data/2026-06-17_00-54-42/*/metrics.json`，对应的可视化图表存放于 `report/figures/`：

- `gflops_comparison.png`：五种配置的 GFLOP/s 柱状图。
- `speedup_comparison.png`：相对 RTX 4090 单卡与 RTX 5080 单卡的加速比。
- `ddot_imbalance.png`：双卡配置的 DDOT `MPI_Allreduce` 最大/最小时间。
- `nsys_breakdown.png`：CUDA kernel、memcpy、memset 时间堆叠图。
- `gpu_metrics_bar.png`：SM active、SM issue、GR active、Compute Warps、DRAM 读写带宽利用率对比。

== 总体性能对比

从 `gflops_comparison.png` 可以看出各配置的性能排序：

- *RTX 4090 单卡*：189.22 GFLOP/s，为所有配置中的峰值。
- *RTX 5080 单卡*：120.70 GFLOP/s，约为 RTX 4090 的 63.8%。
- *DUAL_EQUAL*：178.74 GFLOP/s，低于 RTX 4090 单卡约 5.5%。
- *DUAL_EQUAL_128*：157.49 GFLOP/s，进一步下降。
- *DUAL_HET*：188.96 GFLOP/s，几乎追平 RTX 4090 单卡（差距约 0.14%）。

`speedup_comparison.png` 更直观地展示了加速比：DUAL_EQUAL 相对 RTX 4090 单卡仅有 0.94×，DUAL_EQUAL_128 为 0.83×，而 DUAL_HET 达到 0.999×；相对 RTX 5080 单卡，三种双卡配置分别获得 1.48×、1.30× 和 1.57× 的加速。由此可见，单纯增加 GPU 数量并不能保证性能提升，*负载均衡* 才是决定双卡效率的关键。

#figure(
  image("../figures/gflops_comparison.png", width: 90%),
  caption: [五种配置的 HPCG GFLOP/s 对比]
)

#figure(
  image("../figures/speedup_comparison.png", width: 90%),
  caption: [双卡配置相对单卡的加速比]
)

== 单卡差异：RTX 5080 为何明显慢于 RTX 4090

`gpu_metrics_bar.png` 与 `metrics.json` 中的 GPU 硬件指标揭示了 RTX 5080 在本负载下效率偏低的原因。由于单卡运行时系统中另一张 GPU 处于空闲状态，Nsight Systems 的 "overall" 指标是两卡平均值；以下引用的单卡有效指标来自实际运行该负载的 GPU（即 `per_gpu` 中利用率显著的非空闲 GPU）：

- *SMs Active*：RTX 4090 为 80.90%，RTX 5080 为 42.06%。RTX 5080 的 SM 占用率几乎只有 RTX 4090 的一半，说明其大量 SM 处于空闲或等待状态。
- *SM Issue*：RTX 4090 为 3.39%，RTX 5080 为 2.44%。较低的 SM issue 率意味着 RTX 5080 每个周期实际发出的指令更少。
- *GR Active*：RTX 4090 为 89.32%，RTX 5080 为 69.53%。图形/计算引擎的整体活跃度也存在明显差距。
- *Compute Warps in Flight*：RTX 4090 为 55.08%，RTX 5080 为 29.72%。可同时执行的计算 warp 数量减少，直接削弱了线程级并行。
- *DRAM Read Bandwidth*：RTX 4090 为 70.63%，RTX 5080 为 37.62%。HPCG 是内存密集型负载，DRAM 带宽利用率不足严重限制了 RTX 5080 的吞吐。
- *GPC Clock Frequency*：RTX 4090 平均约 2649.6 MHz，RTX 5080 仅约 2148.1 MHz。运行频率偏低进一步拉大了两者差距。

综合来看，RTX 5080 在该 nvidia-hpcg 代码路径上出现了 SM 利用率低、DRAM 带宽利用率低、运行频率低的现象。结合其 `[CUDA memcpy Device-to-Host]` 在 nsys 时间占比中高达 4.0%（RTX 4090 仅 0.4%），可以判断 DDOT 阶段的 GPU↔CPU 同步开销在 RTX 5080 上被进一步放大，导致其整体性能远低于 RTX 4090。

#figure(
  image("../figures/gpu_metrics_bar.png", width: 95%),
  caption: [单卡与双卡配置的 GPU 硬件指标对比]
)

== 对称双卡为何慢于单卡

DUAL_EQUAL 与 DUAL_EQUAL_128 的 GFLOP/s 均低于 RTX 4090 单卡，核心原因是 *负载不均衡* 导致的 MPI 同步等待。

`ddot_imbalance.png` 展示了双卡配置中 DDOT `MPI_Allreduce` 的最大/最小耗时：

- *DUAL_EQUAL*：Max 31.30 s，Min 0.25 s，差距约 125 倍。
- *DUAL_EQUAL_128*：Max 28.02 s，Min 0.21 s，差距约 133 倍。
- *DUAL_HET*：Max 16.78 s，Min 11.58 s，差距缩小到约 1.45 倍。

#figure(
  image("../figures/ddot_imbalance.png", width: 90%),
  caption: [双卡配置中 DDOT MPI_Allreduce 的最大/最小耗时]
)

在对称配置下，RTX 4090 很快完成本地计算并到达 `MPI_Allreduce` 同步点，而 RTX 5080 仍在执行计算。快卡必须在集合通信处等待慢卡，导致 DDOT 的 Max 时间由慢卡决定、Min 时间由快卡决定。`nsys_breakdown.png` 中的 memcpy 时间也印证了这一点：DUAL_EQUAL 的 memcpy 时间约为 11.94 s，DUAL_EQUAL_128 更高达 16.74 s，而 DUAL_HET 降至 11.21 s，RTX 4090 单卡仅 0.29 s。大量时间被消耗在同步相关的 Device-to-Host 与 Host-to-Device 数据拷贝上。

DUAL_EQUAL_128 的情况更为极端：虽然全局规模保持 128³，但每 rank 本地规模降至 64³，本地计算量减少一半，GPU 无法充分隐藏通信延迟；同时迭代次数从 DUAL_EQUAL 的 17700 增加到 30900，同步开销被进一步放大，因此性能反而更低。

#figure(
  image("../figures/nsys_breakdown.png", width: 90%),
  caption: [Nsight Systems 采集的 CUDA Kernel、Memcpy、Memset 时间分解]
)

== 异构分区的效果

DUAL_HET 通过 `--het-split --het-ratio 1.6` 将快卡子域设为 160、慢卡子域设为 96，实际工作量比例约为 1.667:1。该比例与两卡单卡性能比（189.22 / 120.70 ≈ 1.568）接近，因此两块 GPU 到达 `MPI_Allreduce` 的时间几乎同步，DDOT Max/Min 差距从 125 倍缩小到 1.45 倍。

`nsys_breakdown.png` 显示，DUAL_HET 的总 CUDA kernel 时间约为 56.41 s，低于 DUAL_EQUAL 的 59.65 s 和 DUAL_EQUAL_128 的 49.99 s（后两者因等待导致有效 kernel 利用率下降）。更重要的是，DUAL_HET 的 memcpy 时间从 DUAL_EQUAL 的 11.94 s 降至 11.21 s，说明同步等待减少后，数据搬移开销也随之下降。虽然 DUAL_HET 的单卡 kernel 时间仍高于 RTX 4090 单卡，但两块 GPU 并行工作使 wall-clock 时间缩短，最终 GFLOP/s 达到 188.96，几乎与 RTX 4090 单卡持平。

从 `gpu_metrics_bar.png` 观察，DUAL_HET 的平均 SM active（37.44%）、SM issue（1.81%）、GR active（50.93%）等指标介于 RTX 4090 单卡与 RTX 5080 单卡之间，这是两张卡指标的平均结果。其 DRAM Read 带宽利用率为 32.91%，虽然低于 DUAL_EQUAL 的 35.05%（两卡平均），但 DUAL_EQUAL 的 35.05% 实际上是快卡 35.79% 与慢卡 34.31% 的平均，两者差异较小；而 DUAL_HET 的快卡（25.05%）与慢卡（40.77%）差异较大，反映了按能力分配负载后两卡工作模式的差异。总体而言，DUAL_HET 的快卡不再长时间等待，慢卡也能在相近时刻完成，整体吞吐得到提升。

== 规模与配置选择

综合五种配置，可以得出以下结论：

1. *单卡效率*：RTX 4090 在本负载下的单卡效率最高；RTX 5080 受限于 SM 利用率、DRAM 带宽利用率、运行频率以及较大的 DDOT 同步开销，效率明显偏低。
2. *对称双卡的瓶颈*：当两块 GPU 性能差异较大时，对称划分会导致严重的负载不均衡，DDOT `MPI_Allreduce` 成为主要瓶颈。
3. *异构划分的收益*：通过 `--het-ratio` 按能力比例分配工作量，可以显著减少同步等待，使双卡总吞吐接近甚至追平快卡单卡。
4. *本地规模的重要性*：过小的本地规模（如 64³）会降低 GPU 利用率并增加迭代次数，不适合作为双卡扩展策略。

因此，对于 RTX 4090 + RTX 5080 这类异构双 GPU 工作站，推荐采用 `DUAL_HET` 式的非对称划分，而非简单的等分策略。后续扩展性实验将进一步验证更大问题规模下该结论是否依然成立。
