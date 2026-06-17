= RTX 5080 性能异常偏低的 PCIe 瓶颈分析

RTX 5080 单卡 HPCG 性能仅约 120 GFLOP/s，约为 RTX 4090 的 64%。这一差距远大于两款显卡的理论算力差距，本节通过 PCIe 链路状态和运行时监控，定位其根因。

== 测试方法

使用三种独立手段交叉验证：

1. *`/sys/bus/pci/devices/...`* 内核接口读取显卡的 PCIe 最大/当前链路速度与宽度。
2. *`nvidia-smi`* 在 HPCG 满载时实时采样 `pcie.link.gen.current` 和 `pcie.link.width.current`。
3. *`scripts/monitor_dual_gpu.py`* 在 DUAL_HET 运行期间每 1 秒记录一次 GPU 状态，包括 PCIe 宽度、RX/TX 吞吐、利用率、温度、功耗。

== PCIe 链路状态

=== 空闲与满载对比

#table(
  columns: (3.5cm, 2.5cm, 2.5cm, 2.5cm, 2.5cm),
  align: center,
  table.header([显卡], [最大能力], [空闲状态], [满载状态], [结论]),
  [RTX 5080], [PCIe Gen5 x16], [Gen3 x1], [Gen3 x1], [宽度始终 x1],
  [RTX 4090], [PCIe Gen4 x16], [Gen1 x16], [Gen3 x16], [空闲降速，满载恢复 x16],
)

关键观察：

- RTX 4090 空闲时链路为 Gen1 x16，HPCG 负载下自动升至 Gen3 x16，这是正常的 ASPM 空闲降速。
- RTX 5080 无论空闲还是 97% GPU 利用率下，链路宽度始终为 *x1*，说明不是空闲降速，而是插槽电气通道只有 1 条 lane。

=== 运行时 PCIe 宽度与吞吐

#figure(
  image("../figures/pcie_width_throughput.png", width: 95%),
  caption: [DUAL_HET 运行期间两张显卡的 PCIe 链路宽度与 RX/TX 吞吐]
)

上图显示：

- RTX 5080 的 PCIe 宽度全程为 1 lane（蓝色水平线）。
- RTX 4090 的 PCIe 宽度全程为 16 lanes（橙色水平线）。
- RTX 5080 的 PCIe RX/TX 吞吐峰值约 1 GB/s，已接近 PCIe Gen3 x1 的理论单向带宽（约 0.985 GB/s）。

== 理论带宽估算

PCIe Gen3 每 lane 单向有效带宽约为：

$ 8 "GT/s" times 128/130 "编码效率" approx 984.6 "MB/s/lane" $

因此：

- "RTX 5080 @ Gen3 x1"：单向约 0.985 GB/s，双向约 1.97 GB/s。
- "RTX 4090 @ Gen3 x16"：单向约 15.75 GB/s，双向约 31.5 GB/s。

HPCG 的 halo 交换、`MPI_Allreduce`（DDOT）都需要在 GPU 显存与主机内存之间频繁搬运数据。RTX 5080 的 x1 链路成为显著瓶颈，导致 GPU 算力无法充分发挥。

== 主板插槽规格

本机为 *Dell Precision 7920 Tower*（主板 060K5C），其扩展槽规格如下：

#table(
  columns: (2.5cm, 3cm, 3cm),
  align: center,
  table.header([插槽], [物理尺寸], [电气通道]),
  [Slot 1], [PCIe x8], [open-ended x8],
  [Slot 2], [PCIe x16], [x1],
  [Slot 3], [PCIe x16], [x4],
  [Slot 4], [PCIe x16], [x16],
  [Slot 5], [PCIe x16], [x16],
  [Slot 6–7], [PCIe x16], [x16（需第二颗 CPU）],
)

来源：Dell Precision 7920 Tower 官方规格表。

`lspci` 拓扑显示：

- RTX 4090 位于 `0000:a6:00.0`，挂在 Skylake-E CPU 直连 Root Port，对应全速 x16 插槽。
- RTX 5080 位于 `0000:02:00.0`，挂在 C620 芯片组 Root Port，对应 Slot 2 这种 *x16 物理、x1 电气* 的插槽。

== 结论与建议

RTX 5080 性能偏低的根本原因是它被安装在了主板上一个 *电气仅 x1* 的 PCIe 插槽中，导致 GPU↔CPU 数据传输严重受限。这不是驱动 bug，也不是空闲降速，而是硬件安装位置问题。

修复建议：

1. *物理换槽*：将 RTX 5080 从 Slot 2 移到 Slot 4 或 Slot 5（全速 x16 CPU 直连插槽），与 RTX 4090 一起占用两个全速 x16 插槽。
2. *BIOS 检查*：确认 PCIe ASPM 设置不会导致链路降速；确认没有手动将插槽配置为 x1。
3. *换槽后重跑基准*：预期 RTX 5080 单卡性能和双卡总吞吐都会显著提升，DUAL_EQUAL 的总吞吐应能超过 RTX 4090 单卡。

在换槽之前，软件层面的优化（如 `--het-split` 非异划分、CPU affinity 调整）只能在一定程度上缓解 x1 瓶颈带来的负载不均，无法彻底消除该硬件限制。
