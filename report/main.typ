#set text(font: "Noto Serif CJK SC", size: 11pt)
#set heading(numbering: "1.1 ")
#set figure(numbering: "1")
#set page(numbering: "1", margin: (x: 2.5cm, y: 2.5cm))

#align(center)[
  #text(size: 20pt, weight: "bold")[异构双 GPU 上的 HPCG 基准测试与负载均衡研究]
  #v(0.5cm)
  #text(size: 12pt)[基于 NVIDIA nvidia-hpcg 在 RTX 4090 + RTX 5080 工作站上的实验]
  #v(0.3cm)
  #text(size: 10pt)[实验日期：2026-06-17]
]

#v(1cm)
#outline(title: "目录", indent: auto)
#pagebreak()

#include "sections/intro.typ"
#pagebreak()

#include "sections/implementation.typ"
#pagebreak()

#include "sections/analysis.typ"
#pagebreak()

#include "sections/dim_partition.typ"
#pagebreak()

#include "sections/pcie_bottleneck.typ"
#pagebreak()

#include "sections/scaling.typ"
