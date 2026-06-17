import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "benchmark_data" / "gpu_monitoring" / "2026-06-17_14-55-40_DUAL_HET_rt60"
out_dir = repo_root / "report" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_dir / "gpu_metrics.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["elapsed"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()

# Normalize direction sign for plotting
df_pcie = df.copy()
df_pcie["pcie_tx_signed_mbps"] = -df_pcie["pcie_tx_mbps"]

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

colors = {0: "#1f77b4", 1: "#ff7f0e"}

for gpu_idx, gpu_name in [(0, "RTX 5080"), (1, "RTX 4090")]:
    g = df_pcie[df_pcie["gpu_index"] == gpu_idx]
    axes[0].plot(g["elapsed"], g["pcie_width_current"], label=f"{gpu_name} width", color=colors[gpu_idx])
    axes[1].fill_between(g["elapsed"], 0, g["pcie_rx_mbps"], alpha=0.4, color=colors[gpu_idx], label=f"{gpu_name} RX")
    axes[1].fill_between(g["elapsed"], 0, g["pcie_tx_signed_mbps"], alpha=0.4, color=colors[gpu_idx], label=f"{gpu_name} TX")

axes[0].set_ylabel("PCIe Link Width (lanes)")
axes[0].set_title("PCIe Link Width During DUAL_HET HPCG Run")
axes[0].legend(loc="right")
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel("PCIe Throughput (Mbps)")
axes[1].set_xlabel("Elapsed Time (s)")
axes[1].set_title("PCIe RX/TX Throughput (RTX 5080 x1 vs RTX 4090 x16)")
axes[1].legend(loc="upper right", ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "pcie_width_throughput.png", dpi=200)
print(f"Saved {out_dir / 'pcie_width_throughput.png'}")
