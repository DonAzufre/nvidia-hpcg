# HPCG Report Figures

This Python project generates the figures and data summary for the nvidia-hpcg dual-GPU benchmark report.

## Usage

```bash
cd report
uv run python scripts/generate_plots.py
```

## Outputs

All figures are saved to `figures/`:

- `gflops_comparison.png`
- `ddot_imbalance.png`
- `nsys_breakdown.png`
- `gpu_metrics_bar.png`
- `speedup_comparison.png`
- `data_summary.json`

## Data source

Figures are derived from `../benchmark_data/2026-06-17_00-54-42/*/metrics.json`.
