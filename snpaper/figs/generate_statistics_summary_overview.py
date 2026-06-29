#!/usr/bin/env python3
"""
Generate a compact summary figure from statistics/statistics.json for the paper.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10


def main():
    stats_path = Path("statistics/statistics.json")
    out_path = Path("snpaper/figs/figures_for_article/statistics_summary_overview.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    labels = ["95% var comps", "Compression", "Mean |corr|", "Mean MI"]
    values = [
        stats["dimensionality"]["components_95"],
        stats["dimensionality"]["compression_ratio_95"],
        stats["correlation"]["mean_abs_correlation"],
        stats["discriminative"]["mean_mutual_info"],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_title("Statistical summary of extracted feature space")
    ax.grid(True, axis="y", alpha=0.3)

    for i, b in enumerate(bars):
        val = values[i]
        txt = f"{val:.3f}" if isinstance(val, float) and val < 100 else f"{val:.0f}"
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), txt, ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
