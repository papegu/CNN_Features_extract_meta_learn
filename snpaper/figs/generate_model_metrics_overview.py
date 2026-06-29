#!/usr/bin/env python3
"""
Generate a compact comparison figure from results/metrics.json for inclusion in the paper.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10


def main():
    metrics_path = Path("results/metrics.json")
    out_path = Path("snpaper/figs/figures_for_article/model_metrics_overview.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_names = [k for k in data.keys() if k != "comparisons"]
    acc = [data[m]["accuracy"] * 100 for m in model_names]
    f1m = [data[m]["f1_macro"] * 100 for m in model_names]
    f1w = [data[m]["f1_weighted"] * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, acc, width, label="Accuracy", color="#4C78A8")
    ax.bar(x, f1m, width, label="F1-macro", color="#F58518")
    ax.bar(x + width, f1w, width, label="F1-weighted", color="#54A24B")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel("Score (%)")
    ax.set_title("Comparison of meta-learners on PlantVillage test set")
    ax.set_ylim(90, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
