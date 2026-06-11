"""Correlate metrics with task success and save a scatter plot.

Usage:
    python -m experiments.analyze --csv results.csv --out scatter.png
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out", default="scatter.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"n = {len(df)}, accuracy = {df['correct'].mean():.2%}")

    metrics = ["perplexity", "cot_perplexity", "cot_mean_token_entropy", "prompt_tokens", "cot_tokens"]
    metrics = [m for m in metrics if m in df.columns]
    for metric in metrics:
        r, r_p = pearsonr(df[metric], df["correct"])
        rho, rho_p = spearmanr(df[metric], df["correct"])
        print(f"{metric:>22s}: Pearson r={r:+.3f} (p={r_p:.3f})  "
              f"Spearman rho={rho:+.3f} (p={rho_p:.3f})")

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, metric in zip(axes, metrics):
        ax.scatter(df[metric], df["correct"] + (df.index % 7) * 0.01, alpha=0.7)
        ax.set_xlabel(metric)
        ax.set_ylabel("correct (0/1, jittered)")
        ax.set_title(f"{metric} vs success")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
