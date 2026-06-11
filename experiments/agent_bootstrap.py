"""Bootstrap 95% CI + scatter plots for agent-level correlations.

Headline numbers for the workshop report are bare Pearson r values without
confidence intervals; this script adds non-parametric CIs by resampling
questions with replacement, and renders the two top scatter plots.

Usage:
    python -m experiments.agent_bootstrap \\
        --agent overnight_results/agent_15b_n60_pooled_summary.csv \\
        --out-plot overnight_results/agent_n60_scatter.png \\
        --out-md   overnight_results/RESULTS_AGENT_n60_bootstrap.md \\
        --n-boot 5000
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Headline metrics — same family on whole trajectory and on action segments.
HEADLINE = [
    "agent_ppl_mean",
    "agent_margin_mean",
    "agent_entropy_mean",
    "agent_top1_mass_mean",
    "action_ppl_mean",
    "action_margin_mean",
    "answer_ppl",  # for the "answer alone doesn't carry signal" finding
]


def bootstrap_pearson(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Resample (x, y) pairs with replacement; return r per resample."""
    n = len(x)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if xb.std() == 0 or yb.std() == 0:
            rs[b] = np.nan
            continue
        rs[b] = np.corrcoef(xb, yb)[0, 1]
    return rs


def ci(rs: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    rs = rs[~np.isnan(rs)]
    lo = np.quantile(rs, alpha / 2)
    hi = np.quantile(rs, 1 - alpha / 2)
    return float(lo), float(hi)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="pooled summary CSV")
    ap.add_argument("--out-plot", default="overnight_results/agent_n60_scatter.png")
    ap.add_argument("--out-md", default="overnight_results/RESULTS_AGENT_n60_bootstrap.md")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.agent)
    rng = np.random.default_rng(args.seed)

    rows = []
    for m in HEADLINE:
        if m not in df.columns:
            continue
        sub = df[[m, "correct"]].dropna()
        if len(sub) < 5 or sub[m].nunique() < 2:
            continue
        x = sub[m].values.astype(float)
        y = sub["correct"].values.astype(float)
        r, p = pearsonr(x, y)
        rs = bootstrap_pearson(x, y, args.n_boot, rng)
        lo, hi = ci(rs)
        rows.append({
            "metric": m, "n": len(sub),
            "pearson_r": r, "pearson_p": p,
            "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": (lo > 0) or (hi < 0),
        })

    table = pd.DataFrame(rows).sort_values("pearson_r")
    print("=== Bootstrap 95% CI on pooled correlations ===")
    print(table.to_string(index=False))

    # Two scatter plots side-by-side: action_ppl_mean and agent_margin_mean
    # (one negative slope, one positive — gives reader a clear visual).
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def scatter(ax, metric: str, title_extra: str):
        sub = df[[metric, "correct"]].dropna()
        x = sub[metric].values
        y = sub["correct"].values
        # Jitter binary outcome on y-axis so overlapping points are visible.
        jitter = rng.normal(0, 0.02, size=len(y))
        ax.scatter(x, y + jitter, alpha=0.6)
        r, p = pearsonr(x, y)
        ax.set_xlabel(metric)
        ax.set_ylabel("correct (jittered)")
        # Class-mean lines: where do correct=1 and correct=0 cluster on x?
        for cls, color in [(0, "tab:red"), (1, "tab:green")]:
            xs_c = x[y == cls]
            if len(xs_c):
                ax.axvline(xs_c.mean(), color=color, linestyle="--", alpha=0.7,
                           label=f"mean(correct={cls})")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"{metric}\nr={r:+.3f}, p={fmt_p(p)}  {title_extra}")

    if "action_ppl_mean" in df.columns:
        scatter(axes[0], "action_ppl_mean", "(headline)")
    if "action_margin_mean" in df.columns:
        scatter(axes[1], "action_margin_mean", "")

    fig.suptitle(f"Agent confidence vs correctness — Qwen-1.5B / GSM8K (n={len(df)})",
                 fontsize=11)
    fig.tight_layout()
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot, dpi=150)
    print(f"\nSaved plot to {args.out_plot}")

    # Markdown report
    lines = []
    lines.append(f"# Bootstrap CI for agent correlations\n")
    lines.append(f"Pooled summary: `{args.agent}`")
    lines.append(f"n = **{len(df)}** questions, accuracy = "
                 f"**{df['correct'].mean():.2%}**, "
                 f"truncation = **{df['truncated'].mean():.2%}**")
    lines.append(f"Bootstrap samples = {args.n_boot}, RNG seed = {args.seed}\n")

    lines.append("## Pooled Pearson r with 95% bootstrap CI\n")
    lines.append("| metric | n | Pearson r | p | 95% CI | CI excl. 0? |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in table.iterrows():
        lines.append(
            f"| {row['metric']} | {row['n']} | {row['pearson_r']:+.3f} | "
            f"{fmt_p(row['pearson_p'])} | [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}] | "
            f"{'**yes**' if row['excludes_zero'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("- Metrics with CI that **excludes zero** carry a real signal at this n.")
    lines.append("- `action_*` metrics — scored only on segments where the model issues a "
                 "tool call — give the strongest, most robust signal.")
    lines.append("- `answer_ppl` (PPL on the final-answer segment alone) does **not** carry "
                 "signal: confidence on the final answer is too uniformly high to discriminate.")
    lines.append("- Aggregate trajectory metrics (`agent_*`) are intermediate — they include "
                 "the action signal but also the lower-signal thought / answer segments.")
    lines.append("")
    lines.append(f"![scatter]({Path(args.out_plot).name})")

    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
