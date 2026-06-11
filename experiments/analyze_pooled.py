"""Pool the 3 seed CSVs into a single n=150 dataframe and run the headline analysis.

Usage:
    python -m experiments.analyze_pooled \
        --glob 'overnight_results/results_seed*.csv' \
        --out-plot overnight_results/pooled_scatter.png \
        --out-md overnight_results/RESULTS.md
"""
import argparse
import glob as globlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

METRICS = [
    "perplexity",
    "cot_perplexity",
    "cot_mean_token_entropy",
    "prompt_tokens",
    "cot_tokens",
]


def load_pooled(pattern: str) -> pd.DataFrame:
    paths = sorted(globlib.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched {pattern!r}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["seed_file"] = Path(p).stem
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    return pooled


def correlations(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        r, rp = pearsonr(df[m], df["correct"])
        rho, rhop = spearmanr(df[m], df["correct"])
        rows.append({
            "metric": m,
            "pearson_r": r,
            "pearson_p": rp,
            "spearman_rho": rho,
            "spearman_p": rhop,
            "n": len(df),
        })
    return pd.DataFrame(rows)


def partial_corr(df: pd.DataFrame, x: str, y: str, z: str) -> tuple[float, float]:
    """Partial correlation of x and y controlling for z.

    Residualize x and y on z (linear regression), then Pearson-correlate the
    residuals. This tells us: after accounting for z, does x still carry signal
    about y?
    """
    sub = df[[x, y, z]].dropna()
    z_vec = sub[z].values
    A = np.vstack([z_vec, np.ones_like(z_vec)]).T

    def residualize(target):
        coef, *_ = np.linalg.lstsq(A, sub[target].values, rcond=None)
        pred = A @ coef
        return sub[target].values - pred

    rx, ry = residualize(x), residualize(y)
    r, p = pearsonr(rx, ry)
    return r, p


def per_seed_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for seed_file, grp in df.groupby("seed_file"):
        r, rp = pearsonr(grp[metric], grp["correct"])
        rho, rhop = spearmanr(grp[metric], grp["correct"])
        rows.append({
            "seed_file": seed_file,
            "n": len(grp),
            "accuracy": grp["correct"].mean(),
            "pearson_r": r,
            "pearson_p": rp,
            "spearman_rho": rho,
            "spearman_p": rhop,
        })
    return pd.DataFrame(rows)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="overnight_results/results_seed*.csv")
    ap.add_argument("--out-plot", default="overnight_results/pooled_scatter.png")
    ap.add_argument("--out-md", default="overnight_results/RESULTS.md")
    args = ap.parse_args()

    df = load_pooled(args.glob)
    metrics = [m for m in METRICS if m in df.columns]
    print(f"Pooled n = {len(df)}, accuracy = {df['correct'].mean():.2%}")
    print(f"Per-seed accuracy:")
    print(df.groupby("seed_file")["correct"].mean().to_string())
    print()

    print("=== Pooled correlations (n={}) ===".format(len(df)))
    pooled_corr = correlations(df, metrics)
    print(pooled_corr.to_string(index=False))
    print()

    print("=== Per-seed correlations for cot_perplexity ===")
    per_seed = per_seed_table(df, "cot_perplexity")
    print(per_seed.to_string(index=False))
    print()

    print("=== Partial correlation: cot_perplexity ~ correct | cot_tokens ===")
    pr, pp = partial_corr(df, "cot_perplexity", "correct", "cot_tokens")
    print(f"partial r = {pr:+.3f}  p = {fmt_p(pp)}")
    print("(If |partial r| << |raw r|, cot_perplexity's signal was mostly length.)")
    raw_cot = pooled_corr.loc[pooled_corr["metric"] == "cot_perplexity"].iloc[0]
    print(f"Raw pooled r    = {raw_cot['pearson_r']:+.3f}  p = {fmt_p(raw_cot['pearson_p'])}")
    print()

    # Scatter plot: CoT-PPL vs correctness, colored by seed
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for seed_file, grp in df.groupby("seed_file"):
        axes[0].scatter(
            grp["cot_perplexity"],
            grp["correct"] + (grp.index % 7) * 0.01,
            alpha=0.6,
            label=seed_file,
        )
    axes[0].set_xlabel("CoT perplexity")
    axes[0].set_ylabel("correct (jittered)")
    axes[0].set_title(
        f"CoT-PPL vs correctness (pooled n={len(df)})\n"
        f"r={raw_cot['pearson_r']:+.3f}, p={fmt_p(raw_cot['pearson_p'])}  |  "
        f"partial r (|cot_tokens)={pr:+.3f}"
    )
    axes[0].legend(fontsize=8)

    # CoT tokens vs correctness — the confounder
    tok = pooled_corr.loc[pooled_corr["metric"] == "cot_tokens"].iloc[0]
    axes[1].scatter(df["cot_tokens"], df["correct"] + (df.index % 7) * 0.01, alpha=0.6)
    axes[1].set_xlabel("CoT tokens")
    axes[1].set_ylabel("correct (jittered)")
    axes[1].set_title(
        f"CoT length vs correctness (the confounder)\n"
        f"r={tok['pearson_r']:+.3f}, p={fmt_p(tok['pearson_p'])}"
    )

    fig.tight_layout()
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot, dpi=150)
    print(f"Saved plot to {args.out_plot}")
    lines = []
    lines.append("# Pooled pilot results\n")
    lines.append(f"- Model: Qwen/Qwen2.5-0.5B-Instruct")
    lines.append(f"- Dataset: GSM8K, 3 seeds × 50 examples = **n = {len(df)}**")
    lines.append(f"- Pooled accuracy: **{df['correct'].mean():.2%}**")
    lines.append("")
    lines.append("## Per-seed accuracy")
    for seed_file, acc in df.groupby("seed_file")["correct"].mean().items():
        lines.append(f"- {seed_file}: {acc:.2%}")
    lines.append("")
    lines.append("## Pooled correlations with correctness (n={})".format(len(df)))
    lines.append("")
    lines.append("| metric | Pearson r | p | Spearman ρ | p |")
    lines.append("|---|---|---|---|---|")
    for _, row in pooled_corr.iterrows():
        lines.append(
            f"| {row['metric']} | {row['pearson_r']:+.3f} | {fmt_p(row['pearson_p'])} | "
            f"{row['spearman_rho']:+.3f} | {fmt_p(row['spearman_p'])} |"
        )
    lines.append("")
    lines.append("## Partial correlation (controlling for length)")
    lines.append("")
    lines.append(
        f"- Raw pooled r(cot_perplexity, correct) = **{raw_cot['pearson_r']:+.3f}** "
        f"(p = {fmt_p(raw_cot['pearson_p'])})"
    )
    lines.append(
        f"- Partial r(cot_perplexity, correct | cot_tokens) = **{pr:+.3f}** "
        f"(p = {fmt_p(pp)})"
    )
    lines.append("")
    lines.append(
        "Interpretation: if the partial correlation is much weaker than the raw one, "
        "most of the CoT-PPL signal was explained by length (the Wang 2022 critique). "
        "If it survives, PPL carries information beyond length."
    )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"Wrote summary to {args.out_md}")


if __name__ == "__main__":
    main()
