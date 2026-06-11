"""Calibration analysis — reliability diagrams + ECE for CoT and Agent settings.

Confidence metric: 1 / perplexity. Under greedy decoding the selected token at
each step is the argmax, so 1/PPL equals the geometric mean of the top-1
probabilities over the scored tokens — a well-defined value in (0, 1].

Three columns of confidence are compared:
  - `1 / cot_perplexity`     (1.5B pure CoT, per-question, n=60)
  - `1 / agent_ppl_mean`     (1.5B ReAct agent, full trajectory, n=60)
  - `1 / action_ppl_mean`    (1.5B ReAct agent, action segments only, n≈48)

For each: equal-frequency binning (10 quantile bins). ECE = weighted mean
absolute gap between mean(confidence) and mean(correct) per bin.

Usage:
    python -m experiments.calibration \\
        --cot   overnight_results/cot_15b_n30_seed0.csv,overnight_results/cot_15b_n30_seed1.csv \\
        --agent overnight_results/agent_15b_n60_pooled_summary.csv \\
        --out-plot overnight_results/calibration_15b.png \\
        --out-md   overnight_results/RESULTS_CALIBRATION_15B.md
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def quantile_bins(conf: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency bin assignment (np.digitize-style ids in [0, n_bins))."""
    # `qcut` may collapse bins if many ties; ask for n_bins, accept fewer back.
    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9  # ensure min value falls in bin 0
    return np.digitize(conf, edges[1:-1])


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """Expected Calibration Error + per-bin table.

    ECE = sum_b (n_b / N) * |mean(conf in bin b) - mean(correct in bin b)|.

    Equal-frequency binning is used because confidence in this dataset is
    bunched near 1.0 — equal-width bins would leave most bins nearly empty.
    """
    if len(conf) < n_bins:
        n_bins = max(2, len(conf) // 2)
    bins = quantile_bins(conf, n_bins)
    rows = []
    total = len(conf)
    weighted_gap = 0.0
    for b in range(bins.max() + 1):
        m = bins == b
        if not m.any():
            continue
        c_mean = conf[m].mean()
        a_mean = correct[m].mean()
        gap = abs(c_mean - a_mean)
        weight = m.sum() / total
        weighted_gap += weight * gap
        rows.append({
            "bin": b, "n": int(m.sum()),
            "conf_mean": float(c_mean),
            "acc_mean": float(a_mean),
            "gap": float(c_mean - a_mean),
        })
    return float(weighted_gap), pd.DataFrame(rows)


def reliability_subplot(ax, conf: np.ndarray, correct: np.ndarray, title: str, n_bins: int):
    """One reliability diagram + on-axis ECE text."""
    ece_val, table = ece(conf, correct, n_bins=n_bins)
    # Diagonal = perfect calibration.
    ax.plot([0, 1], [0, 1], color="lightgray", linestyle="--", label="perfect")
    # Per-bin points sized by bin count.
    sizes = (table["n"] / table["n"].sum()) * 600 + 20
    ax.scatter(table["conf_mean"], table["acc_mean"], s=sizes, alpha=0.7,
               color="tab:blue", edgecolor="black")
    # Connect points to show direction.
    ax.plot(table["conf_mean"], table["acc_mean"], color="tab:blue", alpha=0.4)
    ax.set_xlabel("mean confidence (1 / PPL) in bin")
    ax.set_ylabel("mean accuracy in bin")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{title}\nECE = {ece_val:.3f}, n = {len(conf)}")
    ax.legend(loc="lower right", fontsize=8)
    return ece_val, table


def load_concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def fmt_p(x: float) -> str:
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", required=True,
                    help="comma-separated CoT result CSVs to pool")
    ap.add_argument("--agent", required=True, help="pooled agent summary CSV")
    ap.add_argument("--out-plot", default="overnight_results/calibration_15b.png")
    ap.add_argument("--out-md", default="overnight_results/RESULTS_CALIBRATION_15B.md")
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    cot_paths = [Path(p.strip()) for p in args.cot.split(",")]
    cot = load_concat(cot_paths)
    agent = pd.read_csv(args.agent)

    # CoT: 1/cot_perplexity
    cot = cot[cot["cot_perplexity"].notna() & (cot["cot_perplexity"] > 0)].copy()
    cot["conf"] = 1.0 / cot["cot_perplexity"]

    # Agent full-trajectory: 1/agent_ppl_mean
    agent_full = agent[agent["agent_ppl_mean"].notna() & (agent["agent_ppl_mean"] > 0)].copy()
    agent_full["conf"] = 1.0 / agent_full["agent_ppl_mean"]

    # Agent action-only: 1/action_ppl_mean (subset: questions that had ≥1 action)
    agent_act = agent[agent["action_ppl_mean"].notna() & (agent["action_ppl_mean"] > 0)].copy()
    agent_act["conf"] = 1.0 / agent_act["action_ppl_mean"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ece_cot, tab_cot = reliability_subplot(
        axes[0], cot["conf"].values, cot["correct"].values,
        title=f"1.5B CoT (1 / cot_perplexity)\nacc = {cot['correct'].mean():.2%}",
        n_bins=args.n_bins,
    )
    ece_agent_full, tab_af = reliability_subplot(
        axes[1], agent_full["conf"].values, agent_full["correct"].values,
        title=f"1.5B Agent — full trajectory (1 / agent_ppl_mean)\nacc = {agent_full['correct'].mean():.2%}",
        n_bins=args.n_bins,
    )
    ece_agent_act, tab_aa = reliability_subplot(
        axes[2], agent_act["conf"].values, agent_act["correct"].values,
        title=f"1.5B Agent — action segments (1 / action_ppl_mean)\nacc = {agent_act['correct'].mean():.2%}",
        n_bins=args.n_bins,
    )

    fig.suptitle(
        "Reliability diagrams — Qwen-1.5B-Instruct on GSM8K. "
        "Bubble size = bin weight. Above diagonal = under-confident; below = over-confident.",
        fontsize=10,
    )
    fig.tight_layout()
    Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot, dpi=150)
    print(f"Saved {args.out_plot}")

    # Markdown report
    lines = []
    lines.append("# Calibration: ECE + reliability diagrams (Qwen-1.5B on GSM8K)\n")
    lines.append("Confidence = 1 / perplexity. Under greedy decoding this is the geometric "
                 "mean of top-1 probabilities, a value in (0, 1].\n")
    lines.append("Equal-frequency binning with "
                 f"`n_bins={args.n_bins}` (clamps to fewer if ties collapse bins).\n")

    lines.append("## Headline ECE table\n")
    lines.append("| setting | metric used | n | accuracy | ECE |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| 1.5B CoT | 1 / cot_perplexity | {len(cot)} | "
        f"{cot['correct'].mean():.2%} | **{ece_cot:.3f}** |"
    )
    lines.append(
        f"| 1.5B Agent (full trajectory) | 1 / agent_ppl_mean | {len(agent_full)} | "
        f"{agent_full['correct'].mean():.2%} | **{ece_agent_full:.3f}** |"
    )
    lines.append(
        f"| 1.5B Agent (action segments) | 1 / action_ppl_mean | {len(agent_act)} | "
        f"{agent_act['correct'].mean():.2%} | **{ece_agent_act:.3f}** |"
    )
    lines.append("")

    def bin_block(name: str, tab: pd.DataFrame) -> list[str]:
        out = [f"### Per-bin breakdown — {name}\n"]
        out.append("| bin | n | mean confidence | mean accuracy | gap (conf − acc) |")
        out.append("|---|---|---|---|---|")
        for _, r in tab.iterrows():
            out.append(
                f"| {int(r['bin'])} | {int(r['n'])} | {r['conf_mean']:.3f} | "
                f"{r['acc_mean']:.3f} | {r['gap']:+.3f} |"
            )
        out.append("")
        return out

    lines += bin_block("1.5B CoT", tab_cot)
    lines += bin_block("1.5B Agent — full trajectory", tab_af)
    lines += bin_block("1.5B Agent — action segments", tab_aa)

    lines.append("## How to read the numbers\n")
    lines.append("- **ECE close to 0** = confidence numerically matches accuracy. A perfectly "
                 "calibrated model would have ECE = 0.")
    lines.append("- **Gap > 0** in a bin = model is **over-confident** there: it claims "
                 "(say) 0.85 probability of correctness but is only 0.50 accurate.")
    lines.append("- **Gap < 0** = under-confident.")
    lines.append("- Equal-frequency binning is used because 1/PPL is bunched near 1.0; "
                 "equal-width bins would leave most bins empty.")
    lines.append("")
    lines.append(f"![calibration]({Path(args.out_plot).name})")

    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_md}")

    print(f"\nECE summary:")
    print(f"  CoT 1.5B:                 {ece_cot:.3f}")
    print(f"  Agent 1.5B (full):        {ece_agent_full:.3f}")
    print(f"  Agent 1.5B (action only): {ece_agent_act:.3f}")


if __name__ == "__main__":
    main()
