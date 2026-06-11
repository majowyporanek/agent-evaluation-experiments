"""Analyse tool-using agent traces and compare against pure-CoT pilot data.

Inputs:
  --agent       agent summary CSV (one row per question)
  --agent-trace agent trace CSV   (one row per generation segment)
  --cot         pure-CoT CSV from run_pilot (for the same model / seed)
  --out-md      markdown summary file to write

Produces:
  - Accuracy comparison: pure CoT vs tool agent
  - Correlations of every numeric agent-level metric with `correct`
  - Per-step-type breakdown (thought / action / answer) on the trace
  - Format-compliance numbers (parseable actions, truncation rate)

The goal is a one-look summary suitable for a results section: which
metrics, computed where, predict correctness, and how does the agent
setting differ from the CoT setting on the same model.

Usage:
    python -m experiments.analyze_agent \\
        --agent overnight_results/agent_15b_n30_seed0_summary.csv \\
        --agent-trace overnight_results/agent_15b_n30_seed0_trace.csv \\
        --cot overnight_results/results_seed0.csv \\
        --out-md overnight_results/RESULTS_AGENT.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Per-question agent metrics that should correlate with correctness.
AGENT_METRICS = [
    "agent_ppl_mean",
    "agent_entropy_mean",
    "agent_margin_mean",
    "agent_top1_mass_mean",
    "action_ppl_mean",
    "action_margin_mean",
    "thought_ppl_mean",
    "thought_margin_mean",
    "answer_ppl",
    "answer_margin",
    "n_actions",
    "n_actions_ok",
    "total_tokens",
]

# Per-question CoT metrics (from run_pilot CSVs).
COT_METRICS = [
    "cot_perplexity",
    "cot_mean_token_entropy",
    "cot_tokens",
    "cot_gen_margin_mean",
    "cot_gen_top1_mass_mean",
    "cot_gen_top5_mass_mean",
]


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def corr_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Pearson + Spearman correlations against `correct` for every metric
    that exists in `df` and has at least 3 non-null values with variance.
    """
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        sub = df[[m, "correct"]].dropna()
        if len(sub) < 3 or sub[m].nunique() < 2:
            continue
        r, rp = pearsonr(sub[m], sub["correct"])
        rho, rhop = spearmanr(sub[m], sub["correct"])
        rows.append({
            "metric": m, "n": len(sub),
            "pearson_r": r, "pearson_p": rp,
            "spearman_rho": rho, "spearman_p": rhop,
        })
    return pd.DataFrame(rows)


def step_type_breakdown(trace: pd.DataFrame) -> pd.DataFrame:
    """Mean of decoding signals per step_type (thought / action / answer)."""
    cols = ["ppl", "entropy", "margin", "top1_mass", "top5_mass", "n_tokens"]
    rows = []
    for st, grp in trace.groupby("step_type"):
        row = {"step_type": st, "n": len(grp)}
        for c in cols:
            row[f"{c}_mean"] = float(grp[c].mean()) if c in grp.columns else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def action_quality(trace: pd.DataFrame) -> dict:
    """Format-compliance numbers for action segments."""
    actions = trace[trace["step_type"] == "action"]
    if len(actions) == 0:
        return {"n_actions": 0, "n_actions_ok": 0, "parse_rate": float("nan"),
                "action_ppl_ok": float("nan"), "action_ppl_fail": float("nan")}
    ok = actions[actions["action_parse_ok"] == True]
    fail = actions[actions["action_parse_ok"] != True]
    return {
        "n_actions": len(actions),
        "n_actions_ok": len(ok),
        "parse_rate": len(ok) / len(actions),
        "action_ppl_ok": float(ok["ppl"].mean()) if len(ok) else float("nan"),
        "action_ppl_fail": float(fail["ppl"].mean()) if len(fail) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="agent _summary.csv")
    ap.add_argument("--agent-trace", required=True, help="agent _trace.csv")
    ap.add_argument("--cot", help="pure-CoT results.csv (optional)")
    ap.add_argument("--out-md", default="overnight_results/RESULTS_AGENT.md")
    args = ap.parse_args()

    agent = pd.read_csv(args.agent)
    trace = pd.read_csv(args.agent_trace)

    lines = []
    lines.append("# Tool-using agent results\n")
    lines.append(f"Agent summary: `{args.agent}`")
    lines.append(f"Agent trace:   `{args.agent_trace}`")
    if args.cot:
        lines.append(f"Pure CoT:      `{args.cot}`")
    lines.append("")

    # Top-line numbers
    n = len(agent)
    acc = agent["correct"].mean()
    trunc = agent["truncated"].mean()
    quality = action_quality(trace)

    lines.append("## Top-line\n")
    lines.append(f"- n = **{n}**")
    lines.append(f"- agent accuracy = **{acc:.2%}**")
    lines.append(f"- truncation rate = **{trunc:.2%}**")
    lines.append(f"- action format-compliance = **{quality['n_actions_ok']}/{quality['n_actions']}** "
                 f"(parse rate {quality['parse_rate']:.2%})")
    if not np.isnan(quality["action_ppl_ok"]):
        lines.append(f"- mean PPL of parseable actions: {quality['action_ppl_ok']:.3f}")
        lines.append(f"- mean PPL of failed actions:    {quality['action_ppl_fail']:.3f}")
    lines.append("")

    # CoT comparison (same model, same seed assumed)
    if args.cot:
        cot = pd.read_csv(args.cot)
        cot_acc = cot["correct"].mean()
        lines.append("## CoT vs agent on same model\n")
        lines.append(f"- pure CoT accuracy ({len(cot)} ex): **{cot_acc:.2%}**")
        lines.append(f"- tool agent accuracy ({n} ex): **{acc:.2%}**")
        lines.append(f"- delta = **{(acc - cot_acc) * 100:+.1f} pp**")
        lines.append("")

    # Agent-level metric correlations
    lines.append(f"## Agent-level metric correlations with `correct` (n={n})\n")
    agent_corr = corr_table(agent, AGENT_METRICS)
    if len(agent_corr):
        lines.append("| metric | n | Pearson r | p | Spearman ρ | p |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in agent_corr.sort_values("pearson_r").iterrows():
            lines.append(
                f"| {r['metric']} | {r['n']} | {r['pearson_r']:+.3f} | "
                f"{fmt_p(r['pearson_p'])} | {r['spearman_rho']:+.3f} | "
                f"{fmt_p(r['spearman_p'])} |"
            )
    else:
        lines.append("_No metrics had enough variance to correlate (likely n too small "
                     "or single-class outcome)._")
    lines.append("")

    # CoT correlations for reference
    if args.cot:
        cot_corr = corr_table(cot, COT_METRICS)
        if len(cot_corr):
            lines.append(f"## Pure-CoT correlations on same model (n={len(cot)})\n")
            lines.append("| metric | n | Pearson r | p | Spearman ρ | p |")
            lines.append("|---|---|---|---|---|---|")
            for _, r in cot_corr.sort_values("pearson_r").iterrows():
                lines.append(
                    f"| {r['metric']} | {r['n']} | {r['pearson_r']:+.3f} | "
                    f"{fmt_p(r['pearson_p'])} | {r['spearman_rho']:+.3f} | "
                    f"{fmt_p(r['spearman_p'])} |"
                )
            lines.append("")

    # Step-type breakdown
    lines.append("## Per-step-type decoding signals (trace level)\n")
    breakdown = step_type_breakdown(trace)
    if len(breakdown):
        cols = ["step_type", "n", "ppl_mean", "entropy_mean", "margin_mean",
                "top1_mass_mean", "n_tokens_mean"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in breakdown.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    cells.append(f"{v:.3f}" if not np.isnan(v) else "—")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Caveats\n")
    lines.append("- Small n: numbers above are exploratory. Wide confidence intervals.")
    lines.append("- Same-model scoring throughout: PPL on the model's own generation is "
                 "biased low. Cross-model scoring would be cleaner but is out of scope.")
    lines.append("- Truncated runs are included in correlation tables but have no "
                 "`final_answer`; their `correct` defaults to 0.")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_md}")

    # Also dump the key tables to stdout for quick reads.
    print("\n=== Agent-level correlations ===")
    if len(agent_corr):
        print(agent_corr.to_string(index=False))
    print("\n=== Step-type breakdown ===")
    print(breakdown.to_string(index=False))


if __name__ == "__main__":
    main()
