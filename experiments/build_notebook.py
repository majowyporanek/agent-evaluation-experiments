"""Generate the deliverable notebook from existing CSV results.

This script writes `notebooks/main_results.ipynb` — a single self-contained
notebook for the Warsztaty AI submission. The notebook loads precomputed
CSVs from `results/` and `overnight_results/` and reproduces the headline
tables and plots. It does not re-run any model — that machinery lives in
the `experiments/` scripts.

Re-run this script after re-generating any CSV to refresh the notebook.

Usage:
    python -m experiments.build_notebook
"""
from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(
        "# Perplexity measures and LLM agent quality\n"
        "**Warsztaty AI — preliminary results notebook**  \n"
        "Author: Klaudia Chwistek  \n"
        "Supervisor: Luiz  \n"
        "Date: 2026-05  \n\n"
        "This notebook reproduces the headline tables and plots from three "
        "experiments on **GSM8K** with Qwen2.5 models, in two settings: pure "
        "chain-of-thought (CoT) and a single-tool ReAct agent (calculator). "
        "The research question is whether **perplexity-family metrics** "
        "(perplexity, token-level entropy, top-1 margin, top-1/top-5 mass) "
        "carry signal about per-question correctness, and whether *where* in "
        "the trajectory one measures them matters.\n\n"
        "All cells load precomputed CSVs from `results/` and "
        "`overnight_results/`. No model is loaded here — generation code "
        "lives in `experiments/run_pilot.py` and `experiments/agent_calculator.py`.\n\n"
        "**Run order**: top to bottom. Total wall-time on a laptop: under 5 s."
    ))

    cells.append(md(
        "## 0. Setup"
    ))
    cells.append(code(
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from scipy.stats import pearsonr, spearmanr\n"
        "\n"
        "REPO = Path.cwd()\n"
        "while REPO.name and not (REPO / 'experiments').exists():\n"
        "    REPO = REPO.parent\n"
        "print('repo root:', REPO)\n"
        "\n"
        "def fmt_p(p):\n"
        "    return '<0.001' if p < 0.001 else f'{p:.3f}'\n"
    ))

    cells.append(md(
        "## 1. Phase 1 — pilot (Qwen-0.5B-Instruct, pure CoT, n=150)\n\n"
        "Three seeds × 50 GSM8K examples, greedy decoding. Score continuation "
        "with teacher-forced perplexity and mean token entropy of the model's "
        "own reasoning."
    ))
    cells.append(code(
        "pilot = pd.concat([\n"
        "    pd.read_csv(REPO / f'results/results_seed{s}.csv').assign(seed=s)\n"
        "    for s in (0, 1, 2)\n"
        "], ignore_index=True)\n"
        "print(f'n = {len(pilot)}, accuracy = {pilot.correct.mean():.2%}')\n"
        "pilot.groupby('seed')['correct'].mean()\n"
    ))
    cells.append(code(
        "metrics = ['cot_perplexity', 'cot_mean_token_entropy', 'cot_tokens']\n"
        "rows = []\n"
        "for m in metrics:\n"
        "    r, rp = pearsonr(pilot[m], pilot['correct'])\n"
        "    rho, rhop = spearmanr(pilot[m], pilot['correct'])\n"
        "    rows.append({'metric': m, 'pearson_r': r, 'pearson_p': rp,\n"
        "                 'spearman_rho': rho, 'spearman_p': rhop, 'n': len(pilot)})\n"
        "pd.DataFrame(rows)\n"
    ))
    cells.append(md(
        "**Reading**: `cot_mean_token_entropy` already outperforms `cot_perplexity` "
        "(r ≈ −0.32 vs −0.24). First hint that the entropy/margin family beats "
        "geometric mean of chosen-token probabilities. This pattern will replicate "
        "in every subsequent experiment."
    ))

    cells.append(md(
        "## 2. Phase 2 — Tool-using ReAct agent (Qwen-1.5B, n=60)\n\n"
        "ReAct loop with a single calculator tool, max 5 tool calls per question. "
        "Per-segment decoding signals captured directly from "
        "`model.generate(output_scores=True)`. Two seeds × 30 questions; the "
        "pooled summary CSV concatenates them."
    ))
    cells.append(code(
        "agent = pd.read_csv(REPO / 'overnight_results/agent_15b_n60_pooled_summary.csv')\n"
        "trace = pd.read_csv(REPO / 'overnight_results/agent_15b_n60_pooled_trace.csv')\n"
        "print(f'n_questions = {len(agent)}, accuracy = {agent.correct.mean():.2%}, '\n"
        "      f'truncated = {agent.truncated.mean():.2%}')\n"
        "agent.groupby('seed')['correct'].mean()\n"
    ))
    cells.append(code(
        "# Correlations with `correct` for every numeric agent-level metric.\n"
        "AGENT_METRICS = [\n"
        "    'agent_ppl_mean', 'agent_entropy_mean',\n"
        "    'agent_margin_mean', 'agent_top1_mass_mean',\n"
        "    'action_ppl_mean', 'action_margin_mean',\n"
        "    'answer_ppl', 'answer_margin',\n"
        "]\n"
        "rows = []\n"
        "for m in AGENT_METRICS:\n"
        "    sub = agent[[m, 'correct']].dropna()\n"
        "    if len(sub) < 5 or sub[m].nunique() < 2:\n"
        "        continue\n"
        "    r, rp = pearsonr(sub[m], sub['correct'])\n"
        "    rho, rhop = spearmanr(sub[m], sub['correct'])\n"
        "    rows.append({'metric': m, 'n': len(sub),\n"
        "                 'pearson_r': r, 'pearson_p': rp,\n"
        "                 'spearman_rho': rho, 'spearman_p': rhop})\n"
        "pd.DataFrame(rows).sort_values('pearson_r')\n"
    ))
    cells.append(md(
        "**Reading**: `action_*` metrics — measured only on tool-call segments — "
        "give the strongest signal. `answer_ppl` is essentially noise (p > 0.5). "
        "Confidence on the *process* discriminates correctness; confidence on the "
        "*conclusion* does not."
    ))
    cells.append(code(
        "# Per-step-type breakdown — does confidence depend on what the model is doing?\n"
        "step_cols = ['ppl', 'entropy', 'margin', 'top1_mass']\n"
        "rows = []\n"
        "for st, grp in trace.groupby('step_type'):\n"
        "    row = {'step_type': st, 'n': len(grp)}\n"
        "    for c in step_cols:\n"
        "        row[f'{c}_mean'] = float(grp[c].mean())\n"
        "    rows.append(row)\n"
        "pd.DataFrame(rows).sort_values('ppl_mean')\n"
    ))
    cells.append(md(
        "**Reading**: answers are most peaked (lowest PPL, highest margin), thoughts "
        "are least. Order is stable across both seeds — not a single-run artefact."
    ))

    cells.append(md(
        "## 3. Phase 3 — Same-model CoT control (Qwen-1.5B, n=60)\n\n"
        "Same model and questions as the agent run, but no tool loop. Lets us "
        "ask: does putting the model in an agent harness *change* the "
        "perplexity-correctness relationship?"
    ))
    cells.append(code(
        "cot15 = pd.concat([\n"
        "    pd.read_csv(REPO / f'overnight_results/cot_15b_n30_seed{s}.csv').assign(seed=s)\n"
        "    for s in (0, 1)\n"
        "], ignore_index=True)\n"
        "print(f'n = {len(cot15)}, accuracy = {cot15.correct.mean():.2%}')\n"
        "cot15.groupby('seed')['correct'].mean()\n"
    ))
    cells.append(code(
        "metrics = ['cot_perplexity', 'cot_mean_token_entropy', 'cot_tokens', 'prompt_tokens']\n"
        "rows = []\n"
        "for m in metrics:\n"
        "    r, rp = pearsonr(cot15[m], cot15['correct'])\n"
        "    rho, rhop = spearmanr(cot15[m], cot15['correct'])\n"
        "    rows.append({'metric': m, 'pearson_r': r, 'pearson_p': rp,\n"
        "                 'spearman_rho': rho, 'spearman_p': rhop})\n"
        "pd.DataFrame(rows)\n"
    ))
    cells.append(md(
        "**Reading**: `cot_mean_token_entropy` r = −0.49, p < 0.001 — the strongest "
        "single perplexity-family signal across the whole project. Putting the model "
        "in a ReAct loop drops accuracy from 60 % to 28 % but does not destroy the "
        "PPL signal (`action_ppl_mean` still r = −0.44 inside the agent)."
    ))

    cells.append(md(
        "## 4. Headline comparison (same model, two settings)\n"
    ))
    cells.append(code(
        "def first_r(df, m):\n"
        "    sub = df[[m, 'correct']].dropna()\n"
        "    r, p = pearsonr(sub[m], sub['correct'])\n"
        "    return r, p, len(sub)\n"
        "\n"
        "headline = []\n"
        "for label, df, metric in [\n"
        "    ('0.5B CoT (pilot)', pilot, 'cot_perplexity'),\n"
        "    ('0.5B CoT (pilot)', pilot, 'cot_mean_token_entropy'),\n"
        "    ('1.5B CoT', cot15, 'cot_perplexity'),\n"
        "    ('1.5B CoT', cot15, 'cot_mean_token_entropy'),\n"
        "    ('1.5B Agent (full)', agent, 'agent_ppl_mean'),\n"
        "    ('1.5B Agent (full)', agent, 'agent_entropy_mean'),\n"
        "    ('1.5B Agent (action)', agent, 'action_ppl_mean'),\n"
        "    ('1.5B Agent (action)', agent, 'action_margin_mean'),\n"
        "]:\n"
        "    r, p, n = first_r(df, metric)\n"
        "    headline.append({'setting': label, 'metric': metric, 'n': n,\n"
        "                     'pearson_r': r, 'pearson_p': p})\n"
        "pd.DataFrame(headline)\n"
    ))
    cells.append(md(
        "Two consistent patterns visible across every row of this table:\n\n"
        "1. **Entropy / margin > PPL** in every setting and at both model sizes.\n"
        "2. **Where you measure inside an agent matters** — action-segment metrics "
        "carry the strongest signal we see anywhere on 1.5B."
    ))

    cells.append(md(
        "## 5. Bootstrap 95 % CI for the agent headline\n\n"
        "Same procedure as `experiments/agent_bootstrap.py` — resample questions "
        "with replacement 5000 times and report the 2.5 / 97.5 percentiles of the "
        "Pearson r distribution."
    ))
    cells.append(code(
        "def bootstrap_pearson(x, y, n_boot=5000, seed=0):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    n = len(x)\n"
        "    rs = np.empty(n_boot)\n"
        "    for b in range(n_boot):\n"
        "        idx = rng.integers(0, n, size=n)\n"
        "        xb, yb = x[idx], y[idx]\n"
        "        rs[b] = np.corrcoef(xb, yb)[0, 1] if xb.std() and yb.std() else np.nan\n"
        "    return rs\n"
        "\n"
        "rows = []\n"
        "for m in ['action_ppl_mean', 'action_margin_mean', 'agent_ppl_mean',\n"
        "          'agent_margin_mean', 'agent_entropy_mean', 'answer_ppl']:\n"
        "    sub = agent[[m, 'correct']].dropna()\n"
        "    if len(sub) < 5:\n"
        "        continue\n"
        "    x, y = sub[m].values.astype(float), sub['correct'].values.astype(float)\n"
        "    r, p = pearsonr(x, y)\n"
        "    rs = bootstrap_pearson(x, y)\n"
        "    rs = rs[~np.isnan(rs)]\n"
        "    rows.append({'metric': m, 'n': len(sub), 'pearson_r': r, 'pearson_p': p,\n"
        "                 'ci_lo': np.quantile(rs, 0.025),\n"
        "                 'ci_hi': np.quantile(rs, 0.975)})\n"
        "pd.DataFrame(rows).sort_values('pearson_r')\n"
    ))

    cells.append(md(
        "## 6. Calibration — reliability diagrams + ECE\n\n"
        "Confidence = 1 / perplexity (under greedy decoding this is the geometric "
        "mean of top-1 probabilities, a value in (0, 1]). Equal-frequency binning. "
        "ECE = sum_b (n_b / N) · |mean(conf in bin) − mean(correct in bin)|."
    ))
    cells.append(code(
        "def reliability(conf, correct, n_bins=10):\n"
        "    if len(conf) < n_bins:\n"
        "        n_bins = max(2, len(conf) // 2)\n"
        "    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))\n"
        "    edges[0] -= 1e-9\n"
        "    bins = np.digitize(conf, edges[1:-1])\n"
        "    rows = []\n"
        "    ece = 0.0\n"
        "    for b in range(bins.max() + 1):\n"
        "        m = bins == b\n"
        "        if not m.any():\n"
        "            continue\n"
        "        c_mean, a_mean = conf[m].mean(), correct[m].mean()\n"
        "        ece += (m.sum() / len(conf)) * abs(c_mean - a_mean)\n"
        "        rows.append({'n': int(m.sum()), 'conf': c_mean, 'acc': a_mean})\n"
        "    return ece, pd.DataFrame(rows)\n"
        "\n"
        "settings = [\n"
        "    ('1.5B CoT (1/cot_perplexity)', 1 / cot15['cot_perplexity'].values, cot15['correct'].values),\n"
        "    ('1.5B Agent full (1/agent_ppl_mean)', 1 / agent['agent_ppl_mean'].values, agent['correct'].values),\n"
        "]\n"
        "act = agent[agent['action_ppl_mean'].notna()]\n"
        "settings.append(('1.5B Agent action (1/action_ppl_mean)',\n"
        "                 1 / act['action_ppl_mean'].values, act['correct'].values))\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
        "for ax, (label, conf, corr) in zip(axes, settings):\n"
        "    ece_val, tab = reliability(conf, corr)\n"
        "    ax.plot([0, 1], [0, 1], '--', color='lightgray', label='perfect')\n"
        "    sizes = (tab['n'] / tab['n'].sum()) * 600 + 20\n"
        "    ax.scatter(tab['conf'], tab['acc'], s=sizes, alpha=0.7, edgecolor='black')\n"
        "    ax.plot(tab['conf'], tab['acc'], alpha=0.4)\n"
        "    ax.set_xlim(0, 1.02); ax.set_ylim(-0.05, 1.05)\n"
        "    ax.set_xlabel('mean confidence in bin'); ax.set_ylabel('mean accuracy in bin')\n"
        "    ax.set_title(f'{label}\\nECE = {ece_val:.3f}, n = {len(conf)}')\n"
        "    ax.legend(loc='lower right', fontsize=8)\n"
        "fig.suptitle('Reliability diagrams — Qwen-1.5B on GSM8K', fontsize=11)\n"
        "fig.tight_layout()\n"
        "plt.show()\n"
    ))
    cells.append(md(
        "**Reading**: All three settings are visibly mis-calibrated (ECE > 0.2 is "
        "considered poor). The 1.5B CoT model is the **best** calibrated (ECE ≈ 0.27); "
        "putting the same model in an agent loop makes it markedly more over-confident "
        "(ECE ≈ 0.43–0.45). Note the dissociation from the correlation results: "
        "action-segment PPL is the *most discriminative* signal we have, but the agent "
        "is the *least calibrated* setting. Correlation and calibration are different "
        "questions."
    ))

    cells.append(md(
        "## 7. Findings\n\n"
        "1. **Entropy / margin beat PPL** as predictors of correctness in every "
        "setting tested (0.5B-CoT, 1.5B-CoT, 1.5B-Agent). PPL has historical priority "
        "as a calibration signal but is not the best-performing member of its own family.\n\n"
        "2. **Strongest single signal**: `cot_mean_token_entropy` on 1.5B-CoT, r = −0.49, "
        "p < 0.001 (n = 60).\n\n"
        "3. **Putting Qwen-1.5B in a ReAct + calculator loop drops accuracy from 60 % to "
        "28 %** on the same questions. The agent harness imposes overhead that this model "
        "size cannot reliably bear (frequent format errors, ~25 % truncation).\n\n"
        "4. **Inside the agent, where you measure PPL matters**: action segments carry the "
        "strongest signal (r = −0.44, 95 % CI [−0.60, −0.25]); the final-answer segment "
        "alone carries essentially none (r = −0.10, CI straddles zero). Confidence "
        "discriminates on the *process*, not the *conclusion*.\n\n"
        "5. **Confidence depends systematically on ReAct step type**: answers most peaked, "
        "thoughts least. Replicates across both seeds.\n\n"
        "6. **The agent is more over-confident than the same model in pure CoT** "
        "(ECE 0.43 vs 0.27). The PPL signal becomes *more* discriminative inside an "
        "agent but the model also becomes *less* calibrated. These are dissociable.\n\n"
        "## Limitations\n\n"
        "- Small models (0.5B, 1.5B). Calibration is known to improve with scale.\n"
        "- One dataset (GSM8K), one domain (grade-school maths).\n"
        "- Greedy decoding compresses PPL near 1.0. Repeating with sampling and "
        "self-consistency would let `1 / PPL` move further from the ceiling.\n"
        "- Same-model scoring throughout — PPL on the model's own generation is biased "
        "low. A cross-model evaluator (e.g. score 0.5B outputs with 7B) would be cleaner "
        "but was out of scope.\n"
        "- n = 60 for the headline experiments. Sufficient for p < 0.01 on the main "
        "correlations but too small for sub-group analyses.\n"
    ))

    nb.cells = cells
    out_path = Path("notebooks/main_results.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_path))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
