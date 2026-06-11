"""Generate the report-style notebook from existing CSV results.

Sister script to `build_notebook.py`. Produces `notebooks/main_results_report.ipynb`,
a self-contained notebook that includes the research background, methodology,
and the same analysis cells. Markdown sections are written as report headlines
and descriptions (not as speaker notes).

Re-run after re-generating any CSV to refresh the notebook.

Usage:
    python -m experiments.build_notebook_report
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

    # ------------------------------------------------------------------
    # Cover
    # ------------------------------------------------------------------
    cells.append(md(
        "# Perplexity measures and LLM agent quality\n"
        "**Warsztaty AI - preliminary results report**  \n"
        "Author: Klaudia Chwistek  \n"
        "Supervisor: Luiz Miranda  \n"
        "Date: 2026-05  \n\n"
        "This notebook combines the research context, methodology, and headline "
        "experimental results in a single document. All analysis cells load "
        "precomputed CSVs from `results/` and `overnight_results/`; no model is "
        "loaded here. Generation code lives in `experiments/run_pilot.py` and "
        "`experiments/agent_calculator.py`.\n\n"
        "**Run order**: top to bottom. Total wall-time on a laptop: under 5 s."
    ))

    # ------------------------------------------------------------------
    # 1. Background
    # ------------------------------------------------------------------
    cells.append(md(
        "## 1. Background\n\n"
        "### 1.1 The motivating contradiction in the literature\n\n"
        "Perplexity (PPL) was designed as a language-modelling quality measure: "
        "the geometric mean of the inverse probabilities the model assigns to "
        "tokens in a held-out sequence. A natural question is whether it can be "
        "repurposed as an *evaluation* signal for downstream task success - "
        "and the literature is split.\n\n"
        "- **Gonen et al., 2023** - *Demystifying Prompts in Language Models via "
        "Perplexity Estimation* (Findings of EMNLP 2023). Finding: the "
        "performance of a prompt is predicted by how familiar the model is with "
        "its language. **Lower prompt perplexity ⇒ better task performance.** "
        "This is the **pro-perplexity** anchor.\n"
        "- **Wang et al., 2022** - *Perplexity from PLM Is Unreliable for "
        "Evaluating Text Quality* (arXiv:2210.05892). Finding: PPL is a poor "
        "proxy for text quality, with three failure modes - **length bias**, "
        "**repetition sensitivity**, **punctuation dependency**. This is the "
        "**anti-perplexity** anchor.\n\n"
        "The natural research question is **which position wins when we move "
        "from static prompts to multi-step agent trajectories?**\n\n"
        "### 1.2 The newer angle - long contexts and agents\n\n"
        "- **Fang et al., ICLR 2025** - *What Is Wrong with Perplexity for "
        "Long-Context Language Modeling?* (arXiv:2410.23771). Standard PPL "
        "averages over all tokens and dilutes the contribution of \"key tokens\" "
        "that depend on long-range context. Their re-weighted **LongPPL** "
        "achieves Pearson r ≈ −0.96 with long-context benchmarks vs near-zero "
        "for vanilla PPL. Implication: **agent trajectories are long contexts; "
        "the same pathology applies.**\n"
        "- **Mohammadi et al., KDD 2025** - *Evaluation and Benchmarking of LLM "
        "Agents: A Survey* (arXiv:2507.21504). Frames PPL as one signal among "
        "tool-use accuracy, plan quality, trajectory metrics, and safety.\n\n"
        "### 1.3 Agent uncertainty literature\n\n"
        "Three recent papers aggregate token-level probabilities (PPL, "
        "sequence prob, mean token entropy) into a scalar used to **decide "
        "deferral or replanning** in ReAct-style agents:\n\n"
        "- *ReDAct: Uncertainty-Aware Deferral for LLM Agents* (arXiv:2604.07036)\n"
        "- *Every Response Counts: Quantifying Uncertainty of LLM-based "
        "Multi-Agent Systems through Tensor Decomposition* (arXiv:2604.08708)\n"
        "- *Towards Agents That Know When They Don't Know: Uncertainty as a "
        "Control Signal for Structured Reasoning* (arXiv:2509.02401)\n\n"
        "These motivate the question this notebook addresses: **does "
        "token-level confidence carry usable signal about per-question "
        "correctness inside an agent, and where in the trajectory should we "
        "measure it?**"
    ))

    # ------------------------------------------------------------------
    # 2. Research question
    # ------------------------------------------------------------------
    cells.append(md(
        "## 2. Research question\n\n"
        "> Does perplexity (and related token-level signals - entropy, top-1 "
        "margin, top-1 mass) predict per-question correctness of an LLM in "
        "agent vs CoT settings, and does **where in the trajectory** we "
        "measure them matter?\n\n"
        "Two sub-questions inform the design:\n\n"
        "1. **Family**: among PPL, mean token entropy, top-1 margin, and "
        "top-1 mass, which is the best predictor of correctness?\n"
        "2. **Locus**: inside an agent, do action-segment, thought-segment, or "
        "answer-segment metrics carry the most signal?"
    ))

    # ------------------------------------------------------------------
    # 3. Methodology
    # ------------------------------------------------------------------
    cells.append(md(
        "## 3. Methodology\n\n"
        "### 3.1 Dataset - GSM8K\n\n"
        "**GSM8K** (Grade School Math 8K; Cobbe et al., OpenAI, 2021; "
        "arXiv:2110.14168). 8,500 high-quality grade-school math word "
        "problems (7,473 train / 1,319 test) written by human annotators. "
        "Each problem requires 2 to 8 reasoning steps. The format is a "
        "plain-English word problem followed by a step-by-step solution that "
        "ends with `#### <number>`, where the integer is the gold answer.\n\n"
        "**Example (test split):**\n\n"
        "> *Q: Natalia sold clips to 48 of her friends in April, and then she "
        "sold half as many clips in May. How many clips did Natalia sell "
        "altogether in April and May?*\n"
        ">\n"
        "> *A: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 "
        "clips altogether in April and May. #### 72*\n\n"
        "Chosen because grading is exact-match on the final integer (no "
        "LLM-as-judge needed), problems fit in <200 tokens, and difficulty "
        "is calibrated so a 0.5B–1.5B model achieves 20–60% accuracy - not "
        "saturated, not noise. Used as the standard benchmark in essentially "
        "every CoT paper since Wei et al. 2022.\n\n"
        "We use random samples of 50 (pilot) or 60 (headline) problems from "
        "the test split with fixed seeds. No fine-tuning - pure evaluation.\n\n"
        "### 3.2 Models - Qwen2.5 family\n\n"
        "- **Qwen2.5-0.5B-Instruct** - pilot experiments (Phase 1).\n"
        "- **Qwen2.5-1.5B-Instruct** - headline experiments (Phases 2 & 3).\n\n"
        "Same family, two scales - satisfies the *≥2 models* requirement of "
        "Warsztaty AI and lets us check whether findings replicate across "
        "model size.\n\n"
        "### 3.3 Decoding - greedy\n\n"
        "All experiments use greedy decoding (argmax at every step) to remove "
        "sampling noise so PPL is comparable across questions. Trade-off: "
        "PPL is compressed near 1.0 under greedy decoding because the model "
        "is always picking its own top-1 token. Listed as a limitation.\n\n"
        "### 3.4 Scoring - teacher-forced on the model's own continuation\n\n"
        "We compute log-probabilities in a **single forward pass** over the "
        "concatenation `prompt + continuation`, then **mask the prompt** so "
        "metrics aggregate only over the model's own generated tokens. This "
        "scores how confident the model was in its own reasoning, not how "
        "familiar the prompt was. Implemented in `experiments/perplexity.py` "
        "(`score_continuation`).\n\n"
        "### 3.5 Two settings - pure CoT and ReAct + calculator\n\n"
        "- **Pure CoT**: model is prompted to reason step-by-step and produce "
        "a final answer in a single generation. No tools.\n"
        "- **ReAct + calculator**: model alternates `Thought:` / `Action:` / "
        "`Observation:` / `Answer:` segments. Single tool - a calculator. "
        "Maximum 5 tool calls per question.\n\n"
        "Using the **same model and questions** in both settings lets us ask: "
        "does putting the model in an agent change the PPL ↔ correctness "
        "relationship?\n\n"
        "### 3.6 Metrics\n\n"
        "Per-token probabilities are aggregated into four scalars:\n\n"
        "- **Perplexity (PPL)** = `exp(-mean(log p(chosen_token)))`. Geometric "
        "mean of inverse top-1 probabilities. Lower ⇒ more confident.\n"
        "- **Mean token entropy** = `-sum p(v) log p(v)` averaged over the "
        "sequence. Uses the **full distribution**, not just the chosen token. "
        "Lower ⇒ more peaked.\n"
        "- **Top-1 margin** = `p(top-1) - p(top-2)`. Gap to the runner-up. "
        "Higher ⇒ more decisive.\n"
        "- **Top-1 mass** = `p(top-1)`. Probability assigned to the chosen "
        "token. Higher ⇒ more confident.\n\n"
        "For calibration analysis we use **ECE (Expected Calibration Error)**: "
        "equal-frequency binning on `confidence = 1 / PPL`, then compute "
        "`sum_b (n_b / N) · |mean(conf in bin) − mean(correct in bin)|`. "
        "Lower ⇒ better calibrated."
    ))

    # ------------------------------------------------------------------
    # 4. Setup
    # ------------------------------------------------------------------
    cells.append(md(
        "## 4. Setup"
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

    # ------------------------------------------------------------------
    # 5. Phase 1 - pilot
    # ------------------------------------------------------------------
    cells.append(md(
        "## 5. Phase 1 - Pilot (Qwen-0.5B-Instruct, pure CoT, n=150)\n\n"
        "Three seeds × 50 GSM8K examples, greedy decoding. The continuation "
        "is the model's own chain of thought; we score it teacher-forced "
        "and report perplexity and mean token entropy of the reasoning span. "
        "The pilot exists to establish that the perplexity-family signal is "
        "detectable at the smallest model size we use."
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
        "**Phase 1 finding.** `cot_mean_token_entropy` already outperforms "
        "`cot_perplexity` (r ≈ −0.32 vs −0.24) at the smallest model size. "
        "This is the first observation that drives the rest of the project: "
        "**the entropy/margin family carries more signal than the geometric "
        "mean of chosen-token probabilities** even when both are computed "
        "over identical sequences. The pattern replicates in every "
        "subsequent experiment."
    ))

    # ------------------------------------------------------------------
    # 6. Phase 2 - agent
    # ------------------------------------------------------------------
    cells.append(md(
        "## 6. Phase 2 - Tool-using ReAct agent (Qwen-1.5B, n=60)\n\n"
        "ReAct loop with a single calculator tool, max 5 tool calls per "
        "question. Per-segment decoding signals captured directly from "
        "`model.generate(output_scores=True)`. Two seeds × 30 questions; the "
        "pooled summary CSV concatenates them.\n\n"
        "Each trajectory is segmented into `thought`, `action`, "
        "`observation`, and `answer` step types. The summary CSV computes "
        "metrics both at the **whole-trajectory** level (`agent_*`) and "
        "restricted to specific step types (`action_*`, `answer_*`)."
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
        "**Phase 2 finding - segment matters.** `action_*` metrics - "
        "measured only on tool-call segments - give the strongest signal. "
        "`answer_ppl` is essentially noise (p > 0.5). Interpretation: by the "
        "time the model writes the final answer, the answer is already in "
        "the context window from the previous Observation, so the model is "
        "essentially copying. **Confidence on the *process* discriminates "
        "correctness; confidence on the *conclusion* does not.**"
    ))
    cells.append(code(
        "# Per-step-type breakdown - does confidence depend on what the model is doing?\n"
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
        "**Phase 2 finding - confidence depends systematically on step type.** "
        "Answers are most peaked (lowest PPL, highest margin); thoughts are "
        "least. The order is stable across both seeds, so this is not a "
        "single-run artefact but a property of the ReAct format under this "
        "model."
    ))

    # ------------------------------------------------------------------
    # 7. Phase 3 - same-model CoT
    # ------------------------------------------------------------------
    cells.append(md(
        "## 7. Phase 3 - Same-model CoT control (Qwen-1.5B, n=60)\n\n"
        "Same model and questions as the agent run, but no tool loop. This "
        "control lets us ask: **does putting the model in an agent harness "
        "*change* the perplexity-correctness relationship**, beyond simply "
        "changing accuracy?"
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
        "**Phase 3 finding - strongest single signal in the project.** "
        "`cot_mean_token_entropy` r = −0.49, p < 0.001 (n = 60). Putting "
        "Qwen-1.5B in a ReAct + calculator loop drops accuracy from **60% "
        "to 28%** on the same questions, yet does not destroy the "
        "perplexity-family signal - it relocates it (`action_ppl_mean` "
        "still r = −0.44 inside the agent)."
    ))

    # ------------------------------------------------------------------
    # 8. Headline comparison
    # ------------------------------------------------------------------
    cells.append(md(
        "## 8. Headline comparison - same model, two settings\n\n"
        "Single table summarising all three phases. The two patterns "
        "highlighted below are visible in every row."
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
        "**Two consistent patterns across every row of this table.**\n\n"
        "1. **Entropy / margin > PPL** in every setting and at both model "
        "sizes. Mean token entropy uses the full output distribution; "
        "PPL only the chosen token. Under greedy decoding the chosen-token "
        "probabilities are compressed near 1.0, but the distribution still "
        "varies - and entropy captures that variation.\n"
        "2. **Where you measure inside an agent matters.** Action-segment "
        "metrics dominate full-trajectory metrics; answer-segment metrics "
        "carry essentially no signal."
    ))

    # ------------------------------------------------------------------
    # 9. Bootstrap CI
    # ------------------------------------------------------------------
    cells.append(md(
        "## 9. Bootstrap 95% CI for the agent headline\n\n"
        "With n = 60, p-values alone are fragile to outliers. We report "
        "the bootstrap 95% confidence interval for each Pearson correlation "
        "by resampling questions with replacement 5,000 times and taking "
        "the 2.5 / 97.5 percentiles of the resulting r distribution. "
        "Same procedure as `experiments/agent_bootstrap.py`."
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
        "**Bootstrap finding.** `action_ppl_mean` has a 95% CI of "
        "approximately [−0.60, −0.25] - **fully excluding zero**. "
        "`answer_ppl`'s CI straddles zero. This is the formal version of "
        "the segment-locus claim from Phase 2."
    ))

    # ------------------------------------------------------------------
    # 10. Calibration
    # ------------------------------------------------------------------
    cells.append(md(
        "## 10. Calibration - reliability diagrams and ECE\n\n"
        "Correlation answers *can the model rank questions by difficulty?* "
        "Calibration answers a different question: *does the model's "
        "confidence number match its actual accuracy?* The two can dissociate.\n\n"
        "We define **confidence = 1 / perplexity**. Under greedy decoding "
        "this is the geometric mean of top-1 token probabilities, a value "
        "in (0, 1]. We use **equal-frequency binning** (10 quantile bins) "
        "because 1/PPL is bunched near 1.0 and equal-width binning would "
        "leave most bins empty.\n\n"
        "ECE is computed as `sum_b (n_b / N) · |mean(conf in bin) − "
        "mean(correct in bin)|`. ECE > 0.2 is considered poor calibration."
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
        "fig.suptitle('Reliability diagrams - Qwen-1.5B on GSM8K', fontsize=11)\n"
        "fig.tight_layout()\n"
        "plt.show()\n"
    ))
    cells.append(md(
        "**Calibration finding - the dissociation.** All three settings are "
        "visibly mis-calibrated (ECE > 0.2). The 1.5B CoT model is the "
        "**best** calibrated (ECE ≈ 0.27); putting the same model in an "
        "agent loop makes it markedly more over-confident (ECE ≈ 0.43–0.45).\n\n"
        "Note the **dissociation** from the correlation results: "
        "action-segment PPL is the *most discriminative* signal we have, "
        "but the agent is the *least calibrated* setting. Correlation and "
        "calibration are different questions, and **the same model can be "
        "better at ranking yet worse at calibration** when wrapped in an "
        "agent harness."
    ))

    # ------------------------------------------------------------------
    # 11. Findings
    # ------------------------------------------------------------------
    cells.append(md(
        "## 11. Findings\n\n"
        "1. **Entropy / margin beat PPL** as predictors of correctness in "
        "every setting tested (0.5B-CoT, 1.5B-CoT, 1.5B-Agent). PPL has "
        "historical priority as a calibration signal but is not the "
        "best-performing member of its own family.\n\n"
        "2. **Strongest single signal**: `cot_mean_token_entropy` on "
        "1.5B-CoT, r = −0.49, p < 0.001 (n = 60).\n\n"
        "3. **Putting Qwen-1.5B in a ReAct + calculator loop drops accuracy "
        "from 60% to 28%** on the same questions. The agent harness imposes "
        "overhead that this model size cannot reliably bear (frequent format "
        "errors, ~25% truncation).\n\n"
        "4. **Inside the agent, where you measure PPL matters**: action "
        "segments carry the strongest signal (r = −0.44, 95% CI "
        "[−0.60, −0.25]); the final-answer segment alone carries essentially "
        "none (r = −0.10, CI straddles zero). Confidence discriminates on "
        "the *process*, not the *conclusion*.\n\n"
        "5. **Confidence depends systematically on ReAct step type**: "
        "answers most peaked, thoughts least. Replicates across both seeds.\n\n"
        "6. **The agent is more over-confident than the same model in pure "
        "CoT** (ECE 0.43 vs 0.27). The PPL signal becomes *more* "
        "discriminative inside an agent but the model also becomes *less* "
        "calibrated. These are dissociable properties."
    ))

    # ------------------------------------------------------------------
    # 12. Limitations
    # ------------------------------------------------------------------
    cells.append(md(
        "## 12. Limitations\n\n"
        "- **Model scale**. We use 0.5B and 1.5B models. Calibration is "
        "known to improve with scale; the dissociation we observe could "
        "attenuate at 7B+.\n"
        "- **Single dataset and domain**. GSM8K only - grade-school maths. "
        "Findings may not generalise to reading comprehension, coding, or "
        "open-ended generation.\n"
        "- **Greedy decoding compresses PPL near 1.0**. Repeating with "
        "sampling and self-consistency would allow `1 / PPL` to move "
        "further from the ceiling and may strengthen the correlation "
        "signal.\n"
        "- **Same-model scoring throughout**. PPL on the model's own "
        "generation is biased low. A cross-model evaluator (e.g. score "
        "0.5B outputs with 7B) would be cleaner but was out of scope.\n"
        "- **Sample size**. n = 60 for the headline experiments. Sufficient "
        "for p < 0.01 on the main correlations and the bootstrap CI but "
        "too small for sub-group analyses."
    ))

    # ------------------------------------------------------------------
    # 13. References
    # ------------------------------------------------------------------
    cells.append(md(
        "## 13. References\n\n"
        "- Gonen, Iyer, Blevins, Smith, Zettlemoyer. *Demystifying Prompts "
        "in Language Models via Perplexity Estimation*. Findings of EMNLP "
        "2023. https://aclanthology.org/2023.findings-emnlp.679/\n"
        "- Wang, Deng, Sun, Meng. *Perplexity from PLM Is Unreliable for "
        "Evaluating Text Quality*. arXiv:2210.05892. https://arxiv.org/abs/2210.05892\n"
        "- Fang et al. *What Is Wrong with Perplexity for Long-Context "
        "Language Modeling?* ICLR 2025. arXiv:2410.23771. "
        "https://arxiv.org/abs/2410.23771\n"
        "- Mohammadi et al. *Evaluation and Benchmarking of LLM Agents: A "
        "Survey*. KDD 2025. arXiv:2507.21504. https://arxiv.org/abs/2507.21504\n"
        "- Cobbe et al. *Training Verifiers to Solve Math Word Problems* "
        "(GSM8K). arXiv:2110.14168. https://arxiv.org/abs/2110.14168\n"
        "- *ReDAct: Uncertainty-Aware Deferral for LLM Agents*. arXiv:2604.07036.\n"
        "- *Every Response Counts: Quantifying Uncertainty of LLM-based "
        "Multi-Agent Systems through Tensor Decomposition*. arXiv:2604.08708.\n"
        "- *Towards Agents That Know When They Don't Know*. arXiv:2509.02401.\n"
        "- Anthropic - *Demystifying Evals for AI Agents*. "
        "https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents\n"
    ))

    nb.cells = cells
    out_path = Path("notebooks/main_results_report.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_path))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
