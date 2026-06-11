# Perplexity measures and LLM agent quality

Master's-workshop project. The question: does a small LLM's own confidence
(perplexity, token entropy, top-1 margin) predict whether its answer is
correct — in plain chain-of-thought and inside a ReAct agent with a calculator,
on GSM8K.

Short version of what came out: entropy and margin predict correctness a bit
better than raw perplexity; inside the agent the signal sits on the Action
steps and disappears on the final answer; and while confidence orders questions
well, it is badly calibrated — the model is consistently over-confident.

The written report (`report/`) is the main deliverable.

## Where things are

- `report/` — the report (Polish, LaTeX). Start at `main.tex`.
- `experiments/` — all the code (see below).
- `results/` — pilot runs, Qwen2.5-0.5B, 3 seeds (n=150).
- `overnight_results/` — main runs, Qwen2.5-1.5B: CoT and agent CSVs, plus plots.
- `notebooks/` — notebooks bundling the results.
- `notes/` — slides and presentation notes (not part of the deliverable).

## experiments/

- `datasets.py` — load GSM8K, pull out the gold answer.
- `perplexity.py` — load model, teacher-forced scoring (PPL, entropy, margin), prompt masked.
- `run_pilot.py` — pilot: generate CoT, score it, correlate with correctness.
- `agent_calculator.py` — ReAct agent + calculator; writes per-question and per-segment CSVs.
- `analyze.py`, `analyze_pooled.py`, `analyze_agent.py` — correlation analysis.
- `agent_bootstrap.py` — bootstrap CIs for the agent correlations.
- `calibration.py` — ECE and reliability diagrams.
- `build_notebook*.py` — build the notebooks from the CSVs.

## Running

```bash
pip install -r requirements.txt
bash overnight_runs.sh                    # pilot, 0.5B
python -m experiments.agent_calculator --n 60 --seed 0 \
    --model Qwen/Qwen2.5-1.5B-Instruct --out-prefix overnight_results/agent_15b_seed0
```

Everything uses greedy decoding. Numbers in the report come straight from the
CSVs in `results/` and `overnight_results/`.
