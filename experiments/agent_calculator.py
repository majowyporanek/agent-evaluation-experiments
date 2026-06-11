"""Tool-using ReAct agent on GSM8K with a single calculator tool.

The agent alternates between:
  - Thought:   free-text reasoning produced by the model
  - Action: calc(<expression>)     ← parsed and dispatched
  - Observation: <result>          ← injected back into the prompt
  - Answer: #### <number>          ← terminates the loop

Per-segment decoding signals (PPL, entropy, margin, top-1 mass, top-5 mass)
are captured directly from `model.generate(output_scores=True)`, so we can
ask:
  - Is the model more confident on Thought steps that lead to correct
    answers than on those that lead to incorrect ones?
  - Are Action segments (tool-call decisions) systematically more or less
    confident than Thought segments?
  - Does PPL on the final Answer segment carry signal beyond CoT PPL?

Outputs two CSVs:
  - agent_trace_<seed>.csv   — one row per generation segment
  - agent_summary_<seed>.csv — one row per question, with aggregates

Usage:
    python -m experiments.agent_calculator --n 50 --seed 0 \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --out-prefix overnight_results/agent_seed0
"""
import argparse
import ast
import csv
import operator
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from experiments.datasets import extract_final_number, load_gsm8k
from experiments.perplexity import load_model


# ----------------------------------------------------------------------
# Calculator tool — safe arithmetic eval via ast.parse with a whitelist
# ----------------------------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalcError(Exception):
    pass


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression with a whitelist of AST nodes.

    Accepts numbers and +, -, *, /, //, %, ** (with parentheses).
    Rejects names, attribute access, function calls, comprehensions, etc.,
    so the agent cannot accidentally `__import__('os').system(...)`.
    """
    tree = ast.parse(expr.strip(), mode="eval")
    return _eval_node(tree.body)


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise CalcError(f"unsupported op {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise CalcError(f"unsupported op {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise CalcError(f"unsupported node {type(node).__name__}")


# ----------------------------------------------------------------------
# ReAct prompt & parsers
# ----------------------------------------------------------------------

SYSTEM = (
    "You are a careful math problem solver with access to one tool.\n"
    "Use this exact format:\n"
    "Thought: <one short sentence of reasoning>\n"
    "Action: calc(<arithmetic expression>)\n"
    "After each Action you will see Observation: <number>.\n"
    "When you have the final numeric answer, write:\n"
    "Answer: #### <number>"
)

USER_TEMPLATE = "Problem: {question}"

ACTION_RE = re.compile(r"Action:\s*calc\((.+?)\)", re.DOTALL)
# Accept either the GSM8K-style "#### N" or the natural-language "Answer: N"
# pattern that small instruct models produce more reliably.
ANSWER_RE = re.compile(r"(?:####|Answer:)\s*([-+]?\d+(?:\.\d+)?)")


# ----------------------------------------------------------------------
# Per-segment metric extraction from generation scores
# ----------------------------------------------------------------------

def metrics_from_scores(scores) -> dict:
    """Aggregate per-step decoding signals into segment-level metrics.

    `scores` is the tuple returned by `model.generate(output_scores=True)`:
    one (1, vocab) logits tensor per generated token.

    Under greedy decoding the selected token at each step is argmax, so
    the log-prob of the selected token equals log(top1_prob). That makes
    teacher-forced PPL equivalent to generation-time PPL here — no need
    for a second forward pass.
    """
    if not scores:
        return {"n_tokens": 0, "ppl": float("nan"), "entropy": float("nan"),
                "margin": float("nan"), "top1_mass": float("nan"),
                "top5_mass": float("nan")}
    neg_log_top1, entropies, margins, top1s, top5s = [], [], [], [], []
    for step_logits in scores:
        log_probs = torch.log_softmax(step_logits[0], dim=-1)
        probs = log_probs.exp()
        topk = torch.topk(probs, k=5)
        top1s.append(topk.values[0].item())
        top5s.append(topk.values.sum().item())
        margins.append((topk.values[0] - topk.values[1]).item())
        neg_log_top1.append(-torch.log(topk.values[0].clamp_min(1e-12)).item())
        entropies.append(-(probs * log_probs).sum().item())
    n = len(margins)
    return {
        "n_tokens": n,
        "ppl": float(np.exp(sum(neg_log_top1) / n)),
        "entropy": sum(entropies) / n,
        "margin": sum(margins) / n,
        "top1_mass": sum(top1s) / n,
        "top5_mass": sum(top5s) / n,
    }


# ----------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------

MAX_TOOL_CALLS = 5
MAX_NEW_TOKENS_PER_STEP = 128


@torch.no_grad()
def run_agent(question: str, model, tokenizer) -> dict:
    """Run the ReAct loop on one question.

    Strategy: one growing prompt rather than chat-template-per-step.
    Each Observation is appended as plain text, just like the original
    ReAct paper. This keeps parsing simple and avoids the model getting
    confused by alternating role tags.
    """
    base_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TEMPLATE.format(question=question)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = base_prompt
    trace = []
    final_answer = None
    truncated = False

    for step_idx in range(MAX_TOOL_CALLS + 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_PER_STEP,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            stop_strings=["Observation:", "####"],
            tokenizer=tokenizer,
        )
        new_ids = out.sequences[0, inputs.input_ids.shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        metrics = metrics_from_scores(out.scores)

        answer_match = ANSWER_RE.search(text)
        action_match = ACTION_RE.search(text)

        if answer_match:
            final_answer = answer_match.group(1)
            trace.append({
                "step_idx": step_idx,
                "step_type": "answer",
                "text": text,
                "action_parse_ok": None,
                "calc_result": None,
                "calc_error": None,
                **metrics,
            })
            break

        if action_match:
            expr = action_match.group(1).strip()
            try:
                result = safe_eval(expr)
                calc_result = str(result)
                calc_error = None
                parse_ok = True
            except (CalcError, SyntaxError, ZeroDivisionError, ValueError) as e:
                result = None
                calc_result = None
                calc_error = f"{type(e).__name__}: {e}"
                parse_ok = False

            trace.append({
                "step_idx": step_idx,
                "step_type": "action",
                "text": text,
                "action_parse_ok": parse_ok,
                "calc_result": calc_result,
                "calc_error": calc_error,
                **metrics,
            })
            obs = calc_result if parse_ok else f"Error: {calc_error}"
            prompt = prompt + text + f"\nObservation: {obs}\n"
            continue

        # Pure thought (no action, no answer) — usually shouldn't happen
        # because of stop_strings, but capture defensively.
        trace.append({
            "step_idx": step_idx,
            "step_type": "thought",
            "text": text,
            "action_parse_ok": None,
            "calc_result": None,
            "calc_error": None,
            **metrics,
        })
        prompt = prompt + text
        if not text.strip():
            truncated = True
            break
    else:
        truncated = True

    return {"trace": trace, "final_answer": final_answer, "truncated": truncated}


# ----------------------------------------------------------------------
# Aggregation across the trace → one row per question
# ----------------------------------------------------------------------

def summarize_trace(qid: int, gold, result: dict) -> dict:
    """Per-question aggregates suitable for correlation against `correct`."""
    trace = result["trace"]
    answer = result["final_answer"]
    correct = 0
    if answer is not None and gold is not None:
        try:
            correct = int(float(answer) == float(gold))
        except ValueError:
            correct = 0

    thoughts = [t for t in trace if t["step_type"] == "thought"]
    actions = [t for t in trace if t["step_type"] == "action"]
    answers = [t for t in trace if t["step_type"] == "answer"]
    all_steps = trace

    def mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "qid": qid,
        "gold": gold,
        "final_answer": answer,
        "correct": correct,
        "truncated": int(result["truncated"]),
        "n_steps": len(trace),
        "n_actions": len(actions),
        "n_actions_ok": sum(1 for a in actions if a["action_parse_ok"]),
        "agent_ppl_mean": mean(t["ppl"] for t in all_steps),
        "agent_entropy_mean": mean(t["entropy"] for t in all_steps),
        "agent_margin_mean": mean(t["margin"] for t in all_steps),
        "agent_top1_mass_mean": mean(t["top1_mass"] for t in all_steps),
        "action_ppl_mean": mean(t["ppl"] for t in actions),
        "action_margin_mean": mean(t["margin"] for t in actions),
        "thought_ppl_mean": mean(t["ppl"] for t in thoughts),
        "thought_margin_mean": mean(t["margin"] for t in thoughts),
        "answer_ppl": mean(t["ppl"] for t in answers),
        "answer_margin": mean(t["margin"] for t in answers),
        "total_tokens": sum(t["n_tokens"] for t in all_steps),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

TRACE_FIELDS = [
    "qid", "step_idx", "step_type", "n_tokens",
    "ppl", "entropy", "margin", "top1_mass", "top5_mass",
    "action_parse_ok", "calc_result", "calc_error", "text",
]

SUMMARY_FIELDS = [
    "qid", "gold", "final_answer", "correct", "truncated",
    "n_steps", "n_actions", "n_actions_ok",
    "agent_ppl_mean", "agent_entropy_mean",
    "agent_margin_mean", "agent_top1_mass_mean",
    "action_ppl_mean", "action_margin_mean",
    "thought_ppl_mean", "thought_margin_mean",
    "answer_ppl", "answer_margin", "total_tokens",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="overnight_results/agent_seed0")
    args = ap.parse_args()

    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model)

    print(f"Loading {args.n} GSM8K examples (seed={args.seed})...")
    examples = load_gsm8k(n=args.n, seed=args.seed)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trace_path = out_prefix.with_name(out_prefix.name + "_trace.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")

    # Open both files for the full run; flush after each question so a
    # killed process still leaves partial data on disk.
    with trace_path.open("w", newline="") as trace_f, \
         summary_path.open("w", newline="") as summary_f:
        trace_w = csv.DictWriter(trace_f, fieldnames=TRACE_FIELDS)
        summary_w = csv.DictWriter(summary_f, fieldnames=SUMMARY_FIELDS)
        trace_w.writeheader()
        summary_w.writeheader()
        trace_f.flush()
        summary_f.flush()

        n_done = 0
        n_correct = 0
        n_trunc = 0
        for i, ex in enumerate(tqdm(examples, desc="agent")):
            result = run_agent(ex["question"], model, tokenizer)
            for step in result["trace"]:
                row = {"qid": i, **step}
                row["text"] = row["text"].replace("\n", " ")[:300]
                trace_w.writerow({k: row.get(k) for k in TRACE_FIELDS})
            summary = summarize_trace(i, ex["gold"], result)
            summary_w.writerow({k: summary.get(k) for k in SUMMARY_FIELDS})
            trace_f.flush()
            summary_f.flush()
            n_done += 1
            n_correct += summary["correct"]
            n_trunc += summary["truncated"]

    acc = n_correct / n_done if n_done else float("nan")
    print(f"Wrote {n_done} questions to {trace_path} / {summary_path}")
    print(f"Accuracy: {acc:.2%}   Truncated: {n_trunc}/{n_done}")


if __name__ == "__main__":
    main()
