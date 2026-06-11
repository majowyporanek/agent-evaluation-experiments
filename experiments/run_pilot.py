"""Pilot: correlate prompt perplexity with GSM8K task success.

Usage:
    python -m experiments.run_pilot --n 20 --model Qwen/Qwen2.5-1.5B-Instruct --out results.csv
"""
import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from experiments.datasets import extract_final_number, load_gsm8k
from experiments.perplexity import load_model, score_continuation, score_text

PROMPT_TEMPLATE = (
    "Solve this math problem step by step. "
    "End your answer on a new line with '#### <number>'.\n\n"
    "Problem: {question}\n\nSolution:"
)


def build_prompt(question: str, tokenizer) -> str:
    user_msg = PROMPT_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(prompt: str, model, tokenizer, max_new_tokens: int = 384) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def grade(output: str, gold) -> int:
    pred = extract_final_number(output)
    if pred is None or gold is None:
        return 0
    try:
        return int(float(pred) == float(gold))
    except ValueError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model)

    print(f"Loading {args.n} GSM8K examples...")
    examples = load_gsm8k(n=args.n, seed=args.seed)

    rows = []
    for i, ex in enumerate(tqdm(examples, desc="pilot")):
        prompt = build_prompt(ex["question"], tokenizer)
        scores = score_text(prompt, model, tokenizer)
        output = generate(prompt, model, tokenizer)
        cot_scores = score_continuation(prompt, output, model, tokenizer)
        correct = grade(output, ex["gold"])
        rows.append({
            "idx": i,
            "perplexity": scores["perplexity"],
            "mean_token_entropy": scores["mean_token_entropy"],
            "prompt_tokens": scores["n_tokens"],
            "cot_perplexity": cot_scores["cot_perplexity"],
            "cot_mean_token_entropy": cot_scores["cot_mean_token_entropy"],
            "cot_tokens": cot_scores["cot_tokens"],
            "gold": ex["gold"],
            "pred_output": output.replace("\n", " ")[:500],
            "correct": correct,
        })

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()
