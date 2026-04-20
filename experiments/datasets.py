import re

from datasets import load_dataset

GSM8K_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")


def load_gsm8k(n: int = 20, split: str = "test", seed: int = 0):
    ds = load_dataset("gsm8k", "main", split=split)
    ds = ds.shuffle(seed=seed).select(range(n))
    out = []
    for row in ds:
        m = GSM8K_ANSWER_RE.search(row["answer"])
        gold = m.group(1).replace(",", "") if m else None
        out.append({"question": row["question"], "gold": gold, "rationale": row["answer"]})
    return out


def extract_final_number(text: str):
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")
