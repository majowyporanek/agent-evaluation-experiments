import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


@torch.no_grad()
def score_text(text: str, model, tokenizer) -> dict:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    n_tokens = input_ids.shape[1]
    if n_tokens < 2:
        return {"perplexity": float("nan"), "mean_token_entropy": float("nan"), "n_tokens": n_tokens}

    outputs = model(input_ids, labels=input_ids)
    ppl = torch.exp(outputs.loss).item()

    logits = outputs.logits[0, :-1]
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()

    return {"perplexity": ppl, "mean_token_entropy": entropy, "n_tokens": n_tokens}


@torch.no_grad()
def score_continuation(prompt: str, continuation: str, model, tokenizer) -> dict:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(prompt + continuation, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]
    n_cont = full_ids.shape[1] - prompt_len
    if n_cont < 1:
        return {"cot_perplexity": float("nan"), "cot_mean_token_entropy": float("nan"), "cot_tokens": 0}

    outputs = model(full_ids)
    logits = outputs.logits[0, prompt_len - 1 : -1]
    targets = full_ids[0, prompt_len:]
    loss = torch.nn.functional.cross_entropy(logits, targets)
    ppl = torch.exp(loss).item()

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()

    return {"cot_perplexity": ppl, "cot_mean_token_entropy": entropy, "cot_tokens": n_cont}
