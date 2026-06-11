"""
Perplexity and entropy scoring utilities for the CoT-calibration pilot.

Two scoring modes are exposed:

1. score_text(text)
   Perplexity over an arbitrary piece of text. Used for the prompt-side
   perplexity baseline.

2. score_continuation(prompt, continuation)
   Teacher-forced perplexity of `continuation` CONDITIONED on `prompt`.
   Prompt tokens are excluded from the loss, so the resulting number
   reflects the model's confidence in the *generated* reasoning only.

Both functions are no-grad and run on CPU or GPU depending on what
`load_model` detects.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str):
    """Load an instruction-tuned causal LM and its tokenizer.

    Chooses fp16 on CUDA for speed and memory, fp32 on CPU for numerical
    stability. Puts the model in eval mode — we never train here, only
    score. Returns (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # bfloat16 on CPU roughly halves weight memory vs fp32 (Qwen-1.5B:
    # ~3 GB → ~1.5 GB). Numerically robust enough for inference / scoring,
    # unlike fp16 which underflows on long contexts.
    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
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
    """Compute perplexity and mean token entropy of `text`.

    Perplexity = exp(average cross-entropy loss), where the loss is the
    model's negative log-probability of each token given all previous
    tokens, averaged across the full sequence.

    Also returns the mean predictive entropy: at each position, how
    spread-out is the next-token distribution? Low entropy = peaked /
    confident; high entropy = flat / uncertain. Entropy and perplexity
    are related but distinct — entropy looks at the *whole distribution*,
    perplexity looks only at the probability of the chosen token.
    """
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
    """Teacher-forced perplexity of `continuation` given `prompt`.

    This is the core scoring function for the CoT-calibration research
    question. Input is [prompt + continuation]; output is a perplexity
    reflecting ONLY the continuation tokens.

    Mechanics:
      1. Tokenize prompt alone → prompt_len.
      2. Tokenize prompt + continuation → full_ids.
      3. Forward pass over full_ids (no generation — one-shot scoring).
      4. Slice logits to the positions that *predict* continuation
         tokens: logits[prompt_len - 1 : -1] predict full_ids[prompt_len:].
         (Causal-LM logits at position i predict the token at i + 1, so
         the first continuation token is predicted by the logits at the
         last prompt position.)
      5. Cross-entropy between those logits and the actual continuation
         tokens, averaged, then exponentiated → cot_perplexity.
      6. Token-level entropy over the same slice → cot_mean_token_entropy.

    The prompt tokens do NOT contribute to the loss. That is why this
    number reflects the model's confidence in its own reasoning and not
    in the question text.
    """
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
