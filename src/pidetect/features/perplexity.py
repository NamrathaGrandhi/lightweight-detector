"""Stage 2 — perplexity features from a small causal language model.

For a tokenised prompt w_1..w_N scored by model p, the per-token negative
log-likelihood (NLL) is

    nll_i = -log p(w_i | w_<i)

and the aggregate perplexity is

    PPL = exp( (1/N) * sum_i nll_i ).

Five features are extracted per prompt (thesis §3.5.2), exactly as listed
in the approved proposal §7.4:

    1. aggregate perplexity    PPL = exp(mean NLL)
    2. mean per-token NLL      central tendency of surprise
    3. variance of NLL         burstiness of surprise
    4. max per-token NLL       the single most surprising token
    5. argmax position / N     where in the prompt that spike occurs

The distributional features (3-5) are the ones that keep a signal alive when
a long fluent prompt dilutes the global average — the documented failure mode
of perplexity-only filters.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pidetect.config import PERPLEXITY_MAX_TOKENS, PERPLEXITY_MODEL

FEATURE_NAMES = ["ppl", "nll_mean", "nll_var", "nll_max", "nll_argmax_pos"]


class PerplexityScorer:
    """Wraps a small causal LM and yields the 5 perplexity features."""

    def __init__(self, model_name: str = PERPLEXITY_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        )

    @torch.no_grad()
    def token_nlls(self, text: str) -> np.ndarray:
        """Per-token negative log-likelihoods for one prompt."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=PERPLEXITY_MAX_TOKENS,
        ).to(self.device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            # A single-token prompt has no conditional distribution to score.
            return np.array([0.0], dtype=np.float32)

        logits = self.model(**enc).logits  # (1, T, V)
        # Shift: token i is predicted from positions < i.
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        target = input_ids[:, 1:]
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
        return nll[0].float().cpu().numpy()

    def extract_one(self, text: str) -> np.ndarray:
        nll = self.token_nlls(text)
        n = len(nll)
        # Aggregate perplexity per the proposal's equation; float64 exp then
        # clipped so a pathological prompt cannot overflow float32.
        ppl = float(np.clip(np.exp(np.float64(nll.mean())), 0, 1e30))
        return np.array(
            [
                ppl,
                float(nll.mean()),
                float(nll.var()),
                float(nll.max()),
                float(nll.argmax()) / n if n > 0 else 0.0,
            ],
            dtype=np.float32,
        )

    def extract_batch(self, texts: list[str], log_every: int = 500) -> np.ndarray:
        out = np.zeros((len(texts), len(FEATURE_NAMES)), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = self.extract_one(t)
            if log_every and (i + 1) % log_every == 0:
                print(f"  perplexity: {i + 1}/{len(texts)} prompts scored")
        return out
