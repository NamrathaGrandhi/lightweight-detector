"""Stage 1 — text-surface features.

Twelve cheap, model-free features computed directly from the raw prompt
string (thesis §3.5.1). None of these requires a neural network; the whole
block costs microseconds per prompt and gives a strong signal for obfuscated
payloads (base64 blobs, code fences, shouty override instructions) that
fluent-text models can miss.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

import numpy as np

from pidetect.config import BASE64_RUN_MIN, TRIGGER_PHRASES

# Longest run of base64-alphabet characters: a strong indicator of encoded
# payloads or adversarial suffixes.
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{%d,}" % BASE64_RUN_MIN)
_CODE_FENCE = re.compile(r"```")
_PUNCT = set(string.punctuation)

FEATURE_NAMES = [
    "char_len",
    "token_len",
    "avg_token_len",
    "uppercase_ratio",
    "punctuation_ratio",
    "digit_ratio",
    "whitespace_ratio",
    "trigger_phrase_count",
    "newline_count",
    "code_fence_count",
    "base64_run_maxlen",
    "shannon_entropy",
]


def shannon_entropy(text: str) -> float:
    """Character-level Shannon entropy in bits.

    English prose sits around 4.0-4.5 bits; base64/random strings push
    towards 6 bits, and highly repetitive filler drops below 3.5.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_one(text: str) -> np.ndarray:
    """Return the 12-dimensional text-surface vector for a single prompt."""
    n = len(text)
    if n == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    tokens = text.split()
    n_tokens = len(tokens)
    lower = text.lower()

    trigger_count = sum(lower.count(p) for p in TRIGGER_PHRASES)
    base64_runs = _BASE64_RUN.findall(text)

    feats = np.array(
        [
            n,
            n_tokens,
            (n / n_tokens) if n_tokens else 0.0,
            sum(ch.isupper() for ch in text) / n,
            sum(ch in _PUNCT for ch in text) / n,
            sum(ch.isdigit() for ch in text) / n,
            sum(ch.isspace() for ch in text) / n,
            trigger_count,
            text.count("\n"),
            len(_CODE_FENCE.findall(text)),
            max((len(r) for r in base64_runs), default=0),
            shannon_entropy(text),
        ],
        dtype=np.float32,
    )
    return feats


def extract_batch(texts: list[str]) -> np.ndarray:
    """Vectorise a list of prompts -> (n_samples, 12) float32 matrix."""
    return np.vstack([extract_one(t) for t in texts])
