"""Check every mapped file for MIXED classes.

The pipeline assumes each raw file contains a single class. That assumption
held for twelve files and failed silently for predictionguard_df.csv, which
turned out to be a benign block followed by a generated-injection block.
This script tests the assumption on every file rather than trusting it.

For each file it reports the share of rows matching an injection signature
and, crucially, whether that share is uniform across the file or jumps at a
boundary — the signature of two concatenated sources.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pidetect.config import DATA_RAW
from pidetect.data.prepare import FILE_LABEL_MAP, _P

VERB = (r"^(ignore|disregard|skip|neglect|bypass|omit|forget|overlook|do not|"
        r"pay no|discard|dismiss|leave out|set aside|refrain)")


def prompts(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    lookup = {c.lower(): c for c in header}
    col = next((lookup[c] for c in _P if c in lookup), None)
    if col is None:
        return None
    return pd.read_csv(path, usecols=[col], on_bad_lines="skip",
                       encoding_errors="replace", low_memory=False)[col].astype(str)


print(f"{'file':<42} {'rows':>7} {'inj%':>6} {'first-half':>11} {'second-half':>12}  verdict")
print("-" * 100)
for name, (label, _) in sorted(FILE_LABEL_MAP.items()):
    path = DATA_RAW / name
    if not path.exists():
        continue
    p = prompts(path)
    if p is None:
        continue
    sig = ((p.str.len() < 60) & p.str.lower().str.match(VERB)).values
    half = len(sig) // 2
    a, b = sig[:half].mean(), sig[half:].mean()
    # A homogeneous file has a similar rate in both halves; a concatenation
    # of two sources shows a large jump.
    mixed = abs(a - b) > 0.25
    verdict = "MIXED - investigate" if mixed else "homogeneous"
    print(f"{name:<42} {len(p):>7,} {sig.mean()*100:5.1f}% {a*100:10.1f}% {b*100:11.1f}%  {verdict}")
