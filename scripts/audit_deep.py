"""Deeper audit: look for hidden sub-populations inside each file.

The earlier homogeneity check compared a file's first half with its second
half, which catches a single mid-file boundary — how predictionguard_df.csv
was found. That test would miss a boundary near either end, or several
populations, so this scans every file in 20 slices and flags any abrupt
change, plus checks whether files overlap each other.

Every row of every file is examined. Individual prompts are still only
sampled by eye, so this is a structural check, not a semantic one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pidetect.config import DATA_RAW
from pidetect.data.prepare import FILE_LABEL_MAP, _P

VERB = (r"^(ignore|disregard|skip|neglect|bypass|omit|forget|overlook|do not|"
        r"pay no|discard|dismiss|leave out|set aside|refrain)")
SLICES = 20


def prompts(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    lookup = {c.lower(): c for c in header}
    col = next((lookup[c] for c in _P if c in lookup), None)
    return pd.read_csv(path, usecols=[col], on_bad_lines="skip",
                       encoding_errors="replace", low_memory=False)[col].astype(str)


sets = {}
print(f"{'file':<42} {'slices w/ injection signature':>30}   verdict")
print("-" * 96)
for name in sorted(FILE_LABEL_MAP):
    path = DATA_RAW / name
    if not path.exists():
        continue
    p = prompts(path)
    sets[name] = set(p)
    sig = ((p.str.len() < 60) & p.str.lower().str.match(VERB)).values
    parts = np.array_split(sig, SLICES)
    rates = np.array([s.mean() for s in parts])
    # A homogeneous file has a near-constant rate across all 20 slices.
    jump = rates.max() - rates.min()
    profile = "".join("#" if r > 0.5 else ("-" if r > 0.05 else ".") for r in rates)
    verdict = "MIXED - two populations" if jump > 0.5 else "uniform"
    print(f"{name:<42} {profile:>30}   {verdict}")

print("\n('#' = mostly injection strings, '.' = none, each mark is 5% of the file)")

print("\n\nCross-file duplicate prompts (same text appearing in two files):")
names = list(sets)
found = False
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        ov = sets[names[i]] & sets[names[j]]
        if ov:
            found = True
            la = FILE_LABEL_MAP[names[i]][0]
            lb = FILE_LABEL_MAP[names[j]][0]
            flag = "  <-- CONFLICTING LABELS" if la != lb else ""
            print(f"  {len(ov):>6} shared: {names[i]} & {names[j]}{flag}")
if not found:
    print("  none")
