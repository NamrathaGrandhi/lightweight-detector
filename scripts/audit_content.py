"""Characterise what each raw file ACTUALLY contains.

Written after predictionguard_df.csv turned out to hold two concatenated
sources. Filenames are not evidence; this samples every file and reports
what is really in it, so the label assignment can be checked by eye rather
than assumed.
"""

from __future__ import annotations

import pandas as pd

from pidetect.config import DATA_RAW
from pidetect.data.prepare import FILE_LABEL_MAP, ROW_RANGE_LABELS, _P

pd.set_option("display.width", 200)


def prompts(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    lookup = {c.lower(): c for c in header}
    col = next((lookup[c] for c in _P if c in lookup), None)
    return (pd.read_csv(path, usecols=[col], on_bad_lines="skip",
                        encoding_errors="replace", low_memory=False)[col]
            .astype(str), header)


for name in sorted(FILE_LABEL_MAP):
    path = DATA_RAW / name
    if not path.exists():
        continue
    p, header = prompts(path)
    uniq = p.nunique()
    lab = ("MIXED (row ranges)" if name in ROW_RANGE_LABELS
           else ("malicious" if FILE_LABEL_MAP[name][0] == 1 else "benign"))

    print("=" * 96)
    print(f"{name}")
    print(f"  labelled: {lab}   rows: {len(p):,}   unique: {uniq:,}   "
          f"repeat factor: {len(p)/uniq:.1f}x")
    print(f"  columns: {header}")
    print(f"  length  median {int(p.str.len().median()):,}  "
          f"min {p.str.len().min():,}  max {p.str.len().max():,}")

    # Does the corpus look templated? Measure how much of it shares an opening.
    heads = p.str.slice(0, 40).value_counts()
    print(f"  most common opening 40 chars covers {heads.iloc[0]/len(p)*100:.1f}% of rows:")
    print(f"      {heads.index[0]!r}")

    print("  --- 3 distinct samples (truncated) ---")
    for s in p.drop_duplicates().head(3):
        one = " ".join(s.split())
        print(f"      * {one[:150]}")
    print()
