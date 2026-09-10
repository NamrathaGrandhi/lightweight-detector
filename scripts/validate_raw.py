"""Thorough validation of the raw dataset before it is trusted.

Checks, per file: row count, full column list, any candidate label column
(and its value distribution), prompt-length stats, and a few real prompts
so the class content can be eyeballed. Also reports the total corpus size
and whether a benign class exists at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

LABEL_CANDIDATES = ["label", "is_injection", "malicious", "target", "class",
                    "injection", "y", "type", "category"]

raw_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "archive")
total = 0
for p in sorted(raw_dir.glob("*.csv")):
    print("=" * 78)
    print(p.name, f"({p.stat().st_size / 1024**2:.1f} MB)")
    df = pd.read_csv(p, on_bad_lines="skip", encoding_errors="replace",
                     low_memory=False)
    total += len(df)
    print(f"  rows: {len(df):,}")
    print(f"  columns: {list(df.columns)}")

    labels = [c for c in df.columns if c.lower() in LABEL_CANDIDATES]
    if labels:
        for c in labels:
            print(f"  LABEL COLUMN '{c}': {df[c].value_counts().to_dict()}")
    else:
        print("  LABEL COLUMN: none found")

    pcol = next((c for c in df.columns if c.lower() in ("prompt", "text",
                                                        "question")), None)
    if pcol:
        s = df[pcol].astype(str)
        print(f"  prompt column '{pcol}': len min={s.str.len().min()} "
              f"median={int(s.str.len().median())} max={s.str.len().max()}")
        print(f"  unique prompts: {s.nunique():,} of {len(s):,}")
        for i, val in enumerate(s.drop_duplicates().head(3)):
            print(f"    [{i}] {val[:150].replace(chr(10), ' ')}")
    print()

print("=" * 78)
print(f"TOTAL ROWS ACROSS FILES: {total:,}")
