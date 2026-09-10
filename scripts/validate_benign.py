"""Validate the benign companion release and quantify the length confound.

Reports per file: rows, columns, prompt column, and the length distribution.
Then compares the pooled benign and malicious length distributions, which is
the single most important validity question for this corpus (docs/dataset_validation.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("archive")
MALICIOUS = ["forbidden_question_set_with_prompts.csv", "jailbreak_prompts.csv",
             "malicous_deepset.csv", "predictionguard_df.csv"]
BENIGN = ["benign_deepset.csv", "boolq.csv", "code.csv", "docRED.csv",
          "platypus.csv", "puffin.csv", "super_glue_squad_v2.csv", "tapir.csv"]


def prompt_series(path: Path) -> tuple[pd.Series, list[str]]:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    pcol = next((c for c in cols if c.lower() in ("prompt", "text", "question",
                                                  "instruction", "input")), None)
    if pcol is None:
        return pd.Series(dtype=str), cols
    s = pd.read_csv(path, usecols=[pcol], on_bad_lines="skip",
                    encoding_errors="replace", low_memory=False)[pcol]
    return s.astype(str), cols


def describe(name: str, s: pd.Series) -> dict:
    lens = s.str.len()
    return {
        "file": name, "rows": len(s), "unique": s.nunique(),
        "p25": int(lens.quantile(.25)), "median": int(lens.median()),
        "p75": int(lens.quantile(.75)), "p95": int(lens.quantile(.95)),
    }


rows, pooled = [], {"malicious": [], "benign": []}
for group, files in [("malicious", MALICIOUS), ("benign", BENIGN)]:
    print(f"\n{'=' * 78}\n{group.upper()}\n{'=' * 78}")
    for f in files:
        path = RAW / f
        if not path.exists():
            print(f"  MISSING: {f}")
            continue
        s, cols = prompt_series(path)
        if s.empty:
            print(f"  NO PROMPT COLUMN in {f}: {cols}")
            continue
        d = describe(f, s) | {"group": group}
        rows.append(d)
        pooled[group].append(s)
        print(f"  {f:42s} rows={d['rows']:>7,} uniq={d['unique']:>7,} "
              f"len p25/med/p75/p95 = {d['p25']:>5}/{d['median']:>5}/"
              f"{d['p75']:>6}/{d['p95']:>6}")
        print(f"      e.g. {s.iloc[0][:100]!r}")

print(f"\n{'=' * 78}\nPOOLED LENGTH COMPARISON (the confound test)\n{'=' * 78}")
summary = {}
for g, parts in pooled.items():
    allp = pd.concat(parts, ignore_index=True)
    lens = allp.str.len()
    summary[g] = lens
    print(f"  {g:10s} n={len(allp):>7,}  median={int(lens.median()):>6}  "
          f"p25={int(lens.quantile(.25)):>6}  p75={int(lens.quantile(.75)):>6}")

# How separable are the classes on length alone? AUC of a length-only ranker.
m, b = summary["malicious"].values, summary["benign"].values
sample = 20_000
rng = np.random.default_rng(42)
ms = rng.choice(m, min(sample, len(m)), replace=False)
bs = rng.choice(b, min(sample, len(b)), replace=False)
# P(malicious length > benign length) == AUC for a length-only classifier.
auc = (ms[:, None] > bs[None, :]).mean() + 0.5 * (ms[:, None] == bs[None, :]).mean()
print(f"\n  Length-only AUC (malicious vs benign): {auc:.4f}")
print("  1.00 = length alone separates the classes perfectly (severe confound)")
print("  0.50 = length carries no information (no confound)")

pd.DataFrame(rows).to_csv("results/tables/source_length_stats.csv", index=False)
print("\n  per-file stats -> results/tables/source_length_stats.csv")
