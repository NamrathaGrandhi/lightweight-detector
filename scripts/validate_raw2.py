"""Deeper validation: is there ANY benign class or hidden label signal?"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

raw = Path("archive")

print("### 1. forbidden_question_set_df.csv — why 45,504 rows but 22 prompts?")
df = pd.read_csv(raw / "forbidden_question_set_df.csv", low_memory=False,
                 on_bad_lines="skip", encoding_errors="replace")
print("  idx  :", df["idx"].nunique(), "unique;  range",
      df["idx"].min(), "-", df["idx"].max())
print("  Unnamed: 0   :", df["Unnamed: 0"].nunique(), "unique")
print("  Unnamed: 0.1 :", df["Unnamed: 0.1"].nunique(), "unique")
print("  rows per unique prompt:", (len(df) / df["Prompt"].nunique()))
print("  --> the 22 jailbreak templates repeat; only 'idx' varies\n")

print("### 2. Do any files share prompts (overlap)?")
sets = {}
for name in ["forbidden_question_set_df.csv", "forbidden_question_set_with_prompts.csv",
             "jailbreak_prompts.csv", "malicous_deepset.csv", "predictionguard_df.csv"]:
    d = pd.read_csv(raw / name, low_memory=False, on_bad_lines="skip",
                    encoding_errors="replace", usecols=lambda c: c in ("Prompt",))
    sets[name] = set(d["Prompt"].astype(str))
    print(f"  {name:42s} {len(sets[name]):>7,} unique")
names = list(sets)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        ov = len(sets[names[i]] & sets[names[j]])
        if ov:
            print(f"  OVERLAP {names[i]} & {names[j]}: {ov:,}")
print()

print("### 3. Are any prompts plausibly BENIGN? (short, question-like, no attack markers)")
markers = ["ignore", "dan", "jailbreak", "pretend", "you are now", "developer mode",
           "forget", "system prompt", "no restrictions", "amoral", "unfiltered"]
for name, s in sets.items():
    short = [p for p in s if len(p) < 200]
    clean = [p for p in short if not any(m in p.lower() for m in markers)]
    print(f"  {name:42s} <200 chars: {len(short):>6,}  no-attack-marker: {len(clean):>6,}")
    for p in clean[:3]:
        print(f"      | {p[:110]}")
print()

print("### 4. Total unique prompts across the whole corpus")
allp = set().union(*sets.values())
print(f"  {len(allp):,} unique prompts (from 86,576 rows)")
