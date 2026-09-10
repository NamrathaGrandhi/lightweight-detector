"""Label-consistency audit over EVERY row of EVERY file.

The structural audits checked how files are assembled. This one checks
whether each row's content agrees with the label its file implies, so that a
mislabelled block cannot hide the way predictionguard_df.csv did.

Two indicators are computed for every prompt:

  * override signature - an instruction to disregard prior instructions,
    or a role-play framing used to bypass rules. Expected in attacks.
  * task signature     - the phrasing of an ordinary request: a question,
    a maths or coding problem, a passage-comprehension prompt.

Rows whose content contradicts their file's label are counted and sampled,
so any remaining mislabelled population surfaces as a large contradiction
rate rather than having to be stumbled upon.
"""

from __future__ import annotations

import re

import pandas as pd

from pidetect.config import DATA_RAW
from pidetect.data.prepare import FILE_LABEL_MAP, ROW_RANGE_LABELS, _P

OVERRIDE = re.compile(
    r"(ignore|disregard|forget|skip|neglect|bypass|omit|overlook|discard|"
    r"dismiss|pay no (heed|attention)|do not (follow|obey|adhere))\s+"
    r"(all\s+)?(the\s+|your\s+|any\s+|prior|previous|above|preceding|earlier|"
    r"foregoing|initial|former)|"
    r"(you are|act as|pretend to be|roleplay as|role.play as|simulate)\s+"
    r"(now\s+)?(a|an|dan|another|going to)|"
    r"(jailbroken|do anything now|developer mode|without (any )?restrictions|"
    r"no longer bound|opposite mode|unfiltered)", re.I)

TASK = re.compile(
    r"^(read the following|given a context|from the description of a rule|"
    r"is this a piece of news|what label best describes|would you recommend|"
    r"can you (tell|generate|write|add|implement)|how (do|would|can) (i|you)|"
    r"what (is|are|was|were)|why (is|are|did|do)|write a|find |evaluate |"
    r"calculate |compute |solve |explain |describe |summarize |translate )", re.I)


def prompts(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    lookup = {c.lower(): c for c in header}
    col = next((lookup[c] for c in _P if c in lookup), None)
    return pd.read_csv(path, usecols=[col], on_bad_lines="skip",
                       encoding_errors="replace", low_memory=False)[col].astype(str)


def label_of(name, n):
    """Per-row label, honouring the row-range override for mixed files."""
    if name in ROW_RANGE_LABELS:
        lab = pd.Series(-1, index=range(n))
        for a, b, v in ROW_RANGE_LABELS[name]:
            lab.iloc[a:min(b, n)] = v
        return lab
    return pd.Series(FILE_LABEL_MAP[name][0], index=range(n))


print(f"{'file':<40} {'rows':>8} {'label':>6} {'override':>9} {'task-like':>10} "
      f"{'CONTRADICT':>11}")
print("-" * 92)
flagged = {}
for name in sorted(FILE_LABEL_MAP):
    path = DATA_RAW / name
    if not path.exists():
        continue
    p = prompts(path).reset_index(drop=True)
    lab = label_of(name, len(p))
    ovr = p.str.contains(OVERRIDE, regex=True)
    tsk = p.str.match(TASK)

    for value in sorted(lab.unique()):
        m = lab == value
        sub, o, t = p[m], ovr[m], tsk[m]
        # An attack with no override signature, or an ordinary prompt that
        # carries one, contradicts its label.
        bad = (~o) & t if value == 1 else o
        rate = bad.mean()
        mark = "  <-- CHECK" if rate > 0.30 else ""
        print(f"{name:<40} {len(sub):>8,} {value:>6} {o.mean()*100:8.1f}% "
              f"{t.mean()*100:9.1f}% {rate*100:10.1f}%{mark}")
        if bad.sum():
            flagged[f"{name} (label {value})"] = sub[bad]

print("\n\nSamples of contradicting rows (label vs content):")
for key, rows in flagged.items():
    if len(rows) < 5:
        continue
    print(f"\n  {key} - {len(rows):,} rows")
    for s in rows.drop_duplicates().head(3):
        print(f"      * {' '.join(s.split())[:120]}")
