"""Can a targeted word filter rescue the attacks the agreement rule misses?

The agreement rule (lexical AND fused must both flag) catches 166 of 178 test
attacks while blocking only six legitimate requests. The obvious next move is
to look at what it missed, mine vocabulary from those prompts, and OR that
back in.

The trap is that mining vocabulary from the *test* misses and then scoring on
the test set measures nothing: the rule would be fitted to the answers it is
about to be graded on, and would appear to catch almost everything. This
script therefore mines only from the VALIDATION split, which the protocol
already allows models to see, and evaluates once on test. That is the only
version of the idea whose result means anything.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from pidetect.config import DATA_PROCESSED, MODELS_DIR, SEED, TABLES_DIR
from pidetect.models.train import select_threshold


def xy(df):
    return (df.drop(columns=["label", "source_file"], errors="ignore")
              .values.astype(np.float32), df["label"].values)


def ngrams(text: str, n_max: int = 3):
    words = re.findall(r"[a-z']+", text.lower())
    for n in range(1, n_max + 1):
        for i in range(len(words) - n + 1):
            yield " ".join(words[i:i + n])


def report(y, pred, label):
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"rule": label, "caught": tp, "missed": fn, "blocked": fp,
            "recall": round(rec, 4), "F1": round(f1, 4)}


def main() -> None:
    sp = {s: pd.read_parquet(DATA_PROCESSED / f"{s}.parquet") for s in ("val", "test")}
    alt = {s: pd.read_parquet(DATA_PROCESSED / f"{s}_features_alt.parquet")
           for s in ("train", "val", "test")}
    y_val, y_test = sp["val"]["label"].values, sp["test"]["label"].values

    with open(MODELS_DIR / "combined_training_summary.json") as f:
        best = json.load(f)["xgboost"]["best_params"]

    X_tr, y_tr = xy(alt["train"])
    neg, pos = np.bincount(y_tr)
    fused = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          scale_pos_weight=neg / pos, tree_method="hist",
                          random_state=SEED, n_jobs=-1, **best).fit(X_tr, y_tr)
    lex = joblib.load(MODELS_DIR / "tfidf_logreg.joblib")

    f_val = fused.predict_proba(xy(alt["val"])[0])[:, 1]
    f_test = fused.predict_proba(xy(alt["test"])[0])[:, 1]
    l_val = lex.predict_proba(sp["val"]["prompt"])[:, 1]
    l_test = lex.predict_proba(sp["test"]["prompt"])[:, 1]

    t_l, t_f = select_threshold(y_val, l_val), select_threshold(y_val, f_val)
    agree_val = (l_val >= t_l) & (f_val >= t_f)
    agree_test = (l_test >= t_l) & (f_test >= t_f)

    # --- mine vocabulary from the attacks the agreement rule misses on VAL ---
    missed = sp["val"].loc[(y_val == 1) & ~agree_val, "prompt"]
    benign_val = sp["val"].loc[y_val == 0, "prompt"]
    print(f"agreement rule misses {len(missed)} attacks on validation\n")

    miss_counts = Counter()
    for p in missed:
        miss_counts.update(set(ngrams(p)))
    benign_counts = Counter()
    for p in benign_val:
        benign_counts.update(set(ngrams(p)))

    rows = [report(y_test, agree_test.astype(int), "Both must agree (baseline)")]

    # A phrase qualifies if it appears in several missed attacks and is rare
    # among legitimate prompts. Both thresholds are set on validation only.
    for min_hits, max_benign_rate in [(2, 0.002), (2, 0.005), (3, 0.005), (2, 0.010)]:
        phrases = {
            g for g, c in miss_counts.items()
            if c >= min_hits
            and benign_counts.get(g, 0) / max(len(benign_val), 1) <= max_benign_rate
            and len(g.split()) >= 2
        }
        if not phrases:
            continue
        pat = re.compile("|".join(re.escape(p) for p in sorted(phrases)))
        hits_test = sp["test"]["prompt"].str.lower().str.contains(pat, regex=True).values
        rescued = (agree_test | hits_test).astype(int)
        r = report(y_test, rescued,
                   f"Agreement OR phrase list (n={len(phrases)}, "
                   f"seen in >={min_hits} misses, <={max_benign_rate:.1%} of benign)")
        rows.append(r)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(TABLES_DIR / "targeted_rescue.csv", index=False)

    print("\nFor contrast only, the invalid version of this experiment:")
    missed_test = sp["test"].loc[(y_test == 1) & ~agree_test, "prompt"]
    tc = Counter()
    for p in missed_test:
        tc.update(set(ngrams(p)))
    cheat = {g for g, c in tc.items() if c >= 1 and len(g.split()) >= 2
             and benign_counts.get(g, 0) == 0}
    pat = re.compile("|".join(re.escape(p) for p in sorted(cheat)))
    hits = sp["test"]["prompt"].str.lower().str.contains(pat, regex=True).values
    r = report(y_test, (agree_test | hits).astype(int), "MINED FROM TEST (invalid)")
    print(f"  {r}")
    print("  This scores well because it was fitted to the answers. It is "
          "reported only to show what the shortcut would have produced.")


if __name__ == "__main__":
    main()
