"""Do the models fail on the same prompts, or on different ones?

The headline counts say the word-frequency baseline misses 8 attacks, the
fused detector with GPT-2 misses 4 and with Qwen misses 9. Those totals are
compatible with two very different situations: the methods could be failing
on the same hard prompts, or on disjoint sets. Only the second would mean
they carry complementary information, and only the second would make an
ensemble worth building. Nothing in the headline metrics distinguishes them,
so this script measures the overlap directly.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from pidetect.config import DATA_PROCESSED, MODELS_DIR, SEED, TABLES_DIR
from pidetect.models.train import select_threshold


def xy(df: pd.DataFrame):
    y = df["label"].values
    X = df.drop(columns=["label", "source_file"], errors="ignore")
    return X.values.astype(np.float32), y


def fit_predict(train, val, test, params):
    """Refit with the main run's hyperparameters and return test predictions."""
    X_tr, y_tr = xy(train)
    X_val, y_val = xy(val)
    X_te, _ = xy(test)
    neg, pos = np.bincount(y_tr)
    m = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                      scale_pos_weight=neg / pos, tree_method="hist",
                      random_state=SEED, n_jobs=-1, **params)
    m.fit(X_tr, y_tr)
    tau = select_threshold(y_val, m.predict_proba(X_val)[:, 1])
    return (m.predict_proba(X_te)[:, 1] >= tau).astype(int)


def main() -> None:
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet").reset_index(drop=True)
    y = test["label"].values
    n_attack, n_benign = int(y.sum()), int((y == 0).sum())
    print(f"test set: {len(test):,} prompts, {n_attack} attacks, {n_benign} legitimate\n")

    with open(MODELS_DIR / "combined_training_summary.json") as f:
        best = json.load(f)["xgboost"]["best_params"]

    feats = {s: pd.read_parquet(DATA_PROCESSED / f"{s}_features.parquet")
             for s in ["train", "val", "test"]}
    alt = {s: pd.read_parquet(DATA_PROCESSED / f"{s}_features_alt.parquet")
           for s in ["train", "val", "test"]}

    preds = {
        "Fusion+GPT-2": fit_predict(feats["train"], feats["val"], feats["test"], best),
        "Fusion+Qwen": fit_predict(alt["train"], alt["val"], alt["test"], best),
    }

    # The lexical baseline's errors are already recorded row by row.
    mis = pd.read_csv(TABLES_DIR / "misclassified_tfidf.csv")
    tf = np.zeros(len(test), dtype=int)
    prompt_to_row = {p: i for i, p in enumerate(test["prompt"])}
    tf[[prompt_to_row[p] for p in test["prompt"]]] = 0
    for _, r in mis.iterrows():
        i = prompt_to_row.get(r["prompt"])
        if i is not None:
            tf[i] = int(r["pred"])
    # Everything not listed as misclassified was predicted correctly.
    listed = {prompt_to_row[p] for p in mis["prompt"] if p in prompt_to_row}
    for i in range(len(test)):
        if i not in listed:
            tf[i] = y[i]
    preds["Word list"] = tf

    order = ["Word list", "Fusion+GPT-2", "Fusion+Qwen"]
    miss = {k: set(np.where((y == 1) & (preds[k] == 0))[0]) for k in order}
    fp = {k: set(np.where((y == 0) & (preds[k] == 1))[0]) for k in order}

    for name, sets, total in [("MISSED ATTACKS", miss, n_attack),
                              ("WRONGLY BLOCKED", fp, n_benign)]:
        print(f"=== {name} (of {total}) ===")
        for k in order:
            print(f"  {k:<14}{len(sets[k]):>5}")
        u = set().union(*sets.values())
        i = set.intersection(*sets.values())
        print(f"  {'union':<14}{len(u):>5}   (prompts failed by at least one method)")
        print(f"  {'all three':<14}{len(i):>5}   (failed by every method)")
        print("  pairwise shared:")
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                x, z = order[a], order[b]
                print(f"    {x} & {z}: {len(sets[x] & sets[z])}")
        only = {k: sets[k] - set().union(*(sets[j] for j in order if j != k))
                for k in order}
        print("  unique to one method:")
        for k in order:
            print(f"    {k}: {len(only[k])}")
        print()

    # If the misses were disjoint, a perfect union of the two best methods
    # would leave only the prompts that both miss.
    both = miss["Word list"] & miss["Fusion+GPT-2"]
    print(f"Word list and Fusion+GPT-2 both miss {len(both)} attacks. "
          f"An oracle picking the better of the two on each prompt would miss "
          f"{len(both)} of {n_attack}, i.e. catch {n_attack - len(both)}.")

    rows = []
    for name, sets in [("missed_attacks", miss), ("wrongly_blocked", fp)]:
        for k in order:
            rows.append({"error_type": name, "model": k, "count": len(sets[k])})
        rows.append({"error_type": name, "model": "union",
                     "count": len(set().union(*sets.values()))})
        rows.append({"error_type": name, "model": "all_three",
                     "count": len(set.intersection(*sets.values()))})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "error_overlap.csv", index=False)
    print(f"\nwritten: {TABLES_DIR / 'error_overlap.csv'}")


if __name__ == "__main__":
    main()
