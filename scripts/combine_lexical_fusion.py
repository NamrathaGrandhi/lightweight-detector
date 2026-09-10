"""Can the lexical baseline and the fused detector be combined to beat both?

Section 5.7 shows the two do not fail on the same prompts: only three of the
178 attacks defeat both, so an oracle choosing per prompt would catch 175.
This asks whether a rule that has no oracle can get near that, and what it
costs in false positives.

Every combination selects its decision threshold on the validation split
under the same rule as the main protocol (§3.6, step 4), and touches the test
split once. The combinations themselves were conceived after the test set was
opened, so these are exploratory results and are reported as such.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from pidetect.config import DATA_PROCESSED, MODELS_DIR, SEED, TABLES_DIR
from pidetect.models.train import select_threshold


def xy(df: pd.DataFrame):
    y = df["label"].values
    X = df.drop(columns=["label", "source_file"], errors="ignore")
    return X.values.astype(np.float32), y


def fit_scores(train, val, test, params):
    X_tr, y_tr = xy(train)
    neg, pos = np.bincount(y_tr)
    m = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                      scale_pos_weight=neg / pos, tree_method="hist",
                      random_state=SEED, n_jobs=-1, **params)
    m.fit(X_tr, y_tr)
    return (m.predict_proba(xy(val)[0])[:, 1],
            m.predict_proba(xy(test)[0])[:, 1])


def report(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"caught": tp, "missed": fn, "blocked": fp, "F1": round(f1, 4)}


def main() -> None:
    splits = {s: pd.read_parquet(DATA_PROCESSED / f"{s}.parquet")
              for s in ("val", "test")}
    feats = {s: pd.read_parquet(DATA_PROCESSED / f"{s}_features.parquet")
             for s in ("train", "val", "test")}
    alt = {s: pd.read_parquet(DATA_PROCESSED / f"{s}_features_alt.parquet")
           for s in ("train", "val", "test")}
    y_val, y_test = splits["val"]["label"].values, splits["test"]["label"].values

    with open(MODELS_DIR / "combined_training_summary.json") as f:
        best = json.load(f)["xgboost"]["best_params"]

    tfidf = joblib.load(MODELS_DIR / "tfidf_logreg.joblib")
    lex_val = tfidf.predict_proba(splits["val"]["prompt"])[:, 1]
    lex_test = tfidf.predict_proba(splits["test"]["prompt"])[:, 1]
    # Both fused configurations are needed: they differ only in the scorer,
    # but they fail on different prompts, so they combine differently too.
    gpt_val, gpt_test = fit_scores(feats["train"], feats["val"], feats["test"], best)
    qwn_val, qwn_test = fit_scores(alt["train"], alt["val"], alt["test"], best)

    tau = {"lex": select_threshold(y_val, lex_val),
           "gpt": select_threshold(y_val, gpt_val),
           "qwn": select_threshold(y_val, qwn_val)}
    flag = {"lex": lex_test >= tau["lex"],
            "gpt": gpt_test >= tau["gpt"],
            "qwn": qwn_test >= tau["qwn"]}

    rows = []

    def add(name, pred_test, note=""):
        r = {"combination": name} | report(y_test, pred_test) | {"note": note}
        rows.append(r)

    add("Word list alone", flag["lex"].astype(int))
    add("Fused detector alone (GPT-2)", flag["gpt"].astype(int))
    add("Fused detector alone (Qwen)", flag["qwn"].astype(int))

    # "Either flags" is exactly what a cascade produces, since a prompt
    # blocked at stage one never reaches stage two.
    add("Word list or GPT-2 flags", (flag["lex"] | flag["gpt"]).astype(int))
    add("Word list or Qwen flags", (flag["lex"] | flag["qwn"]).astype(int))
    add("Majority of all three",
        (flag["lex"].astype(int) + flag["gpt"] + flag["qwn"] >= 2).astype(int))
    add("Word list and GPT-2 both flag", (flag["lex"] & flag["gpt"]).astype(int))
    add("Word list and Qwen both flag", (flag["lex"] & flag["qwn"]).astype(int))

    # Score-level combinations, thresholded on validation as usual.
    for label, comb in [("Mean of scores", lambda a, b: (a + b) / 2),
                        ("Higher of the two scores", np.maximum),
                        ("Lower of the two scores", np.minimum)]:
        cv, ct = comb(lex_val, qwn_val), comb(lex_test, qwn_test)
        add(label, (ct >= select_threshold(y_val, cv)).astype(int))

    # Weighted mean, weight chosen on validation only.
    best_w, best_f1 = None, -1.0
    for w in np.arange(0, 1.01, 0.05):
        cv = w * lex_val + (1 - w) * qwn_val
        f1 = report(y_val, (cv >= select_threshold(y_val, cv)).astype(int))["F1"]
        if f1 > best_f1:
            best_f1, best_w = f1, w
    cv = best_w * lex_val + (1 - best_w) * qwn_val
    thr = select_threshold(y_val, cv)
    add(f"Weighted mean (w={best_w:.2f} lexical)",
        (best_w * lex_test + (1 - best_w) * qwn_test >= thr).astype(int),
        "weight and threshold from validation")

    tau_lex = tau["lex"]

    out = pd.DataFrame(rows)
    print(f"test set: 178 attacks, 1,462 legitimate\n")
    print(out.to_string(index=False))
    out.to_csv(TABLES_DIR / "combination_exploration.csv", index=False)

    # What a cascade saves: prompts the lexical stage already resolves never
    # need the 214 ms fused stage.
    escalated = int((lex_test < tau_lex).sum())
    print(f"\nCascade cost: {len(y_test) - escalated} of {len(y_test)} prompts "
          f"({(1 - escalated / len(y_test)) * 100:.0f}%) are decided by the "
          f"lexical stage alone.")
    print(f"written: {TABLES_DIR / 'combination_exploration.csv'}")


if __name__ == "__main__":
    main()
