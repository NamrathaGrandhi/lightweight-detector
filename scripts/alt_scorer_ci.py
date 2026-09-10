"""Bootstrap confidence intervals for the modern-scorer variant.

The scorer ablation reported point estimates only. With the Qwen variant
landing inside the lexical baseline's interval, point estimates are not
enough to say whether the two differ, so this recomputes the comparison with
the same 2,000-resample percentile bootstrap used everywhere else, and adds
a paired test on the difference.

The paired form matters: both models score the SAME test prompts, so
resampling prompts jointly and taking the per-resample difference removes
the variance the two models share and answers the question actually being
asked - is one better than the other on this data?
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from pidetect.config import (
    BOOTSTRAP_ITERATIONS,
    DATA_PROCESSED,
    MODELS_DIR,
    SEED,
    TABLES_DIR,
)
from pidetect.models.train import select_threshold


def xy(df):
    y = df["label"].values
    X = df.drop(columns=["label", "source_file"], errors="ignore")
    return X.values.astype(np.float32), y


with open(MODELS_DIR / "combined_training_summary.json") as f:
    params = json.load(f)["xgboost"]["best_params"]

scores = {}
for tag, suffix in [("gpt2", ""), ("qwen", "_alt")]:
    tr = pd.read_parquet(DATA_PROCESSED / f"train_features{suffix}.parquet")
    va = pd.read_parquet(DATA_PROCESSED / f"val_features{suffix}.parquet")
    te = pd.read_parquet(DATA_PROCESSED / f"test_features{suffix}.parquet")
    X_tr, y_tr = xy(tr); X_va, y_va = xy(va); X_te, y_te = xy(te)
    neg, pos = np.bincount(y_tr)
    m = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                      scale_pos_weight=neg / pos, tree_method="hist",
                      random_state=SEED, n_jobs=6, **params)
    m.fit(X_tr, y_tr)
    tau = select_threshold(y_va, m.predict_proba(X_va)[:, 1])
    scores[tag] = (m.predict_proba(X_te)[:, 1], tau)
    print(f"{tag}: tau={tau:.4f}")

# The lexical baseline, scored on the same prompts.
test_txt = pd.read_parquet(DATA_PROCESSED / "test.parquet")
y = test_txt["label"].values
tfidf = joblib.load(MODELS_DIR / "tfidf_logreg.joblib")
with open(MODELS_DIR / "tfidf_training_summary.json") as f:
    tau_tf = json.load(f)["threshold"]
scores["tfidf"] = (tfidf.predict_proba(test_txt["prompt"])[:, 1], tau_tf)

rng = np.random.default_rng(SEED)
n = len(y)
idx = np.array([rng.integers(0, n, n) for _ in range(BOOTSTRAP_ITERATIONS)])

boot = {}
print(f"\n{'model':<10} {'F1':>8}   95% CI")
for tag, (s, tau) in scores.items():
    pred = (s >= tau).astype(int)
    vals = np.array([f1_score(y[i], pred[i], zero_division=0) for i in idx])
    boot[tag] = vals
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"{tag:<10} {f1_score(y, pred, zero_division=0):8.4f}   [{lo:.4f}, {hi:.4f}]")

print("\nPaired differences (same resampled prompts for both models):")
rows = []
for a, b in [("tfidf", "qwen"), ("tfidf", "gpt2"), ("qwen", "gpt2")]:
    d = boot[a] - boot[b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    sig = "DIFFERENT" if lo > 0 or hi < 0 else "not distinguishable"
    rows.append({"comparison": f"{a} - {b}", "mean_diff": round(d.mean(), 4),
                 "ci_low": round(lo, 4), "ci_high": round(hi, 4), "verdict": sig})
    print(f"  {a:>6} - {b:<6} {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}")

pd.DataFrame(rows).to_csv(TABLES_DIR / "alt_scorer_ci.csv", index=False)
print(f"\nWritten to {TABLES_DIR / 'alt_scorer_ci.csv'}")
