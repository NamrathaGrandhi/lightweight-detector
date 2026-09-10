"""Compare the models at matched catch rates, with uncertainty.

The headline comparison fixes recall at 95%. That is one point on a curve,
and it is the point where the fused detector's threshold collapses. This
sweeps the catch rate and asks, at each level, how many legitimate prompts
each model wrongly blocks - and whether the differences are larger than the
noise.

Bootstrap is paired: the same resampled prompts score every model, so the
comparison isolates the models rather than the sample.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from pidetect.config import BOOTSTRAP_ITERATIONS, DATA_PROCESSED, MODELS_DIR, SEED, TABLES_DIR

RECALLS = [0.80, 0.85, 0.90, 0.95, 0.98]


def xy(df):
    y = df["label"].values
    return df.drop(columns=["label", "source_file"], errors="ignore").values.astype(np.float32), y


def fit_variant(suffix):
    params = json.load(open(MODELS_DIR / "combined_training_summary.json"))["xgboost"]["best_params"]
    tr = pd.read_parquet(DATA_PROCESSED / f"train_features{suffix}.parquet")
    te = pd.read_parquet(DATA_PROCESSED / f"test_features{suffix}.parquet")
    X_tr, y_tr = xy(tr)
    X_te, _ = xy(te)
    neg, pos = np.bincount(y_tr)
    m = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                      scale_pos_weight=neg / pos, tree_method="hist",
                      random_state=SEED, n_jobs=6, **params)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
y = test["label"].values
models = {
    "Fusion+GPT-2": fit_variant(""),
    "Fusion+Qwen": fit_variant("_alt"),
    "Word list": joblib.load(MODELS_DIR / "tfidf_logreg.joblib").predict_proba(test["prompt"])[:, 1],
}


def fp_at_recall(scores, yy, target):
    """False positives when the threshold is set to catch `target` of attacks."""
    pos = np.sort(scores[yy == 1])[::-1]
    k = int(np.ceil(target * len(pos)))
    tau = pos[min(k, len(pos)) - 1]
    return int(((scores >= tau) & (yy == 0)).sum())


n_ben = int((y == 0).sum())
print(f"{int((y==1).sum())} attacks, {n_ben:,} legitimate prompts\n")
print("Legitimate prompts wrongly blocked, at matched catch rate")
print(f"{'catch rate':<12}" + "".join(f"{m:>20}" for m in models))
print("-" * 74)
rows = []
for r in RECALLS:
    line = f"{int(r*100)}%".ljust(12)
    for name, s in models.items():
        line += f"{fp_at_recall(s, y, r):>20,}"
        rows.append({"recall": r, "model": name, "false_positives": fp_at_recall(s, y, r)})
    print(line)

# Is the difference between the best fused variant and the word list real?
rng = np.random.default_rng(SEED)
nn = len(y)
print("\n\nWord list minus Fusion+Qwen (negative = fusion blocks fewer)")
print(f"{'catch rate':<12}{'mean diff':>12}{'95% range':>22}   verdict")
print("-" * 70)
out = []
for r in RECALLS:
    diffs = []
    for _ in range(500):
        i = rng.integers(0, nn, nn)
        if y[i].sum() < 10:
            continue
        a = fp_at_recall(models["Word list"][i], y[i], r)
        b = fp_at_recall(models["Fusion+Qwen"][i], y[i], r)
        diffs.append(a - b)
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    verdict = ("word list blocks fewer" if hi < 0 else
               "fusion blocks fewer" if lo > 0 else "not distinguishable")
    print(f"{int(r*100)}%".ljust(12) + f"{d.mean():>+12.1f}" +
          f"[{lo:+.1f}, {hi:+.1f}]".rjust(22) + f"   {verdict}")
    out.append({"recall": r, "mean_diff": round(float(d.mean()), 2),
                "ci_low": round(float(lo), 2), "ci_high": round(float(hi), 2),
                "verdict": verdict})

pd.DataFrame(rows).to_csv(TABLES_DIR / "operating_points.csv", index=False)
pd.DataFrame(out).to_csv(TABLES_DIR / "operating_points_ci.csv", index=False)
print(f"\nWritten to {TABLES_DIR}")
