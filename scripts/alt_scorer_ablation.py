"""Modern-scorer robustness check (proposal §7.4, thesis §5.6).

The proposal commits to re-running the perplexity scorer with a modern
small language model (Qwen2.5-0.5B) so the effect of scorer choice can be
measured directly. GPT-2 remains the primary scorer for all main
experiments; this script produces the single comparison table.

Procedure (deliberately minimal — only the scorer changes):
    1. Recompute the 5 perplexity features for every split with the
       alternative scorer, caching to <split>_features_alt.parquet
       (only the ppl_* columns differ from the main cache).
    2. Retrain XGBoost with the best hyperparameters found in the main run
       (no new grid search: the ablation isolates the scorer, not tuning).
    3. Re-select the threshold on validation (recall >= 0.95) and evaluate
       once on the test set.
    4. Write results/tables/alt_scorer_metrics.csv with GPT-2 vs Qwen rows.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from pidetect.config import (
    ALT_PERPLEXITY_MODEL,
    DATA_PROCESSED,
    MODELS_DIR,
    SEED,
    TABLES_DIR,
)
from pidetect.eval.evaluate import METRICS
from pidetect.features.perplexity import FEATURE_NAMES, PerplexityScorer
from pidetect.models.train import select_threshold


def build_alt_features(split: str, scorer: PerplexityScorer) -> pd.DataFrame:
    out_path = DATA_PROCESSED / f"{split}_features_alt.parquet"
    base = pd.read_parquet(DATA_PROCESSED / f"{split}_features.parquet")

    if out_path.exists():
        cached = pd.read_parquet(out_path)
        # A cache built for a different corpus must never be reused: the
        # scoring is expensive, so the temptation to keep it is exactly why
        # this check exists.
        if len(cached) == len(base):
            print(f"[{split}] alt feature cache matches corpus, reusing")
            return cached
        print(f"[{split}] alt cache is stale ({len(cached):,} rows vs "
              f"{len(base):,} in the current corpus) - rebuilding")
        out_path.unlink()

    prompts = pd.read_parquet(DATA_PROCESSED / f"{split}.parquet")["prompt"].tolist()
    ppl = scorer.extract_batch(prompts)
    alt = base.copy()
    for i in range(len(FEATURE_NAMES)):
        alt[f"ppl_{i}"] = ppl[:, i]
    alt.to_parquet(out_path)
    print(f"[{split}] alt perplexity features cached")
    return alt


def xy(df: pd.DataFrame):
    """Feature matrix and labels. `source_file` is error-analysis metadata
    carried through the cache, never a model input, so it is dropped here."""
    y = df["label"].values
    X = df.drop(columns=["label", "source_file"], errors="ignore")
    return X.values.astype(np.float32), y


def train_eval(train_df, val_df, test_df, params: dict, label: str) -> dict:
    X_tr, y_tr = xy(train_df)
    X_val, y_val = xy(val_df)
    X_te, y_te = xy(test_df)
    neg, pos = np.bincount(y_tr)
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=neg / pos, tree_method="hist",
        random_state=SEED, n_jobs=-1, **params,
    )
    model.fit(X_tr, y_tr)
    tau = select_threshold(y_val, model.predict_proba(X_val)[:, 1])
    scores = model.predict_proba(X_te)[:, 1]
    preds = (scores >= tau).astype(int)
    row = {"scorer": label, "threshold": tau}
    row |= {m: fn(y_te, preds, scores) for m, fn in METRICS.items()}
    return row


def main() -> None:
    with open(MODELS_DIR / "combined_training_summary.json") as f:
        best = json.load(f)["xgboost"]["best_params"]
    print(f"Reusing main-run XGBoost params: {best}")

    main_feats = {s: pd.read_parquet(DATA_PROCESSED / f"{s}_features.parquet")
                  for s in ["train", "val", "test"]}

    print(f"Loading alternative scorer: {ALT_PERPLEXITY_MODEL}")
    scorer = PerplexityScorer(model_name=ALT_PERPLEXITY_MODEL)
    alt_feats = {s: build_alt_features(s, scorer) for s in ["train", "val", "test"]}

    rows = [
        train_eval(*(main_feats[s] for s in ["train", "val", "test"]), best, "gpt2"),
        train_eval(*(alt_feats[s] for s in ["train", "val", "test"]), best,
                   ALT_PERPLEXITY_MODEL),
    ]
    out = pd.DataFrame(rows).round(4)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES_DIR / "alt_scorer_metrics.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
