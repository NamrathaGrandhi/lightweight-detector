"""Baseline detectors (thesis §3.6 steps 6-7, research question 2).

Three progressively simpler alternatives to the combined-feature model,
evaluated under the identical protocol (same splits, same threshold rule,
same metrics) so any improvement is attributable to feature fusion alone:

* perplexity-only  — logistic regression on the 5 perplexity features;
                     stands in for the perplexity-filter line of work.
* surface-only     — logistic regression on the 12 text-surface features;
                     stands in for rule/keyword-style filtering.
* tfidf            — TF-IDF (word 1-2 grams, 20k vocab) + logistic
                     regression; the classic sparse lexical baseline.

A fourth entry, 'length_only', is a VALIDITY DIAGNOSTIC rather than a
competing detector. The malicious prompts in this corpus are far longer
than typical benign prompts (see docs/dataset_validation.md), so a model
could score highly by learning "long text = attack" and nothing else.
Fitting logistic regression on the single char_len feature measures exactly
how much of the headline performance that confound explains: if the
length-only score approaches the fused model's, the result is an artefact
and must be reported as such.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from pidetect.config import (
    CV_FOLDS,
    DATA_PROCESSED,
    LOGREG_GRID,
    LOGREG_MAX_ITER,
    MODELS_DIR,
    SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)
from pidetect.models.train import load_features, select_threshold, train_logreg


def run_feature_baseline(groups: list[str], tag: str,
                         columns: list[str] | None = None) -> None:
    X_train, y_train, names = load_features("train", groups)
    X_val, y_val, _ = load_features("val", groups)
    if columns:  # restrict to named columns (used by the length-only diagnostic)
        idx = [names.index(c) for c in columns]
        X_train, X_val = X_train[:, idx], X_val[:, idx]
    print(f"[{tag}] {X_train.shape[1]} features")
    search = train_logreg(X_train, y_train)
    best = search.best_estimator_
    tau = select_threshold(y_val, best.predict_proba(X_val)[:, 1])
    joblib.dump(best, MODELS_DIR / f"{tag}_logreg.joblib")
    _save_summary(tag, search, tau)


def run_tfidf_baseline(tag: str = "tfidf") -> None:
    train = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    val = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE,
                                  max_features=TFIDF_MAX_FEATURES,
                                  sublinear_tf=TFIDF_SUBLINEAR_TF)),
        ("clf", LogisticRegression(max_iter=LOGREG_MAX_ITER,
                                   class_weight="balanced", random_state=SEED)),
    ])
    grid = {f"clf__{k}": v for k, v in LOGREG_GRID.items()}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
    search.fit(train["prompt"], train["label"])
    best = search.best_estimator_
    tau = select_threshold(
        val["label"].values, best.predict_proba(val["prompt"])[:, 1]
    )
    joblib.dump(best, MODELS_DIR / f"{tag}_logreg.joblib")
    _save_summary(tag, search, tau)


def _save_summary(tag: str, search: GridSearchCV, tau: float) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_params": search.best_params_,
        "cv_f1": float(search.best_score_),
        "threshold": tau,
    }
    with open(MODELS_DIR / f"{tag}_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{tag}] CV F1={search.best_score_:.4f}, tau={tau:.4f}")


def main() -> None:
    run_feature_baseline(["perplexity"], "ppl_only")
    run_feature_baseline(["surface"], "surface_only")
    # ts_0 is char_len (see features/text_surface.FEATURE_NAMES).
    run_feature_baseline(["surface"], "length_only", columns=["ts_0"])
    run_tfidf_baseline()


if __name__ == "__main__":
    main()
