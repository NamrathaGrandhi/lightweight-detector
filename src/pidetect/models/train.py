"""Model training (thesis §3.6, protocol steps 3-4).

Two classifiers are trained on the fused 401-dimensional representation:

* Logistic Regression (baseline) — transparent linear model,
  sigma(w^T x + b), L2 regularisation with C tuned over LOGREG_GRID.
* XGBoost (main model) — gradient-boosted trees, hyperparameters tuned
  over XGB_GRID.

Both are tuned by stratified 5-fold cross-validated grid search on the
TRAINING set only, optimising F1. Class imbalance is handled by
inverse-frequency class weighting (scale_pos_weight for XGBoost).
The decision threshold tau is then chosen on the VALIDATION set as the
smallest threshold whose recall is still >= TARGET_RECALL (0.95).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pidetect.config import (
    CV_FOLDS,
    DATA_PROCESSED,
    GRID_N_JOBS,
    LOGREG_GRID,
    LOGREG_MAX_ITER,
    MODEL_N_THREADS,
    MODELS_DIR,
    SEED,
    TARGET_RECALL,
    XGB_GRID,
)

FEATURE_GROUPS = {
    "surface": lambda c: c.startswith("ts_"),
    "perplexity": lambda c: c.startswith("ppl_"),
    "embedding": lambda c: c.startswith("emb_"),
}


def load_features(split: str, groups: list[str] | None = None):
    """Load a cached feature split, optionally restricted to feature groups.

    `source_file` is metadata for the error analysis, never a model input,
    so it is dropped here rather than at the call sites.
    """
    df = pd.read_parquet(DATA_PROCESSED / f"{split}_features.parquet")
    y = df.pop("label").values
    df = df.drop(columns=["source_file"], errors="ignore")
    if groups:
        cols = [c for c in df.columns
                if any(FEATURE_GROUPS[g](c) for g in groups)]
        df = df[cols]
    return df.values.astype(np.float32), y, list(df.columns)


def select_threshold(y_true: np.ndarray, scores: np.ndarray,
                     target_recall: float = TARGET_RECALL) -> float:
    """Smallest tau such that recall(scores >= tau) >= target_recall.

    Scanning thresholds from high to low, recall is monotonically
    non-decreasing, so the first (highest) tau that satisfies the target
    is the one that also minimises false positives at that recall.
    """
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted)
    recall = tp_cum / y_true.sum()
    idx = np.argmax(recall >= target_recall)
    return float(scores[order][idx])


def train_logreg(X_train, y_train) -> GridSearchCV:
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=LOGREG_MAX_ITER, class_weight="balanced", random_state=SEED)),
    ])
    grid = {f"clf__{k}": v for k, v in LOGREG_GRID.items()}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    return search


def train_xgboost(X_train, y_train) -> GridSearchCV:
    neg, pos = np.bincount(y_train)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=neg / pos,   # inverse-frequency class weighting
        tree_method="hist",
        random_state=SEED,
        n_jobs=MODEL_N_THREADS,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    # verbose=2 reports each fit as it completes: an 81-configuration grid is
    # slow, and silence is indistinguishable from a hang.
    search = GridSearchCV(model, XGB_GRID, scoring="f1", cv=cv,
                          n_jobs=GRID_N_JOBS, verbose=2)
    search.fit(X_train, y_train)
    return search


def main(groups: list[str] | None = None, tag: str = "combined") -> None:
    X_train, y_train, cols = load_features("train", groups)
    X_val, y_val, _ = load_features("val", groups)
    print(f"[{tag}] training on {X_train.shape[0]:,} x {X_train.shape[1]} features")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, trainer in [("logreg", train_logreg), ("xgboost", train_xgboost)]:
        print(f"[{tag}] tuning {name} ({CV_FOLDS}-fold CV grid search) ...")
        search = trainer(X_train, y_train)
        best = search.best_estimator_
        val_scores = best.predict_proba(X_val)[:, 1]
        tau = select_threshold(y_val, val_scores)
        joblib.dump(best, MODELS_DIR / f"{tag}_{name}.joblib")
        summary[name] = {
            "best_params": search.best_params_,
            "cv_f1": float(search.best_score_),
            "threshold": tau,
            "feature_columns": cols,
        }
        print(f"[{tag}] {name}: CV F1={search.best_score_:.4f}, tau={tau:.4f}, "
              f"params={search.best_params_}")

    with open(MODELS_DIR / f"{tag}_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{tag}] saved models + summary to {MODELS_DIR}")


if __name__ == "__main__":
    main()
