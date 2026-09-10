"""EXPLORATORY: would adding lexical features to the fusion have helped?

This is NOT part of the answer to any research question, and it deliberately
never touches the test set.

The main study found a TF-IDF baseline outperforming the 401-dimensional
fusion. The obvious follow-up is whether fusing all four signals — surface,
perplexity, embedding AND lexical — beats either. Answering that after the
test set has been opened would be methodologically improper: an architecture
chosen because it looks good on test data is no longer being measured on
held-out data.

So this script answers the question using cross-validation on the TRAINING
split only, exactly as every model in the main study was tuned. The figures
it produces are comparable with the cross-validated column of thesis Table
4.3 (combined 0.8533, TF-IDF 0.9058) and with nothing else.

Hyperparameters are reused from the main run rather than searched again, so
the comparison isolates the feature set.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pidetect.config import (
    CV_FOLDS,
    DATA_PROCESSED,
    LOGREG_MAX_ITER,
    MODELS_DIR,
    SEED,
    TABLES_DIR,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

# ---- training split only ---------------------------------------------------
train = pd.read_parquet(DATA_PROCESSED / "train.parquet")
feats = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
y = feats.pop("label").values
feats = feats.drop(columns=["source_file"], errors="ignore")
X_dense = feats.values.astype(np.float32)
texts = train["prompt"].tolist()
print(f"Training split: {len(y):,} prompts, {int(y.sum())} malicious")

vec = TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE,
                      max_features=TFIDF_MAX_FEATURES,
                      sublinear_tf=TFIDF_SUBLINEAR_TF)
X_lex = vec.fit_transform(texts)
print(f"Dense fusion features: {X_dense.shape[1]}   lexical features: {X_lex.shape[1]}")

# Dense features are standardised before being combined with the sparse
# lexical block, so that the 401 dense columns are not swamped by scale.
X_dense_s = StandardScaler().fit_transform(X_dense)
X_all = hstack([csr_matrix(X_dense_s), X_lex]).tocsr()
print(f"Combined: {X_all.shape[1]:,} features\n")

with open(MODELS_DIR / "combined_training_summary.json") as f:
    xgb_params = json.load(f)["xgboost"]["best_params"]
neg, pos = np.bincount(y)


def xgb():
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                         scale_pos_weight=neg / pos, tree_method="hist",
                         random_state=SEED, n_jobs=6, **xgb_params)


def logreg():
    return LogisticRegression(max_iter=LOGREG_MAX_ITER, C=1,
                              class_weight="balanced", random_state=SEED)


experiments = [
    ("fusion only (401)", X_dense, Pipeline([("s", StandardScaler()), ("c", logreg())]), "logreg"),
    ("lexical only (20k)", X_lex, logreg(), "logreg"),
    ("fusion + lexical", X_all, logreg(), "logreg"),
    ("fusion only (401)", X_dense, xgb(), "xgboost"),
    ("lexical only (20k)", X_lex, xgb(), "xgboost"),
    ("fusion + lexical", X_all, xgb(), "xgboost"),
]

rows = []
for name, X, model, kind in experiments:
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=1)
    rows.append({"features": name, "model": kind,
                 "cv_f1": scores.mean(), "cv_sd": scores.std()})
    print(f"  {name:22s} {kind:8s} CV F1 = {scores.mean():.4f} (sd {scores.std():.4f})")

out = pd.DataFrame(rows).round(4)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(TABLES_DIR / "exploratory_lexical_fusion.csv", index=False)
print(f"\nWritten to {TABLES_DIR / 'exploratory_lexical_fusion.csv'}")
print("Cross-validated on training data only. No test-set evaluation performed.")
