"""Validity diagnostic: is the corpus separable by template prefix alone?

The length-only diagnostic asked whether prompt length could explain the
results. This asks a sharper question, prompted by an audit finding: several
benign sources are rigidly templated, with every row opening identically
("From the description of a rule: identify the...", 56% of benign training
prompts). If a classifier can separate the classes from the first few
characters alone, it is recognising *which dataset a prompt came from*
rather than whether it is an attack — and every headline number would be
measuring corpus assembly rather than detection.

Cross-validated on the training split only. No test-set evaluation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from pidetect.config import CV_FOLDS, DATA_PROCESSED, LOGREG_MAX_ITER, SEED, TABLES_DIR

train = pd.read_parquet(DATA_PROCESSED / "train.parquet")
y = train["label"].values
text = train["prompt"].astype(str)
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def score(x, label):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20_000,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=LOGREG_MAX_ITER,
                                   class_weight="balanced", random_state=SEED)),
    ])
    s = cross_val_score(pipe, x, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"  {label:<34} CV F1 = {s.mean():.4f}  (sd {s.std():.4f})")
    return {"input": label, "cv_f1": round(s.mean(), 4), "cv_sd": round(s.std(), 4)}


print(f"training split: {len(y):,} prompts, {int(y.sum())} malicious "
      f"({y.mean()*100:.1f}%)\n")
print("How much of the signal is in the opening characters alone?")
rows = [score(text.str.slice(0, n), f"first {n} characters only") for n in (20, 45, 100)]
rows.append(score(text.str.slice(45), "everything EXCEPT first 45 chars"))
rows.append(score(text, "full prompt (reference)"))

pd.DataFrame(rows).to_csv(TABLES_DIR / "prefix_diagnostic.csv", index=False)
print(f"\nWritten to {TABLES_DIR / 'prefix_diagnostic.csv'}")
print("A high score from the opening alone means the classifier can identify")
print("the SOURCE DATASET rather than detect an attack.")
