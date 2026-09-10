"""Feature-importance evidence for thesis §5.3 and §5.7.

Produces three tables, all computed on the TRAINING split only:

  1. token_discrimination.csv - how often override tokens appear in each
     class. Shows directly that the attack signal is lexical.
  2. tfidf_top_features.csv   - the terms the winning baseline actually
     relies on, with their learned weights.
  3. xgb_group_importance.csv - how the fused model distributes its
     attention across the three feature blocks. The proposal committed to
     reporting a feature-importance ranking; this is it.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from pidetect.config import DATA_PROCESSED, MODELS_DIR, TABLES_DIR

TABLES_DIR.mkdir(parents=True, exist_ok=True)
train = pd.read_parquet(DATA_PROCESSED / "train.parquet")
mal = train[train.label == 1]["prompt"].str.lower()
ben = train[train.label == 0]["prompt"].str.lower()
print(f"training split: {len(mal)} attacks, {len(ben)} benign\n")

# --- 1. token discrimination ------------------------------------------------
TOKENS = ["ignore", "previous", "instructions", "rules", "forget", "pretend",
          "bypass", "disregard", "override", "you are now", "act as",
          "account", "how do i", "explain", "please"]
rows = []
for t in TOKENS:
    a, b = mal.str.contains(t, regex=False).mean(), ben.str.contains(t, regex=False).mean()
    rows.append({"token": t, "pct_attacks": round(a * 100, 1),
                 "pct_benign": round(b * 100, 1),
                 "ratio": round(a / b, 1) if b > 0 else float("inf")})
tok = pd.DataFrame(rows).sort_values("ratio", ascending=False)
tok.to_csv(TABLES_DIR / "token_discrimination.csv", index=False)
print("TOKEN DISCRIMINATION (training split)")
print(tok.to_string(index=False))

# --- 2. what the lexical baseline learned ----------------------------------
pipe = joblib.load(MODELS_DIR / "tfidf_logreg.joblib")
vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
names, coefs = np.array(vec.get_feature_names_out()), clf.coef_[0]
top = np.argsort(coefs)[-20:][::-1]
tf = pd.DataFrame({"term": names[top], "weight": coefs[top].round(2)})
tf.to_csv(TABLES_DIR / "tfidf_top_features.csv", index=False)
print("\n\nTOP TERMS PUSHING TOWARDS 'ATTACK' (lexical baseline)")
print(tf.to_string(index=False))

# --- 3. where the fused model puts its attention ---------------------------
model = joblib.load(MODELS_DIR / "combined_xgboost.joblib")
feats = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
cols = [c for c in feats.columns if c not in ("label", "source_file")]
imp = model.feature_importances_

groups = {"text-surface (12)": [i for i, c in enumerate(cols) if c.startswith("ts_")],
          "perplexity (5)": [i for i, c in enumerate(cols) if c.startswith("ppl_")],
          "embedding (384)": [i for i, c in enumerate(cols) if c.startswith("emb_")]}
rows = []
for name, idx in groups.items():
    share = float(imp[idx].sum())
    rows.append({"feature_group": name, "n_features": len(idx),
                 "total_importance": round(share, 4),
                 "importance_per_feature": round(share / len(idx), 6)})
grp = pd.DataFrame(rows)
grp.to_csv(TABLES_DIR / "xgb_group_importance.csv", index=False)
print("\n\nXGBOOST IMPORTANCE BY FEATURE GROUP (fused model)")
print(grp.to_string(index=False))

order = np.argsort(imp)[-10:][::-1]
print("\nTop 10 individual features:")
for i in order:
    print(f"   {cols[i]:<12} {imp[i]:.4f}")
