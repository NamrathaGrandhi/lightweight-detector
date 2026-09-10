"""Attack-family embedding analysis (proposal §7.10, thesis §3.8/§5.6).

Completes the second half of the embedding-space analysis the proposal
commits to: the same cached test-split embeddings, but viewed by attack
source family rather than by binary label.

1. Silhouette coefficient among the malicious prompts only, with source
   family as the cluster label (cosine distance) — measures whether the
   attack families occupy distinct regions of the MiniLM space.
2. UMAP projection (identical settings to embedding_space.py) with benign
   prompts in grey and each attack family in its own colour/marker.

Artefacts:
    results/figures/umap_families.png
    results/tables/silhouette_families.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pidetect.config import DATA_PROCESSED, FIGURES_DIR, SEED, TABLES_DIR

FAMILY_LABELS = {
    "jailbreak_prompts.csv": "jailbreak prompts",
    "forbidden_question_set_with_prompts.csv": "forbidden questions",
    "malicous_deepset.csv": "deepset injections",
    "predictionguard_df.csv": "generated injections",
}

FAMILY_STYLE = {
    "jailbreak prompts": ("tab:red", "^"),
    "forbidden questions": ("tab:purple", "s"),
    "deepset injections": ("tab:orange", "D"),
    "generated injections": ("tab:green", "v"),
}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    emb = df[emb_cols].values.astype(np.float32)
    labels = df["label"].values
    families = np.array([
        FAMILY_LABELS.get(s, s) if y == 1 else "benign"
        for s, y in zip(df["source_file"], labels)
    ])

    mal = labels == 1
    fam_counts = pd.Series(families[mal]).value_counts().to_dict()
    print(f"{mal.sum()} malicious test prompts across {len(fam_counts)} families: {fam_counts}")

    # Silhouette among malicious prompts only, families as clusters
    sil_fam = float(silhouette_score(emb[mal], families[mal], metric="cosine"))
    with open(TABLES_DIR / "silhouette_families.json", "w") as f:
        json.dump({"family_split_malicious_only": sil_fam,
                   "n_malicious": int(mal.sum()),
                   "families": fam_counts}, f, indent=2)
    print(f"Silhouette (attack-family split, malicious only, cosine): {sil_fam:.4f}")

    # UMAP with the identical settings used for umap_labels.png
    import umap

    reducer = umap.UMAP(n_components=2, random_state=SEED, metric="cosine")
    xy = reducer.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    benign_mask = ~mal
    ax.scatter(xy[benign_mask, 0], xy[benign_mask, 1],
               s=6, alpha=0.25, marker="o", color="0.7", label="benign")
    for fam, (color, marker) in FAMILY_STYLE.items():
        m = families == fam
        ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.85,
                   marker=marker, color=color, label=fam)
    ax.set_title("UMAP projection of MiniLM test embeddings, by attack family")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=1.8, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "umap_families.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / 'umap_families.png'}")


if __name__ == "__main__":
    main()
