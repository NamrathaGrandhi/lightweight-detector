"""Embedding-space analysis (thesis §3.8, Chapter 5).

Tests the assumption that malicious prompts form recognisable structure in
the 384-dimensional MiniLM space, using the cached test-split embeddings
(zero additional encoding cost):

1. Dimensionality-reduction visualisation — UMAP projection to 2-D
   (t-SNE as a cross-check), scatter coloured by ground-truth label.
2. Cluster quality — silhouette coefficient for the benign-vs-malicious
   split, giving a single quantitative measure of how much of the
   classifier's job the embedding geometry already does.

Artefacts:
    results/figures/umap_labels.png
    results/figures/tsne_labels.png
    results/tables/silhouette.json
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from pidetect.config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    SEED,
    TABLES_DIR,
    TSNE_PERPLEXITY,
)


def load_test_embeddings():
    df = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    return df[emb_cols].values.astype(np.float32), df["label"].values


def scatter(coords: np.ndarray, labels: np.ndarray, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for value, name, marker in [(0, "benign", "o"), (1, "malicious", "^")]:
        mask = labels == value
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=6, alpha=0.5, marker=marker, label=name)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    emb, labels = load_test_embeddings()
    print(f"Analysing {len(emb):,} test embeddings ({emb.shape[1]}-d)")

    # UMAP (imported lazily: umap-learn pulls in numba)
    import umap

    reducer = umap.UMAP(n_components=2, random_state=SEED, metric="cosine")
    umap_xy = reducer.fit_transform(emb)
    scatter(umap_xy, labels, "UMAP projection of MiniLM test embeddings",
            FIGURES_DIR / "umap_labels.png")

    # t-SNE cross-check
    tsne_xy = TSNE(n_components=2, random_state=SEED, metric="cosine",
                   init="pca", perplexity=TSNE_PERPLEXITY).fit_transform(emb)
    scatter(tsne_xy, labels, "t-SNE projection of MiniLM test embeddings",
            FIGURES_DIR / "tsne_labels.png")

    # Silhouette on the full-dimensional embeddings (cosine distance)
    sil = float(silhouette_score(emb, labels, metric="cosine"))
    with open(TABLES_DIR / "silhouette.json", "w") as f:
        json.dump({"benign_vs_malicious": sil, "n": int(len(emb))}, f, indent=2)
    print(f"Silhouette (benign vs malicious, cosine): {sil:.4f}")


if __name__ == "__main__":
    main()
