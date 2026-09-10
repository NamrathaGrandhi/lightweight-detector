"""Feature extraction runner (thesis §3.5, protocol step 2).

Runs all three feature extractors over every prompt of every split and
caches the resulting 401-dimensional vectors to parquet, so downstream
experiments never recompute features. Column layout:

    ts_0..ts_11    12 text-surface features
    ppl_0..ppl_4    5 perplexity features
    emb_0..emb_383 384 MiniLM embedding dimensions
    label           0 benign / 1 malicious
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from pidetect.config import DATA_PROCESSED
from pidetect.features import embeddings, perplexity, text_surface


def featurize_split(name: str, in_dir: Path = DATA_PROCESSED,
                    scorer: perplexity.PerplexityScorer | None = None,
                    encoder: embeddings.EmbeddingEncoder | None = None) -> None:
    df = pd.read_parquet(in_dir / f"{name}.parquet")
    texts = df["prompt"].tolist()
    print(f"[{name}] extracting features for {len(texts):,} prompts")

    t0 = time.time()
    ts = text_surface.extract_batch(texts)
    print(f"[{name}] text-surface done in {time.time() - t0:.1f}s")

    t0 = time.time()
    scorer = scorer or perplexity.PerplexityScorer()
    ppl = scorer.extract_batch(texts)
    print(f"[{name}] perplexity done in {time.time() - t0:.1f}s")

    t0 = time.time()
    encoder = encoder or embeddings.EmbeddingEncoder()
    emb = encoder.extract_batch(texts)
    print(f"[{name}] embeddings done in {time.time() - t0:.1f}s")

    out = pd.DataFrame(
        {f"ts_{i}": ts[:, i] for i in range(ts.shape[1])}
        | {f"ppl_{i}": ppl[:, i] for i in range(ppl.shape[1])}
        | {f"emb_{i}": emb[:, i] for i in range(emb.shape[1])}
    )
    out["label"] = df["label"].values
    # Carried through (never used as a model input) so Chapter 5 can group
    # misclassifications by attack family / benign source.
    out["source_file"] = df["source_file"].values
    out.to_parquet(in_dir / f"{name}_features.parquet")
    print(f"[{name}] cached {out.shape[1] - 2} features -> {name}_features.parquet")


def main() -> None:
    scorer = perplexity.PerplexityScorer()
    encoder = embeddings.EmbeddingEncoder()
    for split in ["train", "val", "test"]:
        featurize_split(split, scorer=scorer, encoder=encoder)


if __name__ == "__main__":
    main()
