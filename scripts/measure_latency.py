"""Per-prompt inference latency on CPU (thesis §5, deployment suitability).

Measures the wall-clock cost of the full detection path for a single prompt:
feature extraction (surface + perplexity + embedding) plus classifier
scoring, repeated over a sample of test prompts. Reports mean / median /
p95 in milliseconds, which substantiates the 'cheap enough to sit in front
of every request' claim (Research Question 3).
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from pidetect.config import (
    DATA_PROCESSED,
    LATENCY_SAMPLE_SIZE,
    MODELS_DIR,
    SEED,
    TABLES_DIR,
)
from pidetect.features import embeddings, perplexity, text_surface

N_SAMPLE = LATENCY_SAMPLE_SIZE


def main() -> None:
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    prompts = test["prompt"].sample(N_SAMPLE, random_state=SEED).tolist()

    scorer = perplexity.PerplexityScorer()
    encoder = embeddings.EmbeddingEncoder()
    model = joblib.load(MODELS_DIR / "combined_xgboost.joblib")

    # Warm-up (model loading, first-call graph building)
    for p in prompts[:5]:
        _score_one(p, scorer, encoder, model)

    # Per-stage breakdown: CPU deployability is a central claim, so the
    # thesis reports where each millisecond goes, not just the total.
    stages = {"surface": [], "perplexity": [], "embedding": [],
              "classify": [], "total": []}
    for p in prompts:
        t0 = time.perf_counter()
        ts = text_surface.extract_one(p)
        t1 = time.perf_counter()
        ppl = scorer.extract_one(p)
        t2 = time.perf_counter()
        emb = encoder.extract_batch([p])[0]
        t3 = time.perf_counter()
        x = np.concatenate([ts, ppl, emb]).reshape(1, -1)
        model.predict_proba(x)
        t4 = time.perf_counter()
        stages["surface"].append((t1 - t0) * 1000)
        stages["perplexity"].append((t2 - t1) * 1000)
        stages["embedding"].append((t3 - t2) * 1000)
        stages["classify"].append((t4 - t3) * 1000)
        stages["total"].append((t4 - t0) * 1000)

    stats = {"n": len(stages["total"])}
    for name, times in stages.items():
        stats[name] = {
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "latency.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


def _score_one(prompt, scorer, encoder, model) -> float:
    ts = text_surface.extract_one(prompt)
    ppl = scorer.extract_one(prompt)
    emb = encoder.extract_batch([prompt])[0]
    x = np.concatenate([ts, ppl, emb]).reshape(1, -1)
    return float(model.predict_proba(x)[0, 1])


if __name__ == "__main__":
    main()
