"""Stage 3 — sentence embeddings from all-MiniLM-L6-v2.

Each prompt is mapped to a 384-dimensional vector by a distilled
Sentence-BERT encoder and L2-normalised before classification
(thesis §3.5.3). The same cached embeddings are reused for near-duplicate
removal during dataset preparation and for the embedding-space analysis in
Chapter 5, so this module is the single place they are computed.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from pidetect.config import EMBEDDING_BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL


class EmbeddingEncoder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)

    def extract_batch(self, texts: list[str],
                      batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalisation
            show_progress_bar=True,
        )
        return emb.astype(np.float32)
