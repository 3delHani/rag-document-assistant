from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app import config


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return L2-normalised embeddings, so dot product == cosine similarity."""
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def warm_up() -> None:
    _model()
