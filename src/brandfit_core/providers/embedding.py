from __future__ import annotations

import hashlib
import math
from typing import Any

from brandfit_core.config import Settings, get_settings

EMBEDDING_DIMENSIONS = 1024


class DeterministicEmbedder:
    """Offline fixture embedder with stable vectors; never use it for production discovery."""

    model_version = "fake/hash-embedding-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * EMBEDDING_DIMENSIONS
            tokens = text.lower().split() or [""]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                position = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSIONS
                vector[position] += -1.0 if digest[4] & 1 else 1.0
            magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / magnitude for value in vector])
        return vectors


class BgeM3Embedder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_version = self.settings.embedding_model
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_version)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        values = self._load().encode(texts, normalize_embeddings=True)
        return [list(map(float, value)) for value in values]


def configured_embedder(settings: Settings | None = None) -> BgeM3Embedder | DeterministicEmbedder:
    settings = settings or get_settings()
    if settings.provider_chain.startswith("fake:"):
        return DeterministicEmbedder()
    return BgeM3Embedder(settings)
