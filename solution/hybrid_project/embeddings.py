"""Embedding implementations for real retrieval and inexpensive tests."""

import hashlib
import re
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

VectorMatrix = np.ndarray[tuple[int, int], np.dtype[np.float32]]


class Embedder(Protocol):
    """Minimal vectorization contract required by the retriever."""

    @property
    def dimension(self) -> int:
        """Return output-vector dimensionality."""

    def encode(self, texts: list[str]) -> VectorMatrix:
        """Encode texts as normalized vectors."""


class SentenceTransformerEmbedder:
    """Dense semantic vectors from a local Sentence Transformers model."""

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, revision=revision)
        probe = self._model.encode(
            ["embedding dimension probe"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        probe_matrix = np.asarray(probe, dtype=np.float32)
        if probe_matrix.ndim != 2 or probe_matrix.shape[1] == 0:
            raise ValueError("The configured embedding model has no fixed dimension")
        self._dimension = int(probe_matrix.shape[1])

    @property
    def dimension(self) -> int:
        """Return the vector size required by Qdrant."""

        return self._dimension

    def encode(self, texts: list[str]) -> VectorMatrix:
        """Generate unit-length vectors for cosine similarity."""

        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashEmbedder:
    """Deterministic lexical encoder reserved for tests and smoke runs."""

    def __init__(self, dimension: int = 2048) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the fixed smoke-vector size."""

        return self._dimension

    def encode(self, texts: list[str]) -> VectorMatrix:
        """Hash visible terms into normalized vectors without model downloads."""

        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "big") % self.dimension
                matrix[row, column] += 1.0
            length = float(np.linalg.norm(matrix[row]))
            if length:
                matrix[row] /= length
        return matrix
