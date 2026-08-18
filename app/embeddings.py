from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


class HashEmbeddings(Embeddings):
    """Deterministic, dependency-free embeddings for demos and smoke tests.

    This is not a replacement for a semantic embedding model. It preserves
    lexical similarity in a fixed-size vector so the complete RAG stack can be
    deployed and exercised without downloading model weights.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            vector[index] += 1.0 if value & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
