from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from langchain_core.documents import Document

from app.config import (
    DENSE_WEIGHT,
    RERANK_ENABLED,
    RERANK_TOP_K,
    RERANKER_MODEL_NAME,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_K,
    RRF_K,
    SPARSE_WEIGHT,
)

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _chunk_key(document: Document) -> str:
    chunk_id = document.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"{document.metadata.get('source', 'unknown')}::{hash(document.page_content)}"


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None

    def as_dict(self, include_content: bool = False) -> dict:
        payload = {
            "chunk_id": self.document.metadata.get("chunk_id"),
            "source": self.document.metadata.get("source", "unknown"),
            "score": round(self.score, 6),
            "dense_score": round(self.dense_score, 6),
            "sparse_score": round(self.sparse_score, 6),
            "rerank_score": round(self.rerank_score, 6),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
        }
        if include_content:
            payload["content"] = self.document.page_content
        return payload


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    retrieval_ms: float
    rerank_ms: float
    strategy: str = "hybrid_rrf+cross_encoder"
    diagnostics: dict = field(default_factory=dict)


class BM25Index:
    """Small in-process BM25 index over the chunks already stored in Chroma."""

    def __init__(self, documents: Iterable[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(doc.page_content) for doc in self.documents]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        query_terms = tokenize(query)
        scored: list[tuple[Document, float]] = []
        for document, frequencies, length in zip(
            self.documents, self.term_frequencies, self.lengths
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self.avg_length, 1.0)
                )
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            if score > 0:
                scored.append((document, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        if not scored:
            return []
        maximum = scored[0][1]
        return [(document, score / maximum) for document, score in scored[:k]]


class Reranker:
    def __init__(self, enabled: bool = RERANK_ENABLED, model_name: str = RERANKER_MODEL_NAME):
        self.enabled = enabled
        self.model_name = model_name
        self._model = None
        self.backend = "lexical"

    def _load_model(self) -> None:
        if not self.enabled or self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self.backend = "cross_encoder"
        except Exception as exc:
            self.enabled = False
            logger.warning("Cross-encoder unavailable; using lexical reranker: %s", exc)

    @staticmethod
    def _lexical_score(question: str, content: str) -> float:
        query_tokens = set(tokenize(question))
        content_tokens = set(tokenize(content))
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & content_tokens) / len(query_tokens)
        phrase_bonus = 0.15 if question.lower() in content.lower() else 0.0
        return min(1.0, overlap + phrase_bonus)

    def rerank(self, question: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        self._load_model()
        if self._model is not None:
            raw_scores = self._model.predict(
                [(question, item.document.page_content) for item in candidates],
                show_progress_bar=False,
            )
            scores = [1.0 / (1.0 + math.exp(-float(score))) for score in raw_scores]
        else:
            scores = [
                self._lexical_score(question, item.document.page_content)
                for item in candidates
            ]
        for item, score in zip(candidates, scores):
            item.rerank_score = score
            item.score = 0.25 * item.score + 0.75 * score
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:k]


class HybridRetriever:
    def __init__(self, vector_store, reranker: Reranker | None = None):
        self.vector_store = vector_store
        self.reranker = reranker or Reranker()
        self._sparse_indexes: dict[str, BM25Index] = {}

    def invalidate(self, language: str) -> None:
        self._sparse_indexes.pop(language, None)

    def _sparse_index(self, language: str) -> BM25Index:
        if language not in self._sparse_indexes:
            self._sparse_indexes[language] = BM25Index(
                self.vector_store.all_documents(language)
            )
        return self._sparse_indexes[language]

    def retrieve(
        self,
        question: str,
        language: str,
        k: int = RETRIEVAL_K,
        candidate_k: int = RETRIEVAL_CANDIDATES,
    ) -> RetrievalResult:
        started = time.perf_counter()
        dense = self.vector_store.dense_search(language, question, candidate_k)
        sparse = self._sparse_index(language).search(question, candidate_k)

        fused: dict[str, RetrievedChunk] = {}
        for rank, (document, similarity) in enumerate(dense, start=1):
            key = _chunk_key(document)
            item = fused.setdefault(key, RetrievedChunk(document=document, score=0.0))
            item.dense_score = similarity
            item.dense_rank = rank
            item.score += DENSE_WEIGHT / (RRF_K + rank)
        for rank, (document, similarity) in enumerate(sparse, start=1):
            key = _chunk_key(document)
            item = fused.setdefault(key, RetrievedChunk(document=document, score=0.0))
            item.sparse_score = similarity
            item.sparse_rank = rank
            item.score += SPARSE_WEIGHT / (RRF_K + rank)

        candidates = sorted(fused.values(), key=lambda item: item.score, reverse=True)
        retrieval_ms = (time.perf_counter() - started) * 1000
        rerank_started = time.perf_counter()
        chunks = self.reranker.rerank(question, candidates, min(k, RERANK_TOP_K))
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        return RetrievalResult(
            chunks=chunks,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            strategy=f"hybrid_rrf+{self.reranker.backend}",
            diagnostics={
                "dense_candidates": len(dense),
                "sparse_candidates": len(sparse),
                "fused_candidates": len(candidates),
            },
        )
