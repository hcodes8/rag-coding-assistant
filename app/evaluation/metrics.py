from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from app.retrieval import RetrievedChunk, tokenize

ABSTENTION_TEXT = "i couldn't find that in the loaded documentation"
WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _source_matches(actual: str, expected: str) -> bool:
    actual_normalized = actual.replace("\\", "/").lower()
    expected_normalized = expected.replace("\\", "/").lower()
    return actual_normalized == expected_normalized or actual_normalized.endswith(
        expected_normalized
    )


def retrieval_metrics(
    chunks: list[RetrievedChunk], relevant_sources: Iterable[str], k: int | None = None
) -> dict:
    cutoff = k or len(chunks)
    retrieved = [str(item.document.metadata.get("source", "")) for item in chunks[:cutoff]]
    relevant = list(dict.fromkeys(relevant_sources))
    matched_relevant: set[str] = set()
    hits = []
    for source in retrieved:
        matched = next(
            (expected for expected in relevant if _source_matches(source, expected)),
            None,
        )
        is_new_hit = matched is not None and matched not in matched_relevant
        hits.append(is_new_hit)
        if matched is not None:
            matched_relevant.add(matched)
    hit_count = sum(hits)
    first_hit = next((index for index, hit in enumerate(hits, start=1) if hit), None)
    dcg = sum((1.0 / math.log2(index + 1)) for index, hit in enumerate(hits, start=1) if hit)
    ideal_hits = min(len(relevant), cutoff)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return {
        f"precision@{cutoff}": hit_count / cutoff if cutoff else 0.0,
        f"recall@{cutoff}": len(matched_relevant) / len(relevant) if relevant else 0.0,
        f"hit_rate@{cutoff}": float(any(hits)),
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        f"ndcg@{cutoff}": dcg / idcg if idcg else 0.0,
    }


def token_f1(answer: str, reference: str) -> float:
    answer_tokens = tokenize(answer)
    reference_tokens = tokenize(reference)
    if not answer_tokens or not reference_tokens:
        return float(answer_tokens == reference_tokens)
    overlap = sum((Counter(answer_tokens) & Counter(reference_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_relevance(question: str, answer: str, reference: str | None = None) -> float:
    if reference:
        return token_f1(answer, reference)
    question_terms = set(WORD_RE.findall(question.lower()))
    answer_terms = set(WORD_RE.findall(answer.lower()))
    ignored = {"what", "when", "where", "which", "how", "why", "does", "the", "a", "an"}
    meaningful = question_terms - ignored
    return len(meaningful & answer_terms) / len(meaningful) if meaningful else 0.0


def groundedness(answer: str, contexts: Iterable[str]) -> float:
    if ABSTENTION_TEXT in answer.lower():
        return 1.0
    context_terms = set(WORD_RE.findall(" ".join(contexts).lower()))
    prose = re.sub(r"```.*?```", "", answer, flags=re.DOTALL)
    prose = re.split(r"\n\s*Sources:\s*\n", prose, maxsplit=1, flags=re.IGNORECASE)[0]
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", prose) if part.strip()]
    claims = []
    for sentence in sentences:
        terms = set(WORD_RE.findall(sentence.lower()))
        terms -= {"the", "a", "an", "and", "or", "to", "of", "in", "is", "are", "it", "this"}
        if len(terms) >= 3 and not sentence.lower().startswith("sources:"):
            claims.append(terms)
    if not claims:
        return 1.0
    supported = [len(claim & context_terms) / len(claim) >= 0.6 for claim in claims]
    return sum(supported) / len(supported)


def cited_sources(answer: str) -> list[str]:
    candidates = re.findall(r"(?:[\w.-]+/)*[\w.-]+\.(?:md|rst|txt|html?)", answer, re.I)
    return list(dict.fromkeys(candidates))


def citation_metrics(answer: str, retrieved_sources: Iterable[str]) -> dict:
    citations = cited_sources(answer)
    sources = list(dict.fromkeys(retrieved_sources))
    valid = sum(any(_source_matches(source, citation) for source in sources) for citation in citations)
    cited_retrieved = sum(any(_source_matches(source, citation) for citation in citations) for source in sources)
    return {
        "citation_precision": valid / len(citations) if citations else 0.0,
        "citation_recall": cited_retrieved / len(sources) if sources else 0.0,
        "citation_count": len(citations),
    }


def evaluate_answer(
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    reference_answer: str | None = None,
    should_abstain: bool = False,
) -> dict:
    contexts = [item.document.page_content for item in chunks]
    sources = [str(item.document.metadata.get("source", "")) for item in chunks]
    abstained = ABSTENTION_TEXT in answer.lower()
    groundedness_score = groundedness(answer, contexts)
    result = {
        "answer_relevance": answer_relevance(question, answer, reference_answer),
        "groundedness": groundedness_score,
        "hallucination_risk": 1.0 - groundedness_score,
        "abstained": abstained,
        "abstention_accuracy": float(abstained == should_abstain),
        **citation_metrics(answer, sources),
    }
    if reference_answer is not None:
        result["token_f1"] = token_f1(answer, reference_answer)
    return result
