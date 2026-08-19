from langchain_core.documents import Document

from app.evaluation.metrics import evaluate_answer, retrieval_metrics, token_f1
from app.retrieval import RetrievedChunk


def _chunk(source: str, content: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        document=Document(page_content=content, metadata={"source": source}),
        score=score,
    )


def test_retrieval_metrics_rank_relevant_source():
    chunks = [_chunk("python/other.md", "other"), _chunk("python/basics.md", "target")]
    metrics = retrieval_metrics(chunks, ["python/basics.md"], k=2)
    assert metrics["hit_rate@2"] == 1.0
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr"] == 0.5


def test_token_f1_rewards_reference_overlap():
    close = token_f1("Generators yield values lazily", "A generator yields values lazily")
    unrelated = token_f1("A database stores rows", "A generator yields values lazily")
    assert close > unrelated


def test_grounded_answer_has_low_hallucination_risk():
    chunks = [_chunk("python/files.md", "The with statement closes a file automatically.")]
    metrics = evaluate_answer(
        "How are files closed?",
        "The with statement closes a file automatically.\n\nSources:\n- python/files.md",
        chunks,
    )
    assert metrics["groundedness"] == 1.0
    assert metrics["citation_precision"] == 1.0


def test_hallucination_case_requires_explicit_abstention():
    metrics = evaluate_answer(
        "How does imaginary_api work?",
        "I couldn't find that in the loaded documentation.",
        [],
        should_abstain=True,
    )
    assert metrics["abstained"] is True
    assert metrics["abstention_accuracy"] == 1.0
