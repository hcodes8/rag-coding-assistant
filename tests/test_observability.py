from app.observability import TraceRecord, TraceStore, calculate_cost, estimate_tokens


def _trace() -> TraceRecord:
    return TraceRecord(
        question="What is yield?",
        language="python",
        answer="yield creates a generator",
        strategy="hybrid_rrf+lexical",
        sources=[{"source": "python/generators.md", "score": 0.9}],
        retrieval_ms=10,
        rerank_ms=2,
        generation_ms=30,
        total_ms=42,
        ttft_ms=12,
        input_tokens=100,
        output_tokens=20,
        estimated_cost_usd=0,
        token_usage_estimated=False,
        quality={"groundedness": 1.0},
    )


def test_trace_store_records_and_summarizes(tmp_path):
    store = TraceStore(tmp_path / "traces.db")
    store.record(_trace())
    recent = store.recent()
    assert recent[0]["question"] == "What is yield?"
    assert "answer" not in recent[0]
    summary = store.summary()
    assert summary["requests"] == 1
    assert summary["avg_latency_ms"] == 42
    assert summary["input_tokens"] == 100


def test_token_fallback_and_zero_default_cost():
    assert estimate_tokens("abcd") == 1
    assert calculate_cost(100, 100) >= 0
