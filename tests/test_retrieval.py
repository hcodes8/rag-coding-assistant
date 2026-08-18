from langchain_core.documents import Document

from app.retrieval import HybridRetriever, Reranker


class FakeVectorStore:
    def __init__(self):
        self.docs = [
            Document(
                page_content="A Python generator uses yield and produces values lazily.",
                metadata={"source": "python/generators.md", "chunk_id": "generator"},
            ),
            Document(
                page_content="A list comprehension builds a list from an iterable.",
                metadata={"source": "python/lists.md", "chunk_id": "lists"},
            ),
        ]

    def all_documents(self, language):
        return self.docs

    def dense_search(self, language, query, k):
        return [(self.docs[1], 0.9), (self.docs[0], 0.7)]


def test_hybrid_search_recovers_exact_keyword_candidate():
    retriever = HybridRetriever(FakeVectorStore(), Reranker(enabled=False))
    result = retriever.retrieve("How does yield make a generator lazy?", "python", k=1)
    assert result.chunks[0].document.metadata["source"] == "python/generators.md"
    assert result.diagnostics["dense_candidates"] == 2
    assert result.diagnostics["sparse_candidates"] >= 1


def test_sparse_index_is_cached_and_invalidated():
    retriever = HybridRetriever(FakeVectorStore(), Reranker(enabled=False))
    first = retriever._sparse_index("python")
    assert retriever._sparse_index("python") is first
    retriever.invalidate("python")
    assert retriever._sparse_index("python") is not first
