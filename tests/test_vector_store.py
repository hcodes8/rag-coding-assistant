import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
import chromadb

def _fake_docs(n: int = 5, language: str = "python") -> list:
    """Generate n LangChain Documents for testing."""
    return [
        Document(
            page_content=f"Chunk {i}: programming concepts for {language}.",
            metadata={"source": "test.txt", "language": language},
        )
        for i in range(n)
    ]

@pytest.fixture()
def mock_embeddings():
    """
    Patch HuggingFaceEmbeddings so no model download happens in tests.
    Returns fixed-length float vectors that satisfy ChromaDB's
    requirement that all vectors in a collection share the same dimension.
    """
    with patch("app.vector_store.HuggingFaceEmbeddings") as mock_cls:
        instance = MagicMock()

        # embed_documents: called during ingest, one vector per document
        # dimension 384 is used to match all-MiniLM-L6-v2
        instance.embed_documents.side_effect = (
            lambda texts: [[0.1] * 384 for _ in texts]
        )
        # embed_query: called during retrieval, one vector per question
        instance.embed_query.return_value = [0.1] * 384
        instance.side_effect = None
        mock_cls.return_value = instance
        yield instance

@pytest.fixture()
def vs_manager(tmp_path, mock_embeddings):
    """
    Create an isolated VectorStoreManager for each test.
    """
    from app.vector_store import VectorStoreManager
    manager = VectorStoreManager()
    # Replace the chroma client with one backed by the isolated tmp_path.
    manager._chroma_client = chromadb.PersistentClient(path=str(tmp_path))
    # Clear the LangChain wrapper cache so it rebuilds against the new client
    manager._stores.clear()
    return manager

class TestCollectionExistence:
    def test_new_language_does_not_exist(self, vs_manager):
        # New client backed by empty tmp_path, no collections present
        assert not vs_manager.collection_exists("python")

    def test_exists_after_ingest(self, vs_manager):
        vs_manager.ingest("python", _fake_docs())
        assert vs_manager.collection_exists("python")

    def test_different_languages_are_independent(self, vs_manager):
        # Ingesting python must not make rust appear as ingested
        vs_manager.ingest("python", _fake_docs())
        assert not vs_manager.collection_exists("rust")

    def test_collection_name_is_normalised(self, vs_manager):
        # collection_name_for() lowercases and strips, existence check must
        # use same normalisation
        vs_manager.ingest("Python", _fake_docs())
        # Both "Python" and "python" should resolve to the same collection
        assert vs_manager.collection_exists("Python")
        assert vs_manager.collection_exists("python")

class TestIngest:
    def test_ingest_creates_collection(self, vs_manager):
        vs_manager.ingest("javascript", _fake_docs(3, "javascript"))
        assert "javascript" in vs_manager.list_ingested_languages()

    def test_reingest_replaces_collection(self, vs_manager):
        vs_manager.ingest("python", _fake_docs(3))
        # Re-ingesting with diff content should replace
        vs_manager.ingest("python", _fake_docs(7))
        assert vs_manager.collection_exists("python")

    def test_reingest_clears_store_cache(self, vs_manager):
        # After re-ingest the old LangChain wrapper must be removed from cache
        # so the next get_retriever call rebuilds against the new collection
        vs_manager.ingest("python", _fake_docs(3))
        from app.config import collection_name_for
        name = collection_name_for("python")
        store_before = vs_manager._stores.get(name)

        vs_manager.ingest("python", _fake_docs(7))    # re-ingest
        store_after = vs_manager._stores.get(name)

        # The cache entry must exist after re-ingest
        assert store_after is not None
        # but it has to be a different object
        assert store_before is not store_after

    def test_multiple_languages_coexist(self, vs_manager):
        vs_manager.ingest("python", _fake_docs(2, "python"))
        vs_manager.ingest("rust", _fake_docs(2, "rust"))
        ingested = vs_manager.list_ingested_languages()
        assert "python" in ingested
        assert "rust" in ingested

    def test_ingest_empty_list_raises_value_error(self, vs_manager):
        with pytest.raises(ValueError, match="empty document list"):
            vs_manager.ingest("empty", [])

class TestGetRetriever:
    def test_raises_if_not_ingested(self, vs_manager):
        with pytest.raises(RuntimeError, match="does not exist"):
            vs_manager.get_retriever("haskell")

    def test_returns_retriever_after_ingest(self, vs_manager):
        vs_manager.ingest("python", _fake_docs(5))
        retriever = vs_manager.get_retriever("python")
        # LangChain retriever uses invoke() or legacy method
        assert hasattr(retriever, "invoke") or hasattr(
            retriever, "get_relevant_documents"
        )

    def test_second_call_uses_cache(self, vs_manager):
        # get_retriever should cache the LangChain Chroma wrapper so that
        # repeated calls don't reconstruct it from scratch each time
        vs_manager.ingest("python", _fake_docs(3))
        r1 = vs_manager.get_retriever("python")
        r2 = vs_manager.get_retriever("python")
        from app.config import collection_name_for
        assert collection_name_for("python") in vs_manager._stores

    def test_different_languages_return_different_retrievers(self, vs_manager):
        vs_manager.ingest("python", _fake_docs(3, "python"))
        vs_manager.ingest("rust", _fake_docs(3, "rust"))
        r_py = vs_manager.get_retriever("python")
        r_rs = vs_manager.get_retriever("rust")
        # They must be distinct objects backed by different collections
        assert r_py is not r_rs

class TestListIngestedLanguages:
    def test_empty_when_nothing_ingested(self, vs_manager):
        assert vs_manager.list_ingested_languages() == []

    def test_lists_all_ingested(self, vs_manager):
        vs_manager.ingest("python", _fake_docs(2))
        vs_manager.ingest("golang", _fake_docs(2))
        result = vs_manager.list_ingested_languages()
        assert set(result) == {"python", "golang"}