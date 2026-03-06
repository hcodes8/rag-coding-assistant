import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

def _make_mock_vs_manager(language="python", docs=None):
    """Create a mock VectorStoreManager that returns fake docs."""
    if docs is None:
        docs = [
            Document(
                page_content="Variables are symbolic names that act as references to values stored in memory.",
                metadata={"source": "python/basics.txt"},
            )
        ]

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = docs
    # LangChain chains call retriever as a Runnable; mock __or__ for pipe
    mock_retriever.__or__ = lambda self, other: MagicMock(invoke=lambda q: other(docs))

    mock_vs = MagicMock()
    mock_vs.get_retriever.return_value = mock_retriever
    mock_vs.collection_exists.return_value = True
    return mock_vs


class TestRAGPipelineSetLanguage:
    def test_set_language_updates_current_language(self):
        from app.rag_pipeline import RAGPipeline

        mock_vs = _make_mock_vs_manager()
        pipeline = RAGPipeline(mock_vs)

        with patch.object(pipeline, "_llm", MagicMock()):
            pipeline.set_language("python")

        assert pipeline.current_language == "python"

    def test_set_same_language_twice_is_noop(self):
        from app.rag_pipeline import RAGPipeline

        mock_vs = _make_mock_vs_manager()
        pipeline = RAGPipeline(mock_vs)

        with patch.object(pipeline, "_llm", MagicMock()):
            pipeline.set_language("python")
            pipeline.set_language("python")  # second call should not rebuild chain

        # get_retriever should only have been called once
        mock_vs.get_retriever.assert_called_once()

    def test_raises_if_language_not_ingested(self):
        from app.rag_pipeline import RAGPipeline

        mock_vs = MagicMock()
        mock_vs.get_retriever.side_effect = RuntimeError("not ingested")

        pipeline = RAGPipeline(mock_vs)
        with pytest.raises(RuntimeError):
            pipeline.set_language("haskell")


class TestRAGPipelineAsk:
    def test_ask_without_language_raises(self):
        from app.rag_pipeline import RAGPipeline

        mock_vs = _make_mock_vs_manager()
        pipeline = RAGPipeline(mock_vs)

        with pytest.raises(RuntimeError, match="No language selected"):
            pipeline.ask("What is a decorator?")

    def test_empty_question_returns_prompt(self):
        from app.rag_pipeline import RAGPipeline

        mock_vs = _make_mock_vs_manager()
        pipeline = RAGPipeline(mock_vs)
        pipeline._current_language = "python"
        pipeline._chain = MagicMock()  # won't be called for empty input

        result = pipeline.ask("   ")
        assert "Please enter a question" in result

    def test_llm_error_returns_friendly_message(self):
        from app.rag_pipeline import RAGPipeline

        mock_vs = _make_mock_vs_manager()
        pipeline = RAGPipeline(mock_vs)
        pipeline._current_language = "python"

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("API timeout")
        pipeline._chain = mock_chain

        result = pipeline.ask("What is a generator?")
        assert "error" in result.lower()
        assert "API timeout" in result