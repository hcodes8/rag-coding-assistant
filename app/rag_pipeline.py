from __future__ import annotations
import logging
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI

from app.config import (
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from app.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# Prompt
# {context}: retrieved chunks concatenated together
# {question}: the user's question
_SYSTEM_PROMPT = """\
You are an expert technical assistant specialising in programming language \
documentation and practical coding guidance. You have access to retrieved documentation snippets (the Context below) \
to construct your answer. 

Rules:
1. Ground your answer in the provided context. Treat it as the authoritative \
   source of truth for the language's behaviour, syntax, and APIs.
2. You MAY go beyond what appears verbatim in the context to write clear, \
   runnable code examples that illustrate the concept being asked about. \
   Any example you write must be consistent with what the docs describe.
3. If the context genuinely lacks enough information to answer (e.g. the topic \
   is not covered at all), say: "I couldn't find that in the loaded documentation." \
   Do not invent APIs or behaviour not implied by the docs.
4. Format your response with Markdown. Use fenced code blocks for all code.
5. At the end of your answer list the source file(s) you drew from under a \
   "Sources:" heading. 

Context:
{context}
"""

_HUMAN_PROMPT = "Question: {question}"

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM_PROMPT),("human", _HUMAN_PROMPT),])

def _format_docs(docs: list) -> str:
    """
    Concatenate retrieved Document chunks into a single context string. Each
    chunk is separated by a divider and prefixed with its source file.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RAGPipeline:
    """
    Encapsulates the full RAG question-answering pipeline for one language.

    Usage: 
    pipeline = RAGPipeline(vs_manager)
    pipeline.set_language("python")
    answer = pipeline.ask("What is a list comprehension?")
    """

    def __init__(self, vs_manager: VectorStoreManager) -> None:
        self._vs_manager = vs_manager
        self._current_language: str | None = None
        self._chain: Any | None = None  # built lazily when language is set

        # LLM client
        # set request_timeout to avoid hanging the GUI thread
        self._llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            request_timeout=60,
        )

    @property
    def current_language(self) -> str | None:
        return self._current_language

    def set_language(self, language: str) -> None:
        """
        Switches pipeline to a different language's vector store.

        Rebuilds the retrieval chain. Called whenever the user changes
        the active language in the GUI.

        Raises:
            RuntimeError: If the language has not been ingested yet.
        """
        if language == self._current_language:
            return  # if language hasn't changed

        logger.info("Setting active language to '%s'", language)
        retriever = self._vs_manager.get_retriever(language)

        # RunnableParallel runs both branches with the same input dict:
        # "context": retriever fetches docs → _format_docs joins them
        # "question": passed through unchanged
        # Then pipes to _PROMPT -> LLM -> output string parser.
        self._chain = (
            RunnableParallel(
                {
                    "context": retriever | _format_docs,
                    "question": RunnablePassthrough(),
                }
            )
            | _PROMPT
            | self._llm
            | StrOutputParser()
        )

        self._current_language = language
        logger.info("RAG chain ready for '%s'", language)

    def ask(self, question: str) -> str:
        """
        Submits a question to the active language's RAG chain.

        Args:
            question: The user's natural-language question.

        Returns:
            The LLM's answer as a string.

        Raises:
            RuntimeError: If no language has been set yet.
        """
        if self._chain is None or self._current_language is None:
            raise RuntimeError("No language selected. Call set_language() first.")

        if not question.strip():
            return "Please enter a question."

        logger.debug("Asking [%s]: %s", self._current_language, question)

        try:
            answer = self._chain.invoke(question)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            answer = (
                f"An error occurred during the LLM call:\n{exc}\n\n"
                "Check your OPENROUTER_API_KEY and internet connection."
            )

        return answer

    def ask_stream(self, question: str):
        if self._chain is None or self._current_language is None:
            raise RuntimeError("No language selected. Call set_language() first.")
        if not question.strip():
            yield "Please enter a question."
            return

        logger.debug("Streaming ask [%s]: %s", self._current_language, question)
        logger.debug("START STREAM %s", question)
        try:
            # Build a streaming chain without StrOutputParser
            retriever = self._vs_manager.get_retriever(self._current_language)
            stream_chain = (
                RunnableParallel(
                    {
                        "context": retriever | _format_docs,
                        "question": RunnablePassthrough(),
                    }
                )
                | _PROMPT
                | self._llm  # stream directly from LLM, skip StrOutputParser
            )
            for chunk in stream_chain.stream(question):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                logger.debug("TOKEN %s", token[:20])
                if token:
                    yield token
            logger.debug("END STREAM")
        except Exception as exc:
            logger.error("LLM streaming call failed: %s", exc)
            yield f"\nError: {exc}\n"