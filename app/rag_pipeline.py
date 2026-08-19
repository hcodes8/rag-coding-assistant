from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import (
    DEMO_MIN_RELEVANCE,
    DEMO_MODE,
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from app.evaluation.metrics import evaluate_answer
from app.observability import (
    TraceRecord,
    TraceStore,
    calculate_cost,
    estimate_tokens,
    extract_token_usage,
)
from app.retrieval import HybridRetriever, RetrievalResult
from app.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert technical assistant specialising in programming language
documentation and practical coding guidance. Use the retrieved Context below.

Rules:
1. Ground every factual claim in the Context. Treat it as authoritative.
2. You may write examples, but they must be consistent with the Context.
3. If the Context lacks enough information, say exactly: "I couldn't find that
   in the loaded documentation." Do not fill gaps from model memory.
4. Explain the answer clearly, include runnable code when useful, and mention
   caveats supported by the Context.
5. Use Markdown and fenced code blocks.
6. End with a "Sources:" heading listing only source files present in Context.

Context:
{context}
"""
_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", "Question: {question}")]
)


def _format_docs(chunks: list) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {item.document.metadata.get('source', 'unknown')}]\n"
        f"{item.document.page_content}"
        for item in chunks
    )


def _content_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


class RAGPipeline:
    def __init__(
        self,
        vs_manager: VectorStoreManager,
        trace_store: TraceStore | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        llm: Any | None = None,
    ) -> None:
        self._vs_manager = vs_manager
        self._current_language: str | None = None
        self._legacy_retriever: Any | None = None
        self._hybrid_retriever = hybrid_retriever or HybridRetriever(vs_manager)
        self.trace_store = trace_store or TraceStore()
        self.last_trace: TraceRecord | None = None
        self.last_result: dict | None = None
        self._chain: Any | None = None  # compatibility for older integrations
        self._demo_mode = DEMO_MODE and not OPENROUTER_API_KEY
        self._llm = llm
        if self._llm is None and not self._demo_mode:
            self._llm = ChatOpenAI(
                model=LLM_MODEL_NAME,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                timeout=60,
                stream_usage=True,
            )

    @property
    def current_language(self) -> str | None:
        return self._current_language

    def set_language(self, language: str) -> None:
        if language == self._current_language:
            return
        logger.info("Setting active language to '%s'", language)
        self._legacy_retriever = self._vs_manager.get_retriever(language)
        self._current_language = language

    def invalidate_language(self, language: str) -> None:
        self._hybrid_retriever.invalidate(language)

    def retrieve(self, question: str) -> RetrievalResult:
        language = self._require_ready()
        return self._hybrid_retriever.retrieve(question, language)

    def _require_ready(self) -> str:
        if self._current_language is None:
            raise RuntimeError("No language selected. Call set_language() first.")
        return self._current_language

    @staticmethod
    def _demo_answer(question: str, retrieval: RetrievalResult) -> str:
        if (
            not retrieval.chunks
            or retrieval.chunks[0].rerank_score < DEMO_MIN_RELEVANCE
        ):
            return "I couldn't find that in the loaded documentation."
        excerpts = []
        for item in retrieval.chunks[:3]:
            excerpt = " ".join(item.document.page_content.split())[:500]
            excerpts.append(f"- {excerpt}")
        sources = list(
            dict.fromkeys(
                str(item.document.metadata.get("source", "unknown"))
                for item in retrieval.chunks
            )
        )
        return (
            "**Retrieval-only demo mode**\n\n"
            f"The strongest documentation passages for _{question}_ are:\n\n"
            + "\n".join(excerpts)
            + "\n\nSources:\n"
            + "\n".join(f"- {source}" for source in sources)
        )

    def _record_trace(
        self,
        *,
        question: str,
        retrieval: RetrievalResult,
        answer: str,
        started: float,
        generation_started: float,
        ttft_ms: float | None,
        input_tokens: int,
        output_tokens: int,
        usage_estimated: bool,
        error: str | None = None,
    ) -> TraceRecord:
        generation_ms = (time.perf_counter() - generation_started) * 1000
        quality = evaluate_answer(question, answer, retrieval.chunks)
        trace = TraceRecord(
            question=question,
            language=self._current_language or "unknown",
            answer=answer,
            strategy=retrieval.strategy,
            sources=[item.as_dict() for item in retrieval.chunks],
            retrieval_ms=retrieval.retrieval_ms,
            rerank_ms=retrieval.rerank_ms,
            generation_ms=generation_ms,
            total_ms=(time.perf_counter() - started) * 1000,
            ttft_ms=ttft_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=calculate_cost(input_tokens, output_tokens),
            token_usage_estimated=usage_estimated,
            quality=quality,
            error=error,
        )
        self.trace_store.record(trace)
        self.last_trace = trace
        self.last_result = {
            "answer": answer,
            "retrieval": retrieval,
            "trace": trace,
        }
        return trace

    def ask(self, question: str) -> str:
        language = self._require_ready()
        if not question.strip():
            return "Please enter a question."
        if self._chain is not None:
            try:
                return self._chain.invoke(question)
            except Exception as exc:
                return f"An error occurred during the LLM call:\n{exc}"

        started = time.perf_counter()
        retrieval = self._hybrid_retriever.retrieve(question, language)
        context = _format_docs(retrieval.chunks)
        messages = _PROMPT.invoke({"context": context, "question": question}).to_messages()
        generation_started = time.perf_counter()
        error = None
        try:
            if self._demo_mode:
                answer = self._demo_answer(question, retrieval)
                input_tokens = estimate_tokens(context + question)
                output_tokens = estimate_tokens(answer)
                estimated = True
            else:
                response = self._llm.invoke(messages)
                answer = _content_text(response)
                input_tokens, output_tokens = extract_token_usage(response)
                estimated = not (input_tokens or output_tokens)
                if estimated:
                    input_tokens = estimate_tokens(context + question)
                    output_tokens = estimate_tokens(answer)
        except Exception as exc:
            logger.exception("LLM call failed")
            error = str(exc)
            answer = f"An error occurred during the LLM call:\n{exc}"
            input_tokens = estimate_tokens(context + question)
            output_tokens = estimate_tokens(answer)
            estimated = True
        self._record_trace(
            question=question,
            retrieval=retrieval,
            answer=answer,
            started=started,
            generation_started=generation_started,
            ttft_ms=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_estimated=estimated,
            error=error,
        )
        return answer

    def ask_stream_events(self, question: str) -> Iterator[dict]:
        language = self._require_ready()
        if not question.strip():
            yield {"type": "token", "token": "Please enter a question."}
            return

        started = time.perf_counter()
        retrieval = self._hybrid_retriever.retrieve(question, language)
        context = _format_docs(retrieval.chunks)
        messages = _PROMPT.invoke({"context": context, "question": question}).to_messages()
        generation_started = time.perf_counter()
        answer_parts: list[str] = []
        input_tokens = output_tokens = 0
        ttft_ms = None
        error = None
        try:
            if self._demo_mode:
                answer = self._demo_answer(question, retrieval)
                stream = (part + " " for part in answer.split(" "))
            else:
                stream = self._llm.stream(messages)
            for chunk in stream:
                token = chunk if isinstance(chunk, str) else _content_text(chunk)
                if token:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - generation_started) * 1000
                    answer_parts.append(token)
                    yield {"type": "token", "token": token}
                if not isinstance(chunk, str):
                    chunk_input, chunk_output = extract_token_usage(chunk)
                    input_tokens = max(input_tokens, chunk_input)
                    output_tokens = max(output_tokens, chunk_output)
        except Exception as exc:
            logger.exception("LLM streaming call failed")
            error = str(exc)
            token = f"\n\nError: {exc}"
            answer_parts.append(token)
            yield {"type": "token", "token": token}

        answer = "".join(answer_parts).strip()
        estimated = not (input_tokens or output_tokens)
        if estimated:
            input_tokens = estimate_tokens(context + question)
            output_tokens = estimate_tokens(answer)
        trace = self._record_trace(
            question=question,
            retrieval=retrieval,
            answer=answer,
            started=started,
            generation_started=generation_started,
            ttft_ms=ttft_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_estimated=estimated,
            error=error,
        )
        yield {"type": "trace", "trace": trace.as_dict(include_answer=False)}

    def ask_stream(self, question: str) -> Iterator[str]:
        for event in self.ask_stream_events(question):
            if event["type"] == "token":
                yield event["token"]
