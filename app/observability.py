from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import (
    LLM_INPUT_COST_PER_MILLION,
    LLM_OUTPUT_COST_PER_MILLION,
    OBSERVABILITY_DB_PATH,
)


def estimate_tokens(text: str) -> int:
    """Provider-independent fallback when a model omits usage metadata."""
    return max(1, math.ceil(len(text.encode("utf-8")) / 4)) if text else 0


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * LLM_INPUT_COST_PER_MILLION
        + output_tokens * LLM_OUTPUT_COST_PER_MILLION
    ) / 1_000_000


def extract_token_usage(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    if usage:
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    metadata = getattr(message, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    return (
        int(token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))),
        int(token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))),
    )


@dataclass
class TraceRecord:
    question: str
    language: str
    answer: str
    strategy: str
    sources: list[dict]
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float
    ttft_ms: float | None
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    token_usage_estimated: bool
    quality: dict = field(default_factory=dict)
    error: str | None = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self, include_answer: bool = True) -> dict:
        payload = asdict(self)
        if not include_answer:
            payload.pop("answer", None)
        return payload


class TraceStore:
    def __init__(self, path: Path | str = OBSERVABILITY_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    language TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    retrieval_ms REAL NOT NULL,
                    rerank_ms REAL NOT NULL,
                    generation_ms REAL NOT NULL,
                    total_ms REAL NOT NULL,
                    ttft_ms REAL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    token_usage_estimated INTEGER NOT NULL,
                    quality_json TEXT NOT NULL,
                    error TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at DESC)"
            )

    def record(self, trace: TraceRecord) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.created_at,
                    trace.question,
                    trace.language,
                    trace.answer,
                    trace.strategy,
                    json.dumps(trace.sources),
                    trace.retrieval_ms,
                    trace.rerank_ms,
                    trace.generation_ms,
                    trace.total_ms,
                    trace.ttft_ms,
                    trace.input_tokens,
                    trace.output_tokens,
                    trace.estimated_cost_usd,
                    int(trace.token_usage_estimated),
                    json.dumps(trace.quality),
                    trace.error,
                ),
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row, include_answer: bool) -> dict:
        payload = dict(row)
        payload["sources"] = json.loads(payload.pop("sources_json"))
        payload["quality"] = json.loads(payload.pop("quality_json"))
        payload["token_usage_estimated"] = bool(payload["token_usage_estimated"])
        if not include_answer:
            payload.pop("answer", None)
        return payload

    def recent(self, limit: int = 25, include_answer: bool = False) -> list[dict]:
        safe_limit = max(1, min(limit, 200))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        return [self._row_to_dict(row, include_answer) for row in rows]

    def summary(self) -> dict:
        with self._connect() as connection:
            aggregate = connection.execute(
                """
                SELECT COUNT(*) AS requests,
                       SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                       AVG(total_ms) AS avg_latency_ms,
                       AVG(ttft_ms) AS avg_ttft_ms,
                       SUM(input_tokens) AS input_tokens,
                       SUM(output_tokens) AS output_tokens,
                       SUM(estimated_cost_usd) AS cost_usd
                FROM traces
                """
            ).fetchone()
            latencies = [
                row[0]
                for row in connection.execute(
                    "SELECT total_ms FROM traces ORDER BY total_ms"
                ).fetchall()
            ]
        requests = int(aggregate["requests"] or 0)
        p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
        return {
            "requests": requests,
            "errors": int(aggregate["errors"] or 0),
            "error_rate": (aggregate["errors"] or 0) / requests if requests else 0.0,
            "avg_latency_ms": round(aggregate["avg_latency_ms"] or 0.0, 2),
            "p95_latency_ms": round(latencies[p95_index], 2) if latencies else 0.0,
            "avg_ttft_ms": round(aggregate["avg_ttft_ms"] or 0.0, 2),
            "input_tokens": int(aggregate["input_tokens"] or 0),
            "output_tokens": int(aggregate["output_tokens"] or 0),
            "estimated_cost_usd": round(aggregate["cost_usd"] or 0.0, 8),
        }
