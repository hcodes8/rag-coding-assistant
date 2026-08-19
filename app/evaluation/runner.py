from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ensure_dirs
from app.config import RETRIEVAL_K
from app.demo import seed_demo_docs
from app.document_loader import load_documents_for_language
from app.evaluation.metrics import evaluate_answer, retrieval_metrics
from app.rag_pipeline import RAGPipeline
from app.vector_store import VectorStoreManager


def load_dataset(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        missing = {"id", "language", "question"} - set(case)
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
        cases.append(case)
    return cases


def _averages(results: list[dict]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for result in results:
        for group in ("retrieval_metrics", "answer_metrics"):
            for name, value in result.get(group, {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.setdefault(name, []).append(float(value))
    return {name: statistics.fmean(items) for name, items in sorted(values.items())}


def _check_thresholds(averages: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    return [
        f"{metric}={averages.get(metric, 0.0):.3f} < {minimum:.3f}"
        for metric, minimum in thresholds.items()
        if averages.get(metric, 0.0) < minimum
    ]


def run_evaluation(
    dataset: Path,
    output: Path,
    retrieval_only: bool = False,
    thresholds_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    ensure_dirs()
    seed_demo_docs()
    manager = VectorStoreManager()
    pipeline = RAGPipeline(manager)
    results = []
    for case in load_dataset(dataset):
        language = case["language"]
        if not manager.collection_exists(language):
            manager.ingest(language, load_documents_for_language(language))
        pipeline.set_language(language)
        retrieval = pipeline.retrieve(case["question"])
        result = {
            "id": case["id"],
            "language": language,
            "question": case["question"],
            "retrieved_sources": [
                item.document.metadata.get("source") for item in retrieval.chunks
            ],
            "retrieval_metrics": retrieval_metrics(
                retrieval.chunks,
                case.get("relevant_sources", []),
                k=RETRIEVAL_K,
            ),
            "retrieval_ms": retrieval.retrieval_ms,
            "rerank_ms": retrieval.rerank_ms,
        }
        if not retrieval_only:
            answer = pipeline.ask(case["question"])
            generated = pipeline.last_result
            answer_retrieval = generated["retrieval"] if generated else retrieval
            result["answer"] = answer
            result["answer_metrics"] = evaluate_answer(
                case["question"],
                answer,
                answer_retrieval.chunks,
                reference_answer=case.get("reference_answer"),
                should_abstain=bool(case.get("should_abstain", False)),
            )
            if pipeline.last_trace:
                result["trace_id"] = pipeline.last_trace.trace_id

        results.append(result)

    averages = _averages(results)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "retrieval_only": retrieval_only,
        "case_count": len(results),
        "averages": averages,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    thresholds = (
        json.loads(thresholds_path.read_text(encoding="utf-8"))
        if thresholds_path
        else {}
    )
    failures = _check_thresholds(averages, thresholds)
    return report, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality")
    parser.add_argument("--dataset", type=Path, default=Path("evals/demo.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    report, failures = run_evaluation(
        args.dataset, args.output, args.retrieval_only, args.thresholds
    )
    print(json.dumps(report["averages"], indent=2))
    print(f"Report: {args.output}")
    if failures:
        print("Threshold failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
