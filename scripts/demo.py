"""End-to-end demo: ingest the sample docs, run a query, print an eval report.

    python scripts/demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.eval import evaluate
from app.models import EvalSample
from app.service import RAGService

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    service = RAGService(get_settings())

    print("== Ingesting sample documents ==")
    for name in ["sample_policy.txt", "sample_faq.md"]:
        resp = service.ingest_file(ROOT / "data" / name)
        print(f"  {resp.filename}: {resp.num_chunks} chunks, {resp.num_characters} chars")

    print("\n== Query ==")
    q = "How many days per week can employees work remotely?"
    result = service.query(q)
    print(f"  Q: {q}")
    print(f"  A: {result.answer}")
    print(f"  grounded={result.grounded}  citations={len(result.citations)}  "
          f"latency={result.latency_ms}ms")

    print("\n== Evaluation (5 production signals) ==")
    samples = [
        EvalSample(
            question="How many days per week can employees work remotely?",
            expected_keywords=["three", "days"],
        ),
        EvalSample(
            question="What is the home-office stipend?",
            expected_keywords=["500"],
        ),
        EvalSample(
            question="Do you offer refunds?",
            expected_keywords=["14 days", "refund"],
        ),
        EvalSample(
            question="What is the capital of France?",  # out-of-domain
            expected_keywords=[],
        ),
    ]
    report = evaluate(samples, service.query)
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
