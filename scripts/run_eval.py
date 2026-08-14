"""Run the RAGAS-style evaluation against the labeled QA test set.

    python scripts/run_eval.py

Ingests the sample corpus (dedup makes re-runs safe), then scores every labeled
question and prints the aggregate metrics plus a per-question breakdown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.eval import evaluate  # noqa: E402
from app.models import EvalSample  # noqa: E402
from app.service import RAGService  # noqa: E402


def main() -> None:
    service = RAGService(get_settings())

    for name in ["sample_policy.txt", "sample_faq.md"]:
        service.ingest_file(ROOT / "data" / name)

    samples = [
        EvalSample(**row)
        for row in json.loads((ROOT / "data" / "eval" / "qa_test_set.json").read_text())
    ]

    report = evaluate(
        samples,
        service.query,
        embedder=service.retriever.vectors._embeddings,
    )

    print("=" * 60)
    print(f"RAGAS-style evaluation — {report.num_samples} labeled samples")
    print("=" * 60)
    m = report.metrics
    print(f"  faithfulness        : {m.faithfulness:.3f}")
    print(f"  answer_relevancy    : {m.answer_relevancy:.3f}")
    print(f"  answer_correctness  : {m.answer_correctness:.3f}")
    print(f"  context_precision   : {m.context_precision:.3f}")
    print(f"  context_recall      : {m.context_recall:.3f}")
    print(f"  retrieval_relevance : {m.retrieval_relevance:.3f}")
    print(f"  citation_coverage   : {m.citation_coverage:.3f}")
    print(f"  hallucination_rate  : {m.hallucination_rate:.3f}")
    print(f"  avg_latency_ms      : {m.avg_latency_ms:.1f}")
    print(f"  total_cost_usd      : ${m.total_cost_usd:.6f}")
    print()
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
