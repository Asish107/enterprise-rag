"""End-to-end pipeline tests using the local embedding + extractive fallback.

These run without any API key. They do download a small sentence-transformers
model on first run.
"""
import pytest

from app.config import Settings
from app.eval import evaluate
from app.models import EvalSample
from app.service import RAGService


@pytest.fixture
def service(tmp_path):
    settings = Settings(
        index_dir=tmp_path / "index",
        data_dir=tmp_path / "data",
        anthropic_api_key=None,  # force extractive fallback
    )
    settings.ensure_dirs()
    svc = RAGService(settings)
    doc = tmp_path / "policy.txt"
    doc.write_text(
        "Employees may work remotely up to three days per week. "
        "The home-office stipend is 500 USD."
    )
    svc.ingest_file(doc)
    return svc


def test_query_is_grounded(service):
    resp = service.query("How many days can employees work remotely?")
    assert resp.grounded is True
    assert resp.citations, "expected at least one citation"
    assert resp.citations[0].score > 0
    assert "three days" in resp.citations[0].snippet


def test_out_of_domain_not_grounded(service):
    resp = service.query("What is the boiling point of helium?")
    # Weak retrieval should not fabricate an answer.
    assert resp.grounded is False
    assert resp.citations == []


def test_eval_report(service):
    samples = [
        EvalSample(
            question="How many days remote?",
            ground_truth="Employees may work remotely up to three days per week.",
            expected_keywords=["three"],
        ),
        EvalSample(
            question="What is the stipend?",
            ground_truth="The home-office stipend is 500 USD.",
            expected_keywords=["500"],
        ),
    ]
    report = evaluate(samples, service.query)
    assert report.num_samples == 2
    m = report.metrics
    assert 0.0 <= m.citation_coverage <= 1.0
    assert 0.0 <= m.hallucination_rate <= 1.0
    assert 0.0 <= m.faithfulness <= 1.0
    assert 0.0 <= m.context_recall <= 1.0
    assert m.avg_latency_ms >= 0.0
    assert m.total_cost_usd >= 0.0


def test_dedup(service, tmp_path):
    doc = tmp_path / "dup.txt"
    doc.write_text("Unique content about quarterly revenue growth of 12 percent.")
    first = service.ingest_file(doc)
    second = service.ingest_file(doc)
    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.document_id == first.document_id


def test_list_and_delete_documents(service):
    docs = service.list_documents()
    assert docs, "expected at least the fixture document"
    doc_id = docs[0].document_id
    assert service.delete_document(doc_id) is True
    assert all(d.document_id != doc_id for d in service.list_documents())
