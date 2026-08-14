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
        EvalSample(question="How many days remote?", expected_keywords=["three"]),
        EvalSample(question="What is the stipend?", expected_keywords=["500"]),
    ]
    report = evaluate(samples, service.query)
    assert report.num_samples == 2
    assert 0.0 <= report.citation_coverage <= 1.0
    assert 0.0 <= report.hallucination_rate <= 1.0
    assert report.avg_latency_ms >= 0.0
