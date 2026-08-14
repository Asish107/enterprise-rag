# EnterpriseRAG

> Ingestion + retrieval + grounded generation service for enterprise documents — with citations and production evaluation signals.

Built with **FastAPI**, **FAISS**, **LangChain**, and **Docker**. Ingests three
document formats (PDF, DOCX, TXT/Markdown), chunks and embeds them into a FAISS
vector index, runs semantic search, and produces **grounded, cited** answers.
An evaluation endpoint scores five production signals so you can track quality,
latency, and cost over time.

---

## Features

- **Multi-format ingestion** — PDF (`pypdf`), DOCX (`python-docx`), and TXT/MD, with recursive overlapping chunking and **content-hash de-duplication**.
- **Hybrid retrieval + reranking** — dense **FAISS** vectors (`all-MiniLM-L6-v2`) fused with sparse **BM25** via Reciprocal Rank Fusion, then reordered by a **cross-encoder reranker** (`ms-marco-MiniLM`). Runs fully local, no API key required.
- **Grounded generation with citations** — answers cite the exact chunks they use. Backends auto-select: **OpenRouter** (any model) → **Anthropic** (native Claude) → deterministic extractive fallback so the pipeline stays runnable offline.
- **Hallucination guard** — out-of-domain questions fall below a dense-retrieval confidence threshold and return "not enough information" instead of fabricating.
- **Real token & cost tracking** — actual prompt/completion tokens from the API response, priced per-model into a real USD cost on every query.
- **RAGAS-style evaluation** — a labeled QA test set scored on faithfulness, answer relevancy, answer correctness, context precision/recall, plus operational signals (retrieval relevance, citation coverage, hallucination rate, latency, real cost).
- **Document management** — `GET /documents` and `DELETE /documents/{id}`.
- **Containerized API** — one-command Docker / Docker Compose deployment with a persisted index and model cache.

## Architecture

```
          ┌─────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐
 upload → │  Ingestion  │ → │ Chunking │ → │  Embeddings  │ → │  FAISS   │
          │ pdf/docx/txt│   │ (LangCh.)│   │ (MiniLM)     │   │  index   │
          │  dedup(hash)│   └──────────┘   └──────────────┘   └────┬─────┘
          └─────────────┘                    also indexed in       │
                                                BM25 (sparse) ──────┤
                                                                    │
 question ──►  dense (FAISS) + sparse (BM25)  ──► RRF fusion ◄───────┘
                                    │
                                    ▼
                         cross-encoder rerank (top-k)
                                    │
                                    ▼
                    grounded generation + citations
              (OpenRouter → Anthropic → extractive fallback)
                                    │
                                    ▼
              answer + citations + confidence + latency + token cost
```

## Quickstart (local)

```bash
cd enterprise-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the end-to-end demo (ingests sample docs, queries, prints eval report)
python scripts/demo.py

# Or start the API
uvicorn app.main:app --reload
```

Then open the interactive docs at `http://localhost:8000/docs`.

## Quickstart (Docker)

```bash
docker compose up --build
```

## API

| Method | Path                    | Description                                              |
|--------|-------------------------|----------------------------------------------------------|
| GET    | `/health`               | Liveness check.                                          |
| POST   | `/ingest`               | Upload a PDF/DOCX/TXT/MD file (deduped); returns stats.  |
| GET    | `/documents`            | List ingested documents.                                |
| DELETE | `/documents/{id}`       | Remove a document and its chunks from the index.        |
| POST   | `/query`                | Ask a question; grounded, cited answer + token cost.    |
| POST   | `/evaluate`             | Score labeled QA samples on RAGAS-style + ops signals.  |

### Ingest

```bash
curl -F "file=@data/sample_policy.txt" http://localhost:8000/ingest
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days per week can employees work remotely?"}'
```

```json
{
  "question": "How many days per week can employees work remotely?",
  "answer": "Based on the retrieved context, ...",
  "citations": [
    {"chunk_id": "ab12:0", "filename": "sample_policy.txt", "score": 0.71, "snippet": "..."}
  ],
  "grounded": true,
  "latency_ms": 42.7,
  "model": "extractive-fallback"
}
```

### Evaluate

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"samples": [{"question": "What is the home-office stipend?", "expected_keywords": ["500"]}]}'
```

## Evaluation

Run the harness against the labeled QA set in [`data/eval/qa_test_set.json`](data/eval/qa_test_set.json):

```bash
python scripts/run_eval.py
```

RAGAS-style quality signals (0–1):

| Signal              | Meaning                                                                 |
|---------------------|-------------------------------------------------------------------------|
| Faithfulness        | Are the answer's claims supported by the retrieved context?             |
| Answer relevancy    | Embedding cosine between the question and the generated answer.         |
| Answer correctness  | Token-level F1 between the answer and the ground-truth reference.       |
| Context precision   | Fraction of retrieved chunks that are actually relevant.                |
| Context recall      | Fraction of expected facts present in the retrieved context.            |

Operational signals:

| Signal              | Meaning                                                                 |
|---------------------|-------------------------------------------------------------------------|
| Retrieval relevance | Mean dense-retrieval confidence across queries.                         |
| Citation coverage   | Fraction of answers backed by at least one citation.                    |
| Hallucination rate  | Fraction of answers not grounded in retrieved context.                  |
| Avg latency (ms)    | Mean end-to-end query latency.                                          |
| Total cost (USD)    | **Real** spend computed from actual token usage and per-model pricing.  |

> Faithfulness uses a lexical-support proxy so the harness runs offline/in CI;
> it's a drop-in point for an LLM judge. Answer relevancy uses the local
> embedding model.

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)). Key ones:

| Variable                | Default                                        | Purpose                              |
|-------------------------|------------------------------------------------|--------------------------------------|
| `OPENROUTER_API_KEY`    | _(unset)_                                      | Enables generation via OpenRouter.   |
| `ANTHROPIC_API_KEY`     | _(unset)_                                      | Enables native Claude generation.    |
| `RAG_GENERATION_MODEL`  | `anthropic/claude-sonnet-4`                    | Generation model / OpenRouter slug.  |
| `RAG_EMBEDDING_MODEL`   | `sentence-transformers/all-MiniLM-L6-v2`       | Embedding model.                     |
| `RAG_RERANK`            | `true`                                         | Toggle cross-encoder reranking.      |
| `RAG_RERANK_MODEL`      | `cross-encoder/ms-marco-MiniLM-L-6-v2`         | Reranker model.                      |
| `RAG_TOP_K`             | `4`                                            | Chunks retrieved per query.          |
| `RAG_CHUNK_SIZE`        | `800`                                          | Characters per chunk.                |
| `RAG_CHUNK_OVERLAP`     | `120`                                          | Overlap between chunks.              |

Backend priority: **OpenRouter → Anthropic → extractive fallback**.

## Testing

```bash
pytest
```

The suite covers ingestion, the retrieval/generation pipeline (with the
offline extractive fallback), and the HTTP API.

## Project layout

```
enterprise-rag/
├── app/
│   ├── main.py            # FastAPI app + routes
│   ├── service.py         # orchestration (ingest / query)
│   ├── config.py          # env-driven settings
│   ├── models.py          # pydantic schemas
│   ├── ingestion/         # loaders (pdf/docx/txt) + chunker
│   ├── retrieval/         # corpus, FAISS, BM25, reranker, hybrid retriever
│   ├── generation/        # grounded generation + citations + pricing
│   └── eval/              # RAGAS-style + operational metrics
├── tests/                 # pytest suite
├── scripts/demo.py        # end-to-end demo
├── scripts/run_eval.py    # RAGAS-style evaluation runner
├── data/                  # sample documents + labeled QA test set
├── Dockerfile
└── docker-compose.yml
```

## License

MIT
