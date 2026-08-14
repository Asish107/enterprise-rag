# EnterpriseRAG

> Ingestion + retrieval + grounded generation service for enterprise documents — with citations and production evaluation signals.

Built with **FastAPI**, **FAISS**, **LangChain**, and **Docker**. Ingests three
document formats (PDF, DOCX, TXT/Markdown), chunks and embeds them into a FAISS
vector index, runs semantic search, and produces **grounded, cited** answers.
An evaluation endpoint scores five production signals so you can track quality,
latency, and cost over time.

---

## Features

- **Multi-format ingestion** — PDF (`pypdf`), DOCX (`python-docx`), and TXT/MD, with recursive overlapping chunking.
- **Semantic retrieval** — local sentence-transformers embeddings (`all-MiniLM-L6-v2`) indexed in **FAISS**; no API key required.
- **Grounded generation with citations** — answers cite the exact chunks they use. Uses **Claude** when `ANTHROPIC_API_KEY` is set, otherwise a deterministic extractive fallback keeps the pipeline fully runnable offline.
- **Hallucination guard** — out-of-domain questions fall below a similarity threshold and return "not enough information" instead of fabricating.
- **Evaluation harness** — five production signals: retrieval relevance, citation coverage, hallucination rate, latency, and estimated cost.
- **Containerized API** — one-command Docker / Docker Compose deployment with a persisted index and model cache.

## Architecture

```
          ┌─────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐
 upload → │  Ingestion  │ → │ Chunking │ → │  Embeddings  │ → │  FAISS   │
          │ pdf/docx/txt│   │ (LangCh.)│   │ (MiniLM)     │   │  index   │
          └─────────────┘   └──────────┘   └──────────────┘   └────┬─────┘
                                                                    │
 question ───────────────► semantic search (top-k) ◄────────────────┘
                                    │
                                    ▼
                    grounded generation + citations
                    (Claude, or extractive fallback)
                                    │
                                    ▼
                       answer + citations + latency
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

| Method | Path        | Description                                            |
|--------|-------------|--------------------------------------------------------|
| GET    | `/health`   | Liveness check.                                        |
| POST   | `/ingest`   | Upload a PDF/DOCX/TXT/MD file; returns chunk stats.    |
| POST   | `/query`    | Ask a question; returns a grounded, cited answer.      |
| POST   | `/evaluate` | Score a batch of questions on the five signals.        |

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

## Evaluation signals

| Signal                | Meaning                                                        |
|-----------------------|----------------------------------------------------------------|
| Retrieval relevance   | Mean top-1 similarity of retrieved context across queries.     |
| Citation coverage     | Fraction of answers backed by at least one citation.           |
| Hallucination rate    | Fraction of answers not grounded in retrieved context.         |
| Avg latency (ms)      | Mean end-to-end query latency.                                 |
| Estimated cost (USD)  | Order-of-magnitude generation spend based on token estimates.  |

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)). Key ones:

| Variable                | Default                                        | Purpose                          |
|-------------------------|------------------------------------------------|----------------------------------|
| `ANTHROPIC_API_KEY`     | _(unset)_                                      | Enables Claude generation.       |
| `RAG_GENERATION_MODEL`  | `claude-sonnet-5`                              | Generation model.                |
| `RAG_EMBEDDING_MODEL`   | `sentence-transformers/all-MiniLM-L6-v2`       | Embedding model.                 |
| `RAG_TOP_K`             | `4`                                            | Chunks retrieved per query.      |
| `RAG_CHUNK_SIZE`        | `800`                                          | Characters per chunk.            |
| `RAG_CHUNK_OVERLAP`     | `120`                                          | Overlap between chunks.          |

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
│   ├── retrieval/         # embeddings + FAISS vector store
│   ├── generation/        # grounded generation + citations
│   └── eval/              # five production signals
├── tests/                 # pytest suite
├── scripts/demo.py        # end-to-end demo
├── data/                  # sample documents
├── Dockerfile
└── docker-compose.yml
```

## License

MIT
