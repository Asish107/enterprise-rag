"""Grounded answer generation with inline citations.

If an Anthropic API key is configured the service asks Claude to answer *only*
from the retrieved context and cite chunks by ``[n]``. Without a key it falls
back to a deterministic extractive answer that stitches together the most
relevant chunks — this keeps the full pipeline runnable offline and in CI while
still exercising retrieval, citation, and grounding logic.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.retrieval import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a precise enterprise assistant. Answer the user's question using "
    "ONLY the provided context passages. Cite every claim with bracketed "
    "indices like [1] or [2] that refer to the numbered passages. If the "
    "context does not contain the answer, say you don't have enough "
    "information. Never invent facts that are not in the context."
)

# A retrieved chunk whose cosine similarity is below this is treated as too weak
# to ground an answer — the query is considered out-of-domain.
GROUNDING_THRESHOLD = 0.35


@dataclass
class GenerationResult:
    answer: str
    grounded: bool
    model: str


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {c.filename})\n{c.text}" for i, c in enumerate(chunks)
    )


def _extractive_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Deterministic fallback: summarize the top passages with citations."""
    lines = [
        "Based on the retrieved context, the most relevant passages are:",
        "",
    ]
    for i, c in enumerate(chunks):
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:317] + "..."
        lines.append(f"[{i + 1}] {snippet}")
    return "\n".join(lines)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    api_key: str | None,
    model: str,
    max_tokens: int,
) -> GenerationResult:
    grounded = bool(chunks) and chunks[0].score >= GROUNDING_THRESHOLD

    if not grounded:
        return GenerationResult(
            answer=(
                "I don't have enough information in the knowledge base to answer "
                "that question confidently."
            ),
            grounded=False,
            model="none",
        )

    if not api_key:
        return GenerationResult(
            answer=_extractive_answer(question, chunks),
            grounded=True,
            model="extractive-fallback",
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    context = _format_context(chunks)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context passages:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    answer = "".join(block.text for block in message.content if block.type == "text")
    return GenerationResult(answer=answer.strip(), grounded=True, model=model)
