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
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class GenerationResult:
    answer: str
    grounded: bool
    model: str
    usage: TokenUsage = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.usage is None:
            self.usage = TokenUsage()


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


def _usage(model: str, prompt_tokens: int, completion_tokens: int) -> TokenUsage:
    from .pricing import cost_usd

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost_usd(model, prompt_tokens, completion_tokens),
    )


def _generate_openrouter(
    prompt: str, *, api_key: str, base_url: str, model: str, max_tokens: int
) -> tuple[str, TokenUsage]:
    """Grounded generation via OpenRouter's OpenAI-compatible chat API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    answer = (completion.choices[0].message.content or "").strip()
    u = completion.usage
    usage = _usage(model, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
    return answer, usage


def _generate_anthropic(
    prompt: str, *, api_key: str, model: str, max_tokens: int
) -> tuple[str, TokenUsage]:
    """Grounded generation via the native Anthropic Messages API."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = "".join(b.text for b in message.content if b.type == "text").strip()
    usage = _usage(model, message.usage.input_tokens, message.usage.output_tokens)
    return answer, usage


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    confidence: float,
    anthropic_api_key: str | None = None,
    openrouter_api_key: str | None = None,
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    model: str,
    max_tokens: int,
) -> GenerationResult:
    # Grounding uses the absolute dense-retrieval confidence, not the reranker's
    # relative per-candidate score (whose top value is near-constant).
    grounded = bool(chunks) and confidence >= GROUNDING_THRESHOLD

    if not grounded:
        return GenerationResult(
            answer=(
                "I don't have enough information in the knowledge base to answer "
                "that question confidently."
            ),
            grounded=False,
            model="none",
        )

    prompt = f"Context passages:\n{_format_context(chunks)}\n\nQuestion: {question}"

    # Backend priority: OpenRouter -> Anthropic -> extractive fallback. If a
    # configured LLM backend errors (bad key, rate limit, network), we degrade
    # to the extractive answer rather than failing the request outright.
    if openrouter_api_key:
        try:
            answer, usage = _generate_openrouter(
                prompt,
                api_key=openrouter_api_key,
                base_url=openrouter_base_url,
                model=model,
                max_tokens=max_tokens,
            )
            return GenerationResult(
                answer=answer, grounded=True, model=f"openrouter/{model}", usage=usage
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            return GenerationResult(
                answer=_extractive_answer(question, chunks),
                grounded=True,
                model=f"extractive-fallback (openrouter error: {type(exc).__name__})",
            )

    if anthropic_api_key:
        try:
            answer, usage = _generate_anthropic(
                prompt, api_key=anthropic_api_key, model=model, max_tokens=max_tokens
            )
            return GenerationResult(answer=answer, grounded=True, model=model, usage=usage)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            return GenerationResult(
                answer=_extractive_answer(question, chunks),
                grounded=True,
                model=f"extractive-fallback (anthropic error: {type(exc).__name__})",
            )

    return GenerationResult(
        answer=_extractive_answer(question, chunks),
        grounded=True,
        model="extractive-fallback",
    )
