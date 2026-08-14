"""Model pricing and cost computation from real token usage.

Prices are USD per 1M tokens (input, output). Values are approximate public
list prices and easy to update. Unknown models fall back to a conservative
default so cost is estimated, never crashing the request.
"""
from __future__ import annotations

# (input_per_1m, output_per_1m) in USD. Keys are matched by substring so both
# bare ids ("claude-sonnet-4") and OpenRouter slugs ("anthropic/claude-sonnet-4")
# resolve to the same entry.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3.5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3.5-sonnet": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-opus": (15.00, 75.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}

_DEFAULT = (3.00, 15.00)  # conservative fallback


def price_for(model: str) -> tuple[float, float]:
    m = model.lower()
    for key, price in _PRICES.items():
        if key in m:
            return price
    return _DEFAULT


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = price_for(model)
    return round(
        (prompt_tokens / 1_000_000) * in_rate
        + (completion_tokens / 1_000_000) * out_rate,
        6,
    )
