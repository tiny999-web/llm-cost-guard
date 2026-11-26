"""Per-model token pricing and cost computation.

Prices are USD per 1M tokens (matching OpenAI's public pricing convention).
Override or add models freely via ``Pricing(custom={...})``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# USD per 1,000,000 tokens. Update as providers change pricing.
# Values are illustrative defaults -- always confirm with your provider.
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}

_PER = 1_000_000.0


@dataclass
class Pricing:
    """Resolve token counts into a USD cost.

    Args:
        custom: Extra/override model pricing, merged over ``DEFAULT_PRICING``.
        default_input: Fallback input price (USD/1M) for unknown models.
        default_output: Fallback output price (USD/1M) for unknown models.
        strict: If True, raise on unknown models instead of using fallbacks.
    """

    custom: Optional[dict[str, dict[str, float]]] = None
    default_input: float = 1.00
    default_output: float = 2.00
    strict: bool = False

    def __post_init__(self) -> None:
        self._table: dict[str, dict[str, float]] = dict(DEFAULT_PRICING)
        if self.custom:
            self._table.update(self.custom)

    def cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Return the USD cost of one call.

        Raises:
            KeyError: If ``strict`` and ``model`` is not in the table.
        """
        rates = self._table.get(model)
        if rates is None:
            if self.strict:
                raise KeyError(f"Unknown model: {model!r}")
            rates = {"input": self.default_input, "output": self.default_output}
        cost = (
            prompt_tokens * rates.get("input", 0.0) / _PER
            + completion_tokens * rates.get("output", 0.0) / _PER
        )
        return cost

    def known(self, model: str) -> bool:
        return model in self._table

    def add(self, model: str, input_per_1m: float, output_per_1m: float) -> None:
        self._table[model] = {"input": input_per_1m, "output": output_per_1m}