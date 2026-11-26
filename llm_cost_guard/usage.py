"""A single recorded LLM call."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Optional


@dataclass
class Usage:
    """One LLM call's token usage and computed cost.

    Attributes:
        model: Model identifier, e.g. ``gpt-4o``.
        prompt_tokens: Tokens in the prompt.
        completion_tokens: Tokens in the completion.
        cost_usd: Computed cost in USD for this call.
        ts: Unix timestamp of the call.
        meta: Optional free-form metadata (user id, request id, ...).
    """

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    ts: float = field(default_factory=time)
    meta: Optional[dict] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "ts": self.ts,
            "meta": self.meta,
        }