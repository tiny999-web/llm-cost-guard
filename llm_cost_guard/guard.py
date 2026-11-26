"""CostGuard: wrap an OpenAI-compatible client and guard every call.

Works with the official ``openai`` SDK (v1) or any object exposing
``chat.completions.create(...)`` that returns a response with a
``usage`` attribute (``prompt_tokens`` / ``completion_tokens``).

If the ``openai`` package is not installed, you can still use
``CostGuard.record(...)`` to track usage you obtained elsewhere.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from .budget import Budget, BudgetExceeded
from .pricing import Pricing
from .usage import Usage


def _read_field(obj: Any, name: str) -> Any:
    """Read ``name`` from an object (attribute) or a mapping (key)."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)

class CostGuard:
    """Wrap an LLM client to record usage and enforce a budget.

    Args:
        client: An OpenAI-compatible client (or None to use record-only mode).
        budget: A :class:`Budget` to enforce. If None, usage is tracked but not capped.
        pricing: Pricing table; defaults to :class:`Pricing`.
        on_usage: Optional callback fired with each :class:`Usage` (for logging/metrics).
        on_breach: ``"raise"`` (default) raises ``BudgetExceeded``; ``"block"``
            short-circuits future calls once the hard limit is hit; ``"allow"``
            only records and alerts but never blocks.

    Example:
        >>> guard = CostGuard(client, budget=Budget(limit_usd=5.0))
        >>> resp = guard.chat.completions.create(model="gpt-4o", messages=[...])
    """

    def __init__(
        self,
        client: Any = None,
        budget: Optional[Budget] = None,
        pricing: Optional[Pricing] = None,
        on_usage: Optional[Callable[[Usage], None]] = None,
        on_breach: str = "raise",
    ) -> None:
        if on_breach not in ("raise", "block", "allow"):
            raise ValueError("on_breach must be one of: raise, block, allow")
        self.client = client
        self.budget = budget
        self.pricing = pricing or Pricing()
        self.on_usage = on_usage
        self.on_breach = on_breach
        self.history: List[Usage] = []
        self._blocked = False

    # -- public helpers -------------------------------------------------
    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.history)

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.history)

    @property
    def blocked(self) -> bool:
        return self._blocked

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        meta: Optional[dict] = None,
    ) -> Usage:
        """Record one call's usage, update budget, return the Usage.

        Raises ``BudgetExceeded`` when a hard limit is crossed and
        ``on_breach == "raise"``.
        """
        cost = self.pricing.cost_usd(model, prompt_tokens, completion_tokens)
        usage = Usage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            meta=meta,
        )
        self.history.append(usage)
        if self.on_usage:
            self.on_usage(usage)
        if self.budget is not None:
            if self.on_breach == "block" and self._blocked:
                raise BudgetExceeded(
                    f"Budget {self.budget.name!r} already breached; call blocked.",
                    spent_usd=self.budget.spent_usd,
                    limit_usd=self.budget.limit_usd,
                )
            try:
                self.budget.record(cost)
            except BudgetExceeded:
                if self.on_breach in ("raise", "block"):
                    self._blocked = True
                raise
        return usage

    def extract_usage(self, response: Any) -> tuple[str, int, int]:
        """Pull (model, prompt_tokens, completion_tokens) from a SDK response.

        Handles both attribute-style responses (the ``openai`` SDK) and plain
        dicts (proxy clients returning raw JSON).
        """
        model = _read_field(response, "model") or "unknown"
        usage_obj = _read_field(response, "usage")
        if usage_obj is None:
            return model, 0, 0
        prompt = int(_read_field(usage_obj, "prompt_tokens") or 0)
        completion = int(_read_field(usage_obj, "completion_tokens") or 0)
        return model, prompt, completion

    def wrap(self, call: Callable[..., Any], model: str, meta: Optional[dict] = None):
        """Call ``call()`` (an LLM request thunk) and record its usage.

        Usage is extracted from the response after it succeeds, so real
        token counts are captured rather than estimated.
        """

        def _inner(*args: Any, **kwargs: Any) -> Any:
            response = call(*args, **kwargs)
            m, prompt, completion = self.extract_usage(response)
            self.record(m or model, prompt, completion, meta=meta)
            return response

        return _inner

    # -- convenience passthrough ---------------------------------------
    @property
    def chat(self) -> "_GuardedNamespace":
        return _GuardedNamespace(self, ("chat",))


class _GuardedNamespace:
    """Lazy proxy that records usage around ``chat.completions.create``."""

    def __init__(self, guard: CostGuard, path: tuple) -> None:
        self._guard = guard
        self._path = path

    def __getattr__(self, name: str) -> Any:
        new_path = self._path + (name,)
        if name == "create":
            guard = self._guard

            def create(model: str, *args: Any, **kwargs: Any) -> Any:
                target = guard.client
                for part in ("chat", "completions", "create"):
                    target = getattr(target, part)
                return guard.wrap(target, model)(*args, **kwargs)

            return create
        return _GuardedNamespace(self._guard, new_path)