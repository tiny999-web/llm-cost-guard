"""Spend budgets with soft/hard thresholds and alert hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional


class BudgetExceeded(RuntimeError):
    """Raised (or signalled) when a hard budget limit is crossed."""

    def __init__(self, message: str, spent_usd: float, limit_usd: float) -> None:
        super().__init__(message)
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd


AlertFn = Callable[["Budget", float, float], None]
"""Signature: (budget, spent_usd, threshold_usd) -> None"""


@dataclass
class Budget:
    """A spending limit for one scope (global, per-key, per-user, ...).

    Args:
        limit_usd: Hard ceiling in USD. Crossing it raises ``BudgetExceeded``.
        soft_ratio: Fraction of ``limit_usd`` (0-1] that triggers a soft alert.
        on_alert: Callback fired once per threshold when crossed.
        name: Optional label for logs/metrics.
    """

    limit_usd: float
    soft_ratio: float = 0.8
    on_alert: Optional[AlertFn] = None
    name: str = "default"
    spent_usd: float = field(default=0.0, init=False)
    _soft_fired: bool = field(default=False, init=False)
    _hard_fired: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.limit_usd <= 0:
            raise ValueError("limit_usd must be > 0")
        if not (0 < self.soft_ratio <= 1.0):
            raise ValueError("soft_ratio must be in (0, 1]")

    @property
    def soft_limit_usd(self) -> float:
        return self.limit_usd * self.soft_ratio

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    @property
    def utilization(self) -> float:
        return self.spent_usd / self.limit_usd

    def record(self, cost_usd: float) -> None:
        """Add a cost and fire any newly-crossed thresholds.

        Returns normally on soft alert; raises ``BudgetExceeded`` on hard breach.
        """
        if cost_usd < 0:
            raise ValueError("cost_usd must be >= 0")
        self.spent_usd += cost_usd

        if not self._soft_fired and self.spent_usd >= self.soft_limit_usd:
            self._soft_fired = True
            if self.on_alert:
                self.on_alert(self, self.spent_usd, self.soft_limit_usd)

        # Once over the hard limit, stay over: every later record raises too.
        if self.spent_usd >= self.limit_usd:
            if not self._hard_fired:
                self._hard_fired = True
                if self.on_alert:
                    self.on_alert(self, self.spent_usd, self.limit_usd)
            raise BudgetExceeded(
                f"Budget {self.name!r} exceeded: "
                f"${self.spent_usd:.4f} >= ${self.limit_usd:.4f}",
                spent_usd=self.spent_usd,
                limit_usd=self.limit_usd,
            )

    def reset(self) -> None:
        self.spent_usd = 0.0
        self._soft_fired = False
        self._hard_fired = False


@dataclass
class BudgetRegistry:
    """Manage several named budgets (e.g. one per tenant)."""

    _budgets: dict = field(default_factory=dict)

    def add(self, budget: Budget) -> Budget:
        self._budgets[budget.name] = budget
        return budget

    def get(self, name: str) -> Optional[Budget]:
        return self._budgets.get(name)

    def all(self) -> List[Budget]:
        return list(self._budgets.values())