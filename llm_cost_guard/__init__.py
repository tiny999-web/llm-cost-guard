"""llm-cost-guard: track, cap and alert on LLM API costs.

Public API:
    CostGuard  -- the main entry point; wraps an OpenAI-compatible client.
    Budget     -- a spend limit with soft/hard thresholds and alert hooks.
    Usage      -- a single recorded LLM call (tokens + cost).
    Pricing    -- per-model token pricing table.
"""

from .budget import Budget, BudgetExceeded
from .guard import CostGuard
from .pricing import Pricing
from .usage import Usage

__all__ = ["CostGuard", "Budget", "BudgetExceeded", "Usage", "Pricing"]
__version__ = "0.1.0"