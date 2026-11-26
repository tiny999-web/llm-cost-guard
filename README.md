# llm-cost-guard

**Stop surprise LLM bills.** A drop-in guard that wraps any OpenAI-compatible client, tracks per-call token cost, enforces a budget, and fires alerts before you blow past it.

[![CI](https://github.com/tiny999-web/llm-cost-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/tiny999-web/llm-cost-guard/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://img.shields.io/badge/python-3.9%2B-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

```python
from llm_cost_guard import CostGuard, Budget, Pricing

budget = Budget(limit_usd=5.0, soft_ratio=0.8, on_alert=print_alert)
guard = CostGuard(client, budget=budget)

# Real token counts, pulled from the response — not estimated
resp = guard.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)

print(guard.total_cost_usd)   # what you've spent so far
print(guard.total_tokens)     # total tokens across all calls
```

## Why

Your LLM bill is a *runtime* cost with no default guardrail. Existing tools assume you already instrumented everything. `llm-cost-guard` is **30 lines to integrate** and works against the real `openai` SDK (or anything with a `usage` attribute).

- **Real usage, not estimates** — token counts come from the actual API response.
- **Hard budget enforcement** — raises `BudgetExceeded` the moment you cross the limit, and keeps raising on every later call.
- **Soft threshold alerts** — one callback at e.g. 80% so you know before it's too late.
- **Per-model pricing** — ships with a table; add your own models in two lines.
- **Zero dependencies** — core is pure Python stdlib.

## Install

```bash
pip install llm-cost-guard
# or from source
git clone https://github.com/tiny999-web/llm-cost-guard && cd llm-cost-guard
pip install -e .
```

## CLI

```bash
# Estimate the cost of one call
llm-cost-guard price gpt-4o --in 1000000 --out 1000000
# -> {"cost_usd": 12.5, ...}

# List known pricing (USD / 1M tokens)
llm-cost-guard models
```

## Breach policies

| `on_breach` | Behaviour |
|---|---|
| `"raise"` *(default)* | Raise `BudgetExceeded` when the hard limit is crossed |
| `"block"` | Same, plus short-circuit every subsequent call |
| `"allow"` | Record and alert only — never block (observability mode) |

## Per-tenant budgets

```python
from llm_cost_guard import Budget

tenant = Budget(limit_usd=0.50, name="acme-corp", on_alert=notify_ops)
guard = CostGuard(client, budget=tenant)
```

## Pro (paid)

The core is free forever (Apache-2.0). **Pro** adds:
- Team dashboard + per-user/per-key budget scoping
- Cost attribution tags (`user_id`, `feature`) with rollup reports
- Slack/Teams/PagerDuty alert sinks
- Prometheus / OpenTelemetry export

See [PRO.md](PRO.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Contributing

```bash
pip install -e '.[dev]'
pytest
```

Issues and PRs welcome.