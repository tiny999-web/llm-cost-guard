"""Tests for the llm-cost-guard core (pricing, budget, guard, cli)."""

import json

import pytest

from llm_cost_guard import Budget, BudgetExceeded, CostGuard, Pricing
from llm_cost_guard.cli import main as cli_main


# ---------------------------------------------------------------- pricing
def test_known_model_cost():
    p = Pricing()
    # 1M in + 1M out on gpt-4o -> 2.50 + 10.00
    assert p.cost_usd("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.50)


def test_small_call_cost():
    p = Pricing()
    cost = p.cost_usd("gpt-4o-mini", 1000, 2000)
    # 1000*0.15/1e6 + 2000*0.60/1e6 = 0.00015 + 0.0012 = 0.00135
    assert cost == pytest.approx(0.00135)


def test_unknown_model_fallback():
    p = Pricing(default_input=1.0, default_output=2.0)
    assert p.cost_usd("my-custom-model", 1_000_000, 1_000_000) == pytest.approx(3.0)


def test_unknown_model_strict_raises():
    p = Pricing(strict=True)
    with pytest.raises(KeyError):
        p.cost_usd("nope", 10, 10)


def test_custom_override():
    p = Pricing(custom={"gpt-4o": {"input": 1.0, "output": 1.0}})
    assert p.cost_usd("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(2.0)


def test_add_model():
    p = Pricing()
    p.add("brand-new", 5.0, 5.0)
    assert p.known("brand-new")
    assert p.cost_usd("brand-new", 1_000_000, 0) == pytest.approx(5.0)


def test_embedding_output_is_free():
    p = Pricing()
    # embeddings bill only input
    assert p.cost_usd("text-embedding-3-small", 100, 999) == pytest.approx(
        100 * 0.02 / 1_000_000
    )


# ---------------------------------------------------------------- budget
def test_budget_soft_alert_fires_once():
    calls = []
    b = Budget(limit_usd=10.0, soft_ratio=0.8, on_alert=lambda *a: calls.append(a))
    b.record(5.0)   # 50% -- no alert
    assert calls == []
    b.record(3.0)   # 80% -- soft alert
    assert len(calls) == 1
    assert calls[0][2] == pytest.approx(8.0)
    b.record(0.5)    # 85% -- no repeat
    assert len(calls) == 1


def test_budget_hard_breach_raises():
    b = Budget(limit_usd=10.0)
    b.record(9.0)
    with pytest.raises(BudgetExceeded) as ei:
        b.record(2.0)
    assert ei.value.limit_usd == 10.0
    assert ei.value.spent_usd == pytest.approx(11.0)


def test_budget_trackers_utilization_and_remaining():
    b = Budget(limit_usd=10.0)
    b.record(4.0)
    assert b.utilization == pytest.approx(0.4)
    assert b.remaining_usd == pytest.approx(6.0)


def test_budget_reset():
    b = Budget(limit_usd=10.0)
    b.record(9.0)
    with pytest.raises(BudgetExceeded):
        b.record(2.0)  # 11.0 >= 10.0 -> breach
    b.reset()
    b.record(1.0)  # works again after reset
    assert b.spent_usd == pytest.approx(1.0)


def test_budget_invalid_limit():
    with pytest.raises(ValueError):
        Budget(limit_usd=0)


def test_budget_negative_cost_rejected():
    b = Budget(limit_usd=10.0)
    with pytest.raises(ValueError):
        b.record(-1.0)


# ---------------------------------------------------------------- guard
def test_guard_record_only_mode():
    g = CostGuard(budget=Budget(limit_usd=10.0))
    u = g.record("gpt-4o-mini", 1000, 2000)
    assert u.cost_usd == pytest.approx(0.00135)
    assert g.total_tokens == 3000
    assert g.total_cost_usd == pytest.approx(0.00135)


def test_guard_raises_on_breach():
    g = CostGuard(budget=Budget(limit_usd=0.001))
    with pytest.raises(BudgetExceeded):
        g.record("gpt-4o", 1000, 1000)  # ~$0.0125 >> $0.001 limit
    assert g.blocked is True
    with pytest.raises(BudgetExceeded):
        g.record("gpt-4o", 10, 10)  # blocked on every later call too


def test_guard_wrap_records_real_usage():
    class _U:
        prompt_tokens = 100
        completion_tokens = 50

    class _R:
        model = "gpt-4o"
        usage = _U()

    calls = []
    def fake_create(*a, **k):
        calls.append((a, k))
        return _R()

    g = CostGuard(budget=Budget(limit_usd=100.0))
    resp = g.wrap(fake_create, model="gpt-4o")()
    assert resp is not None
    assert g.total_tokens == 150
    assert calls[0][0] == ()  # no extra positional args injected


def test_guard_wrap_handles_dict_response():
    g = CostGuard()
    g.wrap(lambda: {"model": "gpt-4o", "usage": {"prompt_tokens": 10, "completion_tokens": 20}},
           model="gpt-4o")()
    assert g.total_tokens == 30


def test_guard_on_usage_hook():
    seen = []
    g = CostGuard(on_usage=lambda u: seen.append(u.model))
    g.record("gpt-4o", 10, 10)
    assert seen == ["gpt-4o"]


def test_guard_chat_passthrough_records():
    class _U:
        prompt_tokens = 20
        completion_tokens = 30

    class _Completions:
        def create(self, *a, **k):
            return _R()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    class _R:
        model = "gpt-4o"
        usage = _U()

    g = CostGuard(client=_Client(), budget=Budget(limit_usd=100.0))
    g.chat.completions.create(model="gpt-4o", messages=[])
    assert g.total_tokens == 50


def test_guard_rejects_bad_breach_mode():
    with pytest.raises(ValueError):
        CostGuard(on_breach="nope")


# ---------------------------------------------------------------- cli
def test_cli_price(capsys):
    rc = cli_main(["price", "gpt-4o", "--in", "1000000", "--out", "1000000"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cost_usd"] == pytest.approx(12.50)


def test_cli_models(capsys):
    rc = cli_main(["models"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "gpt-4o" in data