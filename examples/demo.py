"""End-to-end demo with a fake client — no API key needed.

Run:  python examples/demo.py
"""
from llm_cost_guard import Budget, CostGuard


class FakeUsage:
    prompt_tokens = 1200
    completion_tokens = 300


class FakeResponse:
    model = "gpt-4o-mini"
    usage = FakeUsage()


class FakeCompletions:
    def create(self, *a, **k):
        return FakeResponse()


class FakeChat:
    completions = FakeCompletions()


class FakeClient:
    chat = FakeChat()


def on_alert(budget, spent, threshold):
    print(f"[ALERT] {budget.name}: ${spent:.4f} crossed ${threshold:.4f}")


def main() -> None:
    budget = Budget(limit_usd=0.001, soft_ratio=0.8, on_alert=on_alert, name="demo")
    guard = CostGuard(client=FakeClient(), budget=budget)

    print("call 1-5:")
    for i in range(1, 6):
        try:
            guard.chat.completions.create(model="gpt-4o-mini", messages=[])
            print(f"  call {i}: ok  spent=${guard.total_cost_usd:.5f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  call {i}: BLOCKED -> {exc}")
            break

    print(f"total tokens: {guard.total_tokens}")
    print(f"utilization : {budget.utilization:.1%}")
    print(f"remaining   : ${budget.remaining_usd:.5f}")


if __name__ == "__main__":
    main()
