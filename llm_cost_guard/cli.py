"""CLI: estimate cost, inspect pricing, and validate a budget config.

Examples:
    llm-cost-guard price gpt-4o --in 1000 --out 500
    llm-cost-guard models
"""

from __future__ import annotations

import argparse
import json
import sys

from .pricing import DEFAULT_PRICING, Pricing


def _cmd_price(args: argparse.Namespace) -> int:
    pricing = Pricing()
    cost = pricing.cost_usd(args.model, args.in_tokens, args.out_tokens)
    print(json.dumps({
        "model": args.model,
        "prompt_tokens": args.in_tokens,
        "completion_tokens": args.out_tokens,
        "cost_usd": round(cost, 6),
        "known_model": pricing.known(args.model),
    }, indent=2))
    return 0


def _cmd_models(_args: argparse.Namespace) -> int:
    print(json.dumps(DEFAULT_PRICING, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-cost-guard",
        description="Estimate and guard LLM API costs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_price = sub.add_parser("price", help="Estimate the USD cost of one call.")
    p_price.add_argument("model", help="Model name, e.g. gpt-4o")
    p_price.add_argument("--in", dest="in_tokens", type=int, default=0, help="prompt tokens")
    p_price.add_argument("--out", dest="out_tokens", type=int, default=0, help="completion tokens")
    p_price.set_defaults(func=_cmd_price)

    p_models = sub.add_parser("models", help="List known model pricing (USD / 1M tokens).")
    p_models.set_defaults(func=_cmd_models)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())