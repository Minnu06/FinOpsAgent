"""CLI entry point: python -m agent.cli "why did EC2 cost go up?" [--provider AWS|Azure]"""

from __future__ import annotations

import json
import sys
from typing import Any

from agent.loop import run_agent
from logging_setup import get_logger, session_scope

_log = get_logger(__name__)


def _print_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
    print(f"\n\033[36m[tool call]\033[0m {name}({json.dumps(args)})")
    print(f"\033[90m[result]\033[0m {json.dumps(result, default=str, indent=2)}")


def main() -> None:
    args = sys.argv[1:]
    provider: str | None = None
    if "--provider" in args:
        idx = args.index("--provider")
        provider = args[idx + 1]
        del args[idx : idx + 2]

    if not args:
        print('Usage: python -m agent.cli "your question" [--provider AWS|Azure]')
        sys.exit(1)

    question = " ".join(args)
    print(f"> {question}" + (f"  [provider={provider}]" if provider else ""))

    with session_scope(label="cli"):
        _log.info('User query: "%s"', question)
        try:
            answer, _history = run_agent(question, on_tool_call=_print_tool_call, provider=provider)
        except Exception as exc:  # noqa: BLE001 - friendly CLI message, not a stack trace
            _log.exception("Agent error")
            print(f"\n\033[31mAgent error:\033[0m {type(exc).__name__}: {exc}")
            print("Check that OPENAI_API_KEY is set (or set LLM_PROVIDER=ollama in .env for a fully local run).")
            sys.exit(1)

        _log.info("Final answer delivered (%d chars)", len(answer))

    print("\n\033[1m=== Answer ===\033[0m")
    print(answer)


if __name__ == "__main__":
    main()
