"""The tool-calling agent loop. The LLM orchestrates and narrates; it never
computes a dollar figure itself — every number comes from a tool result.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from agent.llm import get_llm_client
from agent.tool_schemas import REGISTRY, SCHEMAS

SYSTEM_PROMPT = """You are CloudLens, an autonomous FinOps copilot for AWS and Azure.

Your job: investigate cloud spend, find anomalies and waste, and recommend concrete fixes
with exact dollar savings.

INVESTIGATION ORDER (when investigating cost increases or waste):
1. Find the anomaly (detect_spike) or the spend picture (cost_trend).
2. Identify the driver resources (detect_spike returns driver_resource_ids).
3. Check whether those resources are idle (find_idle_resources).
4. Recommend fixes with savings (recommend).

PROVIDER INFERENCE:
- If the user names an AWS-only service (EC2, EBS, Lambda), filter to provider=AWS.
- If the user names an Azure-only service (Virtual Machine, Azure Functions, Blob Storage),
  filter to provider=Azure.
- If the user says "cloud", "our bill", "everything", or is otherwise ambiguous about which
  cloud, scan BOTH providers (omit the provider filter) and report a per-provider breakdown.
- Never ask the user which cloud to check — infer it or scan both.

HARD RULE — NEVER VIOLATE:
You must never state a dollar figure, percentage, resource count, or resource ID that did
not appear in a tool result. You do not do arithmetic yourself — the tools already computed
exact numbers. If a tool returned nothing or an empty list, say so plainly; do not guess or
extrapolate. Every number you say must be traceable to a tool call in this conversation.

ANSWER FORMAT:
1. One paragraph stating the root cause (or the direct answer if it's a simple lookup).
2. A bulleted list of recommendations, each with its action and monthly saving.
3. The total monthly saving, stated once at the end.
"""

ToolCallHook = Callable[[str, dict[str, Any], dict[str, Any]], None]

_PARAM_TYPES: dict[str, dict[str, str]] = {
    schema["function"]["name"]: {
        key: prop.get("type") for key, prop in schema["function"]["parameters"]["properties"].items()
    }
    for schema in SCHEMAS
}


def _coerce_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Coerce loosely-typed tool arguments (e.g. Ollama returning "30" for an
    integer param) to match the declared JSON schema type. OpenAI's structured
    tool calling rarely needs this; smaller local models often do.
    """
    expected_types = _PARAM_TYPES.get(name, {})
    coerced: dict[str, Any] = {}
    for key, value in args.items():
        expected = expected_types.get(key)
        if expected == "integer" and isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                pass
        elif expected == "array" and isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        coerced[key] = value
    return coerced


def _tool_message(call_id: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(result, default=str),
    }


def run_agent(
    user_message: str,
    on_tool_call: ToolCallHook | None = None,
    max_turns: int = 8,
) -> str:
    """Run the tool-calling loop for a single user message.

    Returns the final assistant text. `on_tool_call(name, args, result)` fires after
    each tool executes, purely for UI rendering (a visible step in the CLI/Chainlit
    app) — it never influences the loop's control flow.
    """
    client = get_llm_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_turns):
        response = client.chat(messages, tools=SCHEMAS)

        if not response.tool_calls:
            return response.content or ""

        messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls],
            }
        )

        for tc in response.tool_calls:
            func = REGISTRY.get(tc.name)
            if func is None:
                result = {"error": f"Unknown tool: {tc.name}"}
            else:
                try:
                    result = func(**_coerce_args(tc.name, tc.arguments))
                except Exception as exc:  # noqa: BLE001 - fed back to the model, not swallowed
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            if on_tool_call:
                on_tool_call(tc.name, tc.arguments, result)
            messages.append(_tool_message(tc.id, tc.name, result))

    return "I wasn't able to reach a final answer within the tool-call budget. Please try narrowing the question."
