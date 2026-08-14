"""The tool-calling agent loop. The LLM orchestrates and narrates; it never
computes a dollar figure itself — every number comes from a tool result.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Callable

from agent.llm import get_llm_client
from agent.tool_schemas import REGISTRY, SCHEMAS
from logging_setup import get_logger
from resolvers.dispatch import resolve_and_execute
from tools.finops_tools import data_date_range

_log = get_logger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """You are CloudLens, an autonomous FinOps copilot for AWS and Azure.

Your job: investigate cloud spend, find anomalies and waste, and recommend concrete fixes
with exact dollar savings.

DATE CONTEXT:
This dataset's most recent day with cost data is {max_date} — treat that as "today" for
this conversation. Data is available from {min_date} to {max_date} — {month_span} only, no other
months or years. When the user says "last 30 days", "this month", "recently", "lately", or
asks about "today", interpret it relative to {max_date}, NOT your own training cutoff or
any other assumed date. Never state or imply a current year other than {max_year}. If the
user names a month, quarter, or year outside {month_span} (earlier or later — e.g. "January",
"last quarter", "this year" if that would reach past {max_date}), the relevant tool will
report a "no_data" or empty result rather than a real number for that period; relay that
plainly instead of computing, estimating, or guessing a figure for a period this dataset
doesn't cover. Tools that take dates (cost_trend) default to the last 30 days of available
data when start/end are omitted — prefer omitting them over guessing a date yourself.

INVESTIGATION ORDER (when investigating cost increases or waste):
1. Find the anomaly (detect_spike) or the spend picture (cost_trend).
2. Identify the driver resources (detect_spike returns driver_resource_ids).
3. Check whether those resources are idle (find_idle_resources).
4. Recommend fixes with savings (recommend).

Never call recommend with resource_ids you invented or guessed — only pass IDs that
already appeared in an earlier tool result in this conversation (e.g. detect_spike's
driver_resource_ids or find_idle_resources' idle_resources). If the user asks for savings
or recommendations and no resource IDs are yet in context, call detect_spike and/or
find_idle_resources first to discover real ones — do not skip straight to recommend on a
guess.

SPIKE SCOPE: detect_spike's baseline_daily and spiked_daily are the cost of the specific
service+region combination that is actually spiking (also returned as the service/region
fields alongside them) — not the provider's total spend across every service and region.
When you cite these figures, always name that scope (e.g. "AWS EC2 in us-east-1 rose from
$318.61/day to $534.63/day"), never describe them as "AWS's total cost" or "the account's
daily spend."

TOOL SELECTION — INVENTORY VS WASTE (do not conflate these):
- "what's running", "show our EC2 instances", "what's stopped in Azure", "list resources
  tagged prod", "what do we have in us-east-1" — these are plain inventory questions with
  no idle/waste framing. Use list_resources. It reports each resource's actual `status`
  ("running" or "stopped") and instance type as recorded — it applies NO utilization
  heuristics, so it is cheap and always answers "what currently exists," never "what's
  wasted."
- "what's idle", "what's wasted", "what can we cut", "find unused resources" — these ask
  you to *judge* a resource as waste using a rule (CPU p95 < 5%, unattached volume,
  zero invocations, cold blob). Use find_idle_resources. Do NOT reach for
  find_idle_resources just because a question mentions "running" or "resources" in
  passing — if there's no waste/idle/unused framing, it's an inventory question, so use
  list_resources instead. Most questions about resources are inventory questions, not
  waste questions — do not default to find_idle_resources.

PROVIDER, SERVICE & INSTANCE-TYPE RESOLUTION:
Provider, service, and instance-type names are all resolved deterministically before a
tool call executes — you do not need to work out which cloud a service or instance type
belongs to yourself, and you must NOT silently pick one when the user didn't say. If the
user's own word for a service is ambiguous about which cloud (e.g. "VM", "instance",
"compute", "server", "storage", "bucket", "blob", "function", "serverless", "kubernetes")
and they did not also name a specific cloud, pass that same ambiguous word through as the
`service` argument — even though it is not one of the exact enum values — instead of
guessing a specific concrete service yourself. The same applies to `instance_type`: a
generic size word ("large", "xlarge", "compute optimized", "memory optimized") with no
cloud named should be passed through as-is, not mapped to a specific concrete type or to
the `service` argument — instance_type and service are different filters (e.g. "what
instance type is this EC2 running?" is an instance_type question about a service you
already know is EC2, not a request to re-filter by service). The resolver will ask the
user to clarify if it's genuinely ambiguous; guessing defeats that safeguard. Only pass an
exact concrete value (EC2, Virtual Machine, m5.2xlarge, Standard_D4s_v5, etc.) when the
user's own words already pin down the cloud (they named the concrete value, or named
AWS/Azure explicitly, or context in this conversation already makes it unambiguous). A
tool call may come back with a `status` field instead of real data:
- "clarification_needed": the service or instance-type name could mean more than one
  cloud (e.g. "VM" -> AWS EC2 or Azure Virtual Machine, "large" -> AWS m5.2xlarge or Azure
  Standard_D4s_v5). Relay the listed `options` to the user as a question. Do not guess
  which one they meant, and do not retry the call yourself — wait for the user to answer.
- "invalid_request": the provider/service/instance-type combination is not valid (e.g.
  Azure does not offer EC2, or does not offer instance type "m5.2xlarge"). State this
  plainly; do not retry with a different value or invent data.
- "data_unavailable": the service is real, but this dataset has no cost data for it.
  Say so plainly — do not estimate or fabricate a number.
- "unresolved_service" / "unresolved_instance_type": the name wasn't recognized. Ask the
  user to clarify which service or instance type they mean.
- "invalid_argument": the tool itself rejected its arguments as malformed (an unparseable
  date, a start date after the end date, a non-positive lookback_days, an unrecognized
  status value). State the problem plainly and ask the user for a corrected value — do not
  silently retry with a guessed correction.
If the user names a service or instance type exclusive to one cloud (EC2/EBS/Lambda/
m5.2xlarge -> AWS; Virtual Machine/Azure Functions/Blob Storage/Standard_D4s_v5 and other
Azure-only values -> Azure), its provider is inferred automatically — just pass the value,
no need to also guess the provider. If the user says "cloud", "our bill", "everything", or
names no specific service, omit the provider filter to scan both clouds; cost_trend
returns a `by_provider` breakdown in that case so you don't need a second call to compare
clouds.

CONVERSATION CONTEXT:
This conversation may include earlier turns. Use prior tool results already in the
conversation to answer follow-up questions ("what about Azure?", "how much would that
save?") instead of re-investigating from scratch — but still call a tool if the follow-up
needs a number that hasn't already appeared in this conversation.

TOOL ERRORS AND EMPTY RESULTS:
If a tool result contains an "error" key, the call failed unexpectedly — this is not a
validation rejection, something went wrong executing it. Tell the user plainly that the
request could not be completed right now; do not guess what the data would have shown,
and do not retry the same call more than once. A result with no "status" key is real data
— narrate it normally. A result that DOES have a "status" key is never real data to
narrate, regardless of which tool produced it or what other fields are alongside it —
treat "no_data" / "no_anomaly" exactly like an empty result (state plainly that nothing
was found), and treat "invalid_argument" like the other rejection statuses above (state
the problem, ask for a correction). Never invent or extrapolate a number to fill the gap.

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


def _month_span_label(min_date: date, max_date: date) -> str:
    """Human-readable list of calendar months the dataset actually spans,
    e.g. "April, May, and June 2026" — derived from the real data range each
    time rather than hardcoded, so this stays correct if the dataset is ever
    regenerated with a different window. Spells out the year per-month if the
    span crosses a year boundary (e.g. "November 2024, December 2024,
    January 2025, and February 2025") rather than tagging every month with
    just `max_date.year`, which would misdate the earlier months.
    """
    months: list[tuple[str, int]] = []
    cursor = min_date.replace(day=1)
    last = max_date.replace(day=1)
    while cursor <= last:
        months.append((cursor.strftime("%B"), cursor.year))
        cursor = date(cursor.year + 1, 1, 1) if cursor.month == 12 else date(cursor.year, cursor.month + 1, 1)

    single_year = len({year for _, year in months}) == 1
    names = [name for name, _ in months] if single_year else [f"{name} {year}" for name, year in months]

    if len(names) == 1:
        joined = names[0]
    elif len(names) == 2:
        joined = " and ".join(names)
    else:
        joined = ", ".join(names[:-1]) + f", and {names[-1]}"
    return f"{joined} {max_date.year}" if single_year else joined


def _build_system_prompt(provider: str | None) -> str:
    min_date, max_date = data_date_range()
    content = _SYSTEM_PROMPT_TEMPLATE.format(
        min_date=min_date.isoformat(),
        max_date=max_date.isoformat(),
        max_year=max_date.year,
        month_span=_month_span_label(min_date, max_date),
    )
    if provider:
        content += (
            f"\n\nSESSION SCOPE OVERRIDE: The user has restricted this session to "
            f"provider={provider} only via a UI control. Every tool call is forced to "
            f"provider='{provider}' regardless of what you request — do not discuss the "
            f"other cloud's numbers unless the user explicitly asks about it."
        )
    return content


def _coerce_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Coerce loosely-typed tool arguments (e.g. Ollama returning "30" for an
    integer param, or "" for an omitted optional param) to match the declared
    JSON schema type. OpenAI's structured tool calling rarely needs this;
    smaller local models often do.
    """
    expected_types = _PARAM_TYPES.get(name, {})
    coerced: dict[str, Any] = {}
    for key, value in args.items():
        if value == "":
            # A local model's way of saying "no value" for an optional param —
            # omit it entirely so the tool's own default applies, instead of
            # passing an empty string that fails enum/type validation.
            continue
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


_FALLBACK_NOTICE = (
    "_(Note: OpenAI was unavailable during this conversation, so this answer was "
    "generated using the local Ollama fallback model.)_\n\n"
)


def _finalize_answer(content: str | None, client: Any) -> str:
    """Prepend a visible fallback notice if `client` failed over to Ollama
    mid-conversation. Reuses the existing answer-text return path — the CLI
    and Chainlit UI both already render this string, so no separate UI
    plumbing is needed to surface that a fallback occurred.
    """
    answer = content or ""
    if getattr(client, "used_fallback", False):
        answer = _FALLBACK_NOTICE + answer
    return answer


def run_agent(
    user_message: str,
    on_tool_call: ToolCallHook | None = None,
    max_turns: int = 8,
    provider: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the tool-calling loop for a single user message.

    Returns (final_assistant_text, updated_history). `on_tool_call(name, args, result)`
    fires after each tool executes, purely for UI rendering (a visible step in the
    CLI/Chainlit app) — it never influences the loop's control flow.

    `provider`, if given (e.g. "AWS" or "Azure" from a UI dropdown), hard-forces every
    tool call's `provider` argument to that value — this is a deterministic filter, not
    a suggestion the model can ignore. Omit (or None) to let the model infer scope per
    the system prompt's provider-inference rules and scan both clouds when ambiguous.

    `history`, if given, is the list of prior turns (user/assistant/tool messages,
    excluding the system prompt) returned by an earlier call — pass it back in so
    follow-up questions have context. Omit for a fresh conversation.
    """
    _log.info('User query: "%s" (provider=%s, history_turns=%d)', user_message, provider or "both/inferred", len(history or []))
    client = get_llm_client()
    system_content = _build_system_prompt(provider)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_message})

    for turn in range(1, max_turns + 1):
        _log.debug("Turn %d: calling LLM with %d messages in context", turn, len(messages))
        response = client.chat(messages, tools=SCHEMAS)

        if not response.tool_calls:
            _log.info("Turn %d: final answer (%d chars), no further tool calls", turn, len(response.content or ""))
            messages.append({"role": "assistant", "content": response.content})
            return _finalize_answer(response.content, client), messages[1:]

        _log.info("Turn %d: model requested %d tool call(s): %s", turn, len(response.tool_calls), [tc.name for tc in response.tool_calls])
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
                _log.warning("Model requested unknown tool: %s", tc.name)
                result = {"error": f"Unknown tool: {tc.name}"}
                args = tc.arguments
            else:
                args = _coerce_args(tc.name, tc.arguments)
                if provider and "provider" in _PARAM_TYPES.get(tc.name, {}):
                    if args.get("provider") != provider:
                        _log.debug("Forcing provider=%s on %s (model requested %s)", provider, tc.name, args.get("provider"))
                    args["provider"] = provider
                _log.debug("Calling %s(%s)", tc.name, args)
                try:
                    result = resolve_and_execute(tc.name, args)
                except Exception as exc:  # noqa: BLE001 - fed back to the model, not swallowed
                    _log.warning("Tool %s raised %s: %s", tc.name, type(exc).__name__, exc)
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                else:
                    _log.debug("%s returned: %s", tc.name, json.dumps(result, default=str))
            if on_tool_call:
                on_tool_call(tc.name, args, result)
            messages.append(_tool_message(tc.id, tc.name, result))

    _log.warning("Max turns (%d) reached without a final answer", max_turns)
    return (
        _finalize_answer(
            "I wasn't able to reach a final answer within the tool-call budget. Please try narrowing the question.",
            client,
        ),
        messages[1:],
    )
