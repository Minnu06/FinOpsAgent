"""Phase 5 regression contract for the provider-aware resolver architecture.

Table-driven cases covering the query classes the whole refactor (Phases
1-4) exists to handle correctly: unambiguous resolution, deterministic
clarification for cross-provider concept words, and plain rejection of
impossible provider/service combos — exercised directly against
`resolvers/dispatch.py::resolve_and_execute`, i.e. the exact seam
`agent/loop.py` calls, with no LLM involved (fast, deterministic).

Each row models the *tool-call arguments* an LLM would have produced for the
described natural-language query — resolution starts from those structured
args, not from parsing English, so the "query" field is documentation of
intent, not something exercised directly here.

A couple of full `run_agent` scripted-client cases at the bottom cover the
one thing the resolution-only cases can't: a clarification or rejection
followed by a corrected follow-up turn actually reaching real data.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent.llm import LLMResponse, ToolCall
from agent.loop import run_agent
from resolvers.dispatch import resolve_and_execute

# --- Successes: resolves and reaches the real tool (no `status` key) ---

_SUCCESS_CASES = [
    pytest.param("cost_trend", {"service": "EC2"}, id="EC2 cost"),
    pytest.param("cost_trend", {"provider": "Azure", "service": "Virtual Machine"}, id="Azure VM cost"),
    pytest.param("cost_trend", {"service": "EC2", "region": "us-east-1"}, id="EC2 in us-east-1"),
    pytest.param(
        "cost_trend",
        {"provider": "Azure", "service": "Virtual Machine", "region": "eastus"},
        id="Azure VM in East US",
    ),
    pytest.param("find_idle_resources", {}, id="top idle resources"),
    pytest.param("cost_trend", {"service": "EC2", "environment": "prod"}, id="prod-tagged EC2 cost"),
    pytest.param("cost_trend", {"service": "EC2", "business_unit": "Platform"}, id="Platform business_unit EC2 cost"),
    pytest.param("cost_trend", {}, id="last month's cost (no service, no provider)"),
    pytest.param("cost_trend", {"service": "EC2", "granularity": "day"}, id="daily EC2 trend"),
]


@pytest.mark.parametrize("tool_name, args", _SUCCESS_CASES)
def test_resolves_and_reaches_the_real_tool(tool_name: str, args: dict[str, Any]) -> None:
    result = resolve_and_execute(tool_name, args)
    assert "status" not in result, f"expected real data, got short-circuit: {result}"


def test_scan_both_cost_trend_includes_by_provider_breakdown() -> None:
    result = resolve_and_execute("cost_trend", {})
    assert result["by_provider"]
    assert set(result["by_provider"]) <= {"AWS", "Azure"}


def test_region_filter_actually_narrows_the_result() -> None:
    unfiltered = resolve_and_execute("cost_trend", {"service": "EC2"})
    filtered = resolve_and_execute("cost_trend", {"service": "EC2", "region": "us-east-1"})
    assert filtered["total"] < unfiltered["total"]


def test_environment_filter_actually_narrows_the_result() -> None:
    unfiltered = resolve_and_execute("cost_trend", {"service": "EC2"})
    filtered = resolve_and_execute("cost_trend", {"service": "EC2", "environment": "prod"})
    assert filtered["total"] < unfiltered["total"]


def test_cost_threshold_filtering_is_not_yet_wired_by_design() -> None:
    """"Resources over $100" has no resolver-level representation: `filters`
    on CanonicalRequest is reserved for a later phase (see
    resolvers/canonical_request.py's module docstring) and no tool schema
    exposes a threshold parameter today. Documented here as a known gap
    rather than silently claimed as covered by the "successes" table above.
    """
    from resolvers.canonical_request import CanonicalRequest

    request = CanonicalRequest(provider="AWS", service="EC2", filters={"min_cost_usd": 100})
    assert "filters" not in request.to_kwargs("cost_trend")


# --- Clarifications: ambiguous cross-provider concept word, no explicit provider ---

_CLARIFICATION_CASES = [
    pytest.param({"service": "vm"}, {"AWS EC2", "Azure Virtual Machine"}, id="VM cost"),
    pytest.param({"service": "compute"}, {"AWS EC2", "Azure Virtual Machine"}, id="compute cost"),
    pytest.param({"service": "storage"}, {"AWS S3", "Azure Blob Storage"}, id="storage cost"),
    pytest.param({"service": "functions"}, {"AWS Lambda", "Azure Functions"}, id="functions cost (Lambda-adjacent, no provider)"),
]


@pytest.mark.parametrize("args, expected_options", _CLARIFICATION_CASES)
def test_ambiguous_service_needs_clarification(args: dict[str, Any], expected_options: set[str]) -> None:
    result = resolve_and_execute("cost_trend", args)
    assert result["status"] == "clarification_needed"
    assert set(result["options"]) == expected_options


# --- Rejections: real service, wrong provider for it ---

_REJECTION_CASES = [
    pytest.param({"provider": "Azure", "service": "EC2"}, id="Azure EC2"),
    pytest.param({"provider": "AWS", "service": "Blob Storage"}, id="AWS Blob Storage"),
    pytest.param({"provider": "Azure", "service": "Lambda"}, id="Azure Lambda"),
]


@pytest.mark.parametrize("args", _REJECTION_CASES)
def test_impossible_combo_is_rejected(args: dict[str, Any]) -> None:
    result = resolve_and_execute("cost_trend", args)
    assert result["status"] == "invalid_request"


# --- Multi-turn: clarification/rejection followed by a corrected follow-up ---


class _ScriptedFollowupClient:
    """First turn asks for an ambiguous/impossible service; once the tool
    result (short-circuit or real data) comes back, produces a natural
    language answer. A second `run_agent` call (fresh instance, `history`
    threaded through per the real UI/CLI flow) then issues a corrected,
    concrete follow-up tool call.
    """

    def __init__(self, first_args: dict[str, Any], corrected_args: dict[str, Any]) -> None:
        self._first_args = first_args
        self._corrected_args = corrected_args
        self._step = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self._step += 1
        last = messages[-1]
        if last["role"] == "user" and last["content"] == "corrected":
            return LLMResponse(
                content=None, tool_calls=[ToolCall(id="2", name="cost_trend", arguments=self._corrected_args)]
            )
        if last["role"] == "tool":
            result = json.loads(last["content"])
            if "status" in result:
                return LLMResponse(content=f"Please clarify: {result['message']}", tool_calls=[])
            return LLMResponse(content=f"total=${result['total']}", tool_calls=[])
        return LLMResponse(content=None, tool_calls=[ToolCall(id="1", name="cost_trend", arguments=self._first_args)])


def test_clarification_then_corrected_followup_reaches_real_data(monkeypatch) -> None:
    client = _ScriptedFollowupClient(
        first_args={"service": "vm"}, corrected_args={"provider": "AWS", "service": "EC2"}
    )
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: client)

    captured: list[dict[str, Any]] = []
    first_answer, history = run_agent(
        "show me VM cost", on_tool_call=lambda n, a, r: captured.append(r), max_turns=2
    )
    assert "AWS EC2" in first_answer and "Azure Virtual Machine" in first_answer
    assert captured[0]["status"] == "clarification_needed"

    second_answer, _ = run_agent(
        "corrected", history=history, on_tool_call=lambda n, a, r: captured.append(r), max_turns=2
    )
    assert "status" not in captured[1]
    assert "total=$" in second_answer


def test_rejection_then_corrected_followup_reaches_real_data(monkeypatch) -> None:
    client = _ScriptedFollowupClient(
        first_args={"provider": "Azure", "service": "EC2"},
        corrected_args={"provider": "Azure", "service": "Virtual Machine"},
    )
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: client)

    captured: list[dict[str, Any]] = []
    first_answer, history = run_agent(
        "show me Azure EC2 cost", on_tool_call=lambda n, a, r: captured.append(r), max_turns=2
    )
    assert captured[0]["status"] == "invalid_request"

    second_answer, _ = run_agent(
        "corrected", history=history, on_tool_call=lambda n, a, r: captured.append(r), max_turns=2
    )
    assert "status" not in captured[1]
    assert "total=$" in second_answer
