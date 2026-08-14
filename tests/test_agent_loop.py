"""Verifies the multi-turn tool-calling mechanics of agent/loop.py using a
scripted fake LLM client — independent of any real model's routing quality.
Confirms detect_spike -> find_idle_resources -> recommend chains correctly
and that tool results actually flow between turns.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from datetime import date

from agent.llm import LLMResponse, ToolCall, _FallbackLLMClient
from agent.loop import _build_system_prompt, _month_span_label, run_agent


class _ScriptedClient:
    def __init__(self) -> None:
        self._step = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self._step += 1
        last_tool_result = json.loads(messages[-1]["content"]) if messages[-1]["role"] == "tool" else None

        if self._step == 1:
            return LLMResponse(content=None, tool_calls=[ToolCall(id="1", name="detect_spike", arguments={})])
        if self._step == 2:
            driver_ids = last_tool_result["driver_resource_ids"]
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="2", name="find_idle_resources", arguments={"resource_ids": driver_ids})],
            )
        if self._step == 3:
            idle_ids = [r["resource_id"] for r in last_tool_result["idle_resources"]]
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="3", name="recommend", arguments={"resource_ids": idle_ids})],
            )
        return LLMResponse(
            content=f"Total monthly saving: ${last_tool_result['total_monthly_saving_usd']}", tool_calls=[]
        )


def test_agent_loop_chains_detect_spike_idle_recommend(monkeypatch):
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: _ScriptedClient())

    calls: list[str] = []

    def on_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        calls.append(name)

    answer, history = run_agent("why did cost go up?", on_tool_call=on_tool_call)

    assert calls == ["detect_spike", "find_idle_resources", "recommend"]
    assert "Total monthly saving: $6634.0" == answer
    assert history[0] == {"role": "user", "content": "why did cost go up?"}
    assert history[-1] == {"role": "assistant", "content": answer}


class _WrongProviderClient:
    """A model that ignores the session's provider scope and asks for the other cloud."""

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="detect_spike", arguments={"provider": "Azure"})],
        )


def test_provider_override_wins_over_model_request(monkeypatch):
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: _WrongProviderClient())

    executed_args: list[dict[str, Any]] = []

    def on_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        executed_args.append(args)

    run_agent("why did cost go up?", on_tool_call=on_tool_call, provider="AWS", max_turns=1)

    assert executed_args[0]["provider"] == "AWS"


class _HistoryInspectingClient:
    """Records a snapshot of the messages list it was called with, to verify
    history threading. Must copy, not just reference, `messages` — run_agent
    keeps mutating that same list object after chat() returns.
    """

    def __init__(self) -> None:
        self.seen_messages: list[dict[str, Any]] | None = None

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.seen_messages = list(messages)
        return LLMResponse(content="ok", tool_calls=[])


def test_prior_history_is_threaded_into_the_next_call(monkeypatch):
    client = _HistoryInspectingClient()
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: client)

    prior_history = [
        {"role": "user", "content": "why did cost go up?"},
        {"role": "assistant", "content": "It was EC2."},
    ]

    run_agent("what about Azure?", history=prior_history)

    assert client.seen_messages is not None
    assert client.seen_messages[0]["role"] == "system"
    assert client.seen_messages[1] == prior_history[0]
    assert client.seen_messages[2] == prior_history[1]
    assert client.seen_messages[-1] == {"role": "user", "content": "what about Azure?"}


def test_no_history_means_fresh_conversation(monkeypatch):
    client = _HistoryInspectingClient()
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: client)

    run_agent("why did cost go up?")

    assert client.seen_messages is not None
    assert len(client.seen_messages) == 2  # system + the new user message only
    assert client.seen_messages[1] == {"role": "user", "content": "why did cost go up?"}


def test_no_provider_override_leaves_model_choice_untouched(monkeypatch):
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: _WrongProviderClient())

    executed_args: list[dict[str, Any]] = []

    def on_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        executed_args.append(args)

    run_agent("why did cost go up?", on_tool_call=on_tool_call, provider=None, max_turns=1)

    assert executed_args[0]["provider"] == "Azure"


# --- Phase 4: resolvers/dispatch.py short-circuits, exercised through the full loop ---


class _RelayStatusClient:
    """Calls one tool with fixed arguments, then relays the tool result's
    `status` field (if any) into its final answer — lets tests assert what
    the LLM would actually see and could narrate to the user.
    """

    def __init__(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self._tool_name = tool_name
        self._arguments = arguments
        self._step = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self._step += 1
        if self._step == 1:
            return LLMResponse(content=None, tool_calls=[ToolCall(id="1", name=self._tool_name, arguments=self._arguments)])
        last_tool_result = json.loads(messages[-1]["content"])
        options = last_tool_result.get("options")
        return LLMResponse(content=f"status={last_tool_result.get('status')} options={options}", tool_calls=[])


def test_impossible_combo_short_circuits_before_reaching_the_real_tool(monkeypatch):
    client = _RelayStatusClient("cost_trend", {"provider": "Azure", "service": "EC2"})
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: client)

    captured: list[dict[str, Any]] = []

    def on_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        captured.append(result)

    run_agent("show me Azure EC2 cost", on_tool_call=on_tool_call, max_turns=1)

    result = captured[0]
    assert result["status"] == "invalid_request"
    assert "total" not in result  # never reached the real cost_trend
    assert "Azure" in result["message"]


def test_ambiguous_service_surfaces_clarification_options_through_the_loop(monkeypatch):
    client = _RelayStatusClient("cost_trend", {"service": "vm"})
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: client)

    captured: list[dict[str, Any]] = []

    def on_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        captured.append(result)

    answer, _ = run_agent("show me VM cost", on_tool_call=on_tool_call, max_turns=2)

    assert captured[0]["status"] == "clarification_needed"
    assert set(captured[0]["options"]) == {"AWS EC2", "Azure Virtual Machine"}
    assert "AWS EC2" in answer and "Azure Virtual Machine" in answer


def test_valid_but_unavailable_service_reports_data_unavailable_through_the_loop(monkeypatch):
    client = _RelayStatusClient("cost_trend", {"provider": "AWS", "service": "S3"})
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: client)

    captured: list[dict[str, Any]] = []

    def on_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        captured.append(result)

    run_agent("show me S3 cost", on_tool_call=on_tool_call, max_turns=1)

    assert captured[0]["status"] == "data_unavailable"
    assert "total" not in captured[0]


def test_ui_provider_override_conflicting_with_requested_service_is_rejected(monkeypatch):
    """Dropdown forces provider=AWS; the model still asks for an Azure-only
    service. The forced provider wins (per test_provider_override_wins_...),
    but the resulting AWS+Blob-Storage combination is genuinely impossible —
    the resolver must catch this rather than silently returning empty data.
    """
    client = _RelayStatusClient("cost_trend", {"service": "Blob Storage"})
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: client)

    captured: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def on_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        captured.append((args, result))

    run_agent("show blob storage cost", on_tool_call=on_tool_call, provider="AWS", max_turns=1)

    args, result = captured[0]
    assert args["provider"] == "AWS"
    assert result["status"] == "invalid_request"


def test_invalid_argument_status_flows_through_the_loop(monkeypatch):
    """A tool-level rejection (bad date, in this case) must reach the model
    the same way a resolver-level rejection does — no crash, no silent
    fabrication, just a status the system prompt tells it how to relay.
    """
    client = _RelayStatusClient("cost_trend", {"start": "not-a-date", "end": "2026-06-30"})
    monkeypatch.setattr("agent.loop.get_llm_client", lambda: client)

    captured: list[dict[str, Any]] = []

    def on_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        captured.append(result)

    run_agent("show cost from not-a-date to today", on_tool_call=on_tool_call, max_turns=1)

    assert captured[0]["status"] == "invalid_argument"


# --- Reliability hardening: system prompt guardrails (items 4 & 5) ---


def test_system_prompt_covers_new_status_and_error_and_fabrication_guardrails():
    prompt = _build_system_prompt(None)
    assert "invalid_argument" in prompt
    assert '"error"' in prompt
    assert "resource_ids you invented" in prompt


def test_system_prompt_states_the_actual_month_coverage():
    # The dataset spans 2026-04-02 -> 2026-06-30 (see tools/finops_tools.py's
    # data_date_range()) — the prompt must name those months explicitly and
    # in plain language, not just as ISO date bounds, and must be computed
    # from the real range rather than hardcoded so it can't drift out of sync
    # with the actual data. Whitespace is normalized before the substring
    # check since the template's fixed source-line wrapping can land a
    # newline mid-phrase once the {month_span} substitution changes line
    # lengths.
    prompt = " ".join(_build_system_prompt(None).split())
    assert "April, May, and June 2026 only, no other months or years" in prompt


def test_month_span_label_is_derived_not_hardcoded():
    # Proves _month_span_label works for any date range, not just the current
    # dataset's April-June 2026 — it must be computed from whatever dates the
    # data actually has, so it stays correct if the dataset is regenerated.
    assert _month_span_label(date(2027, 1, 1), date(2027, 1, 31)) == "January 2027"
    assert _month_span_label(date(2026, 4, 2), date(2026, 6, 30)) == "April, May, and June 2026"


def test_month_span_label_tags_each_month_when_crossing_a_year_boundary():
    # A span like Nov 2024 -> Feb 2025 must not tag every month with just the
    # end year (that would misdate November/December as 2025).
    label = _month_span_label(date(2024, 11, 15), date(2025, 2, 3))
    assert label == "November 2024, December 2024, January 2025, and February 2025"


# --- Reliability hardening: OpenAI -> Ollama automatic fallback (item 1) ---


class _FlakyPrimary:
    """Stands in for OpenAIClient: raises on chat(), like a live outage."""

    def __init__(self, exc: BaseException) -> None:
        self.transient_error_types: tuple[type[BaseException], ...] = (type(exc),)
        self._exc = exc
        self.call_count = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.call_count += 1
        raise self._exc


class _StubFallback:
    def __init__(self) -> None:
        self.call_count = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content="fallback answer", tool_calls=[])


def test_fallback_client_switches_to_ollama_on_primary_failure():
    fallback = _StubFallback()
    client = _FallbackLLMClient(_FlakyPrimary(TimeoutError("boom")), lambda: fallback)

    response = client.chat([], [])

    assert response.content == "fallback answer"
    assert fallback.call_count == 1
    assert client.used_fallback is True


def test_fallback_client_stays_on_fallback_for_the_rest_of_the_session():
    fallback = _StubFallback()
    primary = _FlakyPrimary(TimeoutError("boom"))
    client = _FallbackLLMClient(primary, lambda: fallback)

    client.chat([], [])
    client.chat([], [])

    assert primary.call_count == 1  # never retried after the first failure
    assert fallback.call_count == 2


def test_fallback_client_does_not_swallow_non_transient_errors():
    class _BuggyPrimary:
        transient_error_types = (TimeoutError,)

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
            raise RuntimeError("actual bug, not a provider outage")

    client = _FallbackLLMClient(_BuggyPrimary(), lambda: _StubFallback())

    with pytest.raises(RuntimeError):
        client.chat([], [])


def test_run_agent_prepends_fallback_notice_when_client_used_fallback(monkeypatch):
    class _FellBackClient:
        used_fallback = True

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
            return LLMResponse(content="the answer", tool_calls=[])

    monkeypatch.setattr("agent.loop.get_llm_client", lambda: _FellBackClient())

    answer, _ = run_agent("why did cost go up?")

    assert "Ollama" in answer
    assert answer.endswith("the answer")
