"""Verifies the multi-turn tool-calling mechanics of agent/loop.py using a
scripted fake LLM client — independent of any real model's routing quality.
Confirms detect_spike -> find_idle_resources -> recommend chains correctly
and that tool results actually flow between turns.
"""

from __future__ import annotations

import json
from typing import Any

from agent.llm import LLMResponse, ToolCall
from agent.loop import run_agent


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
