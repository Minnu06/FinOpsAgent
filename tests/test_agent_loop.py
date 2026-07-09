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
