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

    answer = run_agent("why did cost go up?", on_tool_call=on_tool_call)

    assert calls == ["detect_spike", "find_idle_resources", "recommend"]
    assert "Total monthly saving: $6634.0" == answer
