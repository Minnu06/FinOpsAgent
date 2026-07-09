"""Thin, provider-agnostic wrapper over OpenAI and Ollama chat-completions.

Selected via LLM_PROVIDER=openai|ollama (default openai). Both providers speak
the same tool-calling wire format (OpenAI's function-calling JSON schema), so
agent/loop.py never needs to know which one is active.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


def _prepare_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate our neutral assistant tool_calls (dict arguments) into OpenAI's
    wire format (JSON-string arguments). Other message roles pass through unchanged.
    """
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content"),
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                        }
                        for tc in m["tool_calls"]
                    ],
                }
            )
        else:
            out.append(m)
    return out


def _prepare_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate our neutral assistant tool_calls into Ollama's wire format
    (dict arguments, no OpenAI-style 'type'/'id' wrapper).
    """
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [
                        {"function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in m["tool_calls"]
                    ],
                }
            )
        else:
            out.append(m)
    return out


class OpenAIClient:
    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or export it, "
                "or set LLM_PROVIDER=ollama to run fully local."
            )
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI(api_key=api_key)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=_prepare_openai_messages(messages),
            tools=tools or None,
        )
        choice = resp.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in (choice.tool_calls or [])
        ]
        return LLMResponse(content=choice.content, tool_calls=tool_calls)


class OllamaClient:
    def __init__(self, model: str | None = None) -> None:
        import ollama

        self._ollama = ollama
        self._model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        resp = self._ollama.chat(model=self._model, messages=_prepare_ollama_messages(messages), tools=tools or None)
        message = resp["message"]
        raw_calls = message.get("tool_calls") or []
        tool_calls = []
        for i, tc in enumerate(raw_calls):
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCall(id=tc.get("id", f"call_{i}"), name=tc["function"]["name"], arguments=args))
        return LLMResponse(content=message.get("content"), tool_calls=tool_calls)


def get_llm_client() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "openai":
        return OpenAIClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'openai' or 'ollama')")
