"""Thin, provider-agnostic wrapper over OpenAI and Ollama chat-completions.

Selected via LLM_PROVIDER=openai|ollama (default openai). Both providers speak
the same tool-calling wire format (OpenAI's function-calling JSON schema), so
agent/loop.py never needs to know which one is active.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from dotenv import load_dotenv

from logging_setup import get_logger

load_dotenv()

_log = get_logger(__name__)


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
        from openai import OpenAI, OpenAIError

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or export it, "
                "or set LLM_PROVIDER=ollama to run fully local."
            )
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI(api_key=api_key)
        # Failure modes worth failing over to the local Ollama client for:
        # OpenAIError covers timeouts, rate limits, connection drops, and
        # 4xx/5xx API errors; the rest cover a response we can't parse (empty
        # choices, malformed tool-call JSON). Read by _FallbackLLMClient.chat()
        # below — a bug in our own code should still surface normally instead
        # of being silently swallowed as a "provider failure".
        self.transient_error_types: tuple[type[BaseException], ...] = (
            OpenAIError, json.JSONDecodeError, IndexError, KeyError, AttributeError,
        )
        _log.info("LLM backend: OpenAI, model=%s", self._model)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        _log.debug("OpenAI request: %d messages, %d tools", len(messages), len(tools or []))
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
        _log.debug(
            "OpenAI response: content_len=%d, tool_calls=%s",
            len(choice.content or ""), [tc.name for tc in tool_calls],
        )
        return LLMResponse(content=choice.content, tool_calls=tool_calls)


class OllamaClient:
    def __init__(self, model: str | None = None) -> None:
        import ollama

        self._ollama = ollama
        self._model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        _log.info("LLM backend: Ollama (local), model=%s", self._model)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        _log.debug("Ollama request: %d messages, %d tools", len(messages), len(tools or []))
        resp = self._ollama.chat(model=self._model, messages=_prepare_ollama_messages(messages), tools=tools or None)
        message = resp["message"]
        raw_calls = message.get("tool_calls") or []
        tool_calls = []
        for i, tc in enumerate(raw_calls):
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCall(id=tc.get("id", f"call_{i}"), name=tc["function"]["name"], arguments=args))
        _log.debug(
            "Ollama response: content_len=%d, tool_calls=%s",
            len(message.get("content") or ""), [tc.name for tc in tool_calls],
        )
        return LLMResponse(content=message.get("content"), tool_calls=tool_calls)


class _FallbackLLMClient:
    """Wraps the primary OpenAI client with an automatic, transparent fallback
    to the local Ollama client for a single failing call (timeout, rate limit,
    connection error, 5xx, or a malformed/unparseable response).

    This is how OpenAI stays the primary backend while a live session
    survives an OpenAI outage without a manual LLM_PROVIDER flag change and
    restart — the one architectural rule this hardens, not replaces. Once
    triggered, this client stays on Ollama for the rest of its lifetime (one
    run_agent() call) rather than flapping between providers mid-conversation.
    """

    def __init__(self, primary: OpenAIClient, fallback_factory: Callable[[], "OllamaClient"]) -> None:
        self._primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: OllamaClient | None = None
        self.used_fallback = False

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        if self._fallback is not None:
            return self._fallback.chat(messages, tools)
        try:
            return self._primary.chat(messages, tools)
        except self._primary.transient_error_types as exc:
            _log.warning(
                "OpenAI call failed (%s: %s) -- falling back to local Ollama for the rest of this session",
                type(exc).__name__, exc,
            )
            self._fallback = self._fallback_factory()
            self.used_fallback = True
            return self._fallback.chat(messages, tools)


def get_llm_client() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "openai":
        return _FallbackLLMClient(OpenAIClient(), OllamaClient)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'openai' or 'ollama')")
