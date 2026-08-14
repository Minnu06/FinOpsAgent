"""CloudLens Chainlit UI.

Renders every tool call as a visible cl.Step (name, arguments, JSON result),
grouped under a parent "Investigating" step so the tool-calling flow reads as
a sequence — this is the proof the system is agentic. A "Debug: show tool
calls" toggle (gear icon / chat settings, on by default) lets the user turn
that detail off for a plain chatbot view; tool calls still happen and are
still fully logged either way, only the on-screen rendering is suppressed.
When the toggle is on, each tool step also gets a nested "pipeline trace"
child step showing the resolver decision, adapter calls, and row counts
behind that one call (agent.loop.run_agent(debug=True)) — when the toggle is
off, run_agent isn't asked to collect any of that, so there's no extra work
happening in the background either. The final answer streams in afterward. A
"Scan for anomalies" starter injects the proactive investigation prompt; it
is the same agent loop, only the first message differs.

A provider dropdown (gear icon / chat settings) lets the user hard-scope a
session to AWS, Azure, or both — enforced deterministically in agent/loop.py,
not left to the model's discretion.

Conversation history persists across messages within a chat session (stored
in cl.user_session), so follow-up questions have context. Each chat session
gets its own log file under logs/ (see logging_setup.py).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select, Switch

from agent.loop import run_agent
from logging_setup import end_session, get_logger, set_current_session, start_session

_log = get_logger(__name__)

PROACTIVE_SCAN_PROMPT = (
    "Scan the last 30 days for cost anomalies and idle waste across all clouds and report findings."
)

WELCOME_MESSAGE = (
    "**CloudLens** — a local multi-cloud FinOps copilot for AWS and Azure.\n\n"
    "Ask about spend, why costs changed, or what's idle. Every dollar figure I say comes "
    "from a tool call — you'll see each investigation step below, so nothing on screen "
    "is a guess.\n\n"
    "Use the settings (gear icon) to scope this chat to a single cloud provider, or leave "
    "it on **Both** to let the agent infer scope per question. Turn off **Debug: show tool "
    "calls** there if you'd rather just see the final answer."
)

PROVIDER_OPTIONS = ["Both", "AWS", "Azure"]

_TOOL_ICONS = {
    "cost_trend": "📈",
    "detect_spike": "🚨",
    "find_idle_resources": "🧹",
    "recommend": "💡",
}


def _tool_label(name: str) -> str:
    icon = _TOOL_ICONS.get(name, "🔧")
    return f"{icon} {name}"


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="Scan for anomalies",
            message=PROACTIVE_SCAN_PROMPT,
        ),
        cl.Starter(
            label="Why did EC2 cost go up?",
            message="Why did our EC2 cost go up recently?",
        ),
        cl.Starter(
            label="Find idle resources",
            message="What's idle across our AWS and Azure accounts right now?",
        ),
    ]


@cl.on_chat_start
async def start() -> None:
    session_id, handler = start_session(label="chainlit")
    cl.user_session.set("log_session_id", session_id)
    cl.user_session.set("log_handler", handler)
    cl.user_session.set("history", [])
    _log.info("Chat session started")

    await cl.ChatSettings(
        [
            Select(
                id="provider",
                label="Cloud provider",
                values=PROVIDER_OPTIONS,
                initial_index=0,
            ),
            Switch(
                id="debug_mode",
                label="🔧 Debug: show tool calls",
                initial=True,
            ),
        ]
    ).send()
    cl.user_session.set("provider", "Both")
    cl.user_session.set("debug_mode", True)
    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_chat_end
async def on_chat_end() -> None:
    session_id = cl.user_session.get("log_session_id")
    handler = cl.user_session.get("log_handler")
    if session_id and handler:
        end_session(session_id, handler)


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    provider = settings.get("provider", "Both")
    cl.user_session.set("provider", provider)
    _log.info("Provider scope changed to %s", provider)

    debug_mode = settings.get("debug_mode", True)
    cl.user_session.set("debug_mode", debug_mode)
    _log.info("Debug mode set to %s", debug_mode)

    await cl.Message(
        content=f"Provider scope set to **{provider}**. Debug (tool-call detail) is **{'on' if debug_mode else 'off'}**."
    ).send()


async def _invoke_agent(
    message: cl.Message,
    provider: str | None,
    history: list[dict[str, Any]],
    on_tool_call: Any,
    debug: bool = False,
    on_debug_trace: Any = None,
) -> tuple[str, list[dict[str, Any]]] | None:
    """Runs the agent loop and reports an error to the user on failure.

    Returns None (after already sending the error message) on failure, so
    callers can just check for None and return — no error-handling duplicated
    between the debug and non-debug code paths.
    """
    try:
        return await asyncio.to_thread(
            run_agent,
            message.content,
            on_tool_call=on_tool_call,
            provider=provider,
            history=history,
            debug=debug,
            on_debug_trace=on_debug_trace,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        _log.exception("Agent error")
        await cl.Message(
            content=(
                f"⚠️ Agent error: {type(exc).__name__}: {exc}\n\n"
                "Check that OPENAI_API_KEY is set (or switch LLM_PROVIDER=ollama in .env "
                "for a fully local run)."
            )
        ).send()
        return None


async def _stream_answer(answer: str) -> None:
    msg = cl.Message(content="")
    await msg.send()
    for word in answer.split(" "):
        await msg.stream_token(word + " ")
        await asyncio.sleep(0.01)
    await msg.update()


@cl.on_message
async def main(message: cl.Message) -> None:
    session_id = cl.user_session.get("log_session_id")
    if session_id:
        set_current_session(session_id)

    loop = asyncio.get_running_loop()
    provider_setting = cl.user_session.get("provider", "Both")
    provider = None if provider_setting == "Both" else provider_setting
    history = cl.user_session.get("history", [])
    debug_mode = cl.user_session.get("debug_mode", True)

    _log.info(
        'User query: "%s" (provider=%s, history_turns=%d, debug_mode=%s)',
        message.content, provider_setting, len(history), debug_mode,
    )

    if not debug_mode:
        # Toggle off: run silently, no "Investigating" step at all — just the
        # narrated answer, like a plain chatbot. Tool calls still happen and
        # are still fully logged to logs/*.log; only the on-screen detail is
        # suppressed.
        result = await _invoke_agent(message, provider, history, on_tool_call=None)
        if result is None:
            return
        answer, updated_history = result
        cl.user_session.set("history", updated_history)
        _log.info("Final answer delivered (%d chars)", len(answer))
        await _stream_answer(answer)
        return

    tool_calls_made: list[str] = []
    last_step_id: dict[str, str] = {}

    async with cl.Step(name="🔎 Investigating", type="run", default_open=True) as investigation:
        investigation.input = message.content

        def on_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
            tool_calls_made.append(name)

            async def render_step() -> None:
                step = cl.Step(name=_tool_label(name), type="tool", parent_id=investigation.id)
                step.input = json.dumps(args, default=str, indent=2)
                step.output = json.dumps(result, default=str, indent=2)
                await step.send()
                last_step_id["id"] = step.id

            asyncio.run_coroutine_threadsafe(render_step(), loop).result()

        def on_debug_trace(name: str, trace: list[dict[str, Any]]) -> None:
            # Fires right after on_tool_call for the same call, so last_step_id
            # already holds that call's step — nest the trace as its child,
            # the same parent/child pattern "Investigating" already uses for
            # each tool step.
            if not trace:
                return
            parent_id = last_step_id.get("id", investigation.id)

            async def render_trace() -> None:
                step = cl.Step(name="🔬 pipeline trace", type="tool", parent_id=parent_id)
                step.output = json.dumps(trace, default=str, indent=2)
                await step.send()

            asyncio.run_coroutine_threadsafe(render_trace(), loop).result()

        result = await _invoke_agent(
            message, provider, history, on_tool_call, debug=True, on_debug_trace=on_debug_trace
        )
        if result is None:
            investigation.output = "Failed — see error message above."
            return
        answer, updated_history = result
        cl.user_session.set("history", updated_history)

        investigation.output = (
            f"Called {len(tool_calls_made)} tool(s): {', '.join(tool_calls_made)}"
            if tool_calls_made
            else "Answered directly, no tools needed."
        )

    _log.info("Final answer delivered (%d chars)", len(answer))
    await _stream_answer(answer)
