"""CloudLens Chainlit UI.

Renders every tool call as a visible cl.Step (name, arguments, JSON result),
grouped under a parent "Investigating" step so the tool-calling flow reads as
a sequence — this is the proof the system is agentic, so it is never hidden.
The final answer streams in afterward. A "Scan for anomalies" starter injects
the proactive investigation prompt; it is the same agent loop, only the first
message differs.

A provider dropdown (gear icon / chat settings) lets the user hard-scope a
session to AWS, Azure, or both — enforced deterministically in agent/loop.py,
not left to the model's discretion.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select

from agent.loop import run_agent

PROACTIVE_SCAN_PROMPT = (
    "Scan the last 30 days for cost anomalies and idle waste across all clouds and report findings."
)

WELCOME_MESSAGE = (
    "**CloudLens** — a local multi-cloud FinOps copilot for AWS and Azure.\n\n"
    "Ask about spend, why costs changed, or what's idle. Every dollar figure I say comes "
    "from a tool call — you'll see each investigation step below, so nothing on screen "
    "is a guess.\n\n"
    "Use the settings (gear icon) to scope this chat to a single cloud provider, or leave "
    "it on **Both** to let the agent infer scope per question."
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
    await cl.ChatSettings(
        [
            Select(
                id="provider",
                label="Cloud provider",
                values=PROVIDER_OPTIONS,
                initial_index=0,
            )
        ]
    ).send()
    cl.user_session.set("provider", "Both")
    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    provider = settings.get("provider", "Both")
    cl.user_session.set("provider", provider)
    await cl.Message(content=f"Provider scope set to **{provider}**.").send()


@cl.on_message
async def main(message: cl.Message) -> None:
    loop = asyncio.get_running_loop()
    provider_setting = cl.user_session.get("provider", "Both")
    provider = None if provider_setting == "Both" else provider_setting

    tool_calls_made: list[str] = []

    async with cl.Step(name="🔎 Investigating", type="run", default_open=True) as investigation:
        investigation.input = message.content

        def on_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
            tool_calls_made.append(name)

            async def render_step() -> None:
                step = cl.Step(name=_tool_label(name), type="tool", parent_id=investigation.id)
                step.input = json.dumps(args, default=str, indent=2)
                step.output = json.dumps(result, default=str, indent=2)
                await step.send()

            asyncio.run_coroutine_threadsafe(render_step(), loop).result()

        try:
            answer = await asyncio.to_thread(run_agent, message.content, on_tool_call=on_tool_call, provider=provider)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            investigation.output = f"Failed: {type(exc).__name__}: {exc}"
            await cl.Message(
                content=(
                    f"⚠️ Agent error: {type(exc).__name__}: {exc}\n\n"
                    "Check that OPENAI_API_KEY is set (or switch LLM_PROVIDER=ollama in .env "
                    "for a fully local run)."
                )
            ).send()
            return

        investigation.output = (
            f"Called {len(tool_calls_made)} tool(s): {', '.join(tool_calls_made)}"
            if tool_calls_made
            else "Answered directly, no tools needed."
        )

    msg = cl.Message(content="")
    await msg.send()
    for word in answer.split(" "):
        await msg.stream_token(word + " ")
        await asyncio.sleep(0.01)
    await msg.update()
