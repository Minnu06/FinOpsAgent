"""CloudLens Chainlit UI.

Renders every tool call as a visible cl.Step (name, arguments, JSON result) —
this is the proof the system is agentic, so it is never hidden. The final
answer streams in afterward. A "Scan for anomalies" starter injects the
proactive investigation prompt; it is the same agent loop, only the first
message differs.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import chainlit as cl

from agent.loop import run_agent

PROACTIVE_SCAN_PROMPT = (
    "Scan the last 30 days for cost anomalies and idle waste across all clouds and report findings."
)

WELCOME_MESSAGE = (
    "**CloudLens** — a local multi-cloud FinOps copilot for AWS and Azure.\n\n"
    "Ask about spend, why costs changed, or what's idle. Every dollar figure I say comes "
    "from a tool call — you'll see each one below as I investigate, so nothing on screen "
    "is a guess."
)


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
    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def main(message: cl.Message) -> None:
    loop = asyncio.get_running_loop()

    def on_tool_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        async def render_step() -> None:
            async with cl.Step(name=name, type="tool") as step:
                step.input = json.dumps(args, default=str, indent=2)
                step.output = json.dumps(result, default=str, indent=2)

        asyncio.run_coroutine_threadsafe(render_step(), loop).result()

    try:
        answer = await asyncio.to_thread(run_agent, message.content, on_tool_call)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        await cl.Message(
            content=(
                f"⚠️ Agent error: {type(exc).__name__}: {exc}\n\n"
                "Check that OPENAI_API_KEY is set (or switch LLM_PROVIDER=ollama in .env "
                "for a fully local run)."
            )
        ).send()
        return

    msg = cl.Message(content="")
    await msg.send()
    for word in answer.split(" "):
        await msg.stream_token(word + " ")
        await asyncio.sleep(0.01)
    await msg.update()
