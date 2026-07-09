"""OpenAI-format function-calling schemas for the four FinOps tools.

Descriptions state trigger conditions and explicit negatives so the model
routes correctly instead of guessing which tool answers a given question.
"""

from __future__ import annotations

from typing import Any, Callable

from tools.finops_tools import cost_trend, detect_spike, find_idle_resources, recommend

_SERVICE_ENUM = ["EC2", "EBS", "Lambda", "Virtual Machine", "Azure Functions", "Blob Storage"]
_PROVIDER_ENUM = ["AWS", "Azure"]

SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cost_trend",
            "description": (
                "Return daily cost totals and summary stats (total, avg_daily, "
                "pct_change_first_last) for a date range. Use for plain spend questions: "
                "'what did we spend', 'show me the cost trend', 'how much did X cost'. "
                "Do NOT use this to find anomalies or root causes — use detect_spike for "
                "'why did cost go up'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start date, YYYY-MM-DD."},
                    "end": {"type": "string", "description": "End date, YYYY-MM-DD."},
                    "service": {
                        "type": "string",
                        "enum": _SERVICE_ENUM,
                        "description": "Optional service filter.",
                    },
                    "provider": {
                        "type": "string",
                        "enum": _PROVIDER_ENUM,
                        "description": "Optional provider filter. Omit to scan both clouds.",
                    },
                    "granularity": {
                        "type": "string",
                        "enum": ["day", "week", "month"],
                        "description": "Bucket size. Auto-upgrades to week if the range would exceed ~30 points.",
                    },
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_spike",
            "description": (
                "Find the day cloud cost anomalously increased and identify the resources "
                "driving it. Use when the user asks why cost went up, what caused a spike, "
                "or to investigate an anomaly. Do NOT use for plain spend totals — use "
                "cost_trend for that. Returns driver_resource_ids that feed directly into "
                "find_idle_resources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lookback_days": {
                        "type": "integer",
                        "description": "How many days back to scan for anomalies. Default 30.",
                    },
                    "provider": {
                        "type": "string",
                        "enum": _PROVIDER_ENUM,
                        "description": "Optional provider filter. Omit to scan both clouds.",
                    },
                    "service": {
                        "type": "string",
                        "enum": _SERVICE_ENUM,
                        "description": "Optional service filter.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_idle_resources",
            "description": (
                "Find resources that are running but wasted: idle compute (CPU p95 < 5%), "
                "unattached EBS volumes, zero-invocation functions, and cold blobs (no "
                "access in 90+ days). Use after detect_spike to check whether driver "
                "resources are idle, or standalone to sweep for waste. Do NOT use this to "
                "compute savings — use recommend for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "enum": _PROVIDER_ENUM,
                        "description": "Optional provider filter. Omit to scan both clouds.",
                    },
                    "service": {
                        "type": "string",
                        "enum": _SERVICE_ENUM,
                        "description": "Optional service filter.",
                    },
                    "resource_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional specific resource IDs to check, e.g. "
                            "driver_resource_ids from detect_spike. Omit to sweep all resources."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend",
            "description": (
                "Turn facts about specific resources into concrete cost-saving actions with "
                "dollar savings: terminate, downsize, delete_volume, decommission, or "
                "move_to_archive_tier. Use after find_idle_resources to price out the fix "
                "for resources it flagged. Do NOT use this to discover waste in the first "
                "place — use find_idle_resources for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Resource IDs to evaluate, e.g. from find_idle_resources.",
                    },
                },
                "required": ["resource_ids"],
            },
        },
    },
]

REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "cost_trend": cost_trend,
    "detect_spike": detect_spike,
    "find_idle_resources": find_idle_resources,
    "recommend": recommend,
}
