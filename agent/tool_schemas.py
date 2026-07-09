"""OpenAI-format function-calling schemas for the four FinOps tools.

Descriptions state trigger conditions and explicit negatives so the model
routes correctly instead of guessing which tool answers a given question.
"""

from __future__ import annotations

from typing import Any, Callable

from resolvers import instance_type_registry, service_registry
from tools.finops_tools import cost_trend, detect_spike, find_idle_resources, list_resources, recommend

# Generated from the centralized service registry (resolvers/service_registry.py)
# instead of a hand-typed list — covers the full catalog (17 services across
# both providers), not just the 6 this dataset happens to have rows for. A
# service the model asks for that isn't in the CSV still resolves through the
# schema; resolvers/dispatch.py reports "data not available" rather than the
# model being unable to name it in the first place.
_SERVICE_ENUM = service_registry.all_concrete_names()
_PROVIDER_ENUM = service_registry.all_providers()
_INSTANCE_TYPE_ENUM = instance_type_registry.all_concrete_names()

_REGION_PROPERTY = {
    "type": "string",
    "description": "Optional region filter, e.g. 'us-east-1' or 'eastus'.",
}
_ENVIRONMENT_PROPERTY = {
    "type": "string",
    "description": "Optional environment filter, e.g. 'prod', 'staging', 'test', 'dev'.",
}
_BUSINESS_UNIT_PROPERTY = {
    "type": "string",
    "description": "Optional business unit / tag filter, e.g. 'Finance', 'Platform', 'Security'.",
}
_INSTANCE_TYPE_PROPERTY = {
    "type": "string",
    "enum": _INSTANCE_TYPE_ENUM,
    "description": (
        "Optional instance-type/size filter (compute services only), e.g. 'm5.2xlarge' or "
        "'Standard_D4s_v5'. If the user's own word is a generic size ('large', 'xlarge', "
        "'compute optimized') with no cloud named, pass that word as-is rather than picking a "
        "specific concrete type yourself — do not guess the cloud."
    ),
}
_STATUS_PROPERTY = {
    "type": "string",
    "enum": ["running", "stopped"],
    "description": "Optional status filter. Omit to return both running and stopped resources.",
}

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
                "'why did cost go up'. Omit start/end for 'recent'/'last 30 days'/'lately' "
                "questions — the tool defaults to the most recent 30 days of available data. "
                "Do NOT guess or compute dates yourself; only pass start/end if the user "
                "names an explicit date or range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Start date, YYYY-MM-DD. Omit to default to 30 days before end.",
                    },
                    "end": {
                        "type": "string",
                        "description": "End date, YYYY-MM-DD. Omit to default to the most recent available date.",
                    },
                    "service": {
                        "type": "string",
                        "enum": _SERVICE_ENUM,
                        "description": (
                            "Optional service filter. If the user's own word is ambiguous "
                            "about which cloud (e.g. 'VM', 'compute', 'storage', 'function'), "
                            "pass that word as-is rather than picking a specific concrete "
                            "service yourself — do not guess the cloud."
                        ),
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
                    "region": _REGION_PROPERTY,
                    "environment": _ENVIRONMENT_PROPERTY,
                    "business_unit": _BUSINESS_UNIT_PROPERTY,
                    "instance_type": _INSTANCE_TYPE_PROPERTY,
                },
                "required": [],
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
                        "description": (
                            "Optional service filter. If the user's own word is ambiguous "
                            "about which cloud (e.g. 'VM', 'compute', 'storage', 'function'), "
                            "pass that word as-is rather than picking a specific concrete "
                            "service yourself — do not guess the cloud."
                        ),
                    },
                    "region": _REGION_PROPERTY,
                    "environment": _ENVIRONMENT_PROPERTY,
                    "business_unit": _BUSINESS_UNIT_PROPERTY,
                    "instance_type": _INSTANCE_TYPE_PROPERTY,
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
                "compute savings — use recommend for that. Do NOT use this for a plain "
                "'what's running'/'show our resources' question with no idle/waste framing — "
                "use list_resources for that; it's much cheaper and doesn't apply any "
                "utilization heuristics."
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
                        "description": (
                            "Optional service filter. If the user's own word is ambiguous "
                            "about which cloud (e.g. 'VM', 'compute', 'storage', 'function'), "
                            "pass that word as-is rather than picking a specific concrete "
                            "service yourself — do not guess the cloud."
                        ),
                    },
                    "resource_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional specific resource IDs to check, e.g. "
                            "driver_resource_ids from detect_spike. Omit to sweep all resources."
                        ),
                    },
                    "region": _REGION_PROPERTY,
                    "environment": _ENVIRONMENT_PROPERTY,
                    "business_unit": _BUSINESS_UNIT_PROPERTY,
                    "instance_type": _INSTANCE_TYPE_PROPERTY,
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_resources",
            "description": (
                "List resources with their current status (running/stopped), instance type, "
                "and monthly cost — a plain inventory with NO idle/waste heuristics applied. "
                "Use for 'what's running', 'show our EC2 instances', 'what's stopped in "
                "Azure', 'list resources tagged prod' — anything asking what exists or its "
                "current state. Do NOT use this to find waste — use find_idle_resources for "
                "'idle'/'unused'/'wasted' questions instead; do NOT use find_idle_resources "
                "for a plain 'what's currently running' question — use this tool for that."
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
                        "description": (
                            "Optional service filter. If the user's own word is ambiguous "
                            "about which cloud (e.g. 'VM', 'compute', 'storage', 'function'), "
                            "pass that word as-is rather than picking a specific concrete "
                            "service yourself — do not guess the cloud."
                        ),
                    },
                    "resource_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional specific resource IDs to check. Omit to sweep all resources.",
                    },
                    "region": _REGION_PROPERTY,
                    "environment": _ENVIRONMENT_PROPERTY,
                    "business_unit": _BUSINESS_UNIT_PROPERTY,
                    "instance_type": _INSTANCE_TYPE_PROPERTY,
                    "status": _STATUS_PROPERTY,
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
    "list_resources": list_resources,
    "recommend": recommend,
}
