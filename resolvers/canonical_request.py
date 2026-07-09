"""The one normalized request shape that flows out of resolution/validation.

`CanonicalRequest` is built and validated by `resolvers/validation.py`, then
`to_kwargs(tool_name)` projects it down onto each existing tool function's
flat keyword arguments. Every tool call is *derived from* one validated
object, even though each tool's own signature stays flat (flat parameters
are what OpenAI/Ollama tool-calling schemas — and the `_coerce_args` type
coercion in agent/loop.py — are built around; nesting a request object there
would trade reliability for architectural purity with no real benefit here).

`filters` is reserved for cost-threshold/tag-style filters (e.g. "resources
over $100") planned for a later phase — no tool currently has a parameter to
receive it, so it is deliberately *not* projected by `to_kwargs` yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

# Which CanonicalRequest fields each existing tool function actually accepts,
# in the order they matter for readability (not evaluation order — to_kwargs
# only includes fields whose value is not None).
_TOOL_FIELDS: dict[str, tuple[str, ...]] = {
    "cost_trend": ("start", "end", "service", "provider", "granularity"),
    "detect_spike": ("lookback_days", "provider", "service"),
    "find_idle_resources": ("provider", "service", "resource_ids"),
    "recommend": ("resource_ids",),
}


@dataclass(frozen=True)
class CanonicalRequest:
    provider: str | None = None  # resolved provider ("AWS" | "Azure"), or None = scan both
    service: str | None = None  # resolved CONCRETE service name (e.g. "EC2"), or None = no service filter
    service_concept: str | None = None  # matched concept (e.g. "compute") — for logging/narration only

    region: str | None = None
    resource_ids: list[str] | None = None
    instance_type: str | None = None
    environment: str | None = None
    business_unit: str | None = None

    start: str | date | None = None
    end: str | date | None = None
    granularity: str | None = None
    lookback_days: int | None = None

    filters: dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self, tool_name: str) -> dict[str, Any]:
        """Project this request onto `tool_name`'s existing flat kwargs.

        Only includes fields the tool actually accepts, and only those whose
        value is not None (so each tool's own defaults still apply).
        """
        fields = _TOOL_FIELDS.get(tool_name)
        if fields is None:
            raise ValueError(f"Unknown tool {tool_name!r}; expected one of {sorted(_TOOL_FIELDS)}")
        return {name: value for name in fields if (value := getattr(self, name)) is not None}
