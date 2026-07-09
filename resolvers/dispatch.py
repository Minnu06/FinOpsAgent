"""The single seam between the LLM's tool-call arguments and actual tool
execution.

Builds a CanonicalRequest via resolvers/validation.py (which may
short-circuit with a clarification/rejection instead of a real answer), then
calls the real tool function with the validated, projected kwargs. This is
the only module that imports agent.tool_schemas.REGISTRY and
adapters.factory — service_registry, provider_resolver, validation, and
canonical_request all stay decoupled from both the concrete adapter
implementations and the tool registry itself.

Called from agent/loop.py's existing try/except in place of `func(**args)`,
after `_coerce_args` and any provider-override forcing have already run — so
this module only ever sees already-coerced arguments, and a UI dropdown
override that conflicts with a requested service (e.g. dropdown=AWS but the
model asked for "Blob Storage") is correctly caught here as an impossible
combo rather than silently producing an empty result.
"""

from __future__ import annotations

from typing import Any

from adapters import factory
from agent.tool_schemas import REGISTRY
from logging_setup import get_logger
from resolvers.validation import validate

_log = get_logger(__name__)


def _available_services(provider: str) -> set[str]:
    return set(factory.get(provider).list_services())


def resolve_and_execute(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate `args` into a CanonicalRequest, then execute `name`'s tool
    function with the projected kwargs.

    Returns either the tool's real result or a short-circuit dict describing
    why execution was skipped:
    `{"status": "clarification_needed" | "invalid_request" | "data_unavailable"
    | "unresolved_service", "message": str, "options": [...]}`. Never raises
    for a resolution failure — only lets a genuine tool execution error
    propagate, so agent/loop.py's existing try/except keeps handling that
    exactly as before.
    """
    func = REGISTRY.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}

    result = validate(
        raw_service=args.get("service"),
        raw_provider=args.get("provider"),
        available_services=_available_services,
        region=args.get("region"),
        resource_ids=args.get("resource_ids"),
        instance_type=args.get("instance_type"),
        environment=args.get("environment"),
        business_unit=args.get("business_unit"),
        start=args.get("start"),
        end=args.get("end"),
        granularity=args.get("granularity"),
        lookback_days=args.get("lookback_days"),
    )

    if result.kind is not None:
        _log.info("resolve_and_execute(%s) short-circuited: %s — %s", name, result.kind, result.message)
        payload: dict[str, Any] = {"status": result.kind, "message": result.message}
        if result.options:
            payload["options"] = list(result.options)
        return payload

    assert result.request is not None  # kind is None => accepted => request is populated
    kwargs = result.request.to_kwargs(name)
    _log.debug("resolve_and_execute(%s) -> validated kwargs: %s", name, kwargs)
    return func(**kwargs)
