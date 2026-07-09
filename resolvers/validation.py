"""Combines the service registry, provider resolver, and a data-availability
check into one accept/reject decision, producing a validated
`CanonicalRequest` on success.

`available_services` is an **injected callable** (`provider -> set[str]`),
never an adapter or `adapters.factory` import — this module stays decoupled
from any concrete adapter implementation, matching the Protocol boundary the
README's v2 real-cloud story depends on. The caller (`resolvers/dispatch.py`
in a later phase) is responsible for wiring it to
`lambda p: set(adapters.factory.get(p).list_services())`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from logging_setup import get_logger
from resolvers.canonical_request import CanonicalRequest
from resolvers.provider_resolver import resolve_provider_and_service

_log = get_logger(__name__)

AvailableServices = Callable[[str], set[str]]

_ALL_PROVIDERS = ("AWS", "Azure")


@dataclass(frozen=True)
class ValidationResult:
    """`kind is None` means accepted — `request` is populated. Any other
    `kind` means the request was short-circuited before touching a tool;
    `message` is meant to be relayed to the user as-is or narrated by the LLM.
    """

    kind: str | None = None  # None (accept) | "invalid_request" | "data_unavailable" | "clarification_needed" | "unresolved_service"
    message: str = ""
    request: CanonicalRequest | None = None
    options: tuple[str, ...] = field(default_factory=tuple)


def _availability_for(provider: str | None, available_services: AvailableServices) -> set[str]:
    """Services available for `provider`, or the union across all known
    providers if `provider` is None (scan-both case). Never calls
    `available_services(None)` — there is no adapter registered under that key.
    """
    if provider is not None:
        return available_services(provider)
    union: set[str] = set()
    for p in _ALL_PROVIDERS:
        union |= available_services(p)
    return union


def validate(
    raw_service: str | None,
    raw_provider: str | None,
    available_services: AvailableServices,
    *,
    region: str | None = None,
    resource_ids: list[str] | None = None,
    instance_type: str | None = None,
    environment: str | None = None,
    business_unit: str | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    granularity: str | None = None,
    lookback_days: int | None = None,
    filters: dict[str, Any] | None = None,
) -> ValidationResult:
    resolution = resolve_provider_and_service(raw_service, raw_provider)

    if resolution.outcome == "unresolved_service":
        message = (
            f"I don't recognize the service {raw_service!r}. "
            "Could you clarify which AWS or Azure service you mean?"
        )
        _log.debug("validate: unresolved_service raw_service=%r", raw_service)
        return ValidationResult(kind="unresolved_service", message=message)

    if resolution.outcome == "impossible_combo":
        message = f"{resolution.provider} does not offer {raw_service!r} — that combination isn't valid."
        _log.debug("validate: invalid_request provider=%s service=%r", resolution.provider, raw_service)
        return ValidationResult(kind="invalid_request", message=message)

    if resolution.outcome == "clarification_needed":
        options_text = " or ".join(resolution.clarification_options)
        message = f"{raw_service!r} could mean {options_text} — which cloud do you mean?"
        _log.debug("validate: clarification_needed raw_service=%r options=%s", raw_service, resolution.clarification_options)
        return ValidationResult(kind="clarification_needed", message=message, options=resolution.clarification_options)

    # resolution.outcome == "resolved"
    if resolution.service is not None:
        available = _availability_for(resolution.provider, available_services)
        if resolution.service not in available:
            message = (
                f"{resolution.service} is a valid {resolution.provider} service, "
                "but this dataset has no cost data for it."
            )
            _log.debug("validate: data_unavailable provider=%s service=%s", resolution.provider, resolution.service)
            return ValidationResult(kind="data_unavailable", message=message)

    request = CanonicalRequest(
        provider=resolution.provider,
        service=resolution.service,
        service_concept=resolution.concept,
        region=region,
        resource_ids=resource_ids,
        instance_type=instance_type,
        environment=environment,
        business_unit=business_unit,
        start=start,
        end=end,
        granularity=granularity,
        lookback_days=lookback_days,
        filters=filters or {},
    )
    _log.debug("validate: accepted provider=%s service=%s", resolution.provider, resolution.service)
    return ValidationResult(kind=None, request=request)
