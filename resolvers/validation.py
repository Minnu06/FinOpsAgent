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
from resolvers import instance_type_registry
from resolvers.canonical_request import CanonicalRequest
from resolvers.provider_resolver import option_label, resolve_provider_and_service

_log = get_logger(__name__)

AvailableServices = Callable[[str], set[str]]

_ALL_PROVIDERS = ("AWS", "Azure")


@dataclass(frozen=True)
class ValidationResult:
    """`kind is None` means accepted — `request` is populated. Any other
    `kind` means the request was short-circuited before touching a tool;
    `message` is meant to be relayed to the user as-is or narrated by the LLM.
    """

    kind: str | None = None  # None (accept) | "invalid_request" | "data_unavailable" | "clarification_needed" | "unresolved_service" | "unresolved_instance_type"
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


@dataclass(frozen=True)
class _InstanceTypeResolution:
    outcome: str  # "resolved" | "unresolved_instance_type" | "invalid_request" | "clarification_needed"
    concrete_name: str | None = None
    provider: str | None = None
    message: str = ""
    options: tuple[str, ...] = ()


def _resolve_instance_type(raw_instance_type: str, known_provider: str | None) -> _InstanceTypeResolution:
    """Resolve a raw instance-type string, reconciling it against a provider
    already pinned by service resolution (if any). `known_provider` narrows
    an otherwise-ambiguous size word (e.g. "large") down to one candidate the
    same way an explicit provider narrows an ambiguous service; it never
    overrides a provider the service already resolved — a mismatch is a
    rejection, not a silent override.
    """
    match = instance_type_registry.resolve(raw_instance_type)
    if not match.matched:
        return _InstanceTypeResolution(
            outcome="unresolved_instance_type",
            message=(
                f"I don't recognize the instance type {raw_instance_type!r}. "
                "Could you clarify which AWS or Azure instance type you mean?"
            ),
        )

    if known_provider is not None:
        for candidate in match.candidates:
            if candidate.provider == known_provider:
                return _InstanceTypeResolution(
                    outcome="resolved", concrete_name=candidate.concrete_name, provider=known_provider
                )
        return _InstanceTypeResolution(
            outcome="invalid_request",
            message=f"{known_provider} does not offer instance type {raw_instance_type!r} — that combination isn't valid.",
        )

    if len(match.candidates) == 1:
        candidate = match.candidates[0]
        return _InstanceTypeResolution(
            outcome="resolved", concrete_name=candidate.concrete_name, provider=candidate.provider
        )

    options = tuple(option_label(c.provider, c.concrete_name) for c in match.candidates)
    options_text = " or ".join(options)
    return _InstanceTypeResolution(
        outcome="clarification_needed",
        message=f"{raw_instance_type!r} could mean {options_text} — which cloud do you mean?",
        options=options,
    )


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
    status: str | None = None,
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

    final_provider = resolution.provider
    resolved_instance_type: str | None = None
    if instance_type is not None and instance_type.strip():
        it_resolution = _resolve_instance_type(instance_type, resolution.provider)
        if it_resolution.outcome != "resolved":
            _log.debug(
                "validate: %s raw_instance_type=%r known_provider=%s",
                it_resolution.outcome, instance_type, resolution.provider,
            )
            return ValidationResult(kind=it_resolution.outcome, message=it_resolution.message, options=it_resolution.options)
        resolved_instance_type = it_resolution.concrete_name
        # Naming an exact instance type pins the provider just like naming an
        # exact service does — only takes effect when service resolution left
        # it unset (scan-both case); a provider already resolved from the
        # service was already reconciled inside _resolve_instance_type above.
        final_provider = final_provider or it_resolution.provider

    request = CanonicalRequest(
        provider=final_provider,
        service=resolution.service,
        service_concept=resolution.concept,
        region=region,
        resource_ids=resource_ids,
        instance_type=resolved_instance_type,
        environment=environment,
        business_unit=business_unit,
        status=status,
        start=start,
        end=end,
        granularity=granularity,
        lookback_days=lookback_days,
        filters=filters or {},
    )
    _log.debug(
        "validate: accepted provider=%s service=%s instance_type=%s", final_provider, resolution.service, resolved_instance_type
    )
    return ValidationResult(kind=None, request=request)
