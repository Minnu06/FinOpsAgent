"""Deterministic provider resolution policy.

Precedence: explicit provider wins > a service name that maps to exactly one
provider infers it directly > a service name that maps to multiple providers
(e.g. "vm") with no explicit provider asks for clarification > no service
named at all scans both clouds (nothing ambiguous to clarify).

This is real Python logic, not prompt text — it runs whether or not the LLM
"remembers" to ask, and gives the same answer every time for the same input.
"""

from __future__ import annotations

from dataclasses import dataclass

from logging_setup import get_logger
from resolvers import service_registry

_log = get_logger(__name__)


@dataclass(frozen=True)
class ProviderResolution:
    provider: str | None
    service: str | None
    concept: str | None = None
    outcome: str = "resolved"  # "resolved" | "clarification_needed" | "impossible_combo" | "unresolved_service"
    clarification_options: tuple[str, ...] = ()


def resolve_provider_and_service(
    raw_service: str | None,
    explicit_provider: str | None = None,
) -> ProviderResolution:
    """Resolve provider + concrete service from a raw service string and an
    optional explicit provider (e.g. a UI dropdown override).
    """
    if not raw_service or not raw_service.strip():
        # Nothing named -> nothing to disambiguate. Scan both unless the
        # caller forced a provider explicitly.
        _log.debug("resolve_provider_and_service(service=None, provider=%s) -> scan both/explicit", explicit_provider)
        return ProviderResolution(provider=explicit_provider, service=None)

    match = service_registry.resolve(raw_service)

    if not match.matched:
        _log.debug("resolve_provider_and_service(service=%r) -> unresolved", raw_service)
        return ProviderResolution(provider=explicit_provider, service=None, outcome="unresolved_service")

    if explicit_provider is not None:
        for candidate in match.candidates:
            if candidate.provider == explicit_provider:
                _log.debug(
                    "resolve_provider_and_service(service=%r, provider=%s) -> %s",
                    raw_service, explicit_provider, candidate.concrete_name,
                )
                return ProviderResolution(
                    provider=explicit_provider, service=candidate.concrete_name, concept=match.concept
                )
        # The service is real, just not offered by the explicitly forced provider.
        _log.debug(
            "resolve_provider_and_service(service=%r, provider=%s) -> impossible combo (candidates: %s)",
            raw_service, explicit_provider, [c.provider for c in match.candidates],
        )
        return ProviderResolution(
            provider=explicit_provider, service=None, concept=match.concept, outcome="impossible_combo"
        )

    if len(match.candidates) == 1:
        candidate = match.candidates[0]
        _log.debug(
            "resolve_provider_and_service(service=%r) -> inferred %s (%s)",
            raw_service, candidate.concrete_name, candidate.provider,
        )
        return ProviderResolution(provider=candidate.provider, service=candidate.concrete_name, concept=match.concept)

    options = tuple(f"{c.provider} {c.concrete_name}" for c in match.candidates)
    _log.debug("resolve_provider_and_service(service=%r) -> clarification needed: %s", raw_service, options)
    return ProviderResolution(
        provider=None,
        service=None,
        concept=match.concept,
        outcome="clarification_needed",
        clarification_options=options,
    )
