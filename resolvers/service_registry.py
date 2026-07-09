"""Centralized cloud-service catalog: canonical services, provider mapping,
synonyms, and fuzzy-match resolution.

This replaces hardcoded per-file service lists (previously only
`agent/tool_schemas.py`'s `_SERVICE_ENUM`) with one source of truth other
modules query. It models the full service catalog from the product spec (17
services across AWS + Azure) — independent of which services actually have
rows in the CSV. Whether a resolved service has real data is a separate
concern, checked later by `resolvers/validation.py` against each adapter's
`list_services()`.

Resolution precedence (see `resolve()`): exact concrete name > service-specific
synonym (unambiguous) > cross-provider concept synonym (may be ambiguous,
e.g. "vm" -> {EC2, Virtual Machine}) > fuzzy fallback.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from logging_setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class ServiceEntry:
    """One concrete, queryable service: an exact provider-native name plus the
    words a user might use to refer to it.
    """

    concrete_name: str  # exact string used by adapters/CSV/tool schemas, e.g. "EC2"
    provider: str  # "AWS" | "Azure"
    concept: str  # cross-provider grouping key, e.g. "compute"
    synonyms: tuple[str, ...] = ()  # lowercase, service-specific synonym tokens


@dataclass(frozen=True)
class ServiceMatch:
    """Result of resolving a raw user string against the registry.

    `candidates` has 0 entries when unresolved, 1 when unambiguous, 2+ when
    the raw string names a cross-provider concept with no provider specified
    (e.g. "vm") — the caller (provider_resolver) decides what to do with
    multiple candidates.
    """

    matched: bool
    concept: str | None = None
    candidates: tuple[ServiceEntry, ...] = ()
    confidence: str = "exact"  # "exact" | "synonym" | "fuzzy"


_SERVICES: tuple[ServiceEntry, ...] = (
    # --- Cross-provider concept pairs (genuinely equivalent services) ---
    ServiceEntry("EC2", "AWS", "compute", ("ec2", "ec2 instance")),
    ServiceEntry("Virtual Machine", "Azure", "compute", ("azure vm",)),
    ServiceEntry("EBS", "AWS", "block_storage", ("ebs", "ebs volume", "elastic block store")),
    ServiceEntry("Managed Disk", "Azure", "block_storage", ("managed disk", "azure disk")),
    ServiceEntry("Lambda", "AWS", "serverless_functions", ("lambda", "aws lambda")),
    ServiceEntry("Azure Functions", "Azure", "serverless_functions", ("azure functions", "azure function")),
    ServiceEntry("S3", "AWS", "object_storage", ("s3", "s3 bucket")),
    ServiceEntry("Blob Storage", "Azure", "object_storage", ("azure blob", "azure blob storage")),
    ServiceEntry("EKS", "AWS", "kubernetes", ("eks", "elastic kubernetes service")),
    ServiceEntry("AKS", "Azure", "kubernetes", ("aks", "azure kubernetes service")),
    # --- Single-provider concepts (no forced cross-provider pairing) ---
    ServiceEntry("RDS", "AWS", "relational_database", ("rds", "relational database service")),
    ServiceEntry("ELB", "AWS", "load_balancer", ("elb", "load balancer", "elastic load balancer", "load balancing")),
    ServiceEntry("CloudFront", "AWS", "cdn", ("cloudfront", "cdn")),
    ServiceEntry("ECS", "AWS", "container_service", ("ecs", "elastic container service")),
    ServiceEntry("SQL Database", "Azure", "managed_sql_database", ("sql database", "azure sql", "azure sql database")),
    ServiceEntry("Cosmos DB", "Azure", "nosql_database", ("cosmos db", "cosmos", "cosmosdb", "cosmos database")),
    ServiceEntry("App Service", "Azure", "paas_app_service", ("app service", "azure app service", "paas")),
)

# Ambiguous, cross-provider synonym -> concept. Only used when the raw string
# doesn't exactly match a concrete_name and isn't a service-specific synonym.
# Each of these concepts spans 2+ providers, so resolving one to a single
# provider without the user saying which cloud would be a silent guess.
_CONCEPT_SYNONYMS: dict[str, str] = {
    "vm": "compute",
    "instance": "compute",
    "instances": "compute",
    "compute": "compute",
    "server": "compute",
    "servers": "compute",
    "volume": "block_storage",
    "disk": "block_storage",
    "disks": "block_storage",
    "block storage": "block_storage",
    "function": "serverless_functions",
    "functions": "serverless_functions",
    "serverless": "serverless_functions",
    "storage": "object_storage",
    "bucket": "object_storage",
    "blob": "object_storage",
    "object storage": "object_storage",
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",
    "container orchestration": "kubernetes",
}

_CONCRETE_BY_LOWER: dict[str, ServiceEntry] = {e.concrete_name.lower(): e for e in _SERVICES}
_SYNONYM_TO_ENTRY: dict[str, ServiceEntry] = {syn: entry for entry in _SERVICES for syn in entry.synonyms}
_CONCEPT_TO_ENTRIES: dict[str, tuple[ServiceEntry, ...]] = {}
for _entry in _SERVICES:
    _CONCEPT_TO_ENTRIES[_entry.concept] = _CONCEPT_TO_ENTRIES.get(_entry.concept, ()) + (_entry,)

_FUZZY_CANDIDATES: dict[str, ServiceEntry] = {**_CONCRETE_BY_LOWER, **_SYNONYM_TO_ENTRY}
_FUZZY_MATCH_CUTOFF = 0.75


def resolve(raw: str) -> ServiceMatch:
    """Resolve a raw user-supplied service string to candidate ServiceEntry
    objects. Never raises — an unresolvable string returns
    ServiceMatch(matched=False, candidates=()).
    """
    if not raw or not raw.strip():
        return ServiceMatch(matched=False)

    key = raw.strip().lower()

    # 1. Exact concrete name (case-insensitive) always wins, even if the same
    #    string is also an ambiguous concept synonym (e.g. "Virtual Machine"
    #    is both Azure's exact name and would otherwise fuzzy-match "vm").
    if key in _CONCRETE_BY_LOWER:
        entry = _CONCRETE_BY_LOWER[key]
        _log.debug("resolve(%r) -> exact: %s (%s)", raw, entry.concrete_name, entry.provider)
        return ServiceMatch(matched=True, concept=entry.concept, candidates=(entry,), confidence="exact")

    # 2. Service-specific synonym — unambiguous, maps to exactly one entry.
    if key in _SYNONYM_TO_ENTRY:
        entry = _SYNONYM_TO_ENTRY[key]
        _log.debug("resolve(%r) -> synonym: %s (%s)", raw, entry.concrete_name, entry.provider)
        return ServiceMatch(matched=True, concept=entry.concept, candidates=(entry,), confidence="synonym")

    # 3. Ambiguous cross-provider concept synonym — may yield multiple candidates.
    if key in _CONCEPT_SYNONYMS:
        concept = _CONCEPT_SYNONYMS[key]
        candidates = _CONCEPT_TO_ENTRIES.get(concept, ())
        _log.debug(
            "resolve(%r) -> concept %s (%d candidate(s): %s)",
            raw, concept, len(candidates), [c.concrete_name for c in candidates],
        )
        return ServiceMatch(matched=True, concept=concept, candidates=candidates, confidence="synonym")

    # 4. Fuzzy fallback over both concrete names and known synonyms.
    close = difflib.get_close_matches(key, _FUZZY_CANDIDATES.keys(), n=1, cutoff=_FUZZY_MATCH_CUTOFF)
    if close:
        entry = _FUZZY_CANDIDATES[close[0]]
        _log.debug("resolve(%r) -> fuzzy %r -> %s (%s)", raw, close[0], entry.concrete_name, entry.provider)
        return ServiceMatch(matched=True, concept=entry.concept, candidates=(entry,), confidence="fuzzy")

    _log.debug("resolve(%r) -> unresolved", raw)
    return ServiceMatch(matched=False)


def providers_for(concept_or_concrete: str) -> list[str]:
    """Providers that offer the given concept or exact concrete service name."""
    key = concept_or_concrete.strip().lower()
    if key in _CONCRETE_BY_LOWER:
        return [_CONCRETE_BY_LOWER[key].provider]
    entries = _CONCEPT_TO_ENTRIES.get(concept_or_concrete, ())
    return sorted({e.provider for e in entries})


def concrete_name_for(concept: str, provider: str) -> str | None:
    """The exact concrete service name for a concept under a specific provider, or None."""
    for entry in _CONCEPT_TO_ENTRIES.get(concept, ()):
        if entry.provider == provider:
            return entry.concrete_name
    return None


def all_concrete_names(provider: str | None = None) -> list[str]:
    """All known concrete service names, optionally filtered to one provider."""
    return sorted(e.concrete_name for e in _SERVICES if provider is None or e.provider == provider)


def all_providers() -> list[str]:
    return sorted({e.provider for e in _SERVICES})
