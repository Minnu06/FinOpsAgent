"""Centralized instance-type catalog: canonical types, provider mapping,
synonyms, and fuzzy-match resolution — mirrors `resolvers/service_registry.py`
exactly, one level down (service -> instance size within that service).

AWS and Azure instance-type strings never collide in format (`m5.2xlarge` vs
`Standard_D4s_v5`), so an *exact* match always infers its provider directly,
with zero ambiguity. The ambiguity this module actually resolves is generic
size words a user might say without naming a cloud ("large", "xlarge",
"compute optimized") — those map to a `size_tier` concept spanning one AWS
and one Azure type each, loosely paired by instance family (general purpose
/ compute-optimized / memory-optimized), not exact vCPU parity. Good enough
to disambiguate "which cloud do you mean," not a claim of real-world
cross-cloud sizing equivalence.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from logging_setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class InstanceTypeEntry:
    concrete_name: str  # exact string used by adapters/data, e.g. "m5.2xlarge"
    provider: str  # "AWS" | "Azure"
    size_tier: str  # cross-provider grouping key, e.g. "large"
    synonyms: tuple[str, ...] = ()  # lowercase, type-specific synonym tokens


@dataclass(frozen=True)
class InstanceTypeMatch:
    """Result of resolving a raw user string against the registry.

    `candidates` has 0 entries when unresolved, 1 when unambiguous, 2+ when
    the raw string names a cross-provider size tier with no provider
    specified (e.g. "large") — the caller decides what to do with multiple
    candidates, same pattern as `service_registry.ServiceMatch`.
    """

    matched: bool
    size_tier: str | None = None
    candidates: tuple[InstanceTypeEntry, ...] = ()
    confidence: str = "exact"  # "exact" | "synonym" | "fuzzy"


_INSTANCE_TYPES: tuple[InstanceTypeEntry, ...] = (
    # --- AWS (m5 general purpose, c5 compute-optimized, r5 memory-optimized) ---
    InstanceTypeEntry("m5.large", "AWS", "small", ("m5 large",)),
    InstanceTypeEntry("m5.xlarge", "AWS", "medium", ("m5 xlarge",)),
    InstanceTypeEntry("m5.2xlarge", "AWS", "large", ("m5 2xlarge", "m5 2x large")),
    InstanceTypeEntry("m5.4xlarge", "AWS", "xlarge", ("m5 4xlarge", "m5 4x large")),
    InstanceTypeEntry("c5.4xlarge", "AWS", "compute_optimized", ("c5 4xlarge", "compute optimized instance")),
    InstanceTypeEntry("r5.2xlarge", "AWS", "memory_optimized", ("r5 2xlarge", "memory optimized instance")),
    # --- Azure (Dsv5 general purpose, Fsv2 compute-optimized, Esv5 memory-optimized, B-series burstable) ---
    InstanceTypeEntry("Standard_B2s", "Azure", "small", ("b2s", "burstable")),
    InstanceTypeEntry("Standard_D2s_v5", "Azure", "medium", ("d2s v5", "d2s")),
    InstanceTypeEntry("Standard_D4s_v5", "Azure", "large", ("d4s v5", "d4s")),
    InstanceTypeEntry("Standard_D8s_v5", "Azure", "xlarge", ("d8s v5", "d8s")),
    InstanceTypeEntry("Standard_F8s_v2", "Azure", "compute_optimized", ("f8s v2", "f8s")),
    InstanceTypeEntry("Standard_E4s_v5", "Azure", "memory_optimized", ("e4s v5", "e4s")),
)

# Ambiguous, cross-provider size word -> size_tier. Only used when the raw
# string doesn't exactly match a concrete_name and isn't a type-specific
# synonym. Each tier spans both providers, so resolving one to a single
# provider without the user saying which cloud would be a silent guess.
_SIZE_TIER_SYNONYMS: dict[str, str] = {
    "small": "small",
    "medium": "medium",
    "large": "large",
    "xlarge": "xlarge",
    "compute optimized": "compute_optimized",
    "compute-optimized": "compute_optimized",
    "memory optimized": "memory_optimized",
    "memory-optimized": "memory_optimized",
}

_CONCRETE_BY_LOWER: dict[str, InstanceTypeEntry] = {e.concrete_name.lower(): e for e in _INSTANCE_TYPES}
_SYNONYM_TO_ENTRY: dict[str, InstanceTypeEntry] = {syn: entry for entry in _INSTANCE_TYPES for syn in entry.synonyms}
_TIER_TO_ENTRIES: dict[str, tuple[InstanceTypeEntry, ...]] = {}
for _entry in _INSTANCE_TYPES:
    _TIER_TO_ENTRIES[_entry.size_tier] = _TIER_TO_ENTRIES.get(_entry.size_tier, ()) + (_entry,)

_FUZZY_CANDIDATES: dict[str, InstanceTypeEntry] = {**_CONCRETE_BY_LOWER, **_SYNONYM_TO_ENTRY}
_FUZZY_MATCH_CUTOFF = 0.75


def resolve(raw: str) -> InstanceTypeMatch:
    """Resolve a raw user-supplied instance-type string to candidate
    InstanceTypeEntry objects. Never raises — an unresolvable string returns
    InstanceTypeMatch(matched=False, candidates=()).
    """
    if not raw or not raw.strip():
        return InstanceTypeMatch(matched=False)

    key = raw.strip().lower()

    if key in _CONCRETE_BY_LOWER:
        entry = _CONCRETE_BY_LOWER[key]
        _log.debug("resolve(%r) -> exact: %s (%s)", raw, entry.concrete_name, entry.provider)
        return InstanceTypeMatch(matched=True, size_tier=entry.size_tier, candidates=(entry,), confidence="exact")

    if key in _SYNONYM_TO_ENTRY:
        entry = _SYNONYM_TO_ENTRY[key]
        _log.debug("resolve(%r) -> synonym: %s (%s)", raw, entry.concrete_name, entry.provider)
        return InstanceTypeMatch(matched=True, size_tier=entry.size_tier, candidates=(entry,), confidence="synonym")

    if key in _SIZE_TIER_SYNONYMS:
        tier = _SIZE_TIER_SYNONYMS[key]
        candidates = _TIER_TO_ENTRIES.get(tier, ())
        _log.debug(
            "resolve(%r) -> size_tier %s (%d candidate(s): %s)",
            raw, tier, len(candidates), [c.concrete_name for c in candidates],
        )
        return InstanceTypeMatch(matched=True, size_tier=tier, candidates=candidates, confidence="synonym")

    close = difflib.get_close_matches(key, _FUZZY_CANDIDATES.keys(), n=1, cutoff=_FUZZY_MATCH_CUTOFF)
    if close:
        entry = _FUZZY_CANDIDATES[close[0]]
        _log.debug("resolve(%r) -> fuzzy %r -> %s (%s)", raw, close[0], entry.concrete_name, entry.provider)
        return InstanceTypeMatch(matched=True, size_tier=entry.size_tier, candidates=(entry,), confidence="fuzzy")

    _log.debug("resolve(%r) -> unresolved", raw)
    return InstanceTypeMatch(matched=False)


def providers_for(size_tier_or_concrete: str) -> list[str]:
    """Providers that offer the given size tier or exact concrete type name."""
    key = size_tier_or_concrete.strip().lower()
    if key in _CONCRETE_BY_LOWER:
        return [_CONCRETE_BY_LOWER[key].provider]
    entries = _TIER_TO_ENTRIES.get(size_tier_or_concrete, ())
    return sorted({e.provider for e in entries})


def concrete_name_for(size_tier: str, provider: str) -> str | None:
    """The exact concrete instance-type name for a size tier under a specific provider, or None."""
    for entry in _TIER_TO_ENTRIES.get(size_tier, ()):
        if entry.provider == provider:
            return entry.concrete_name
    return None


def all_concrete_names(provider: str | None = None) -> list[str]:
    """All known concrete instance-type names, optionally filtered to one provider."""
    return sorted(e.concrete_name for e in _INSTANCE_TYPES if provider is None or e.provider == provider)
