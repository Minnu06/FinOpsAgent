"""Provider -> CloudAdapter registry.

Formalizes what used to be a flat `_ADAPTERS = {"AWS": ..., "Azure": ...}`
dict living inside tools/finops_tools.py. Registering a new provider (e.g. a
v2 AWSAdapter backed by boto3) is now a one-line `register()` call here —
tools/, agent/, and app.py never need to change, since they only ever ask
this module for "the adapter for provider X."
"""

from __future__ import annotations

from adapters.base import CloudAdapter
from adapters.synthetic import SyntheticAdapter
from logging_setup import get_logger

_log = get_logger(__name__)

_REGISTRY: dict[str, CloudAdapter] = {}


def register(provider: str, adapter: CloudAdapter) -> None:
    """Register (or replace) the adapter instance used for `provider`."""
    _REGISTRY[provider] = adapter
    _log.debug("Registered adapter for provider=%s (%s)", provider, type(adapter).__name__)


def get(provider: str) -> CloudAdapter:
    """The adapter registered for `provider`.

    Raises KeyError with the list of registered providers if none is
    registered — callers wanting a soft failure should check
    `provider in all_providers()` first.
    """
    if provider not in _REGISTRY:
        raise KeyError(f"No adapter registered for provider {provider!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[provider]


def all_providers() -> list[str]:
    return sorted(_REGISTRY)


def all_adapters() -> list[CloudAdapter]:
    return [_REGISTRY[p] for p in all_providers()]


# Default registration: the synthetic CSV-backed adapters, constructed eagerly
# at import time (matching the previous module-level singleton behavior in
# tools/finops_tools.py). A v2 real-cloud swap only needs to replace these two
# lines (or call register() again after importing this module).
register("AWS", SyntheticAdapter("AWS"))
register("Azure", SyntheticAdapter("Azure"))
