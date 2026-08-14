from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class CloudAdapter(Protocol):
    """Interface every cloud data source must implement.

    Only this layer may know how the underlying data is fetched (CSV today,
    boto3/Azure SDK calls in v2). Everything above it works with plain
    pandas.DataFrame results and provider-agnostic filters.

    Every concrete implementation of get_cost/get_utilization/get_metadata
    should call `logging_setup.record_trace("adapter", provider=..., source=...,
    method=..., rows_read=...)` right after fetching, before returning — this
    is what powers the Chainlit UI's debug-mode pipeline trace (which
    provider/source was actually hit, how many rows came back). It's a no-op
    unless something upstream is collecting, so it's safe to call
    unconditionally; a future boto3/Azure SDK-backed adapter should keep
    calling it the same way SyntheticAdapter does, with `source` naming
    whatever the real data source is (an API endpoint, a table) instead of a
    file name.
    """

    provider: str

    def get_cost(
        self,
        start: date,
        end: date,
        service: str | None = None,
        region: str | None = None,
        group_by: list[str] | None = None,
        extra_filters: dict[str, str] | None = None,
    ) -> pd.DataFrame: ...

    def get_utilization(self, resource_ids: list[str]) -> pd.DataFrame: ...

    def get_metadata(self, resource_ids: list[str]) -> pd.DataFrame: ...

    def list_services(self) -> list[str]:
        """Concrete service names this adapter actually has data for.

        Used by the validation layer to distinguish "valid service, wrong
        provider" (e.g. Azure EC2 — rejected) from "valid service, provider
        supports it, but this dataset has no rows for it" (e.g. AWS S3 —
        reported as unavailable, not silently empty). Protocols aren't
        runtime-enforced, so callers should treat a missing implementation as
        "unknown, assume available" rather than crash.
        """
        ...
