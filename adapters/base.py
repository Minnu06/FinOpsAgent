from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class CloudAdapter(Protocol):
    """Interface every cloud data source must implement.

    Only this layer may know how the underlying data is fetched (CSV today,
    boto3/Azure SDK calls in v2). Everything above it works with plain
    pandas.DataFrame results and provider-agnostic filters.
    """

    provider: str

    def get_cost(
        self,
        start: date,
        end: date,
        service: str | None = None,
        region: str | None = None,
        group_by: list[str] | None = None,
    ) -> pd.DataFrame: ...

    def get_utilization(self, resource_ids: list[str]) -> pd.DataFrame: ...

    def get_metadata(self, resource_ids: list[str]) -> pd.DataFrame: ...
