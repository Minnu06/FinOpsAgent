from __future__ import annotations

from datetime import date

import pandas as pd

from adapters.base import CloudAdapter


class MultiCloudAdapter:
    """Fans a query out to every wrapped CloudAdapter and concatenates results.

    Lets tools query "all clouds" without knowing which providers exist.
    """

    provider = "multi"

    def __init__(self, adapters: list[CloudAdapter]) -> None:
        self._adapters = adapters

    def get_cost(
        self,
        start: date,
        end: date,
        service: str | None = None,
        region: str | None = None,
        group_by: list[str] | None = None,
    ) -> pd.DataFrame:
        frames = [
            a.get_cost(start, end, service=service, region=region, group_by=group_by)
            for a in self._adapters
        ]
        return pd.concat(frames, ignore_index=True)

    def get_utilization(self, resource_ids: list[str]) -> pd.DataFrame:
        frames = [a.get_utilization(resource_ids) for a in self._adapters]
        return pd.concat(frames, ignore_index=True)

    def get_metadata(self, resource_ids: list[str]) -> pd.DataFrame:
        frames = [a.get_metadata(resource_ids) for a in self._adapters]
        return pd.concat(frames, ignore_index=True)
