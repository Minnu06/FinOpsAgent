from __future__ import annotations

from datetime import date

import pandas as pd

from adapters.base import CloudAdapter
from logging_setup import get_logger

_log = get_logger(__name__)


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
        _log.debug("MultiCloudAdapter.get_cost fanning out to %d adapters", len(self._adapters))
        frames = [
            a.get_cost(start, end, service=service, region=region, group_by=group_by)
            for a in self._adapters
        ]
        result = pd.concat(frames, ignore_index=True)
        _log.debug("MultiCloudAdapter.get_cost -> %d total rows across all providers", len(result))
        return result

    def get_utilization(self, resource_ids: list[str]) -> pd.DataFrame:
        frames = [a.get_utilization(resource_ids) for a in self._adapters]
        result = pd.concat(frames, ignore_index=True)
        _log.debug("MultiCloudAdapter.get_utilization -> %d total rows across all providers", len(result))
        return result

    def get_metadata(self, resource_ids: list[str]) -> pd.DataFrame:
        frames = [a.get_metadata(resource_ids) for a in self._adapters]
        result = pd.concat(frames, ignore_index=True)
        _log.debug("MultiCloudAdapter.get_metadata -> %d total rows across all providers", len(result))
        return result

    def list_services(self) -> list[str]:
        services: set[str] = set()
        for a in self._adapters:
            services.update(a.list_services())
        return sorted(services)
