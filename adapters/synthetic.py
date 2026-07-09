from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from logging_setup import get_logger

_log = get_logger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "finops_combined.csv"

_METADATA_COLUMNS = [
    "resource_id",
    "resource_name",
    "provider",
    "account_id",
    "account_name",
    "region",
    "service",
    "instance_type",
    "environment",
    "business_unit",
    "application",
    "owner",
    "status",
    "attached_to",
    "storage_gb",
]

_COST_COLUMNS = [
    "date",
    "provider",
    "account_id",
    "account_name",
    "region",
    "service",
    "resource_id",
    "resource_name",
    "cost_usd",
]

_UTILIZATION_COLUMNS = [
    "date",
    "resource_id",
    "service",
    "status",
    "cpu_utilization_p95",
    "invocations",
    "last_access_days",
]


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df


class SyntheticAdapter:
    """CloudAdapter backed by the local finops_combined.csv fixture.

    Filters the combined CSV down to a single provider. This is the only
    class in the codebase that touches pandas file I/O directly; every
    other layer consumes the DataFrames it returns.
    """

    def __init__(self, provider: str, data_path: Path | str = DEFAULT_DATA_PATH) -> None:
        self.provider = provider
        self._path = Path(data_path)
        full = _load_csv(self._path)
        self.df = full[full["provider"] == provider].reset_index(drop=True)
        _log.debug("SyntheticAdapter(%s) loaded %d rows from %s", provider, len(self.df), self._path.name)

    def get_cost(
        self,
        start: date,
        end: date,
        service: str | None = None,
        region: str | None = None,
        group_by: list[str] | None = None,
    ) -> pd.DataFrame:
        df = self.df
        mask = (df["date"] >= start) & (df["date"] <= end)
        if service is not None:
            mask &= df["service"] == service
        if region is not None:
            mask &= df["region"] == region
        df = df.loc[mask]

        result = df.groupby(group_by, as_index=False)["cost_usd"].sum() if group_by else df[_COST_COLUMNS].copy()
        _log.debug(
            "%s.get_cost(start=%s, end=%s, service=%s, region=%s, group_by=%s) -> %d rows",
            self.provider, start, end, service, region, group_by, len(result),
        )
        return result

    def get_utilization(self, resource_ids: list[str]) -> pd.DataFrame:
        df = self.df[self.df["resource_id"].isin(resource_ids)]
        result = df[_UTILIZATION_COLUMNS].copy()
        _log.debug("%s.get_utilization(%d resource_ids) -> %d rows", self.provider, len(resource_ids), len(result))
        return result

    def get_metadata(self, resource_ids: list[str]) -> pd.DataFrame:
        df = self.df[self.df["resource_id"].isin(resource_ids)]
        if df.empty:
            _log.debug("%s.get_metadata(%d resource_ids) -> 0 rows", self.provider, len(resource_ids))
            return df[_METADATA_COLUMNS].copy()
        latest = df.sort_values("date").groupby("resource_id", as_index=False).last()
        result = latest[_METADATA_COLUMNS].copy()
        _log.debug("%s.get_metadata(%d resource_ids) -> %d rows", self.provider, len(resource_ids), len(result))
        return result

    def list_services(self) -> list[str]:
        return sorted(self.df["service"].unique())
