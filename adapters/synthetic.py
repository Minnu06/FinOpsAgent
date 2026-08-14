from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from logging_setup import get_logger, record_trace

_log = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Each provider now ships its own source-of-truth Excel workbook at the repo
# root (aws_finops_data.xlsx / azure_finops_data.xlsx) instead of one combined
# CSV under data/. Same cost/region/instance_type/environment/business_unit
# values as the old data/finops_combined.csv — the only real difference is
# `status`, simplified to a uniform "running"/"stopped" for every service
# (the old CSV used archetype-specific values like "in-use"/"available"/
# "active"/"deallocated").
DEFAULT_DATA_PATHS: dict[str, Path] = {
    "AWS": _REPO_ROOT / "aws_finops_data.xlsx",
    "Azure": _REPO_ROOT / "azure_finops_data.xlsx",
}

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


def _load_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Excel sheets can carry trailing fully-blank rows past the last real
    # record; a blank resource_id/date is never valid data, so drop them
    # rather than let them show up as phantom NaN-provider rows downstream.
    df = df.dropna(subset=["resource_id", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


class SyntheticAdapter:
    """CloudAdapter backed by a per-provider Excel workbook fixture
    (aws_finops_data.xlsx / azure_finops_data.xlsx).

    This is the only class in the codebase that touches pandas file I/O
    directly; every other layer consumes the DataFrames it returns.
    """

    def __init__(self, provider: str, data_path: Path | str | None = None) -> None:
        self.provider = provider
        self._path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATHS[provider]
        full = _load_excel(self._path)
        self.df = full[full["provider"] == provider].reset_index(drop=True)
        _log.debug("SyntheticAdapter(%s) loaded %d rows from %s", provider, len(self.df), self._path.name)

    def get_cost(
        self,
        start: date,
        end: date,
        service: str | None = None,
        region: str | None = None,
        group_by: list[str] | None = None,
        extra_filters: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        df = self.df
        mask = (df["date"] >= start) & (df["date"] <= end)
        if service is not None:
            mask &= df["service"] == service
        if region is not None:
            mask &= df["region"] == region
        if extra_filters:
            for column, value in extra_filters.items():
                mask &= df[column] == value
        df = df.loc[mask]

        result = df.groupby(group_by, as_index=False)["cost_usd"].sum() if group_by else df[_COST_COLUMNS].copy()
        _log.debug(
            "%s.get_cost(start=%s, end=%s, service=%s, region=%s, group_by=%s, extra_filters=%s) -> %d rows",
            self.provider, start, end, service, region, group_by, extra_filters, len(result),
        )
        record_trace(
            "adapter", provider=self.provider, source=self._path.name, method="get_cost",
            rows_read=len(result), source_total_rows=len(self.df),
        )
        return result

    def get_utilization(self, resource_ids: list[str]) -> pd.DataFrame:
        df = self.df[self.df["resource_id"].isin(resource_ids)]
        result = df[_UTILIZATION_COLUMNS].copy()
        _log.debug("%s.get_utilization(%d resource_ids) -> %d rows", self.provider, len(resource_ids), len(result))
        record_trace(
            "adapter", provider=self.provider, source=self._path.name, method="get_utilization",
            rows_read=len(result), source_total_rows=len(self.df),
        )
        return result

    def get_metadata(self, resource_ids: list[str]) -> pd.DataFrame:
        df = self.df[self.df["resource_id"].isin(resource_ids)]
        if df.empty:
            _log.debug("%s.get_metadata(%d resource_ids) -> 0 rows", self.provider, len(resource_ids))
            record_trace(
                "adapter", provider=self.provider, source=self._path.name, method="get_metadata",
                rows_read=0, source_total_rows=len(self.df),
            )
            return df[_METADATA_COLUMNS].copy()
        latest = df.sort_values("date").groupby("resource_id", as_index=False).last()
        result = latest[_METADATA_COLUMNS].copy()
        _log.debug("%s.get_metadata(%d resource_ids) -> %d rows", self.provider, len(resource_ids), len(result))
        record_trace(
            "adapter", provider=self.provider, source=self._path.name, method="get_metadata",
            rows_read=len(result), source_total_rows=len(self.df),
        )
        return result

    def list_services(self) -> list[str]:
        return sorted(self.df["service"].unique())
