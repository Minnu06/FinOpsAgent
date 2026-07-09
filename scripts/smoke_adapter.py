"""Smoke test for the adapter layer: prints 5 rows per method per adapter."""

from __future__ import annotations

from datetime import date

import pandas as pd

from adapters.multi import MultiCloudAdapter
from adapters.synthetic import SyntheticAdapter

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n--- {title} (rows={len(df)}) ---")
    print(df.head(5).to_string(index=False))


def main() -> None:
    aws = SyntheticAdapter("AWS")
    azure = SyntheticAdapter("Azure")
    multi = MultiCloudAdapter([aws, azure])

    start, end = date(2026, 6, 10), date(2026, 6, 20)

    show("AWS get_cost", aws.get_cost(start, end))
    show("AWS get_cost EC2 grouped by date", aws.get_cost(start, end, service="EC2", group_by=["date"]))

    spike_resources = aws.get_cost(
        date(2026, 6, 16), date(2026, 6, 16), service="EC2", group_by=["resource_id"]
    )["resource_id"].tolist()
    show("AWS get_utilization (loadtest-worker candidates)", aws.get_utilization(spike_resources))
    show("AWS get_metadata (loadtest-worker candidates)", aws.get_metadata(spike_resources))

    show("Azure get_cost", azure.get_cost(start, end))

    azure_resources = azure.get_cost(start, end, group_by=["resource_id"])["resource_id"].head(5).tolist()
    show("Azure get_utilization", azure.get_utilization(azure_resources))
    show("Azure get_metadata", azure.get_metadata(azure_resources))

    show("Multi get_cost (both providers, concatenated)", multi.get_cost(start, end, group_by=["date", "provider"]))
    show("Multi get_utilization", multi.get_utilization(spike_resources + azure_resources))
    show("Multi get_metadata", multi.get_metadata(spike_resources + azure_resources))


if __name__ == "__main__":
    main()
