"""Deterministic FinOps tools. No LLM inside this module.

Every function queries the adapter layer, does exact arithmetic with pandas,
and returns a small JSON-serializable dict (never more than ~30 records).
The agent layer only calls these functions and narrates their output — it
never computes a dollar figure itself.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import date, timedelta
from typing import Any

import pandas as pd

from adapters import factory
from adapters.base import CloudAdapter
from adapters.multi import MultiCloudAdapter
from logging_setup import get_logger

_log = get_logger(__name__)

_ALL = MultiCloudAdapter(factory.all_adapters())

_COMPUTE_SERVICES = ("EC2", "Virtual Machine")
_FUNCTION_SERVICES = ("Lambda", "Azure Functions")
_VOLUME_SERVICES = ("EBS",)
_BLOB_SERVICES = ("Blob Storage",)
_ALL_SERVICES = _COMPUTE_SERVICES + _VOLUME_SERVICES + _FUNCTION_SERVICES + _BLOB_SERVICES

_IDLE_CPU_THRESHOLD = 5.0
_UNDERUTILIZED_CPU_THRESHOLD = 40.0
_COLD_BLOB_DAYS = 90
_UTILIZATION_WINDOW_DAYS = 7
_SPIKE_TRAILING_WINDOW_DAYS = 14
_SPIKE_STD_MULTIPLIER = 2.0
_SPIKE_PCT_THRESHOLD = 25.0
_MAX_RECORDS = 30
_DEFAULT_TREND_WINDOW_DAYS = 30


def _adapter_for(provider: str | None) -> CloudAdapter:
    if provider is None:
        return _ALL
    if provider not in factory.all_providers():
        raise ValueError(
            f"Unknown provider {provider!r}; expected one of {factory.all_providers()} or omit it to scan both."
        )
    return factory.get(provider)


def _max_available_date() -> date:
    return max(factory.get("AWS").df["date"].max(), factory.get("Azure").df["date"].max())


def _min_available_date() -> date:
    return min(factory.get("AWS").df["date"].min(), factory.get("Azure").df["date"].min())


def data_date_range() -> tuple[date, date]:
    """(min_date, max_date) of available cost data — used to ground the
    agent's system prompt so it doesn't guess "today" from its own training
    cutoff (e.g. assuming the current year is wrong for this dataset).
    """
    return _min_available_date(), _max_available_date()


def _to_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _mode(series: pd.Series) -> Any:
    clean = series.dropna()
    if clean.empty:
        return None
    return clean.mode().iloc[0]


def _extra_filters(
    environment: str | None, business_unit: str | None, instance_type: str | None = None
) -> dict[str, str] | None:
    filters = {}
    if environment is not None:
        filters["environment"] = environment
    if business_unit is not None:
        filters["business_unit"] = business_unit
    if instance_type is not None:
        filters["instance_type"] = instance_type
    return filters or None


def _monthly_cost(adapter: CloudAdapter, resource_id: str, window_start: date, window_end: date) -> float:
    window_days = (window_end - window_start).days + 1
    cost = adapter.get_cost(window_start, window_end, group_by=["resource_id"])
    row = cost.loc[cost["resource_id"] == resource_id, "cost_usd"]
    total = float(row.iloc[0]) if not row.empty else 0.0
    return round((total / window_days) * 30, 2)


def cost_trend(
    start: str | date | None = None,
    end: str | date | None = None,
    service: str | None = None,
    provider: str | None = None,
    granularity: str = "day",
    region: str | None = None,
    environment: str | None = None,
    business_unit: str | None = None,
    instance_type: str | None = None,
) -> dict[str, Any]:
    """Daily (or auto-aggregated) cost totals with summary stats.

    Use for plain spend questions ("what did we spend on X", "show me the
    trend"). Does NOT identify anomalies or drivers — use detect_spike for
    "why did cost go up". If start/end are omitted, defaults to the most
    recent 30 days of available data — do not guess dates yourself. When
    `provider` is omitted (scanning both clouds), the result includes a
    `by_provider` breakdown so a per-cloud split doesn't require a second call.
    """
    end_d = _to_date(end) if end is not None else _max_available_date()
    start_d = (
        _to_date(start)
        if start is not None
        else max(_min_available_date(), end_d - timedelta(days=_DEFAULT_TREND_WINDOW_DAYS - 1))
    )
    _log.info(
        "cost_trend(start=%s, end=%s, service=%s, provider=%s, granularity=%s, region=%s, environment=%s, business_unit=%s, instance_type=%s)",
        start_d, end_d, service, provider, granularity, region, environment, business_unit, instance_type,
    )
    adapter = _adapter_for(provider)
    filters = _extra_filters(environment, business_unit, instance_type)

    df = adapter.get_cost(start_d, end_d, service=service, region=region, group_by=["date"], extra_filters=filters)
    df = df.groupby("date", as_index=False)["cost_usd"].sum().sort_values("date")

    by_provider: dict[str, float] | None = None
    if provider is None:
        provider_totals = adapter.get_cost(
            start_d, end_d, service=service, region=region, group_by=["provider"], extra_filters=filters
        )
        by_provider = {row["provider"]: round(float(row["cost_usd"]), 2) for _, row in provider_totals.iterrows()}

    # Guarantee the ~30 record cap regardless of requested granularity/range.
    if granularity == "day" and len(df) > _MAX_RECORDS:
        granularity = "week"

    if granularity in ("week", "month") and not df.empty:
        freq = "W" if granularity == "week" else "MS"
        indexed = df.set_index(pd.to_datetime(df["date"]))["cost_usd"]
        resampled = indexed.resample(freq).sum()
        df = pd.DataFrame({"date": resampled.index.date, "cost_usd": resampled.values})

    series = [{"date": d.isoformat(), "cost": round(float(c), 2)} for d, c in zip(df["date"], df["cost_usd"])]
    total = round(float(df["cost_usd"].sum()), 2)
    avg_daily = round(float(df["cost_usd"].mean()), 2) if len(df) else 0.0
    pct_change_first_last = 0.0
    if len(df) >= 2 and df["cost_usd"].iloc[0] != 0:
        pct_change_first_last = round(
            (df["cost_usd"].iloc[-1] - df["cost_usd"].iloc[0]) / df["cost_usd"].iloc[0] * 100, 1
        )

    _log.info(
        "cost_trend -> total=$%.2f avg_daily=$%.2f pct_change=%.1f%% (%d points) by_provider=%s",
        total, avg_daily, pct_change_first_last, len(series), by_provider,
    )
    return {
        "series": series[:_MAX_RECORDS],
        "total": total,
        "avg_daily": avg_daily,
        "pct_change_first_last": pct_change_first_last,
        "by_provider": by_provider,
    }


def detect_spike(
    lookback_days: int = 30,
    provider: str | None = None,
    service: str | None = None,
    region: str | None = None,
    environment: str | None = None,
    business_unit: str | None = None,
    instance_type: str | None = None,
) -> dict[str, Any]:
    """Find the day cost anomalously increased and identify the resources driving it.

    Use when the user asks why cost went up or what caused a spike. Compares
    each day against its trailing 14-day mean; flags days that exceed 2
    standard deviations or a 25% jump. Do NOT use for plain spend totals —
    use cost_trend for that.
    """
    _log.info(
        "detect_spike(lookback_days=%s, provider=%s, service=%s, region=%s, environment=%s, business_unit=%s, instance_type=%s)",
        lookback_days, provider, service, region, environment, business_unit, instance_type,
    )
    adapter = _adapter_for(provider)
    filters = _extra_filters(environment, business_unit, instance_type)
    max_date = _max_available_date()
    min_date = _min_available_date()

    window_start = max(min_date, max_date - timedelta(days=lookback_days - 1))
    query_start = max(min_date, window_start - timedelta(days=_SPIKE_TRAILING_WINDOW_DAYS))

    raw = adapter.get_cost(
        query_start, max_date, service=service, region=region, group_by=["date", "provider", "service", "region"],
        extra_filters=filters,
    )
    if raw.empty:
        _log.info("detect_spike -> no cost data available for the given filters")
        return {"spike_date": None, "message": "No cost data available for the given filters."}

    full_dates = pd.date_range(query_start, max_date, freq="D").date
    report_dates = pd.date_range(window_start, max_date, freq="D").date

    best: dict[str, Any] | None = None

    for (prov, svc, reg), group in raw.groupby(["provider", "service", "region"]):
        series = group.set_index("date")["cost_usd"].reindex(full_dates, fill_value=0.0)
        trailing_mean = series.rolling(window=_SPIKE_TRAILING_WINDOW_DAYS, min_periods=7).mean().shift(1)
        trailing_std = series.rolling(window=_SPIKE_TRAILING_WINDOW_DAYS, min_periods=7).std(ddof=0).shift(1)

        for day in report_dates:
            mean = trailing_mean.loc[day]
            std = trailing_std.loc[day]
            today_cost = series.loc[day]
            if pd.isna(mean) or mean <= 0:
                continue
            std = 0.0 if pd.isna(std) else float(std)
            pct_increase = (today_cost - mean) / mean * 100
            is_spike = (std > 0 and today_cost > mean + _SPIKE_STD_MULTIPLIER * std) or pct_increase > _SPIKE_PCT_THRESHOLD
            if not is_spike:
                continue
            delta = today_cost - mean
            if best is None or delta > best["delta"]:
                best = {
                    "date": day,
                    "provider": prov,
                    "service": svc,
                    "region": reg,
                    "baseline": mean,
                    "spiked": today_cost,
                    "pct_increase": pct_increase,
                    "delta": delta,
                }

    if best is None:
        _log.info("detect_spike -> no anomalies detected in the given window")
        return {"spike_date": None, "message": "No anomalies detected in the given window."}

    spike_date = best["date"]
    driver_adapter = _adapter_for(best["provider"])
    resource_costs = driver_adapter.get_cost(
        query_start, spike_date, service=best["service"], region=best["region"], group_by=["date", "resource_id"]
    )
    spike_ids = set(resource_costs.loc[resource_costs["date"] == spike_date, "resource_id"])
    baseline_ids = set(resource_costs.loc[resource_costs["date"] < spike_date, "resource_id"])
    driver_ids = sorted(spike_ids - baseline_ids)

    driver_summary: dict[str, Any] = {
        "count": len(driver_ids),
        "instance_type": None,
        "environment": None,
        "first_seen": spike_date.isoformat(),
    }
    if driver_ids:
        meta = driver_adapter.get_metadata(driver_ids)
        driver_summary["instance_type"] = _mode(meta["instance_type"])
        driver_summary["environment"] = _mode(meta["environment"])

    _log.info(
        "detect_spike -> spike_date=%s service=%s provider=%s region=%s baseline=$%.2f spiked=$%.2f (+%.1f%%) drivers=%d",
        spike_date, best["service"], best["provider"], best["region"], best["baseline"], best["spiked"], best["pct_increase"], len(driver_ids),
    )
    return {
        "spike_date": spike_date.isoformat(),
        "service": best["service"],
        "provider": best["provider"],
        "region": best["region"],
        "baseline_daily": round(float(best["baseline"]), 2),
        "spiked_daily": round(float(best["spiked"]), 2),
        "pct_increase": round(float(best["pct_increase"]), 1),
        "delta_usd_per_day": round(float(best["delta"]), 2),
        "driver_resource_ids": driver_ids[:_MAX_RECORDS],
        "driver_summary": driver_summary,
    }


def _cap_round_robin(items: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    """Cap a list to `limit` items, round-robin across `key` groups.

    Prevents one large group (e.g. idle_compute) from crowding out smaller
    ones (e.g. unattached_volume) when a flat truncation would.
    """
    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for item in items:
        groups[item[key]].append(item)

    capped: list[dict[str, Any]] = []
    while groups and len(capped) < limit:
        for group_key in list(groups.keys()):
            if len(capped) >= limit:
                break
            capped.append(groups[group_key].popleft())
            if not groups[group_key]:
                del groups[group_key]
    return capped


def _candidate_resource_ids(
    adapter: CloudAdapter,
    service: str | None,
    window_start: date,
    window_end: date,
    region: str | None = None,
    extra_filters: dict[str, str] | None = None,
) -> list[str]:
    services = [service] if service else list(_ALL_SERVICES)
    ids: set[str] = set()
    for svc in services:
        df = adapter.get_cost(
            window_start, window_end, service=svc, region=region, group_by=["resource_id"], extra_filters=extra_filters
        )
        ids.update(df["resource_id"].tolist())
    return sorted(ids)


def find_idle_resources(
    provider: str | None = None,
    service: str | None = None,
    resource_ids: list[str] | None = None,
    region: str | None = None,
    environment: str | None = None,
    business_unit: str | None = None,
    instance_type: str | None = None,
) -> dict[str, Any]:
    """Find resources that are running but wasted, per service archetype.

    Rules: idle compute (CPU p95 < 5% while running), unattached EBS volumes,
    zero-invocation functions, and cold blobs (last access > 90 days). Use
    after detect_spike to check whether the driver resources are idle, or on
    its own to sweep for waste. Do NOT use this to compute savings — use
    recommend for that. `region`/`environment`/`business_unit`/`instance_type`
    only narrow the initial sweep — they're ignored when `resource_ids` is
    given explicitly.
    """
    _log.info(
        "find_idle_resources(provider=%s, service=%s, resource_ids=%s, region=%s, environment=%s, business_unit=%s, instance_type=%s)",
        provider, service, f"{len(resource_ids)} given" if resource_ids else "none (sweeping all)",
        region, environment, business_unit, instance_type,
    )
    adapter = _adapter_for(provider)
    filters = _extra_filters(environment, business_unit, instance_type)
    max_date = _max_available_date()
    window_start = max_date - timedelta(days=_UTILIZATION_WINDOW_DAYS - 1)
    window_days = _UTILIZATION_WINDOW_DAYS

    candidate_ids = list(dict.fromkeys(resource_ids)) if resource_ids else _candidate_resource_ids(
        adapter, service, window_start, max_date, region=region, extra_filters=filters
    )
    if not candidate_ids:
        _log.info("find_idle_resources -> no candidate resources in scope")
        return {"idle_resources": [], "count": 0}

    meta = adapter.get_metadata(candidate_ids).set_index("resource_id")
    util = adapter.get_utilization(candidate_ids)

    idle: list[dict[str, Any]] = []
    for rid in candidate_ids:
        if rid not in meta.index:
            continue
        row = meta.loc[rid]
        svc = row["service"]
        u = util[util["resource_id"] == rid].sort_values("date")
        monthly_cost = _monthly_cost(adapter, rid, window_start, max_date)

        if svc in _COMPUTE_SERVICES:
            avg_cpu = u["cpu_utilization_p95"].mean()
            last_status = u["status"].iloc[-1] if not u.empty else row["status"]
            if pd.notna(avg_cpu) and last_status == "running" and avg_cpu < _IDLE_CPU_THRESHOLD:
                idle.append(
                    {
                        "resource_id": rid,
                        "service": svc,
                        "reason": "idle_compute",
                        "evidence": {
                            "avg_cpu_utilization_p95": round(float(avg_cpu), 1),
                            "status": last_status,
                            "window_days": window_days,
                        },
                        "monthly_cost_usd": monthly_cost,
                    }
                )
        elif svc in _VOLUME_SERVICES:
            if pd.isna(row["attached_to"]) or row["attached_to"] in (None, ""):
                idle.append(
                    {
                        "resource_id": rid,
                        "service": svc,
                        "reason": "unattached_volume",
                        "evidence": {"attached_to": None, "storage_gb": row["storage_gb"]},
                        "monthly_cost_usd": monthly_cost,
                    }
                )
        elif svc in _FUNCTION_SERVICES:
            total_invocations = u["invocations"].fillna(0).sum()
            if total_invocations == 0:
                idle.append(
                    {
                        "resource_id": rid,
                        "service": svc,
                        "reason": "zero_invocations",
                        "evidence": {"invocations": 0, "window_days": window_days},
                        "monthly_cost_usd": monthly_cost,
                    }
                )
        elif svc in _BLOB_SERVICES:
            last_access = u["last_access_days"].iloc[-1] if not u.empty else None
            if last_access is not None and pd.notna(last_access) and last_access > _COLD_BLOB_DAYS:
                idle.append(
                    {
                        "resource_id": rid,
                        "service": svc,
                        "reason": "cold_blob",
                        "evidence": {"last_access_days": int(last_access), "storage_gb": row["storage_gb"]},
                        "monthly_cost_usd": monthly_cost,
                    }
                )

    reason_counts = {r: sum(1 for i in idle if i["reason"] == r) for r in dict.fromkeys(i["reason"] for i in idle)}
    _log.info("find_idle_resources -> %d idle of %d candidates checked, by reason: %s", len(idle), len(candidate_ids), reason_counts)
    return {"idle_resources": _cap_round_robin(idle, "reason", _MAX_RECORDS), "count": len(idle)}


def list_resources(
    provider: str | None = None,
    service: str | None = None,
    resource_ids: list[str] | None = None,
    region: str | None = None,
    environment: str | None = None,
    business_unit: str | None = None,
    instance_type: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """List resources with their current status, instance type, and monthly
    cost — a plain inventory, with NO idle/waste heuristics applied.

    Use for "what's running", "show our EC2 instances", "what's stopped in
    Azure", "list resources tagged prod" — anything asking what exists or
    its current state. Do NOT use this to find waste — use
    find_idle_resources for "idle"/"unused"/"wasted" questions; unlike that
    tool, this one never inspects CPU/invocations/last-access, it only
    reports `status` ("running" or "stopped") as recorded. Omit `status` to
    return both.
    """
    _log.info(
        "list_resources(provider=%s, service=%s, resource_ids=%s, region=%s, environment=%s, business_unit=%s, instance_type=%s, status=%s)",
        provider, service, f"{len(resource_ids)} given" if resource_ids else "none (sweeping all)",
        region, environment, business_unit, instance_type, status,
    )
    adapter = _adapter_for(provider)
    filters = _extra_filters(environment, business_unit, instance_type)
    max_date = _max_available_date()
    window_start = max_date - timedelta(days=_UTILIZATION_WINDOW_DAYS - 1)

    candidate_ids = list(dict.fromkeys(resource_ids)) if resource_ids else _candidate_resource_ids(
        adapter, service, window_start, max_date, region=region, extra_filters=filters
    )
    if not candidate_ids:
        _log.info("list_resources -> no candidate resources in scope")
        return {"resources": [], "count": 0}

    meta = adapter.get_metadata(candidate_ids).set_index("resource_id")

    resources: list[dict[str, Any]] = []
    for rid in candidate_ids:
        if rid not in meta.index:
            continue
        row = meta.loc[rid]
        if status is not None and row["status"] != status:
            continue
        resources.append(
            {
                "resource_id": rid,
                "resource_name": row["resource_name"],
                "service": row["service"],
                "provider": row["provider"],
                "region": row["region"],
                "instance_type": row["instance_type"] if pd.notna(row["instance_type"]) else None,
                "environment": row["environment"],
                "business_unit": row["business_unit"],
                "status": row["status"],
                "monthly_cost_usd": _monthly_cost(adapter, rid, window_start, max_date),
            }
        )

    resources.sort(key=lambda r: r["monthly_cost_usd"], reverse=True)
    _log.info("list_resources -> %d of %d candidates matched (status=%s)", len(resources), len(candidate_ids), status)
    return {"resources": resources[:_MAX_RECORDS], "count": len(resources)}


def recommend(resource_ids: list[str]) -> dict[str, Any]:
    """Turn facts about specific resources into concrete cost-saving actions.

    Pure rules over facts (never a data column): idle compute -> terminate,
    underutilized compute -> downsize, unattached volume -> delete_volume,
    zero-invocation function -> decommission, cold blob -> move_to_archive_tier.
    Use after find_idle_resources to price out the recommended fix. Do NOT
    use this to discover waste — use find_idle_resources for that.
    """
    _log.info("recommend(%d resource_ids)", len(resource_ids) if resource_ids else 0)
    if not resource_ids:
        _log.info("recommend -> no resource_ids given")
        return {"recommendations": [], "total_monthly_saving_usd": 0.0}

    adapter = _ALL
    max_date = _max_available_date()
    window_start = max_date - timedelta(days=_UTILIZATION_WINDOW_DAYS - 1)
    window_days = _UTILIZATION_WINDOW_DAYS

    unique_ids = list(dict.fromkeys(resource_ids))
    meta = adapter.get_metadata(unique_ids).set_index("resource_id")
    util = adapter.get_utilization(unique_ids)

    recommendations: list[dict[str, Any]] = []
    total_saving = 0.0

    for rid in unique_ids:
        if rid not in meta.index:
            continue
        row = meta.loc[rid]
        svc = row["service"]
        u = util[util["resource_id"] == rid].sort_values("date")
        monthly_cost = _monthly_cost(adapter, rid, window_start, max_date)

        rec: dict[str, Any] | None = None

        if svc in _COMPUTE_SERVICES:
            avg_cpu = u["cpu_utilization_p95"].mean()
            if pd.notna(avg_cpu):
                if avg_cpu < _IDLE_CPU_THRESHOLD:
                    saving = monthly_cost
                    rec = {
                        "action": "terminate",
                        "reason": f"avg CPU utilization {avg_cpu:.1f}% over last {window_days} days is idle (<5%)",
                        "evidence": {"avg_cpu_utilization_p95": round(float(avg_cpu), 1), "instance_type": row["instance_type"]},
                    }
                elif avg_cpu < _UNDERUTILIZED_CPU_THRESHOLD:
                    saving = round(monthly_cost * 0.5, 2)
                    rec = {
                        "action": "downsize",
                        "reason": f"avg CPU utilization {avg_cpu:.1f}% over last {window_days} days is underutilized (5-40%)",
                        "evidence": {"avg_cpu_utilization_p95": round(float(avg_cpu), 1), "instance_type": row["instance_type"]},
                    }
        elif svc in _VOLUME_SERVICES:
            if pd.isna(row["attached_to"]) or row["attached_to"] in (None, ""):
                saving = monthly_cost
                rec = {
                    "action": "delete_volume",
                    "reason": "volume is unattached (attached_to is null)",
                    "evidence": {"attached_to": None, "storage_gb": row["storage_gb"]},
                }
        elif svc in _FUNCTION_SERVICES:
            total_invocations = u["invocations"].fillna(0).sum()
            if total_invocations == 0:
                saving = monthly_cost
                rec = {
                    "action": "decommission",
                    "reason": f"zero invocations over last {window_days} days",
                    "evidence": {"invocations": 0},
                }
        elif svc in _BLOB_SERVICES:
            last_access = u["last_access_days"].iloc[-1] if not u.empty else None
            if last_access is not None and pd.notna(last_access) and last_access > _COLD_BLOB_DAYS:
                saving = round(monthly_cost * 0.8, 2)
                rec = {
                    "action": "move_to_archive_tier",
                    "reason": f"last accessed {int(last_access)} days ago (>90)",
                    "evidence": {"last_access_days": int(last_access), "storage_gb": row["storage_gb"]},
                }

        if rec is not None:
            rec = {"resource_id": rid, "service": svc, "monthly_saving_usd": saving, **rec}
            recommendations.append(rec)
            total_saving += saving

    _log.info("recommend -> %d recommendation(s), total_monthly_saving_usd=$%.2f", len(recommendations), total_saving)
    return {
        "recommendations": _cap_round_robin(recommendations, "action", _MAX_RECORDS),
        "total_monthly_saving_usd": round(total_saving, 2),
    }
