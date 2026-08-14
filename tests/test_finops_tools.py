from tools.finops_tools import cost_trend, detect_spike, find_idle_resources, list_resources, recommend


def test_detect_spike_finds_seeded_anomaly():
    result = detect_spike()
    assert result["spike_date"] == "2026-06-16"
    assert result["service"] == "EC2"
    assert result["provider"] == "AWS"
    assert result["region"] == "us-east-1"
    assert len(result["driver_resource_ids"]) == 12
    assert result["driver_summary"]["count"] == 12
    assert result["driver_summary"]["instance_type"] == "m5.4xlarge"
    assert result["driver_summary"]["environment"] == "test"
    assert result["pct_increase"] > 25


def test_find_idle_resources_covers_all_four_archetypes():
    result = find_idle_resources()
    reasons = {r["reason"] for r in result["idle_resources"]}
    assert result["count"] > 0
    assert "idle_compute" in reasons
    assert "unattached_volume" in reasons
    assert "zero_invocations" in reasons
    assert "cold_blob" in reasons


def test_find_idle_resources_flags_spike_drivers_as_idle():
    spike = detect_spike()
    result = find_idle_resources(resource_ids=spike["driver_resource_ids"])
    assert result["count"] == 12
    assert all(r["reason"] == "idle_compute" for r in result["idle_resources"])


def test_recommend_positive_total_saving():
    spike = detect_spike()
    result = recommend(spike["driver_resource_ids"])
    assert result["total_monthly_saving_usd"] > 0
    assert len(result["recommendations"]) == 12
    assert all(r["action"] == "terminate" for r in result["recommendations"])


def test_recommend_empty_input():
    result = recommend([])
    assert result["recommendations"] == []
    assert result["total_monthly_saving_usd"] == 0.0


def test_list_resources_returns_both_statuses_when_omitted():
    result = list_resources(provider="AWS", service="EC2")
    assert result["count"] > 0
    statuses = {r["status"] for r in result["resources"]}
    assert statuses <= {"running", "stopped"}


def test_list_resources_status_filter_actually_filters():
    running = list_resources(provider="AWS", service="EC2", status="running")
    stopped = list_resources(provider="AWS", service="EC2", status="stopped")
    assert all(r["status"] == "running" for r in running["resources"])
    assert all(r["status"] == "stopped" for r in stopped["resources"])
    assert running["count"] + stopped["count"] == list_resources(provider="AWS", service="EC2")["count"]


def test_list_resources_instance_type_filter():
    result = list_resources(provider="AWS", service="EC2", instance_type="m5.2xlarge")
    assert result["count"] > 0
    assert all(r["instance_type"] == "m5.2xlarge" for r in result["resources"])


def test_list_resources_applies_no_idle_heuristics():
    # The spike drivers are idle_compute per find_idle_resources' CPU rule,
    # but list_resources has no utilization opinion — it must still report
    # all of them, just tagged with their recorded status, not a waste reason.
    spike = detect_spike()
    idle = find_idle_resources(resource_ids=spike["driver_resource_ids"])
    assert idle["count"] == len(spike["driver_resource_ids"])  # sanity: they are in fact idle

    result = list_resources(resource_ids=spike["driver_resource_ids"])
    assert result["count"] == len(spike["driver_resource_ids"])
    assert all(r["status"] == "running" for r in result["resources"])
    assert all("reason" not in r for r in result["resources"])


def test_list_resources_no_candidates_in_scope():
    result = list_resources(resource_ids=["not-a-real-id"])
    assert result == {"resources": [], "count": 0}


def test_cost_trend_basic():
    result = cost_trend("2026-06-01", "2026-06-30", service="EC2", provider="AWS")
    assert result["total"] > 0
    assert len(result["series"]) <= 30
    assert result["avg_daily"] > 0


def test_tool_results_stay_under_record_cap():
    assert len(detect_spike()["driver_resource_ids"]) <= 30
    assert len(find_idle_resources()["idle_resources"]) <= 30
    assert len(cost_trend("2026-04-02", "2026-06-30")["series"]) <= 30


# --- Reliability hardening: bad-argument validation and no-data contract ---


def test_cost_trend_rejects_unparseable_date():
    result = cost_trend(start="not-a-date", end="2026-06-30")
    assert result["status"] == "invalid_argument"
    assert "message" in result


def test_cost_trend_rejects_reversed_range():
    result = cost_trend(start="2026-06-30", end="2026-06-01")
    assert result["status"] == "invalid_argument"


def test_cost_trend_rejects_bad_granularity():
    # A value outside the day/week/month enum (e.g. from a local model that
    # doesn't strictly honor schema enums) must not silently fall through the
    # record-cap safeguard and return an unbounded series.
    result = cost_trend("2026-06-01", "2026-06-30", granularity="fortnight")
    assert result["status"] == "invalid_argument"


def test_cost_trend_no_data_is_distinguishable_from_zero_cost():
    result = cost_trend("2026-01-01", "2026-01-31", service="EC2", provider="AWS")
    assert result["status"] == "no_data"
    assert result["series"] == []


def test_cost_trend_success_has_no_status_key():
    # `"status" not in result` is the existing, tested signal (see
    # test_regression_queries.py) that a result is real data, not a
    # short-circuit — successful calls must not carry a status key.
    result = cost_trend("2026-06-01", "2026-06-30", service="EC2", provider="AWS")
    assert "status" not in result


def test_detect_spike_rejects_non_positive_lookback_days():
    result = detect_spike(lookback_days=0)
    assert result["status"] == "invalid_argument"
    result = detect_spike(lookback_days=-5)
    assert result["status"] == "invalid_argument"


def test_list_resources_rejects_bad_status():
    result = list_resources(status="deleted")
    assert result["status"] == "invalid_argument"
