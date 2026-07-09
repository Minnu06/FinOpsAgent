from tools.finops_tools import cost_trend, detect_spike, find_idle_resources, recommend


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


def test_cost_trend_basic():
    result = cost_trend("2026-06-01", "2026-06-30", service="EC2", provider="AWS")
    assert result["total"] > 0
    assert len(result["series"]) <= 30
    assert result["avg_daily"] > 0


def test_tool_results_stay_under_record_cap():
    assert len(detect_spike()["driver_resource_ids"]) <= 30
    assert len(find_idle_resources()["idle_resources"]) <= 30
    assert len(cost_trend("2026-04-02", "2026-06-30")["series"]) <= 30
