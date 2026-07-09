import pytest

from resolvers.canonical_request import CanonicalRequest
from tools.finops_tools import cost_trend, detect_spike, find_idle_resources, recommend


def test_to_kwargs_omits_none_fields():
    request = CanonicalRequest(provider="AWS", service="EC2")
    kwargs = request.to_kwargs("cost_trend")
    assert kwargs == {"service": "EC2", "provider": "AWS"}


def test_to_kwargs_per_tool_field_projection():
    request = CanonicalRequest(
        provider="AWS", service="EC2", region="us-east-1", resource_ids=["i-1"], lookback_days=14
    )
    assert request.to_kwargs("cost_trend") == {"service": "EC2", "provider": "AWS", "region": "us-east-1"}
    assert request.to_kwargs("detect_spike") == {
        "lookback_days": 14, "provider": "AWS", "service": "EC2", "region": "us-east-1"
    }
    assert request.to_kwargs("find_idle_resources") == {
        "provider": "AWS", "service": "EC2", "resource_ids": ["i-1"], "region": "us-east-1"
    }
    assert request.to_kwargs("recommend") == {"resource_ids": ["i-1"]}


def test_to_kwargs_projects_environment_and_business_unit_for_every_filterable_tool():
    # Regression: _TOOL_FIELDS was written in Phase 3 before cost_trend/
    # detect_spike/find_idle_resources accepted region/environment/
    # business_unit (added in Phase 4). Without these three in the table,
    # a validated CanonicalRequest.environment/business_unit is silently
    # dropped before the real tool call, so "Production-tagged" or
    # "Finance business_unit" queries would resolve fine but filter nothing.
    request = CanonicalRequest(
        provider="AWS", service="EC2", environment="prod", business_unit="Finance"
    )
    assert request.to_kwargs("cost_trend") == {
        "service": "EC2", "provider": "AWS", "environment": "prod", "business_unit": "Finance"
    }
    assert request.to_kwargs("detect_spike") == {
        "provider": "AWS", "service": "EC2", "environment": "prod", "business_unit": "Finance"
    }
    assert request.to_kwargs("find_idle_resources") == {
        "provider": "AWS", "service": "EC2", "environment": "prod", "business_unit": "Finance"
    }


def test_to_kwargs_unknown_tool_raises():
    request = CanonicalRequest()
    with pytest.raises(ValueError):
        request.to_kwargs("not_a_real_tool")


def test_to_kwargs_output_is_actually_callable_against_the_real_tools():
    # Proves to_kwargs()'s output is a valid call for each tool's *current*
    # signature, not just a plausible-looking dict.
    request = CanonicalRequest(provider="AWS", service="EC2", start="2026-06-01", end="2026-06-30")
    result = cost_trend(**request.to_kwargs("cost_trend"))
    assert result["total"] > 0

    request = CanonicalRequest(provider="AWS", lookback_days=30)
    result = detect_spike(**request.to_kwargs("detect_spike"))
    assert result["spike_date"] == "2026-06-16"

    request = CanonicalRequest(provider="AWS", service="EC2")
    result = find_idle_resources(**request.to_kwargs("find_idle_resources"))
    assert "idle_resources" in result

    request = CanonicalRequest(resource_ids=[])
    result = recommend(**request.to_kwargs("recommend"))
    assert result["recommendations"] == []
