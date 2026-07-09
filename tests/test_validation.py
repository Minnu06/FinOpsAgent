from resolvers.validation import validate

# Fake data-availability: mirrors what factory.get(provider).list_services()
# returns today (6 of the 17 catalog services actually have CSV rows).
_FAKE_AVAILABLE = {
    "AWS": {"EC2", "EBS", "Lambda"},
    "Azure": {"Virtual Machine", "Azure Functions", "Blob Storage"},
}


def _available(provider: str) -> set[str]:
    return _FAKE_AVAILABLE[provider]


def test_accepts_a_plain_unambiguous_query():
    result = validate("EC2", None, _available, region="us-east-1")
    assert result.kind is None
    assert result.request is not None
    assert result.request.provider == "AWS"
    assert result.request.service == "EC2"
    assert result.request.region == "us-east-1"


def test_accepts_no_service_named_scans_both():
    result = validate(None, None, _available)
    assert result.kind is None
    assert result.request.provider is None
    assert result.request.service is None


def test_accepts_explicit_provider_no_service():
    result = validate(None, "AWS", _available)
    assert result.kind is None
    assert result.request.provider == "AWS"
    assert result.request.service is None


def test_rejects_impossible_combos():
    cases = [
        ("EC2", "Azure"),
        ("Blob Storage", "AWS"),
        ("Lambda", "Azure"),
        ("Virtual Machine", "AWS"),
    ]
    for raw_service, explicit_provider in cases:
        result = validate(raw_service, explicit_provider, _available)
        assert result.kind == "invalid_request", f"{raw_service}+{explicit_provider} should be rejected"
        assert result.request is None
        assert explicit_provider in result.message


def test_valid_but_absent_service_reports_data_unavailable_not_invalid():
    # S3 is a real AWS service per the registry, just not in this dataset —
    # must be distinguishable from an impossible combo like "Azure EC2".
    result = validate("S3", "AWS", _available)
    assert result.kind == "data_unavailable"
    assert result.request is None
    assert "S3" in result.message

    result = validate("Cosmos DB", "Azure", _available)
    assert result.kind == "data_unavailable"


def test_data_unavailable_also_triggers_with_no_explicit_provider():
    # "S3" resolves unambiguously to AWS even with no explicit provider given.
    result = validate("S3", None, _available)
    assert result.kind == "data_unavailable"


def test_ambiguous_service_needs_clarification():
    result = validate("vm", None, _available)
    assert result.kind == "clarification_needed"
    assert result.request is None
    assert set(result.options) == {"AWS EC2", "Azure Virtual Machine"}


def test_ambiguous_service_with_explicit_provider_does_not_need_clarification():
    result = validate("vm", "AWS", _available)
    assert result.kind is None
    assert result.request.provider == "AWS"
    assert result.request.service == "EC2"


def test_unrecognized_service_name():
    result = validate("whatzit service", None, _available)
    assert result.kind == "unresolved_service"
    assert result.request is None


def test_instance_type_exact_match_infers_provider_with_no_service_named():
    result = validate(None, None, _available, instance_type="m5.2xlarge")
    assert result.kind is None
    assert result.request.provider == "AWS"
    assert result.request.instance_type == "m5.2xlarge"


def test_instance_type_ambiguous_size_word_needs_clarification():
    result = validate(None, None, _available, instance_type="large")
    assert result.kind == "clarification_needed"
    assert result.request is None
    assert set(result.options) == {"AWS m5.2xlarge", "Azure Standard_D4s_v5"}


def test_instance_type_ambiguous_size_word_resolved_by_service_provider():
    # "EC2" pins the provider to AWS, which narrows the otherwise-ambiguous
    # "large" down to exactly one candidate instead of asking to clarify.
    result = validate("EC2", None, _available, instance_type="large")
    assert result.kind is None
    assert result.request.provider == "AWS"
    assert result.request.instance_type == "m5.2xlarge"


def test_instance_type_mismatched_with_resolved_service_provider_is_rejected():
    # EC2 resolves to AWS; Standard_D4s_v5 is an Azure-only instance type.
    result = validate("EC2", None, _available, instance_type="Standard_D4s_v5")
    assert result.kind == "invalid_request"
    assert result.request is None
    assert "AWS" in result.message


def test_instance_type_mismatched_with_explicit_provider_is_rejected():
    result = validate(None, "Azure", _available, instance_type="m5.large")
    assert result.kind == "invalid_request"
    assert result.request is None


def test_unresolved_instance_type():
    result = validate(None, None, _available, instance_type="banana.xlarge")
    assert result.kind == "unresolved_instance_type"
    assert result.request is None


def test_scan_both_availability_check_never_calls_available_services_with_none():
    calls: list[str] = []

    def spying_available(provider: str) -> set[str]:
        calls.append(provider)
        return _FAKE_AVAILABLE[provider]

    # A service that only resolves for one provider anyway (EC2), but forces
    # the internal union-of-all-providers code path is instead exercised via
    # a case where provider stays None with a service present: not reachable
    # through provider_resolver today (documented as defensive), so we assert
    # the normal single-provider path only ever queries with real provider keys.
    validate("EC2", None, spying_available)
    assert None not in calls
    assert "multi" not in calls
