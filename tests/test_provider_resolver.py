from resolvers.provider_resolver import resolve_provider_and_service


def test_no_service_named_no_provider_scans_both():
    r = resolve_provider_and_service(None, None)
    assert r.provider is None
    assert r.service is None
    assert r.outcome == "resolved"


def test_no_service_named_with_explicit_provider_keeps_the_provider():
    r = resolve_provider_and_service(None, "AWS")
    assert r.provider == "AWS"
    assert r.service is None
    assert r.outcome == "resolved"


def test_unambiguous_service_infers_provider_directly():
    r = resolve_provider_and_service("EC2", None)
    assert r.provider == "AWS"
    assert r.service == "EC2"
    assert r.outcome == "resolved"

    r = resolve_provider_and_service("azure functions", None)
    assert r.provider == "Azure"
    assert r.service == "Azure Functions"
    assert r.outcome == "resolved"


def test_ambiguous_service_with_no_explicit_provider_needs_clarification():
    r = resolve_provider_and_service("vm", None)
    assert r.outcome == "clarification_needed"
    assert r.provider is None
    assert r.service is None
    assert set(r.clarification_options) == {"AWS EC2", "Azure Virtual Machine"}


def test_ambiguous_service_with_explicit_provider_resolves_directly_no_clarification():
    r = resolve_provider_and_service("vm", "Azure")
    assert r.outcome == "resolved"
    assert r.provider == "Azure"
    assert r.service == "Virtual Machine"


def test_explicit_provider_matching_the_service_resolves():
    r = resolve_provider_and_service("EC2", "AWS")
    assert r.outcome == "resolved"
    assert r.provider == "AWS"
    assert r.service == "EC2"


def test_impossible_combos_are_flagged_not_silently_passed_through():
    cases = [
        ("EC2", "Azure"),
        ("Blob Storage", "AWS"),
        ("Lambda", "Azure"),
        ("Virtual Machine", "AWS"),
    ]
    for raw_service, explicit_provider in cases:
        r = resolve_provider_and_service(raw_service, explicit_provider)
        assert r.outcome == "impossible_combo", f"{raw_service} + {explicit_provider} should be impossible"
        assert r.provider == explicit_provider
        assert r.service is None


def test_unresolvable_service_name():
    r = resolve_provider_and_service("whatzit service", None)
    assert r.outcome == "unresolved_service"
    assert r.service is None


def test_unresolvable_service_name_preserves_explicit_provider():
    r = resolve_provider_and_service("whatzit service", "AWS")
    assert r.outcome == "unresolved_service"
    assert r.provider == "AWS"
