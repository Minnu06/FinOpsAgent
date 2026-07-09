from resolvers.instance_type_registry import all_concrete_names, concrete_name_for, providers_for, resolve


def test_catalog_covers_all_12_instance_types():
    assert len(all_concrete_names()) == 12
    assert len(all_concrete_names("AWS")) == 6
    assert len(all_concrete_names("Azure")) == 6


def test_exact_match_infers_provider_unambiguously():
    match = resolve("m5.2xlarge")
    assert match.matched
    assert match.confidence == "exact"
    assert len(match.candidates) == 1
    assert match.candidates[0].provider == "AWS"

    match = resolve("Standard_D4s_v5")
    assert match.matched
    assert len(match.candidates) == 1
    assert match.candidates[0].provider == "Azure"


def test_exact_match_case_insensitive():
    match = resolve("standard_b2s")
    assert match.matched
    assert match.candidates[0].concrete_name == "Standard_B2s"
    assert match.candidates[0].provider == "Azure"


def test_type_specific_synonyms_are_unambiguous():
    cases = {
        "d8s v5": ("Standard_D8s_v5", "Azure"),
        "burstable": ("Standard_B2s", "Azure"),
        "m5 4xlarge": ("m5.4xlarge", "AWS"),
        "compute optimized instance": ("c5.4xlarge", "AWS"),
        "memory optimized instance": ("r5.2xlarge", "AWS"),
    }
    for raw, (expected_name, expected_provider) in cases.items():
        match = resolve(raw)
        assert match.matched, f"{raw!r} should resolve"
        assert len(match.candidates) == 1, f"{raw!r} should be unambiguous"
        assert match.candidates[0].concrete_name == expected_name
        assert match.candidates[0].provider == expected_provider


def test_ambiguous_size_words_yield_multiple_candidates_across_providers():
    cases = {
        "small": {"m5.large", "Standard_B2s"},
        "medium": {"m5.xlarge", "Standard_D2s_v5"},
        "large": {"m5.2xlarge", "Standard_D4s_v5"},
        "xlarge": {"m5.4xlarge", "Standard_D8s_v5"},
        "compute optimized": {"c5.4xlarge", "Standard_F8s_v2"},
        "memory optimized": {"r5.2xlarge", "Standard_E4s_v5"},
    }
    for raw, expected_names in cases.items():
        match = resolve(raw)
        assert match.matched, f"{raw!r} should resolve"
        assert len(match.candidates) == 2, f"{raw!r} should be ambiguous (2 candidates)"
        assert {c.concrete_name for c in match.candidates} == expected_names
        assert {c.provider for c in match.candidates} == {"AWS", "Azure"}


def test_exact_concrete_name_wins_over_ambiguous_size_word():
    # "m5.large" is an exact concrete name even though "large" alone is an
    # ambiguous cross-provider size tier — exact match must take precedence.
    match = resolve("m5.large")
    assert match.confidence == "exact"
    assert len(match.candidates) == 1
    assert match.candidates[0].provider == "AWS"


def test_fuzzy_fallback_handles_typos():
    match = resolve("m5.2xlrage")  # transposed letters
    assert match.matched
    assert match.confidence == "fuzzy"
    assert match.candidates[0].concrete_name == "m5.2xlarge"


def test_unresolvable_string_returns_no_candidates():
    match = resolve("whatzit instance")
    assert not match.matched
    assert match.candidates == ()


def test_empty_string_returns_no_candidates():
    assert not resolve("").matched
    assert not resolve("   ").matched


def test_providers_for():
    assert providers_for("large") == ["AWS", "Azure"]
    assert providers_for("m5.2xlarge") == ["AWS"]


def test_concrete_name_for():
    assert concrete_name_for("large", "AWS") == "m5.2xlarge"
    assert concrete_name_for("large", "Azure") == "Standard_D4s_v5"
    assert concrete_name_for("large", "GCP") is None
