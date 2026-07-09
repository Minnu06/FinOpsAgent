from resolvers.service_registry import all_concrete_names, all_providers, concrete_name_for, providers_for, resolve


def test_catalog_covers_all_17_services():
    assert len(all_concrete_names()) == 17
    assert len(all_concrete_names("AWS")) == 9
    assert len(all_concrete_names("Azure")) == 8
    assert set(all_providers()) == {"AWS", "Azure"}


def test_exact_match_is_unambiguous_even_when_also_a_concept_word():
    match = resolve("Virtual Machine")
    assert match.matched
    assert match.confidence == "exact"
    assert [c.concrete_name for c in match.candidates] == ["Virtual Machine"]
    assert match.candidates[0].provider == "Azure"


def test_exact_match_case_insensitive():
    match = resolve("ec2")
    assert match.matched
    assert match.candidates[0].concrete_name == "EC2"
    assert match.candidates[0].provider == "AWS"


def test_service_specific_synonyms_are_unambiguous():
    cases = {
        "lambda": ("Lambda", "AWS"),
        "s3": ("S3", "AWS"),
        "eks": ("EKS", "AWS"),
        "aks": ("AKS", "Azure"),
        "rds": ("RDS", "AWS"),
        "elb": ("ELB", "AWS"),
        "cloudfront": ("CloudFront", "AWS"),
        "ecs": ("ECS", "AWS"),
        "sql database": ("SQL Database", "Azure"),
        "cosmos db": ("Cosmos DB", "Azure"),
        "app service": ("App Service", "Azure"),
        "azure functions": ("Azure Functions", "Azure"),
        "managed disk": ("Managed Disk", "Azure"),
    }
    for raw, (expected_name, expected_provider) in cases.items():
        match = resolve(raw)
        assert match.matched, f"{raw!r} should resolve"
        assert len(match.candidates) == 1, f"{raw!r} should be unambiguous"
        assert match.candidates[0].concrete_name == expected_name
        assert match.candidates[0].provider == expected_provider


def test_ambiguous_concept_words_yield_multiple_candidates():
    cases = {
        "vm": {"EC2", "Virtual Machine"},
        "instance": {"EC2", "Virtual Machine"},
        "compute": {"EC2", "Virtual Machine"},
        "server": {"EC2", "Virtual Machine"},
        "storage": {"S3", "Blob Storage"},
        "bucket": {"S3", "Blob Storage"},
        "blob": {"S3", "Blob Storage"},
        "object storage": {"S3", "Blob Storage"},
        "function": {"Lambda", "Azure Functions"},
        "functions": {"Lambda", "Azure Functions"},
        "serverless": {"Lambda", "Azure Functions"},
        "kubernetes": {"EKS", "AKS"},
        "k8s": {"EKS", "AKS"},
    }
    for raw, expected_names in cases.items():
        match = resolve(raw)
        assert match.matched, f"{raw!r} should resolve"
        assert len(match.candidates) == 2, f"{raw!r} should be ambiguous (2 candidates)"
        assert {c.concrete_name for c in match.candidates} == expected_names
        assert {c.provider for c in match.candidates} == {"AWS", "Azure"}


def test_single_provider_concepts_are_never_ambiguous():
    for raw in ("rds", "elb", "cloudfront", "ecs", "sql database", "cosmos db", "app service"):
        match = resolve(raw)
        assert len(match.candidates) == 1


def test_fuzzy_fallback_handles_typos():
    match = resolve("lamda")  # missing 'b'
    assert match.matched
    assert match.confidence == "fuzzy"
    assert match.candidates[0].concrete_name == "Lambda"


def test_unresolvable_string_returns_no_candidates():
    match = resolve("whatzit service")
    assert not match.matched
    assert match.candidates == ()


def test_empty_string_returns_no_candidates():
    match = resolve("")
    assert not match.matched
    match = resolve("   ")
    assert not match.matched


def test_providers_for():
    assert providers_for("compute") == ["AWS", "Azure"]
    assert providers_for("EC2") == ["AWS"]
    assert providers_for("relational_database") == ["AWS"]


def test_concrete_name_for():
    assert concrete_name_for("compute", "AWS") == "EC2"
    assert concrete_name_for("compute", "Azure") == "Virtual Machine"
    assert concrete_name_for("compute", "GCP") is None
