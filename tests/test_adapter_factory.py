from adapters import factory
from adapters.base import CloudAdapter
from adapters.synthetic import SyntheticAdapter


def test_aws_and_azure_are_registered_by_default():
    assert set(factory.all_providers()) == {"AWS", "Azure"}


def test_get_returns_the_registered_adapter():
    aws = factory.get("AWS")
    assert isinstance(aws, SyntheticAdapter)
    assert aws.provider == "AWS"


def test_get_unknown_provider_raises_key_error():
    try:
        factory.get("GCP")
    except KeyError as exc:
        assert "GCP" in str(exc)
        assert "AWS" in str(exc) and "Azure" in str(exc)
    else:
        raise AssertionError("expected KeyError for unregistered provider")


def test_all_adapters_returns_every_registered_instance():
    adapters = factory.all_adapters()
    assert len(adapters) == 2
    assert {a.provider for a in adapters} == {"AWS", "Azure"}


def test_register_can_swap_an_adapter():
    class _FakeAdapter:
        provider = "AWS"

        def get_cost(self, *a, **k): ...
        def get_utilization(self, *a, **k): ...
        def get_metadata(self, *a, **k): ...
        def list_services(self) -> list[str]:
            return ["EC2"]

    original = factory.get("AWS")
    try:
        fake = _FakeAdapter()
        factory.register("AWS", fake)
        assert factory.get("AWS") is fake
        assert factory.get("AWS").list_services() == ["EC2"]
    finally:
        factory.register("AWS", original)  # restore, since factory state is module-global


def test_synthetic_adapters_expose_list_services():
    aws_services = factory.get("AWS").list_services()
    azure_services = factory.get("Azure").list_services()
    assert set(aws_services) == {"EC2", "EBS", "Lambda"}
    assert set(azure_services) == {"Virtual Machine", "Azure Functions", "Blob Storage"}


def test_registered_adapters_satisfy_the_cloud_adapter_protocol():
    for provider in factory.all_providers():
        adapter: CloudAdapter = factory.get(provider)
        assert hasattr(adapter, "get_cost")
        assert hasattr(adapter, "get_utilization")
        assert hasattr(adapter, "get_metadata")
        assert hasattr(adapter, "list_services")
