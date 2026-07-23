from types import SimpleNamespace

from scripts import cache_models
from src.model import ModelUnavailableError


def test_cache_models_loads_each_requested_model(monkeypatch) -> None:
    loaded: list[str] = []

    def fake_load_model(model_key: str):
        loaded.append(model_key)
        return SimpleNamespace(label=model_key)

    fake_load_model.cache_clear = lambda: None
    monkeypatch.setattr(cache_models, "load_model", fake_load_model)

    assert cache_models.cache_models(("alexnet", "resnet50")) == []
    assert loaded == ["alexnet", "resnet50"]


def test_cache_models_reports_unavailable_models(monkeypatch) -> None:
    def fake_load_model(model_key: str):
        raise ModelUnavailableError(f"{model_key} unavailable")

    fake_load_model.cache_clear = lambda: None
    monkeypatch.setattr(cache_models, "load_model", fake_load_model)

    assert cache_models.cache_models(("alexnet",)) == ["alexnet"]


def test_cache_models_cli_defaults_to_every_supported_model(monkeypatch) -> None:
    requested: list[tuple[str, ...]] = []

    def fake_cache_models(model_keys) -> list[str]:
        requested.append(tuple(model_keys))
        return []

    monkeypatch.setattr(cache_models, "cache_models", fake_cache_models)

    assert cache_models.main([]) == 0
    assert requested == [cache_models.SUPPORTED_MODEL_KEYS]
