from src.activations import layer_specs_for_model, normalise_model_key
from src.model import supported_model_options


def test_supported_model_options_include_better_prediction_models() -> None:
    options = supported_model_options()
    keys = {option["key"] for option in options}

    assert keys == {"alexnet", "resnet50", "mobilenet_v3_large"}
    assert any(option["label"] == "ResNet-50" for option in options)
    assert all(option["layers"] for option in options)


def test_model_layer_specs_include_average_pool_summary() -> None:
    for model_key in ("alexnet", "resnet50", "mobilenet_v3_large"):
        labels = [spec.label for spec in layer_specs_for_model(model_key)]

        assert "Avg pool" in labels


def test_unknown_model_key_falls_back_to_alexnet() -> None:
    assert normalise_model_key("unknown") == "alexnet"
