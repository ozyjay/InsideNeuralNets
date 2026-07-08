from pathlib import Path

import app as demo_app
from app import RunRequest


def test_index_html_mentions_fastapi_demo_title() -> None:
    html = demo_app.index()

    assert "How Does a Neural Network See?" in html
    assert "trained vision model" in html


def test_index_html_uses_selectable_diagram_for_model_layers() -> None:
    html = demo_app.index()

    for layer in (
        "Conv 1",
        "Pool 1",
        "Conv 2",
        "Residual 1",
        "Residual 4",
        "Early block",
        "Final conv",
        "Avg pool",
    ):
        assert layer in html

    assert "Model layer explorer" in html
    assert "Input image preview" not in html
    assert "Layer timeline" not in html
    assert "Top-5 predictions" not in html
    assert 'id="predictions"' not in html
    assert "Predictions are shown by selecting the Prediction stage" in html


def test_index_html_includes_classifier_visual_sketch() -> None:
    html = demo_app.index()

    assert "Classifier layer visual sketch" in html
    assert "classifier-flow" in html
    assert "Top classifier score bars" in html
    assert "Scores, not certainty" in html


def test_index_html_includes_colour_theme_selector() -> None:
    html = demo_app.index()

    assert 'id="themeSelect"' in html
    assert "Colour theme selector" in html
    assert "Laboratory microscope" in html
    assert "Warm classroom" in html
    assert "Neural neon" in html
    assert "Calm deep learning" in html
    assert "Monochrome signal" in html
    assert 'body[data-theme="calm"]' in html
    assert "insideAlexNetColourTheme" in html


def test_index_html_includes_model_selector() -> None:
    html = demo_app.index()
    models = demo_app.get_models()

    assert 'id="modelSelect"' in html
    assert "Vision model selector" in html
    assert "ResNet-50" in html
    assert "MobileNetV3 Large" in html
    assert models["default"] == "alexnet"
    assert {model["key"] for model in models["models"]} == {"alexnet", "resnet50", "mobilenet_v3_large"}


def test_index_html_includes_activation_colour_selector() -> None:
    html = demo_app.index()

    assert 'id="activationColourSelect"' in html
    assert "Activation image colour selector" in html
    assert "Layer image colours" in html
    assert "activation_colour_map" in html
    assert "insideAlexNetActivationColourMap" in html
    assert "It does not change the model result" in html


def test_list_images_handles_empty_folder(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(demo_app, "DEMO_IMAGES_DIR", tmp_path)

    response = demo_app.list_images()

    assert response["images"] == []
    assert "No curated images" in response["empty_message"]


def test_fallback_run_returns_graceful_placeholder(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"discovery only")
    monkeypatch.setattr(demo_app, "DEMO_IMAGES_DIR", tmp_path)

    response = demo_app.run_demo(RunRequest(image_name="sample.jpg", fallback=True))

    assert response.status_code == 200
    assert b"Fallback replay assets" in response.body

import base64
from io import BytesIO

from PIL import Image

from app import CameraRunRequest, _decode_camera_image, run_camera_demo
from src.model import AlexNetAnalysis, Prediction


def test_index_html_includes_local_camera_privacy_wording() -> None:
    html = demo_app.index()

    assert "Live camera" in html
    assert "are not saved" in html
    assert "/api/run-camera" in html
    assert "Start continuous model" in html
    assert "Updates with each live frame" in html
    assert "camera-capture-source" in html
    assert "Live local camera input" in html


def test_decode_camera_image_accepts_data_url() -> None:
    image = Image.new("RGB", (8, 6), color=(10, 20, 30))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    decoded = _decode_camera_image(data_url)

    assert decoded.mode == "RGB"
    assert decoded.size == (8, 6)


def test_camera_fallback_returns_graceful_message() -> None:
    response = run_camera_demo(CameraRunRequest(image_data="not-used", fallback=True))

    assert response.status_code == 200
    assert b"Fallback replay uses curated" in response.body


def test_camera_request_can_disable_visualisations() -> None:
    request = CameraRunRequest(image_data="not-used", include_visualisations=False)

    assert request.include_visualisations is False


def test_camera_request_accepts_selected_visualisation_keys() -> None:
    request = CameraRunRequest(image_data="not-used", visualisation_keys=["conv1"])

    assert request.visualisation_keys == ["conv1"]


def test_run_request_accepts_model_and_activation_colour_map() -> None:
    request = RunRequest(image_name="sample.jpg", model_key="resnet50", activation_colour_map="neon")

    assert request.model_key == "resnet50"
    assert request.activation_colour_map == "neon"


def test_live_response_passes_selected_visualisation_keys(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_analysis(
        image,
        *,
        include_visualisations=True,
        visualisation_keys=None,
        activation_colour_map="aurora",
        model_key="alexnet",
    ):
        captured["model_key"] = model_key
        captured["include_visualisations"] = include_visualisations
        captured["visualisation_keys"] = visualisation_keys
        captured["activation_colour_map"] = activation_colour_map
        return AlexNetAnalysis(predictions=[Prediction(label="test", probability=0.5)], visualisations=[])

    monkeypatch.setattr(demo_app, "run_model_analysis", fake_analysis)
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))

    response = demo_app._run_live_analysis_response(
        image,
        source="camera",
        include_visualisations=True,
        visualisation_keys={"conv1"},
        activation_colour_map="calm",
        model_key="resnet50",
    )

    assert response.status_code == 200
    assert captured == {
        "model_key": "resnet50",
        "include_visualisations": True,
        "visualisation_keys": {"conv1"},
        "activation_colour_map": "calm",
    }
