from pathlib import Path

import app as demo_app
from app import RunRequest


def test_index_html_mentions_fastapi_demo_title() -> None:
    html = demo_app.index()

    assert "How Does a Neural Network See?" in html
    assert "trained vision model" in html


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


def test_index_html_includes_local_camera_privacy_wording() -> None:
    html = demo_app.index()

    assert "Live camera" in html
    assert "are not saved" in html
    assert "/api/run-camera" in html
    assert "Start continuous AlexNet" in html
    assert "Updates with each live frame" in html


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
