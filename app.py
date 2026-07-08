"""FastAPI app for the Open Day vision model demo."""

from __future__ import annotations

import base64
import binascii
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from src.captions import CAPTIONS
from src.demo_images import DEMO_IMAGES_DIR, discover_images, format_image_label
from src.activations import normalise_model_key
from src.model import ModelUnavailableError, run_model_analysis, supported_model_options
from src.visualise import (
    ACTIVATION_COLOUR_MAP_OPTIONS,
    DEFAULT_ACTIVATION_COLOUR_MAP,
    normalise_activation_colour_map,
)

APP_TITLE = "InsideNeuralNets"
PREDICTION_LABEL_COUNT = 5
CLASSIFIER_LABEL_COUNT = 20
THEME_OPTIONS = (
    ("aurora", "Aurora booth"),
    ("laboratory", "Laboratory microscope"),
    ("classroom", "Warm classroom"),
    ("neon", "Neural neon"),
    ("calm", "Calm deep learning"),
    ("signal", "Monochrome signal"),
)

app = FastAPI(
    title="Vision Model Demo",
    description="Local-first Open Day demo showing vision model layer responses and predictions.",
    version="0.2.0",
)


class RunRequest(BaseModel):
    """Request body for running a supported model on a curated image."""

    image_name: str = Field(..., min_length=1)
    fallback: bool = False
    model_key: str = "alexnet"
    activation_colour_map: str = Field(DEFAULT_ACTIVATION_COLOUR_MAP, min_length=1, max_length=32)


class CameraRunRequest(BaseModel):
    """Request body for running a supported model on a locally captured camera frame."""

    image_data: str = Field(..., min_length=1)
    fallback: bool = False
    model_key: str = "alexnet"
    include_visualisations: bool = True
    visualisation_keys: list[str] | None = None
    activation_colour_map: str = Field(DEFAULT_ACTIVATION_COLOUR_MAP, min_length=1, max_length=32)


def _image_lookup() -> dict[str, Path]:
    """Return curated images keyed by filename."""
    return {path.name: path for path in discover_images(DEMO_IMAGES_DIR)}


def _find_demo_image(image_name: str) -> Path:
    """Return a validated curated image path or raise a 404."""
    image_path = _image_lookup().get(Path(image_name).name)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Curated image not found.")
    return image_path


def _decode_camera_image(image_data: str) -> Image.Image:
    """Decode a browser camera data URL without saving it to disk."""
    if "," in image_data:
        header, encoded = image_data.split(",", 1)
        if not header.startswith("data:image/"):
            raise ValueError("Camera frame must be an image data URL.")
    else:
        encoded = image_data

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Camera frame was not valid base64 image data.") from exc

    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Camera frame is too large for this local demo.")

    try:
        return Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Camera frame could not be opened as an image.") from exc


def _run_live_analysis_response(
    image: Image.Image,
    *,
    source: str,
    model_key: str = "alexnet",
    include_visualisations: bool = True,
    visualisation_keys: set[str] | None = None,
    activation_colour_map: str = DEFAULT_ACTIVATION_COLOUR_MAP,
) -> JSONResponse:
    """Run model analysis and return a consistent JSON response."""
    model_key = normalise_model_key(model_key)
    activation_colour_map = normalise_activation_colour_map(activation_colour_map)
    try:
        analysis = run_model_analysis(
            image,
            model_key=model_key,
            top_k=CLASSIFIER_LABEL_COUNT,
            include_visualisations=include_visualisations,
            visualisation_keys=visualisation_keys,
            activation_colour_map=activation_colour_map,
        )
    except ModelUnavailableError as exc:
        return JSONResponse(
            {
                "ok": False,
                "mode": "live",
                "source": source,
                "message": str(exc),
                "help": "Run the setup script on a networked machine once to cache the selected model weights, or choose another cached model.",
                "predictions": [],
                "visualisations": [],
                "visualisations_included": include_visualisations,
                "activation_colour_map": activation_colour_map,
                "model_key": model_key,
            }
        )

    return JSONResponse(
        {
            "ok": True,
            "mode": "live",
            "source": source,
            "message": "The selected model returned likely ImageNet labels. These can be wrong.",
            "predictions": [prediction.to_dict() for prediction in analysis.predictions],
            "visualisations": [visualisation.to_dict() for visualisation in analysis.visualisations],
            "visualisations_included": include_visualisations,
            "activation_colour_map": activation_colour_map,
            "model_key": model_key,
        }
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-screen demo UI."""
    return _render_index_html()


@app.get("/api/images")
def list_images() -> dict[str, Any]:
    """List curated demo images for the selector."""
    images = [
        {
            "name": path.name,
            "label": format_image_label(path),
            "url": f"/api/images/{path.name}",
        }
        for path in discover_images(DEMO_IMAGES_DIR)
    ]
    return {
        "images": images,
        "empty_message": (
            "No curated images were found. Add .jpg, .jpeg, .png, or .webp files to "
            f"{DEMO_IMAGES_DIR} and refresh the page."
        ),
    }


@app.get("/api/images/{image_name}")
def get_image(image_name: str) -> FileResponse:
    """Serve a validated curated image file."""
    image_path = _find_demo_image(image_name)
    return FileResponse(image_path)


@app.get("/api/captions")
def get_captions() -> dict[str, str]:
    """Return editable public captions for the selectable layer diagram."""
    return CAPTIONS


@app.get("/api/models")
def get_models() -> dict[str, object]:
    """Return supported local model options for the UI."""
    return {"models": list(supported_model_options()), "default": "alexnet"}


@app.post("/api/run")
def run_demo(request: RunRequest) -> JSONResponse:
    """Run live inference for a curated image.

    Fallback replay is kept as a visible mode, but full fallback asset playback
    will be implemented in the next phase. Live inference failures are returned
    as safe UI messages instead of crashing the app.
    """
    image_path = _find_demo_image(request.image_name)

    if request.fallback:
        return JSONResponse(
            {
                "ok": False,
                "mode": "fallback",
                "source": "curated",
                "message": "Fallback replay assets will be wired in Phase 5. Turn replay mode off for live model inference.",
                "predictions": [],
                "visualisations": [],
                "activation_colour_map": request.activation_colour_map,
                "model_key": normalise_model_key(request.model_key),
            }
        )

    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return JSONResponse(
            {
                "ok": False,
                "mode": "live",
                "source": "curated",
                "message": f"Could not open the curated image: {exc}",
                "predictions": [],
                "visualisations": [],
            },
            status_code=400,
        )

    return _run_live_analysis_response(
        image,
        source="curated",
        model_key=request.model_key,
        activation_colour_map=request.activation_colour_map,
    )


@app.post("/api/run-camera")
def run_camera_demo(request: CameraRunRequest) -> JSONResponse:
    """Run a supported model on a browser-captured camera frame without saving it."""
    if request.fallback:
        return JSONResponse(
            {
                "ok": False,
                "mode": "fallback",
                "source": "camera",
                "message": "Fallback replay uses curated precomputed assets, not live camera frames. Turn replay mode off for camera inference.",
                "predictions": [],
                "visualisations": [],
                "activation_colour_map": request.activation_colour_map,
                "model_key": normalise_model_key(request.model_key),
            }
        )

    try:
        image = _decode_camera_image(request.image_data)
    except ValueError as exc:
        return JSONResponse(
            {
                "ok": False,
                "mode": "live",
                "source": "camera",
                "message": str(exc),
                "predictions": [],
                "visualisations": [],
            },
            status_code=400,
        )

    return _run_live_analysis_response(
        image,
        source="camera",
        model_key=request.model_key,
        include_visualisations=request.include_visualisations,
        visualisation_keys=set(request.visualisation_keys or ()) or None,
        activation_colour_map=request.activation_colour_map,
    )


def _render_index_html() -> str:
    """Return dependency-free HTML, CSS, and JavaScript for the booth UI."""
    theme_options_html = "\n".join(
        f'          <option value="{value}">{label}</option>' for value, label in THEME_OPTIONS
    )
    activation_colour_options_html = "\n".join(
        f'          <option value="{value}">{label}</option>' for value, label in ACTIVATION_COLOUR_MAP_OPTIONS
    )
    model_options_json = json.dumps(supported_model_options())
    return f"""
<!doctype html>
<html lang="en-AU">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{APP_TITLE}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080814;
      --bg-2: #10142a;
      --bg-3: #070711;
      --panel: rgba(15, 23, 42, 0.82);
      --panel-2: #172033;
      --panel-3: #0d1324;
      --text: #f8fbff;
      --muted: #b9c4d8;
      --accent: #22d3ee;
      --accent-2: #8b5cf6;
      --accent-3: #f59e0b;
      --accent-rgb: 34, 211, 238;
      --accent-2-rgb: 139, 92, 246;
      --accent-3-rgb: 245, 158, 11;
      --heading-2: #67e8f9;
      --heading-3: #c4b5fd;
      --heading-4: #fbbf24;
      --button-text: #03111d;
      --danger: #fecaca;
      --ok: #bbf7d0;
      --border: rgba(148, 163, 184, 0.22);
      --glow: rgba(34, 211, 238, 0.22);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body[data-theme="laboratory"] {{
      --bg: #06111f;
      --bg-2: #0b2535;
      --bg-3: #020712;
      --panel-2: #102b3d;
      --panel-3: #071522;
      --muted: #c2d3df;
      --accent: #38bdf8;
      --accent-2: #14b8a6;
      --accent-3: #facc15;
      --accent-rgb: 56, 189, 248;
      --accent-2-rgb: 20, 184, 166;
      --accent-3-rgb: 250, 204, 21;
      --heading-2: #7dd3fc;
      --heading-3: #5eead4;
      --heading-4: #fef08a;
      --button-text: #02131f;
    }}
    body[data-theme="classroom"] {{
      --bg: #111827;
      --bg-2: #2b1b13;
      --bg-3: #120b08;
      --panel-2: #382517;
      --panel-3: #21140c;
      --muted: #ead0bd;
      --accent: #fb923c;
      --accent-2: #f97316;
      --accent-3: #fde68a;
      --accent-rgb: 251, 146, 60;
      --accent-2-rgb: 249, 115, 22;
      --accent-3-rgb: 253, 230, 138;
      --heading-2: #fed7aa;
      --heading-3: #fdba74;
      --heading-4: #fde68a;
      --button-text: #1b1007;
    }}
    body[data-theme="neon"] {{
      --bg: #05010f;
      --bg-2: #160329;
      --bg-3: #05000b;
      --panel-2: #2a1248;
      --panel-3: #12051f;
      --muted: #d9c5ef;
      --accent: #ec4899;
      --accent-2: #8b5cf6;
      --accent-3: #22d3ee;
      --accent-rgb: 236, 72, 153;
      --accent-2-rgb: 139, 92, 246;
      --accent-3-rgb: 34, 211, 238;
      --heading-2: #f9a8d4;
      --heading-3: #c4b5fd;
      --heading-4: #67e8f9;
      --button-text: #140414;
    }}
    body[data-theme="calm"] {{
      --bg: #071a1f;
      --bg-2: #0b2731;
      --bg-3: #041014;
      --panel-2: #12313d;
      --panel-3: #071820;
      --muted: #c0d7dc;
      --accent: #2dd4bf;
      --accent-2: #60a5fa;
      --accent-3: #a7f3d0;
      --accent-rgb: 45, 212, 191;
      --accent-2-rgb: 96, 165, 250;
      --accent-3-rgb: 167, 243, 208;
      --heading-2: #5eead4;
      --heading-3: #93c5fd;
      --heading-4: #a7f3d0;
      --button-text: #041715;
    }}
    body[data-theme="signal"] {{
      --bg: #0a0a0a;
      --bg-2: #18181b;
      --bg-3: #050505;
      --panel-2: #27272a;
      --panel-3: #18181b;
      --muted: #d4d4d8;
      --accent: #84cc16;
      --accent-2: #bef264;
      --accent-3: #f8fafc;
      --accent-rgb: 132, 204, 22;
      --accent-2-rgb: 190, 242, 100;
      --accent-3-rgb: 248, 250, 252;
      --heading-2: #bef264;
      --heading-3: #84cc16;
      --heading-4: #f8fafc;
      --button-text: #071004;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at 12% 6%, rgba(var(--accent-rgb), 0.22) 0, transparent 28%), radial-gradient(circle at 86% 12%, rgba(var(--accent-2-rgb), 0.24) 0, transparent 30%), linear-gradient(135deg, var(--bg) 0%, var(--bg-2) 56%, var(--bg-3) 100%); color: var(--text); }}
    body::before {{ content: ""; position: fixed; inset: 0; pointer-events: none; background-image: linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px); background-size: 42px 42px; mask-image: radial-gradient(circle at top, black, transparent 72%); }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 28px 0 44px; position: relative; }}
    header {{ margin-bottom: 22px; }}
    h1 {{ font-size: clamp(2rem, 4vw, 4.5rem); line-height: 1; margin: 0 0 14px; letter-spacing: -0.05em; background: linear-gradient(90deg, var(--text), var(--heading-2) 45%, var(--heading-3) 76%, var(--heading-4)); -webkit-background-clip: text; background-clip: text; color: transparent; }}
    h2 {{ margin: 0 0 14px; font-size: 1.25rem; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .grid {{ display: grid; grid-template-columns: 360px 1fr; gap: 20px; align-items: start; }}
    .card {{ background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(10,15,30,0.88)); border: 1px solid var(--border); border-radius: 24px; padding: 20px; box-shadow: 0 24px 70px rgba(0,0,0,0.38), 0 0 38px rgba(var(--accent-rgb), 0.07); backdrop-filter: blur(14px); }}
    .stack {{ display: grid; gap: 16px; }}
    label {{ display: block; margin-bottom: 8px; font-weight: 700; }}
    select, button {{ width: 100%; border-radius: 14px; border: 1px solid var(--border); padding: 12px 14px; font: inherit; }}
    select {{ background-color: var(--panel-3); background-image: linear-gradient(180deg, var(--panel-2), var(--panel-3)); color: var(--text); color-scheme: dark; }}
    select option {{ background-color: var(--panel-3); color: var(--text); }}
    select option:checked {{ background-color: var(--panel-2); color: var(--text); }}
    button {{ background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: var(--button-text); font-weight: 900; cursor: pointer; box-shadow: 0 12px 30px rgba(var(--accent-rgb), 0.22); transition: transform 120ms ease, filter 120ms ease, box-shadow 120ms ease; }}
    button:hover:not(:disabled) {{ transform: translateY(-1px); filter: brightness(1.08); box-shadow: 0 16px 38px rgba(var(--accent-2-rgb), 0.26); }}
    button.secondary {{ background: linear-gradient(180deg, var(--panel-2), var(--panel-3)); color: var(--text); box-shadow: none; }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; box-shadow: none; }}
    .toggle {{ display: flex; gap: 10px; align-items: center; color: var(--muted); }}
    .toggle input {{ width: auto; transform: scale(1.2); accent-color: var(--accent); }}
    .camera-actions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .camera-actions .wide {{ grid-column: 1 / -1; }}
    .camera-note {{ font-size: 0.88rem; margin: 8px 0 0; color: var(--muted); }}
    .message {{ padding: 12px 14px; border-radius: 14px; background: linear-gradient(180deg, rgba(30,41,59,0.98), rgba(17,24,39,0.98)); color: var(--muted); border: 1px solid rgba(148,163,184,0.16); }}
    .message.error {{ background: linear-gradient(180deg, rgba(127,29,29,0.46), rgba(69,10,10,0.34)); color: var(--danger); border-color: rgba(248,113,113,0.3); }}
    .message.ok {{ background: linear-gradient(180deg, rgba(20,83,45,0.45), rgba(6,78,59,0.34)); color: var(--ok); border-color: rgba(74,222,128,0.28); }}
    .bar {{ grid-column: 1 / -1; height: 9px; background: #1f2941; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3)); box-shadow: 0 0 18px rgba(var(--accent-rgb), 0.55); }}
    .muted {{ color: var(--muted); }}
    .network {{ margin-bottom: 16px; padding: 16px; border-radius: 18px; background: radial-gradient(circle at 20% 45%, rgba(var(--accent-rgb), 0.18), transparent 32%), radial-gradient(circle at 78% 50%, rgba(var(--accent-3-rgb), 0.13), transparent 35%), #030611; border: 1px solid var(--border); overflow: hidden; }}
    .network svg {{ width: 100%; height: auto; display: block; filter: drop-shadow(0 0 16px rgba(var(--accent-rgb), 0.12)); overflow: visible; }}
    .network .layer {{ outline: none; }}
    .network .stage-hit {{ fill: transparent; stroke: none; }}
    .network .volume-front {{ fill: rgba(var(--accent-rgb), 0.16); stroke: rgba(226, 232, 240, 0.82); stroke-width: 2; }}
    .network .volume-top {{ fill: rgba(var(--accent-2-rgb), 0.22); stroke: rgba(226, 232, 240, 0.62); stroke-width: 1.5; }}
    .network .volume-side {{ fill: rgba(var(--accent-3-rgb), 0.18); stroke: rgba(226, 232, 240, 0.58); stroke-width: 1.5; }}
    .network .volume-vector {{ fill: rgba(var(--accent-rgb), 0.13); stroke: rgba(226, 232, 240, 0.78); stroke-width: 2; }}
    .network .layer.active .volume-front, .network .layer:focus .volume-front, .network .layer.active .volume-vector, .network .layer:focus .volume-vector {{ fill: rgba(var(--accent-3-rgb), 0.38); stroke: var(--heading-4); }}
    .network .layer.active .volume-top, .network .layer:focus .volume-top {{ fill: rgba(var(--accent-2-rgb), 0.34); stroke: var(--heading-4); }}
    .network .layer.active .volume-side, .network .layer:focus .volume-side {{ fill: rgba(var(--accent-3-rgb), 0.28); stroke: var(--heading-4); }}
    .network .connector {{ stroke: rgba(148, 163, 184, 0.5); stroke-width: 3; stroke-linecap: round; }}
    .network .size-guide {{ stroke: rgba(226, 232, 240, 0.26); stroke-width: 1; stroke-dasharray: 4 6; }}
    .network text {{ fill: #f8fbff; font-size: 12px; font-weight: 900; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }}
    .network .stage-note {{ fill: #d6deed; font-size: 10px; font-weight: 800; }}
    .layer-detail {{ margin: 0 0 16px; display: grid; grid-template-columns: minmax(260px, 1.4fr) minmax(220px, 0.8fr); gap: 16px; align-items: start; background: linear-gradient(180deg, rgba(8,16,36,0.98), rgba(4,7,16,0.98)); border: 1px solid rgba(var(--accent-rgb), 0.2); border-radius: 18px; padding: 16px; }}
    .layer-detail.placeholder {{ display: block; }}
    .layer-detail img, .layer-detail video {{ width: 100%; display: block; border-radius: 14px; background: #020208; border: 1px solid rgba(255,255,255,0.09); box-shadow: 0 24px 60px rgba(0,0,0,0.32); }}
    .layer-detail video {{ transform: scaleX(-1); }}
    .camera-capture-source {{ position: fixed; width: 1px; height: 1px; left: -10px; top: -10px; opacity: 0; pointer-events: none; }}
    .detail-copy h3 {{ margin: 0 0 8px; font-size: 1.18rem; color: var(--heading-4); }}
    .detail-copy p {{ margin: 0 0 10px; }}
    .detail-pill {{ display: inline-flex; align-items: center; margin: 4px 6px 4px 0; padding: 6px 9px; border-radius: 999px; background: rgba(var(--accent-rgb), 0.12); border: 1px solid rgba(var(--accent-rgb), 0.22); color: var(--heading-2); font-size: 0.82rem; font-weight: 800; }}
    .prediction-detail {{ list-style: none; padding: 0; margin: 8px 0 0; display: grid; gap: 8px; }}
    .prediction-detail li {{ display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; padding: 10px; border-radius: 12px; background: rgba(15,23,42,0.74); border: 1px solid rgba(148,163,184,0.14); }}
    .classifier-visual {{ display: grid; gap: 14px; padding: 14px; border-radius: 16px; background: radial-gradient(circle at 24% 22%, rgba(var(--accent-rgb), 0.16), transparent 38%), rgba(2,6,23,0.58); border: 1px solid rgba(var(--accent-rgb), 0.18); }}
    .classifier-flow {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; align-items: stretch; }}
    .classifier-block {{ position: relative; display: grid; place-items: center; min-height: 92px; padding: 12px 8px; border-radius: 14px; text-align: center; background: linear-gradient(180deg, rgba(var(--accent-rgb), 0.16), rgba(var(--accent-2-rgb), 0.08)); border: 1px solid rgba(226,232,240,0.18); font-weight: 900; }}
    .classifier-block:not(:last-child)::after {{ content: "→"; position: absolute; right: -16px; top: 50%; transform: translateY(-50%); color: var(--heading-4); font-size: 1.35rem; z-index: 2; }}
    .classifier-block small {{ display: block; margin-top: 6px; color: var(--muted); font-weight: 700; line-height: 1.25; }}
    .score-note {{ margin: 0 0 10px; color: var(--muted); font-size: 0.9rem; }}
    .classifier-scores {{ display: grid; gap: 8px; max-height: 440px; overflow: auto; padding-right: 4px; }}
    .classifier-score {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; font-size: 0.9rem; }}
    .classifier-score .bar {{ grid-column: 1 / -1; }}
    .score-rank {{ color: var(--heading-4); font-weight: 900; margin-right: 4px; }}
    @media (max-width: 980px) {{ .layer-detail {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 620px) {{ .classifier-flow {{ grid-template-columns: 1fr; gap: 22px; }} .classifier-block:not(:last-child)::after {{ content: "↓"; right: auto; left: 50%; top: auto; bottom: -18px; transform: translateX(-50%); }} }}
    @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body data-theme="aurora">
<main>
  <header>
    <h1>{APP_TITLE}</h1>
    <p>This local demo shows how a trained vision model responds at different layers. Early layers often respond to simple visual patterns, while deeper layers combine those patterns into features useful for classification. The final prediction is a likely label, not guaranteed truth.</p>
  </header>

  <section class="grid">
    <aside class="card stack">
      <div>
        <h2>Vision model</h2>
        <label for="modelSelect">Select a model</label>
        <select id="modelSelect" aria-label="Vision model selector"></select>
        <p id="modelDescription" class="camera-note">AlexNet is the classic option; newer models usually predict better.</p>
      </div>

      <div>
        <h2>Curated image</h2>
        <label for="imageSelect">Select a booth image</label>
        <select id="imageSelect"><option value="">Loading images…</option></select>
      </div>

      <div>
        <h2>Live camera</h2>
        <div class="camera-actions">
          <button id="startCameraButton" class="secondary" type="button">Start camera</button>
          <button id="cameraRunButton" type="button" disabled>Capture + run</button>
          <button id="liveRunButton" class="wide" type="button" disabled>Start continuous model</button>
        </div>
        <p class="camera-note">Opt-in local camera mode. Frames are sent only to this local app for analysis and are not saved.</p>
      </div>

      <label class="toggle"><input id="fallbackToggle" type="checkbox" /> Fallback / replay mode</label>

      <div>
        <h2>Layer image colours</h2>
        <label for="activationColourSelect">Select activation image colours</label>
        <select id="activationColourSelect" aria-label="Activation image colour selector">
{activation_colour_options_html}
        </select>
        <p class="camera-note">This changes only the layer visualisation colours. It does not change the model result.</p>
      </div>

      <div>
        <h2>Page colours</h2>
        <label for="themeSelect">Select a colour theme</label>
        <select id="themeSelect" aria-label="Colour theme selector">
{theme_options_html}
        </select>
      </div>

      <button id="runButton" disabled>Run selected model</button>
      <button id="resetButton" class="secondary">Reset demo</button>
    </aside>

    <section class="stack">
      <div class="card">
        <h2>Model layer explorer</h2>
        <div id="status" class="message" hidden></div>
        <p id="caption" class="message">Choose any layer in the diagram, including the input.</p>
        <div id="networkDiagram" class="network" aria-label="Selectable model layer diagram"></div>
        <div id="layerDetail" class="layer-detail placeholder">
          <p class="message">Choose a curated image or start the camera, then select any model layer in the diagram.</p>
        </div>
      </div>
    </section>
  </section>
</main>

<script>
const MODEL_OPTIONS = {model_options_json};
const PREDICTION_LABEL_COUNT = {PREDICTION_LABEL_COUNT};
const CLASSIFIER_LABEL_COUNT = {CLASSIFIER_LABEL_COUNT};
const MODEL_STORAGE_KEY = 'insideAlexNetModelKey';
const modelSelect = document.getElementById('modelSelect');
const modelDescription = document.getElementById('modelDescription');
const imageSelect = document.getElementById('imageSelect');
const runButton = document.getElementById('runButton');
const startCameraButton = document.getElementById('startCameraButton');
const cameraRunButton = document.getElementById('cameraRunButton');
const liveRunButton = document.getElementById('liveRunButton');
const resetButton = document.getElementById('resetButton');
const fallbackToggle = document.getElementById('fallbackToggle');
const activationColourSelect = document.getElementById('activationColourSelect');
const themeSelect = document.getElementById('themeSelect');
const statusBox = document.getElementById('status');
const layerDetail = document.getElementById('layerDetail');
const caption = document.getElementById('caption');
const networkDiagram = document.getElementById('networkDiagram');
const ACTIVATION_COLOUR_STORAGE_KEY = 'insideAlexNetActivationColourMap';
const THEME_STORAGE_KEY = 'insideAlexNetColourTheme';
let captions = {{}};
let visualisationsByLabel = new Map();
let currentStage = 'Input';
let inputState = {{kind: 'empty'}};
let lastPredictions = [];
let cameraStream = null;
let cameraVideo = null;
let liveRunActive = false;
let liveFrameIndex = 0;

function escapeHtml(value) {{
  return String(value).replace(/[&<>"']/g, character => ({{
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }}[character]));
}}

function setStatus(text, kind = '') {{
  statusBox.hidden = false;
  statusBox.className = `message ${{kind}}`;
  statusBox.textContent = text;
}}

function clearStatus() {{
  statusBox.hidden = true;
  statusBox.className = 'message';
  statusBox.textContent = '';
}}

function selectedModel() {{
  return MODEL_OPTIONS.find(model => model.key === modelSelect.value) || MODEL_OPTIONS[0];
}}

function selectedModelKey() {{
  return selectedModel().key;
}}

function selectedModelLayer(stage) {{
  return selectedModel().layers.find(layer => layer.label === stage);
}}

function selectedVisualisationKeyForStage(stage) {{
  const layer = selectedModelLayer(stage);
  return layer ? layer.key : undefined;
}}

function diagramStages() {{
  return [
    {{label: 'Input'}},
    ...selectedModel().layers,
    {{label: 'Classifier'}},
    {{label: 'Prediction'}}
  ];
}}

function diagramStageLabels() {{
  return diagramStages().map(stage => stage.label);
}}

function selectedModelName() {{
  return selectedModel().label;
}}

function renderModelOptions() {{
  modelSelect.innerHTML = '';
  MODEL_OPTIONS.forEach(model => {{
    const option = document.createElement('option');
    option.value = model.key;
    option.textContent = `${{model.label}} — ${{model.short_label}}`;
    modelSelect.appendChild(option);
  }});
}}

function applyModelSelection(modelKey, options = {{}}) {{
  const known = MODEL_OPTIONS.some(model => model.key === modelKey);
  modelSelect.value = known ? modelKey : 'alexnet';
  const model = selectedModel();
  modelDescription.textContent = `${{model.description}} ${{model.recommendation}}`;
  renderNetworkDiagram();
  try {{
    window.localStorage.setItem(MODEL_STORAGE_KEY, model.key);
  }} catch (error) {{
    // Local storage can be disabled; the selector still works for this page view.
  }}
  if (options.clear !== false) {{
    clearResults();
    selectStage('Input');
  }}
}}

function initialiseModelSelection() {{
  renderModelOptions();
  let savedModel = '';
  try {{
    savedModel = window.localStorage.getItem(MODEL_STORAGE_KEY) || '';
  }} catch (error) {{
    savedModel = '';
  }}
  applyModelSelection(savedModel || 'alexnet', {{clear: false}});
}}

function splitStageLabel(label) {{
  const parts = String(label).split(' ');
  if (parts.length <= 1) {{
    return [label, ''];
  }}
  return [parts[0], parts.slice(1).join(' ')];
}}

function stageVisualSpec(stage, index) {{
  const label = stage.label;
  const lowerLabel = label.toLowerCase();
  if (label === 'Input') {{
    return {{kind: 'volume', width: 58, height: 122, depth: 10}};
  }}
  if (label === 'Classifier') {{
    return {{kind: 'vector', width: 88, height: 66, depth: 0}};
  }}
  if (label === 'Prediction') {{
    return {{kind: 'vector', width: 86, height: 54, depth: 0}};
  }}
  if (lowerLabel.includes('avg')) {{
    return {{kind: 'volume', width: 60, height: 42, depth: 18}};
  }}

  const layerCount = selectedModel().layers.length;
  const layerIndex = Math.max(0, index - 1);
  const progress = layerCount <= 1 ? 0 : layerIndex / (layerCount - 1);
  let spatialHeight = 96 - progress * 46;
  let frontWidth = 48 + progress * 26;
  let depth = 10 + progress * 26;

  if (lowerLabel.includes('pool')) {{
    spatialHeight -= 14;
    frontWidth -= 4;
    depth += 3;
  }}
  if (lowerLabel.includes('residual') || lowerLabel.includes('deep')) {{
    depth += 5;
  }}
  if (lowerLabel.includes('final')) {{
    spatialHeight -= 6;
    depth += 8;
  }}

  return {{
    kind: 'volume',
    width: Math.round(Math.max(44, frontWidth)),
    height: Math.round(Math.max(38, spatialHeight)),
    depth: Math.round(Math.max(8, depth))
  }};
}}

function renderStageShape(stage, index, centre, baseline) {{
  const spec = stageVisualSpec(stage, index);
  const x = centre - spec.width / 2;
  const y = baseline - spec.height / 2;
  const [top, bottom] = splitStageLabel(stage.label);
  const textY = baseline + (spec.height < 48 ? 0 : 2);

  if (spec.kind === 'vector') {{
    return `
      <g class="layer" data-layer="${{escapeHtml(stage.label)}}" tabindex="0" role="button" aria-label="${{escapeHtml(stage.label)}} stage">
        <rect class="volume-vector" x="${{x}}" y="${{y}}" width="${{spec.width}}" height="${{spec.height}}" rx="12" />
        <text x="${{centre}}" y="${{textY - (bottom ? 8 : 0)}}">${{escapeHtml(top)}}</text>
        ${{bottom ? `<text x="${{centre}}" y="${{textY + 9}}" class="stage-note">${{escapeHtml(bottom)}}</text>` : ''}}
        <rect class="stage-hit" x="${{centre - 64}}" y="34" width="128" height="176" rx="16" />
      </g>`;
  }}

  const dx = spec.depth * 0.58;
  const dy = -spec.depth * 0.42;
  const sidePoints = `${{x + spec.width}},${{y}} ${{x + spec.width + dx}},${{y + dy}} ${{x + spec.width + dx}},${{y + spec.height + dy}} ${{x + spec.width}},${{y + spec.height}}`;
  const topPoints = `${{x}},${{y}} ${{x + dx}},${{y + dy}} ${{x + spec.width + dx}},${{y + dy}} ${{x + spec.width}},${{y}}`;

  return `
    <g class="layer layer-volume" data-layer="${{escapeHtml(stage.label)}}" tabindex="0" role="button" aria-label="${{escapeHtml(stage.label)}} stage">
      <polygon class="volume-side" points="${{sidePoints}}" />
      <polygon class="volume-top" points="${{topPoints}}" />
      <rect class="volume-front" x="${{x}}" y="${{y}}" width="${{spec.width}}" height="${{spec.height}}" rx="10" />
      <text x="${{centre}}" y="${{textY - (bottom ? 8 : 0)}}">${{escapeHtml(top)}}</text>
      ${{bottom ? `<text x="${{centre}}" y="${{textY + 9}}" class="stage-note">${{escapeHtml(bottom)}}</text>` : ''}}
      <rect class="stage-hit" x="${{centre - 64}}" y="34" width="128" height="176" rx="16" />
    </g>`;
}}

function renderNetworkDiagram() {{
  const stages = diagramStages();
  const width = 1180;
  const height = 250;
  const baseline = 126;
  const margin = 82;
  const step = (width - margin * 2) / Math.max(1, stages.length - 1);
  const shapes = stages.map((stage, index) => renderStageShape(stage, index, margin + step * index, baseline)).join('');
  networkDiagram.innerHTML = `
    <svg viewBox="0 0 ${{width}} ${{height}}" role="img">
      <title>Selectable ${{escapeHtml(selectedModelName())}} feature-volume path from input through demo layers</title>
      <line class="connector" x1="${{margin}}" y1="${{baseline}}" x2="${{width - margin}}" y2="${{baseline}}" />
      <line class="size-guide" x1="${{margin}}" y1="64" x2="${{width - margin}}" y2="64" />
      <line class="size-guide" x1="${{margin}}" y1="188" x2="${{width - margin}}" y2="188" />
      ${{shapes}}
    </svg>`;
  attachLayerEvents();
}}

function attachLayerEvents() {{
  document.querySelectorAll('.network .layer').forEach(layer => {{
    layer.style.cursor = 'pointer';
    layer.addEventListener('click', () => selectStage(layer.dataset.layer, {{scroll: true}}));
    layer.addEventListener('keydown', event => {{
      if (event.key === 'Enter' || event.key === ' ') {{
        event.preventDefault();
        selectStage(layer.dataset.layer, {{scroll: true}});
      }}
    }});
  }});
}}

function clearResults() {{
  visualisationsByLabel = new Map();
  lastPredictions = [];
  renderSelectedLayerDetail();
  clearStatus();
}}

function isKnownTheme(theme) {{
  return Array.from(themeSelect.options).some(option => option.value === theme);
}}

function applyTheme(theme) {{
  const nextTheme = isKnownTheme(theme) ? theme : 'aurora';
  document.body.dataset.theme = nextTheme;
  themeSelect.value = nextTheme;
  try {{
    window.localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
  }} catch (error) {{
    // Local storage can be disabled; the selector still works for this page view.
  }}
}}

function initialiseTheme() {{
  let savedTheme = '';
  try {{
    savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY) || '';
  }} catch (error) {{
    savedTheme = '';
  }}
  applyTheme(savedTheme || themeSelect.value || 'aurora');
}}

function isKnownActivationColourMap(colourMap) {{
  return Array.from(activationColourSelect.options).some(option => option.value === colourMap);
}}

function applyActivationColourMapSelection(colourMap) {{
  const nextColourMap = isKnownActivationColourMap(colourMap) ? colourMap : 'aurora';
  activationColourSelect.value = nextColourMap;
  try {{
    window.localStorage.setItem(ACTIVATION_COLOUR_STORAGE_KEY, nextColourMap);
  }} catch (error) {{
    // Local storage can be disabled; the selector still works for this page view.
  }}
}}

function initialiseActivationColourMap() {{
  let savedColourMap = '';
  try {{
    savedColourMap = window.localStorage.getItem(ACTIVATION_COLOUR_STORAGE_KEY) || '';
  }} catch (error) {{
    savedColourMap = '';
  }}
  applyActivationColourMapSelection(savedColourMap || activationColourSelect.value || 'aurora');
}}

function selectedActivationColourMap() {{
  return activationColourSelect.value || 'aurora';
}}

function resetDemo() {{
  stopCamera({{clearInput: true}});
  imageSelect.value = '';
  inputState = {{kind: 'empty'}};
  fallbackToggle.checked = false;
  runButton.disabled = true;
  clearResults();
  selectStage('Input');
}}

function selectStage(stage, options = {{}}) {{
  currentStage = diagramStageLabels().includes(stage) ? stage : 'Input';
  document.querySelectorAll('.network .layer').forEach(layer => layer.classList.toggle('active', layer.dataset.layer === currentStage));
  caption.textContent = stageCaption(currentStage);
  if (options.render !== false) {{
    renderSelectedLayerDetail({{scroll: options.scroll === true}});
  }}
}}

function stageCaption(stage) {{
  const modelLayer = selectedModelLayer(stage);
  if (modelLayer) {{
    return captions[modelLayer.caption_key] || modelLayer.note || 'This shows how a trained vision model responds at this stage.';
  }}
  return captions[stage] || 'This shows how a trained vision model responds at this stage.';
}}

function renderSelectedLayerDetail(options = {{}}) {{
  if (currentStage === 'Input') {{
    renderInputDetail(options);
  }} else if (currentStage === 'Prediction') {{
    renderPredictionDetail(options);
  }} else if (currentStage === 'Classifier') {{
    renderClassifierDetail(options);
  }} else {{
    const item = visualisationsByLabel.get(currentStage);
    if (item) {{
      renderActivationDetail(item, options);
    }} else {{
      renderLayerPlaceholder(currentStage, options);
    }}
  }}
}}

function scrollLayerDetailIfNeeded(options = {{}}) {{
  if (options.scroll === true) {{
    layerDetail.scrollIntoView({{behavior: 'smooth', block: 'nearest'}});
  }}
}}

function renderInputDetail(options = {{}}) {{
  const captionText = captions.Input || 'The image is resized and normalised before entering the network.';
  layerDetail.className = inputState.kind === 'empty' ? 'layer-detail placeholder' : 'layer-detail';
  if (cameraStream && cameraVideo) {{
    layerDetail.innerHTML = `
      <div class="input-media" aria-label="Local camera preview">
        <video autoplay playsinline muted aria-label="Live local camera input"></video>
      </div>
      <div class="detail-copy">
        <h3>Input</h3>
        <p>${{escapeHtml(captionText)}}</p>
        <p>This is the live local camera frame before model preprocessing. Frames are analysed in memory and are not saved.</p>
        <span class="detail-pill">Opt-in camera</span>
        <span class="detail-pill">Local only</span>
        <span class="detail-pill">Not saved</span>
      </div>`;
    const visibleVideo = layerDetail.querySelector('.input-media video');
    visibleVideo.srcObject = cameraStream;
    visibleVideo.play().catch(() => {{}});
  }} else if (inputState.kind === 'image') {{
    layerDetail.innerHTML = `
      <img src="${{escapeHtml(inputState.url)}}" alt="Selected curated input image" />
      <div class="detail-copy">
        <h3>Input</h3>
        <p>${{escapeHtml(captionText)}}</p>
        <p>${{escapeHtml(inputState.label || 'Selected curated image')}}</p>
        <span class="detail-pill">Curated image</span>
        <span class="detail-pill">224 × 224 model input after preprocessing</span>
      </div>`;
  }} else {{
    layerDetail.innerHTML = '<p class="message">Choose a curated image or start the camera. The selected input will appear here as the first selectable model stage.</p>';
  }}
  scrollLayerDetailIfNeeded(options);
}}

function renderActivationDetail(item, options = {{}}) {{
  const captionText = captions[item.caption_key] || item.note || 'This shows fixed channels from this layer response.';
  const tensorShape = (item.tensor_shape || []).join(' × ');
  layerDetail.className = 'layer-detail';
  layerDetail.innerHTML = `
    <img class="activation-detail-image" data-layer="${{escapeHtml(item.label)}}" src="${{item.image_data}}" alt="Large ${{escapeHtml(item.label)}} activation grid" />
    <div class="detail-copy">
      <h3>${{escapeHtml(item.label)}}</h3>
      <p>${{escapeHtml(captionText)}}</p>
      <p>${{escapeHtml(item.note || 'Each square is one fixed channel from this layer, so the tile position stays stable across frames. Brighter regions indicate stronger responses after normalising that channel for display.')}}</p>
      <span class="detail-pill">${{escapeHtml(tensorShape)}}</span>
      <span class="detail-pill">Fixed channel positions</span>
      <span class="detail-pill">Brighter colours = stronger</span>
      <span class="detail-pill">Normalised for display</span>
      <span class="detail-pill">Updates with each live frame</span>
    </div>`;
  scrollLayerDetailIfNeeded(options);
}}

function renderLayerPlaceholder(stage, options = {{}}) {{
  const captionText = captions[stage] || 'This shows how a trained vision model responds at this stage.';
  layerDetail.className = 'layer-detail placeholder';
  layerDetail.innerHTML = `<p class="message"><strong>${{escapeHtml(stage)}}:</strong> ${{escapeHtml(captionText)}} Run the selected model to show this layer from the selected input.</p>`;
  scrollLayerDetailIfNeeded(options);
}}

function classifierScoresHtml() {{
  if (!lastPredictions.length) {{
    return '<p class="message">Run the selected model to show how classifier scores flow into likely labels.</p>';
  }}
  const classifierPredictions = lastPredictions.slice(0, CLASSIFIER_LABEL_COUNT);
  const topProbability = Math.max(...classifierPredictions.map(item => item.probability), 0);
  const items = classifierPredictions.map((item, index) => {{
    const pct = Math.round(item.probability * 1000) / 10;
    const relativeWidth = topProbability > 0 ? Math.max(2, Math.round((item.probability / topProbability) * 100)) : 0;
    return `
      <div class="classifier-score">
        <strong><span class="score-rank">#${{index + 1}}</span> ${{escapeHtml(item.label)}}</strong>
        <span>${{pct}}%</span>
        <div class="bar" title="Relative to the top classifier score"><span style="width: ${{relativeWidth}}%"></span></div>
      </div>`;
  }}).join('');
  return `
    <div>
      <p class="score-note">Top ${{classifierPredictions.length}} ImageNet labels, with bar widths scaled relative to the highest current score.</p>
      <div class="classifier-scores" aria-label="Top classifier score bars">${{items}}</div>
    </div>`;
}}

function renderClassifierDetail(options = {{}}) {{
  const captionText = captions.Classifier || 'The classifier turns compact feature values into scores for ImageNet training labels.';
  const avgPool = visualisationsByLabel.get('Avg pool');
  const shape = avgPool ? avgPool.tensor_shape.join(' × ') : 'Run the selected model to show incoming feature shape';
  layerDetail.className = 'layer-detail';
  layerDetail.innerHTML = `
    <div class="classifier-visual" aria-label="Classifier layer visual sketch">
      <div class="classifier-flow" aria-hidden="true">
        <div class="classifier-block">Avg pool<small>${{escapeHtml(shape)}}</small></div>
        <div class="classifier-block">Dense layer<small>combines features</small></div>
        <div class="classifier-block">Dense layer<small>refines scores</small></div>
        <div class="classifier-block">1000 labels<small>ImageNet scores</small></div>
      </div>
      ${{classifierScoresHtml()}}
    </div>
    <div class="detail-copy">
      <h3>Classifier</h3>
      <p>${{escapeHtml(captionText)}}</p>
      <p>The model turns compact feature values into class scores. This layer shows more candidate labels than the final prediction summary so visitors can see alternatives the model considered plausible.</p>
      <span class="detail-pill">Input: ${{escapeHtml(shape)}}</span>
      <span class="detail-pill">Fully connected layers</span>
      <span class="detail-pill">Top ${{CLASSIFIER_LABEL_COUNT}} label scores</span>
      <span class="detail-pill">Scores, not certainty</span>
    </div>`;
  scrollLayerDetailIfNeeded(options);
}}

function predictionListHtml() {{
  if (!lastPredictions.length) {{
    return '<p class="message">Run the selected model to show the top likely ImageNet labels for this input.</p>';
  }}
  const items = lastPredictions.slice(0, PREDICTION_LABEL_COUNT).map(item => {{
    const pct = Math.round(item.probability * 1000) / 10;
    return `<li><strong>${{escapeHtml(item.label)}}</strong><span>${{pct}}%</span><div class="bar"><span style="width: ${{pct}}%"></span></div></li>`;
  }}).join('');
  return `<ol class="prediction-detail">${{items}}</ol>`;
}}

function renderPredictionDetail(options = {{}}) {{
  const captionText = captions.Prediction || 'The final prediction is a likely class from the model’s training labels. It can be wrong.';
  layerDetail.className = lastPredictions.length ? 'layer-detail' : 'layer-detail placeholder';
  layerDetail.innerHTML = `
    <div data-role="prediction-list">
      ${{predictionListHtml()}}
    </div>
    <div class="detail-copy">
      <h3>Prediction</h3>
      <p>${{escapeHtml(captionText)}}</p>
      <p>These labels come from the model’s training categories and can be wrong, especially for unusual, ambiguous, or out-of-distribution images.</p>
      <span class="detail-pill">Top-${{PREDICTION_LABEL_COUNT}} labels</span>
      <span class="detail-pill">Likely, not guaranteed</span>
    </div>`;
  scrollLayerDetailIfNeeded(options);
}}

function updateLiveLayerDetail() {{
  if (currentStage === 'Input') {{
    return;
  }}
  if (currentStage === 'Classifier') {{
    renderClassifierDetail({{scroll: false}});
    return;
  }}
  if (currentStage === 'Prediction') {{
    const predictionList = layerDetail.querySelector('[data-role="prediction-list"]');
    if (predictionList) {{
      predictionList.innerHTML = predictionListHtml();
    }} else {{
      renderPredictionDetail({{scroll: false}});
    }}
    return;
  }}

  const item = visualisationsByLabel.get(currentStage);
  if (!item) {{
    return;
  }}
  const image = layerDetail.querySelector('.activation-detail-image');
  if (image && image.dataset.layer === currentStage) {{
    image.src = item.image_data;
  }} else {{
    renderActivationDetail(item, {{scroll: false}});
  }}
}}

function renderAnalysisResult(data, options = {{}}) {{
  setStatus(data.message || 'Run complete.', data.ok ? 'ok' : 'error');
  if (data.help) {{
    setStatus(`${{data.message}} ${{data.help}}`, data.ok ? 'ok' : 'error');
  }}
  lastPredictions = data.predictions || [];
  if (options.live !== true) {{
    visualisationsByLabel = new Map();
  }}
  const visualisationsIncluded = data.visualisations_included !== false;
  if (visualisationsIncluded && data.visualisations) {{
    data.visualisations.forEach(item => visualisationsByLabel.set(item.label, item));
  }}
  if (options.live === true) {{
    updateLiveLayerDetail();
  }} else {{
    renderSelectedLayerDetail();
  }}
}}

function stopLiveRun() {{
  liveRunActive = false;
  liveRunButton.textContent = 'Start continuous model';
  liveRunButton.classList.remove('secondary');
}}

function stopCamera(options = {{}}) {{
  stopLiveRun();
  if (cameraStream) {{
    cameraStream.getTracks().forEach(track => track.stop());
  }}
  if (cameraVideo) {{
    cameraVideo.srcObject = null;
    cameraVideo.remove();
  }}
  cameraStream = null;
  cameraVideo = null;
  cameraRunButton.disabled = true;
  liveRunButton.disabled = true;
  startCameraButton.textContent = 'Start camera';
  if (options.clearInput === true) {{
    inputState = {{kind: 'empty'}};
  }}
  if (currentStage === 'Input') {{
    renderSelectedLayerDetail();
  }}
}}

async function startCamera() {{
  if (cameraStream) {{
    stopCamera({{clearInput: true}});
    clearResults();
    selectStage('Input');
    return;
  }}
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
    setStatus('This browser does not support local camera access.', 'error');
    return;
  }}
  clearResults();
  imageSelect.value = '';
  inputState = {{kind: 'empty'}};
  runButton.disabled = true;
  try {{
    cameraStream = await navigator.mediaDevices.getUserMedia({{video: {{width: {{ideal: 960}}, height: {{ideal: 720}}, facingMode: 'user'}}, audio: false}});
    cameraVideo = document.createElement('video');
    cameraVideo.autoplay = true;
    cameraVideo.playsInline = true;
    cameraVideo.muted = true;
    cameraVideo.className = 'camera-capture-source';
    cameraVideo.setAttribute('aria-hidden', 'true');
    cameraVideo.tabIndex = -1;
    cameraVideo.srcObject = cameraStream;
    document.body.appendChild(cameraVideo);
    inputState = {{kind: 'camera'}};
    selectStage('Input');
    await waitForCameraReady();
    cameraRunButton.disabled = false;
    liveRunButton.disabled = false;
    startCameraButton.textContent = 'Stop camera';
    setStatus('Camera preview is local. Capture one frame or start continuous model analysis.', 'ok');
  }} catch (error) {{
    setStatus(`Camera access was not available: ${{error}}`, 'error');
    stopCamera({{clearInput: true}});
  }}
}}

function waitForCameraReady() {{
  return new Promise((resolve, reject) => {{
    if (!cameraVideo) {{
      reject(new Error('Camera preview was not created.'));
      return;
    }}
    const done = () => {{
      cleanup();
      resolve();
    }};
    const fail = () => {{
      cleanup();
      reject(new Error('Camera preview did not become ready in time.'));
    }};
    const cleanup = () => {{
      window.clearTimeout(timeout);
      cameraVideo.removeEventListener('loadedmetadata', done);
      cameraVideo.removeEventListener('canplay', done);
    }};
    const timeout = window.setTimeout(fail, 4000);
    cameraVideo.addEventListener('loadedmetadata', done, {{once: true}});
    cameraVideo.addEventListener('canplay', done, {{once: true}});
    cameraVideo.play().catch(() => {{}});
    if (cameraVideo.videoWidth && cameraVideo.videoHeight) {{
      done();
    }}
  }});
}}

function captureCameraFrame() {{
  if (!cameraVideo || !cameraVideo.videoWidth || !cameraVideo.videoHeight) {{
    throw new Error('Camera preview is not ready yet.');
  }}
  const canvas = document.createElement('canvas');
  const maxSide = 900;
  const scale = Math.min(1, maxSide / Math.max(cameraVideo.videoWidth, cameraVideo.videoHeight));
  canvas.width = Math.round(cameraVideo.videoWidth * scale);
  canvas.height = Math.round(cameraVideo.videoHeight * scale);
  const ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.9);
}}

async function analyseCameraFrame({{includeVisualisations = true, live = false, visualisationKeys = []}} = {{}}) {{
  const imageData = captureCameraFrame();
  const response = await fetch('/api/run-camera', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{
      image_data: imageData,
      fallback: fallbackToggle.checked,
      model_key: selectedModelKey(),
      include_visualisations: includeVisualisations,
      visualisation_keys: visualisationKeys,
      activation_colour_map: selectedActivationColourMap()
    }})
  }});
  const data = await response.json();
  renderAnalysisResult(data, {{live}});
  if (live && data.ok) {{
    if (liveFrameIndex === 1 || liveFrameIndex % 5 === 0) {{
      setStatus(`Continuous ${{selectedModelName()}} is running locally. Analysed frame ${{liveFrameIndex}}.`, 'ok');
    }}
  }}
  return data;
}}

async function runCameraFrame() {{
  cameraRunButton.disabled = true;
  setStatus(`Capturing one local camera frame and selected ${{selectedModelName()}} layers…`);
  try {{
    await analyseCameraFrame({{includeVisualisations: true, live: false}});
  }} catch (error) {{
    setStatus(`The local app could not analyse the camera frame: ${{error}}`, 'error');
  }} finally {{
    cameraRunButton.disabled = !cameraStream;
  }}
}}

async function liveRunLoop() {{
  if (!liveRunActive || !cameraStream) return;
  liveFrameIndex += 1;
  const selectedVisualisationKey = selectedVisualisationKeyForStage(currentStage);
  try {{
    await analyseCameraFrame({{
      includeVisualisations: Boolean(selectedVisualisationKey),
      live: true,
      visualisationKeys: selectedVisualisationKey ? [selectedVisualisationKey] : []
    }});
  }} catch (error) {{
    setStatus(`Continuous model analysis stopped: ${{error}}`, 'error');
    stopLiveRun();
    return;
  }}
  if (liveRunActive) {{
    window.setTimeout(liveRunLoop, 150);
  }}
}}

function toggleLiveRun() {{
  if (liveRunActive) {{
    stopLiveRun();
    setStatus('Continuous model analysis stopped. Camera preview is still local.', 'ok');
    return;
  }}
  if (!cameraStream) {{
    setStatus('Start the camera before continuous model analysis.', 'error');
    return;
  }}
  liveRunActive = true;
  liveFrameIndex = 0;
  liveRunButton.textContent = 'Stop continuous model';
  liveRunButton.classList.add('secondary');
  setStatus(`Continuous ${{selectedModelName()}} is starting. Predictions and the selected diagram layer update on each analysed frame.`, 'ok');
  liveRunLoop();
}}

async function loadCaptions() {{
  captions = await fetch('/api/captions').then(response => response.json());
  selectStage('Input');
}}

async function loadImages() {{
  const data = await fetch('/api/images').then(response => response.json());
  imageSelect.innerHTML = '<option value="">Choose a curated image</option>';
  if (!data.images.length) {{
    const option = document.createElement('option');
    option.value = '';
    option.textContent = data.empty_message;
    imageSelect.appendChild(option);
    imageSelect.disabled = true;
    setStatus(data.empty_message, 'error');
    return;
  }}
  data.images.forEach(image => {{
    const option = document.createElement('option');
    option.value = image.name;
    option.textContent = image.label;
    option.dataset.url = image.url;
    imageSelect.appendChild(option);
  }});
}}

imageSelect.addEventListener('change', () => {{
  stopCamera({{clearInput: false}});
  clearResults();
  const selected = imageSelect.selectedOptions[0];
  runButton.disabled = !imageSelect.value;
  if (!imageSelect.value) {{
    inputState = {{kind: 'empty'}};
    selectStage('Input');
    return;
  }}
  inputState = {{kind: 'image', url: selected.dataset.url, label: selected.textContent}};
  selectStage('Input');
  setStatus(`Input image ready. Run ${{selectedModelName()}}, then use the diagram to inspect every layer.`, 'ok');
}});

runButton.addEventListener('click', async () => {{
  if (!imageSelect.value) return;
  runButton.disabled = true;
  setStatus(`Running ${{selectedModelName()}} locally and capturing selectable layer responses…`);
  try {{
    const response = await fetch('/api/run', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        image_name: imageSelect.value,
        fallback: fallbackToggle.checked,
        model_key: selectedModelKey(),
        activation_colour_map: selectedActivationColourMap()
      }})
    }});
    const data = await response.json();
    renderAnalysisResult(data);
  }} catch (error) {{
    setStatus(`The local app could not complete the run: ${{error}}`, 'error');
  }} finally {{
    runButton.disabled = !imageSelect.value;
  }}
}});

startCameraButton.addEventListener('click', startCamera);
cameraRunButton.addEventListener('click', runCameraFrame);
liveRunButton.addEventListener('click', toggleLiveRun);
resetButton.addEventListener('click', resetDemo);
modelSelect.addEventListener('change', () => {{
  stopLiveRun();
  applyModelSelection(modelSelect.value);
}});
activationColourSelect.addEventListener('change', () => {{
  applyActivationColourMapSelection(activationColourSelect.value);
  if (!liveRunActive) {{
    visualisationsByLabel = new Map();
    renderSelectedLayerDetail();
    setStatus('Layer image colours set. Run the selected model again to redraw the layer images with this palette.', 'ok');
  }}
}});
themeSelect.addEventListener('change', () => applyTheme(themeSelect.value));

initialiseModelSelection();
initialiseActivationColourMap();
initialiseTheme();
loadCaptions();
loadImages();
</script>
</body>
</html>
"""
