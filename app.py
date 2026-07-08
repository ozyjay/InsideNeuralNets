"""FastAPI app for the AlexNet Open Day demo."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from src.captions import CAPTIONS
from src.demo_images import DEMO_IMAGES_DIR, discover_images, format_image_label
from src.model import ModelUnavailableError, run_alexnet_analysis

APP_TITLE = "How Does a Neural Network See?"

app = FastAPI(
    title="AlexNet Vision Demo",
    description="Local-first Open Day demo showing AlexNet layer responses and predictions.",
    version="0.2.0",
)


class RunRequest(BaseModel):
    """Request body for running AlexNet on a curated image."""

    image_name: str = Field(..., min_length=1)
    fallback: bool = False


class CameraRunRequest(BaseModel):
    """Request body for running AlexNet on a locally captured camera frame."""

    image_data: str = Field(..., min_length=1)
    fallback: bool = False
    include_visualisations: bool = True


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
    include_visualisations: bool = True,
) -> JSONResponse:
    """Run AlexNet analysis and return a consistent JSON response."""
    try:
        analysis = run_alexnet_analysis(image, include_visualisations=include_visualisations)
    except ModelUnavailableError as exc:
        return JSONResponse(
            {
                "ok": False,
                "mode": "live",
                "source": source,
                "message": str(exc),
                "help": "Run the setup script and pre-download AlexNet weights, or use fallback replay once assets have been precomputed.",
                "predictions": [],
                "visualisations": [],
                "visualisations_included": include_visualisations,
            }
        )

    return JSONResponse(
        {
            "ok": True,
            "mode": "live",
            "source": source,
            "message": "AlexNet returned likely ImageNet labels. These can be wrong.",
            "predictions": [prediction.to_dict() for prediction in analysis.predictions],
            "visualisations": [visualisation.to_dict() for visualisation in analysis.visualisations],
            "visualisations_included": include_visualisations,
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
    """Return editable public captions for the layer timeline."""
    return CAPTIONS


@app.post("/api/run")
def run_demo(request: RunRequest) -> JSONResponse:
    """Run AlexNet top-5 inference for a curated image.

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
                "message": "Fallback replay assets will be wired in Phase 5. Turn replay mode off for live AlexNet inference.",
                "predictions": [],
                "visualisations": [],
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

    return _run_live_analysis_response(image, source="curated")


@app.post("/api/run-camera")
def run_camera_demo(request: CameraRunRequest) -> JSONResponse:
    """Run AlexNet on a browser-captured camera frame without saving it."""
    if request.fallback:
        return JSONResponse(
            {
                "ok": False,
                "mode": "fallback",
                "source": "camera",
                "message": "Fallback replay uses curated precomputed assets, not live camera frames. Turn replay mode off for camera inference.",
                "predictions": [],
                "visualisations": [],
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
        include_visualisations=request.include_visualisations,
    )


def _render_index_html() -> str:
    """Return dependency-free HTML, CSS, and JavaScript for the booth UI."""
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
      --panel: rgba(15, 23, 42, 0.82);
      --panel-2: #172033;
      --panel-3: #0d1324;
      --text: #f8fbff;
      --muted: #b9c4d8;
      --accent: #22d3ee;
      --accent-2: #8b5cf6;
      --accent-3: #f59e0b;
      --danger: #fecaca;
      --ok: #bbf7d0;
      --border: rgba(148, 163, 184, 0.22);
      --glow: rgba(34, 211, 238, 0.22);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at 12% 6%, rgba(34,211,238,0.22) 0, transparent 28%), radial-gradient(circle at 86% 12%, rgba(139,92,246,0.24) 0, transparent 30%), linear-gradient(135deg, var(--bg) 0%, var(--bg-2) 56%, #070711 100%); color: var(--text); }}
    body::before {{ content: ""; position: fixed; inset: 0; pointer-events: none; background-image: linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px); background-size: 42px 42px; mask-image: radial-gradient(circle at top, black, transparent 72%); }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 28px 0 44px; position: relative; }}
    header {{ margin-bottom: 22px; }}
    h1 {{ font-size: clamp(2rem, 4vw, 4.5rem); line-height: 1; margin: 0 0 14px; letter-spacing: -0.05em; background: linear-gradient(90deg, #f8fbff, #67e8f9 45%, #c4b5fd 76%, #fbbf24); -webkit-background-clip: text; background-clip: text; color: transparent; }}
    h2 {{ margin: 0 0 14px; font-size: 1.25rem; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .grid {{ display: grid; grid-template-columns: 360px 1fr; gap: 20px; align-items: start; }}
    .card {{ background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(10,15,30,0.88)); border: 1px solid var(--border); border-radius: 24px; padding: 20px; box-shadow: 0 24px 70px rgba(0,0,0,0.38), 0 0 38px rgba(34,211,238,0.07); backdrop-filter: blur(14px); }}
    .stack {{ display: grid; gap: 16px; }}
    label {{ display: block; margin-bottom: 8px; font-weight: 700; }}
    select, button {{ width: 100%; border-radius: 14px; border: 1px solid var(--border); padding: 12px 14px; font: inherit; }}
    select {{ background: linear-gradient(180deg, #1b2540, #121a2f); color: var(--text); }}
    button {{ background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #03111d; font-weight: 900; cursor: pointer; box-shadow: 0 12px 30px rgba(34,211,238,0.22); transition: transform 120ms ease, filter 120ms ease, box-shadow 120ms ease; }}
    button:hover:not(:disabled) {{ transform: translateY(-1px); filter: brightness(1.08); box-shadow: 0 16px 38px rgba(139,92,246,0.26); }}
    button.secondary {{ background: linear-gradient(180deg, #27324d, #1a2338); color: var(--text); box-shadow: none; }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; box-shadow: none; }}
    .toggle {{ display: flex; gap: 10px; align-items: center; color: var(--muted); }}
    .toggle input {{ width: auto; transform: scale(1.2); accent-color: var(--accent); }}
    .preview {{ min-height: 340px; display: grid; place-items: center; background: radial-gradient(circle at 50% 25%, rgba(34,211,238,0.13), transparent 34%), #050814; border-radius: 18px; overflow: hidden; border: 1px solid var(--border); }}
    .preview img, .preview video {{ max-width: 100%; max-height: 520px; display: block; }}
    .preview video {{ width: 100%; transform: scaleX(-1); }}
    .camera-actions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .camera-actions .wide {{ grid-column: 1 / -1; }}
    .camera-note {{ font-size: 0.88rem; margin: 8px 0 0; color: var(--muted); }}
    .message {{ padding: 12px 14px; border-radius: 14px; background: linear-gradient(180deg, rgba(30,41,59,0.98), rgba(17,24,39,0.98)); color: var(--muted); border: 1px solid rgba(148,163,184,0.16); }}
    .message.error {{ background: linear-gradient(180deg, rgba(127,29,29,0.46), rgba(69,10,10,0.34)); color: var(--danger); border-color: rgba(248,113,113,0.3); }}
    .message.ok {{ background: linear-gradient(180deg, rgba(20,83,45,0.45), rgba(6,78,59,0.34)); color: var(--ok); border-color: rgba(74,222,128,0.28); }}
    .predictions {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 10px; }}
    .predictions li {{ display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; background: linear-gradient(180deg, #10182b, #0a1020); padding: 12px; border-radius: 14px; border: 1px solid rgba(148,163,184,0.12); }}
    .bar {{ grid-column: 1 / -1; height: 9px; background: #1f2941; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: linear-gradient(90deg, #22d3ee, #a78bfa, #f59e0b); box-shadow: 0 0 18px rgba(34,211,238,0.55); }}
    .timeline {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .timeline button {{ width: auto; padding: 9px 12px; background: linear-gradient(180deg, #25304a, #182239); color: var(--text); box-shadow: none; }}
    .timeline button.active {{ background: linear-gradient(135deg, #22d3ee, #f59e0b); color: #06101c; }}
    .muted {{ color: var(--muted); }}
    .network {{ margin-bottom: 16px; padding: 16px; border-radius: 18px; background: radial-gradient(circle at 20% 45%, rgba(34,211,238,0.18), transparent 32%), radial-gradient(circle at 78% 50%, rgba(245,158,11,0.13), transparent 35%), #030611; border: 1px solid var(--border); overflow: hidden; }}
    .network svg {{ width: 100%; height: auto; display: block; filter: drop-shadow(0 0 16px rgba(34,211,238,0.12)); }}
    .network .layer {{ fill: rgba(34, 211, 238, 0.13); stroke: rgba(226, 232, 240, 0.78); stroke-width: 2; }}
    .network .layer.active {{ fill: rgba(245, 158, 11, 0.38); stroke: #fbbf24; }}
    .network text {{ fill: #f8fbff; font-size: 15px; font-weight: 900; text-anchor: middle; dominant-baseline: middle; }}
    .feature-maps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
    .feature-card {{ background: linear-gradient(180deg, #081024, #060912); border: 1px solid rgba(34,211,238,0.16); border-radius: 18px; padding: 14px; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04); cursor: pointer; }}
    .feature-card:hover {{ border-color: rgba(245,158,11,0.42); }}
    .feature-card h3 {{ margin: 0 0 8px; font-size: 1rem; color: #f8fbff; }}
    .feature-card.active {{ border-color: rgba(245,158,11,0.72); box-shadow: 0 0 0 1px rgba(245,158,11,0.28), 0 18px 44px rgba(0,0,0,0.28); }}
    .feature-card img {{ display: block; width: 100%; border-radius: 12px; image-rendering: auto; background: #020208; border: 1px solid rgba(255,255,255,0.08); }}
    .feature-card small {{ display: block; color: var(--muted); margin-top: 8px; line-height: 1.4; }}
    .layer-detail {{ margin: 0 0 16px; display: grid; grid-template-columns: minmax(260px, 1.4fr) minmax(220px, 0.8fr); gap: 16px; align-items: start; background: linear-gradient(180deg, rgba(8,16,36,0.98), rgba(4,7,16,0.98)); border: 1px solid rgba(34,211,238,0.2); border-radius: 18px; padding: 16px; }}
    .layer-detail.placeholder {{ display: block; }}
    .layer-detail img {{ width: 100%; display: block; border-radius: 14px; background: #020208; border: 1px solid rgba(255,255,255,0.09); box-shadow: 0 24px 60px rgba(0,0,0,0.32); }}
    .detail-copy h3 {{ margin: 0 0 8px; font-size: 1.18rem; color: #fef3c7; }}
    .detail-copy p {{ margin: 0 0 10px; }}
    .detail-pill {{ display: inline-flex; align-items: center; margin: 4px 6px 4px 0; padding: 6px 9px; border-radius: 999px; background: rgba(34,211,238,0.12); border: 1px solid rgba(34,211,238,0.22); color: #cffafe; font-size: 0.82rem; font-weight: 800; }}
    @media (max-width: 980px) {{ .layer-detail {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>{APP_TITLE}</h1>
    <p>This local demo shows how a trained vision model responds at different layers. Early layers often respond to simple visual patterns, while deeper layers combine those patterns into features useful for classification. The final prediction is a likely label, not guaranteed truth.</p>
  </header>

  <section class="grid">
    <aside class="card stack">
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
          <button id="liveRunButton" class="wide" type="button" disabled>Start continuous AlexNet</button>
        </div>
        <p class="camera-note">Opt-in local camera mode. Frames are sent only to this local app for analysis and are not saved.</p>
      </div>

      <label class="toggle"><input id="fallbackToggle" type="checkbox" /> Fallback / replay mode</label>

      <button id="runButton" disabled>Run AlexNet</button>
      <button id="resetButton" class="secondary">Reset demo</button>

      <div>
        <h2>Layer timeline</h2>
        <div class="timeline" id="timeline"></div>
        <p id="caption" class="message">Choose a stage to read the caption.</p>
      </div>
    </aside>

    <section class="stack">
      <div class="card">
        <h2>Input image preview</h2>
        <div id="preview" class="preview"><p class="muted">Choose a curated image to preview it here.</p></div>
      </div>

      <div class="card">
        <h2>Top-5 predictions</h2>
        <div id="status" class="message">Predictions will appear here after you run AlexNet.</div>
        <ol id="predictions" class="predictions"></ol>
      </div>

      <div class="card">
        <h2>Feature visualisation</h2>
        <div class="network" aria-label="Simplified AlexNet layer diagram">
          <svg viewBox="0 0 760 190" role="img">
            <title>Simplified AlexNet path from input through selected layers</title>
            <polygon class="layer" data-layer="Input" points="20,30 105,45 105,155 20,170" />
            <polygon class="layer" data-layer="Early layer" points="95,50 185,66 185,140 95,156" />
            <polygon class="layer" data-layer="Middle layer" points="185,72 360,78 360,132 185,138" />
            <polygon class="layer" data-layer="Deep layer" points="365,78 535,84 535,126 365,132" />
            <polygon class="layer" data-layer="Prediction" points="545,76 730,88 730,122 545,134" />
            <text x="62" y="100">Input</text>
            <text x="140" y="103">Early</text>
            <text x="272" y="105">Middle</text>
            <text x="450" y="105">Deep</text>
            <text x="638" y="105">Prediction</text>
          </svg>
        </div>
        <div id="layerDetail" class="layer-detail placeholder">
          <p class="message">Click a feature-map card after running AlexNet to enlarge that layer and read a more detailed explanation.</p>
        </div>
        <div id="featureMaps" class="feature-maps">
          <p class="message">Run AlexNet to show selected activation grids for early, middle, and deep convolution layers.</p>
        </div>
      </div>
    </section>
  </section>
</main>

<script>
const imageSelect = document.getElementById('imageSelect');
const preview = document.getElementById('preview');
const runButton = document.getElementById('runButton');
const startCameraButton = document.getElementById('startCameraButton');
const cameraRunButton = document.getElementById('cameraRunButton');
const liveRunButton = document.getElementById('liveRunButton');
const resetButton = document.getElementById('resetButton');
const fallbackToggle = document.getElementById('fallbackToggle');
const statusBox = document.getElementById('status');
const predictions = document.getElementById('predictions');
const featureMaps = document.getElementById('featureMaps');
const layerDetail = document.getElementById('layerDetail');
const timeline = document.getElementById('timeline');
const caption = document.getElementById('caption');
let captions = {{}};
let visualisationsByLabel = new Map();
let selectedDetailLabel = null;
let cameraStream = null;
let cameraVideo = null;
let liveRunActive = false;
let liveFrameIndex = 0;

function setStatus(text, kind = '') {{
  statusBox.className = `message ${{kind}}`;
  statusBox.textContent = text;
}}

function clearResults() {{
  predictions.innerHTML = '';
  visualisationsByLabel = new Map();
  selectedDetailLabel = null;
  layerDetail.className = 'layer-detail placeholder';
  layerDetail.innerHTML = '<p class="message">Click a feature-map card after running AlexNet to enlarge that layer and read a more detailed explanation.</p>';
  featureMaps.innerHTML = '<p class="message">Run AlexNet to show selected activation grids for early, middle, and deep convolution layers.</p>';
  setStatus('Predictions will appear here after you run AlexNet.');
}}

function resetDemo() {{
  stopCamera();
  imageSelect.value = '';
  fallbackToggle.checked = false;
  runButton.disabled = true;
  preview.innerHTML = '<p class="muted">Choose a curated image to preview it here.</p>';
  clearResults();
  selectStage('Input');
}}

function selectStage(stage) {{
  [...timeline.querySelectorAll('button')].forEach(button => button.classList.toggle('active', button.dataset.stage === stage));
  document.querySelectorAll('.network .layer').forEach(layer => layer.classList.toggle('active', layer.dataset.layer === stage));
  caption.textContent = captions[stage] || 'This shows how a trained vision model responds at this stage.';
}}

function showLayerDetail(item, options = {{}}) {{
  selectedDetailLabel = item.label;
  selectStage(item.label);
  document.querySelectorAll('.feature-card').forEach(card => card.classList.toggle('active', card.dataset.layer === item.label));
  const captionText = captions[item.caption_key] || item.note || 'This shows fixed channels from this layer response.';
  layerDetail.className = 'layer-detail';
  layerDetail.innerHTML = `
    <img src="${{item.image_data}}" alt="Large ${{item.label}} activation grid" />
    <div class="detail-copy">
      <h3>${{item.label}}</h3>
      <p>${{captionText}}</p>
      <p>${{item.note || 'Each square is one fixed channel from this layer, so the tile position stays stable across frames. Cyan, yellow, and white regions indicate stronger responses after normalising that channel for display.'}}</p>
      <span class="detail-pill">${{item.tensor_shape.join(' × ')}}</span>
      <span class="detail-pill">Fixed channel positions</span>
      <span class="detail-pill">Cyan/yellow/white = stronger</span>
      <span class="detail-pill">Normalised for display</span>
      <span class="detail-pill">Updates with each live frame</span>
    </div>`;
  if (options.scroll !== false) {{
    layerDetail.scrollIntoView({{behavior: 'smooth', block: 'nearest'}});
  }}
}}

function selectLayerFromDiagram(stage) {{
  selectStage(stage);
  const item = visualisationsByLabel.get(stage);
  if (item) {{
    showLayerDetail(item);
  }}
}}

function renderAnalysisResult(data) {{
  setStatus(data.message || 'Run complete.', data.ok ? 'ok' : 'error');
  if (data.help) {{
    setStatus(`${{data.message}} ${{data.help}}`, data.ok ? 'ok' : 'error');
  }}
  predictions.innerHTML = '';
  visualisationsByLabel = new Map();
  data.predictions.forEach(item => {{
    const li = document.createElement('li');
    const pct = Math.round(item.probability * 1000) / 10;
    li.innerHTML = `<strong>${{item.label}}</strong><span>${{pct}}%</span><div class="bar"><span style="width: ${{pct}}%"></span></div>`;
    predictions.appendChild(li);
  }});
  const visualisationsIncluded = data.visualisations_included !== false;
  if (visualisationsIncluded) {{
    featureMaps.innerHTML = '';
    if (data.visualisations && data.visualisations.length) {{
      data.visualisations.forEach(item => {{
        visualisationsByLabel.set(item.label, item);
        const card = document.createElement('article');
        card.className = 'feature-card';
        card.dataset.layer = item.label;
        const captionText = captions[item.caption_key] || item.note || '';
        card.innerHTML = `<h3>${{item.label}}</h3><img src="${{item.image_data}}" alt="${{item.label}} activation grid" /><small>${{captionText}}</small><small>Tensor shape: ${{item.tensor_shape.join(' × ')}}</small><small>Click to enlarge this layer.</small>`;
        card.addEventListener('click', () => showLayerDetail(item));
        featureMaps.appendChild(card);
      }});
      const detailLabel = selectedDetailLabel || data.visualisations[0].label;
      const detailItem = visualisationsByLabel.get(detailLabel);
      if (detailItem) {{
        showLayerDetail(detailItem, {{scroll: false}});
      }}
    }} else if (data.ok) {{
      layerDetail.className = 'layer-detail placeholder';
      layerDetail.innerHTML = '<p class="message">Click a feature-map card after running AlexNet to enlarge that layer and read a more detailed explanation.</p>';
      featureMaps.innerHTML = '<p class="message">The model ran, but no selected activation grids were captured.</p>';
    }} else {{
      layerDetail.className = 'layer-detail placeholder';
      layerDetail.innerHTML = '<p class="message">Detailed layer views will appear when live inference or fallback replay is available.</p>';
      featureMaps.innerHTML = '<p class="message">Activation grids will appear here when live inference or fallback replay is available.</p>';
    }}
  }}
}}

function stopLiveRun() {{
  liveRunActive = false;
  liveRunButton.textContent = 'Start continuous AlexNet';
  liveRunButton.classList.remove('secondary');
}}

function stopCamera() {{
  stopLiveRun();
  if (cameraStream) {{
    cameraStream.getTracks().forEach(track => track.stop());
  }}
  cameraStream = null;
  cameraVideo = null;
  cameraRunButton.disabled = true;
  liveRunButton.disabled = true;
  startCameraButton.textContent = 'Start camera';
}}

async function startCamera() {{
  if (cameraStream) {{
    stopCamera();
    preview.innerHTML = '<p class="muted">Choose a curated image or start the camera preview.</p>';
    return;
  }}
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
    setStatus('This browser does not support local camera access.', 'error');
    return;
  }}
  clearResults();
  imageSelect.value = '';
  runButton.disabled = true;
  try {{
    cameraStream = await navigator.mediaDevices.getUserMedia({{video: {{width: {{ideal: 960}}, height: {{ideal: 720}}, facingMode: 'user'}}, audio: false}});
    cameraVideo = document.createElement('video');
    cameraVideo.autoplay = true;
    cameraVideo.playsInline = true;
    cameraVideo.muted = true;
    cameraVideo.srcObject = cameraStream;
    preview.innerHTML = '';
    preview.appendChild(cameraVideo);
    cameraRunButton.disabled = false;
    liveRunButton.disabled = false;
    startCameraButton.textContent = 'Stop camera';
    setStatus('Camera preview is local. Capture one frame or start continuous AlexNet.', 'ok');
  }} catch (error) {{
    setStatus(`Camera access was not available: ${{error}}`, 'error');
    stopCamera();
  }}
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

async function analyseCameraFrame({{includeVisualisations = true, live = false}} = {{}}) {{
  const imageData = captureCameraFrame();
  const response = await fetch('/api/run-camera', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{image_data: imageData, fallback: fallbackToggle.checked, include_visualisations: includeVisualisations}})
  }});
  const data = await response.json();
  renderAnalysisResult(data);
  if (live && data.ok) {{
    setStatus(`Continuous AlexNet is running locally. Analysed frame ${{liveFrameIndex}}.`, 'ok');
  }}
  return data;
}}

async function runCameraFrame() {{
  cameraRunButton.disabled = true;
  predictions.innerHTML = '';
  featureMaps.innerHTML = '<p class="message">Capturing selected AlexNet layer responses from the camera frame…</p>';
  setStatus('Capturing one local camera frame…');
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
  try {{
    await analyseCameraFrame({{includeVisualisations: true, live: true}});
  }} catch (error) {{
    setStatus(`Continuous AlexNet stopped: ${{error}}`, 'error');
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
    setStatus('Continuous AlexNet stopped. Camera preview is still local.', 'ok');
    return;
  }}
  if (!cameraStream) {{
    setStatus('Start the camera before continuous AlexNet.', 'error');
    return;
  }}
  liveRunActive = true;
  liveFrameIndex = 0;
  liveRunButton.textContent = 'Stop continuous AlexNet';
  liveRunButton.classList.add('secondary');
  predictions.innerHTML = '';
  featureMaps.innerHTML = '<p class="message">Continuous AlexNet is analysing camera frames locally and refreshing feature maps each frame…</p>';
  setStatus('Continuous AlexNet is starting. Predictions and detailed feature maps update on each analysed frame.', 'ok');
  liveRunLoop();
}}

async function loadCaptions() {{
  captions = await fetch('/api/captions').then(response => response.json());
  timeline.innerHTML = '';
  Object.keys(captions).forEach(stage => {{
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = stage;
    button.dataset.stage = stage;
    button.addEventListener('click', () => selectStage(stage));
    timeline.appendChild(button);
  }});
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
  stopCamera();
  clearResults();
  const selected = imageSelect.selectedOptions[0];
  runButton.disabled = !imageSelect.value;
  if (!imageSelect.value) {{
    preview.innerHTML = '<p class="muted">Choose a curated image to preview it here.</p>';
    return;
  }}
  preview.innerHTML = `<img src="${{selected.dataset.url}}" alt="Selected curated image" />`;
}});

runButton.addEventListener('click', async () => {{
  if (!imageSelect.value) return;
  runButton.disabled = true;
  predictions.innerHTML = '';
  featureMaps.innerHTML = '<p class="message">Capturing selected AlexNet layer responses…</p>';
  setStatus('Running AlexNet locally…');
  try {{
    const response = await fetch('/api/run', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{image_name: imageSelect.value, fallback: fallbackToggle.checked}})
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
document.querySelectorAll('.network .layer').forEach(layer => {{
  layer.style.cursor = 'pointer';
  layer.addEventListener('click', () => selectLayerFromDiagram(layer.dataset.layer));
}});

loadCaptions();
loadImages();
</script>
</body>
</html>
"""
