# How Does a Neural Network See?

A local-first Open Day demo that uses AlexNet to explain how a trained vision model responds at different layers.

The app now uses FastAPI with a simple dependency-free browser UI. The current MVP supports curated image discovery, optional local camera capture, image preview, reset behaviour, top-5 AlexNet predictions when pretrained weights are available locally, selected activation-grid visualisations, and graceful messaging when live weights are unavailable. Full fallback replay will be added in a later phase.

## What the demo shows

The demo is designed for a public university booth. A staff member or visitor selects a curated local image or explicitly starts the local camera mode, then can view:

- the selected input image;
- a single opt-in camera frame, if camera mode is used;
- top-5 AlexNet predictions;
- selected layer responses;
- simple feature-map grid visualisations;
- short public captions for each stage.

Use this wording when explaining the app:

> “This demo shows how a trained vision model responds at different layers. Early layers respond to simple visual patterns, while deeper layers combine those patterns into features useful for classification. The final prediction is a likely label, not guaranteed truth.”

## What it does not show

This demo does **not** show private reasoning, human-like sight, guaranteed truth, or a model that is always correct. It does not use visitor uploads or visitor data storage. Camera mode is opt-in, local-only, and analyses a captured still frame without saving it.

Avoid phrases such as:

- “The AI is thinking”
- “The AI sees exactly like a human”
- “This shows the model’s private reasoning”
- “The AI is always correct”

## Setup

Use Python 3.11 or newer.

Recommended setup uses the project script so packages are installed into the
local `.venv`, not global Python:

```bash
bash scripts/setup.sh
```

PowerShell:

```powershell
pwsh -NoProfile -File scripts/setup.ps1
```

Then activate the environment:

```bash
source .venv/bin/activate
```

PowerShell:

```powershell
. .venv/bin/Activate.ps1
```

Manual setup is also possible:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You can run a local setup check with:

```bash
python scripts/check_setup.py
```

## Add curated images

Place booth-safe curated images in:

```text
assets/demo_images/
```

Supported extensions are:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

The MVP intentionally does not include visitor photo upload. Optional camera mode is local-only and does not save captured frames. If the image folder is empty, the app shows a clear message explaining where to place images.

## Live camera mode

Camera mode was added as an explicit opt-in booth feature.

- The browser asks for camera permission.
- The preview stays in the local browser.
- Press **Capture + run** to send one still frame to the local FastAPI app.
- Press **Start continuous AlexNet** to repeatedly analyse the latest camera frame while the button is active.
- Continuous mode runs one local AlexNet request at a time and updates predictions plus the selected detailed feature-map view as fast as the model completes frames. Feature-map tile positions are fixed so the mini images do not swap places between frames.
- Frames are analysed in memory and are not written to disk.
- Do not use camera mode for visitors who do not consent.

## Precompute fallback assets

Fallback replay mode will be implemented in a later phase. The intended command is:

```bash
python scripts/precompute_fallback.py
```

When complete, this will save prediction JSON and feature-map images into:

```text
assets/fallback/
```

## Run the app

The Open Day port is fixed by convention:

```bash
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 3450
```

Or use:

```bash
bash scripts/run_dev.sh
```

PowerShell:

```powershell
pwsh -NoProfile -File scripts/run_dev.ps1
```

The run scripts first stop any previous `uvicorn app:app` demo process for this project on the configured port, then start a fresh server.

To stop the demo without starting it again:

```bash
bash scripts/stop_dev.sh
```

PowerShell:

```powershell
pwsh -NoProfile -File scripts/stop_dev.ps1
```

The FastAPI app reads these defaults from `.env` when using the run scripts. `.env.example` documents them:

```env
DEMO_NAME=alexnet-vision-demo
FRONTEND_HOST=127.0.0.1
FRONTEND_PORT=3450
```

Do not use random fallback ports for Open Day mode.

## Reset and fallback behaviour

The reset button clears the current image selection, stops any active camera preview and continuous run loop, clears results, and returns the app to the default instruction screen.

Fallback/replay mode is shown in the UI now, but full fallback asset playback is part of a later build phase. Live AlexNet prediction failures are displayed as public setup messages rather than app crashes.

Live mode currently shows fixed-channel feature-map grids for early, middle, and deep AlexNet convolution layers. Each tile position represents the same channel across frames, and each channel is normalised for display. Dark navy/purple means quieter response; cyan, yellow, and white mean stronger response.

## Public booth script

> “This demo shows how a trained vision model responds at different layers. Early layers respond to simple visual patterns, while deeper layers combine those patterns into features useful for classification. The final prediction is a likely label, not guaranteed truth.”

## Troubleshooting

### No images appear

Add `.jpg`, `.jpeg`, `.png`, or `.webp` files to `assets/demo_images/`, then refresh the browser page.

### FastAPI or Uvicorn is not installed

Activate your virtual environment and run:

```bash
python -m pip install -r requirements.txt
```

Or rerun the setup script:

```powershell
pwsh -NoProfile -File scripts/setup.ps1
```

### The app uses the wrong port or says the port is already in use

The run scripts should stop previous demo processes automatically. You can also stop them manually:

```powershell
pwsh -NoProfile -File scripts/stop_dev.ps1
```

Then run the app with the explicit Open Day command:

```bash
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 3450
```

### AlexNet weights are unavailable

The app fails gracefully if pretrained AlexNet weights are unavailable locally. Run setup on a networked machine once so torchvision can cache the weights, or use fallback replay after Phase 5 assets are precomputed.

## Open Day readiness checklist

- [ ] Curated booth-safe images added to `assets/demo_images/`.
- [ ] App starts on `127.0.0.1:3450`.
- [ ] Reset button returns to the default instruction screen.
- [ ] No visitor upload or visitor data storage is present.
- [ ] Camera mode is used only with explicit visitor consent.
- [ ] Staff can explain the demo in under 45 seconds.
- [ ] Fallback replay assets are precomputed once Phase 5 is complete.
- [ ] Tests pass before the booth session.
