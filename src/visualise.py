"""Feature-map visualisation helpers for AlexNet activations."""

from __future__ import annotations

import base64
from io import BytesIO
from math import ceil

import numpy as np
from PIL import Image
import torch


def normalise_layer_name(layer_name: str) -> str:
    """Return a display-safe layer name."""
    return layer_name.strip()


def activation_grid_png_base64(
    activation: torch.Tensor,
    *,
    max_channels: int = 64,
    columns: int = 8,
    tile_size: int = 56,
    gap: int = 4,
) -> str:
    """Render fixed activation channels as a vivid feature-map grid.

    Grid positions are stable across frames: tile 1 is always the same channel,
    tile 2 is always the same channel, and so on. Each channel is normalised
    independently so visitors can see spatial response patterns. The result is
    a PNG data URI suitable for a local web UI.
    """
    if activation.ndim != 4:
        raise ValueError("Expected activation tensor with shape (batch, channels, height, width).")

    maps = activation[0].float()
    if maps.numel() == 0:
        raise ValueError("Activation tensor is empty.")

    selected = select_fixed_channels(maps, max_channels=max_channels)
    channel_count = selected.shape[0]
    rows = ceil(channel_count / columns)
    grid_width = columns * tile_size + (columns + 1) * gap
    grid_height = rows * tile_size + (rows + 1) * gap
    grid = Image.new("RGB", (grid_width, grid_height), color=(4, 6, 22))

    for idx, feature_map in enumerate(selected):
        row, col = divmod(idx, columns)
        tile = _render_feature_tile(feature_map.numpy(), tile_size=tile_size)
        x = gap + col * (tile_size + gap)
        y = gap + row * (tile_size + gap)
        grid.paste(tile, (x, y))

    buffer = BytesIO()
    grid.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def fixed_channel_indices(channel_count: int, *, max_channels: int) -> list[int]:
    """Return stable channel indices for a layer.

    The first `max_channels` channels are used intentionally. This makes the
    grid steady in live camera mode because each tile position represents the
    same AlexNet channel on every frame.
    """
    if channel_count < 1:
        return []
    keep = min(max_channels, channel_count)
    return list(range(keep))


def select_fixed_channels(maps: torch.Tensor, *, max_channels: int) -> torch.Tensor:
    """Pick a stable set of channels instead of re-ranking them per frame."""
    indices = fixed_channel_indices(maps.shape[0], max_channels=max_channels)
    if not indices:
        return maps[:0]
    return maps[indices]


def _render_feature_tile(feature_map: np.ndarray, *, tile_size: int) -> Image.Image:
    """Convert one feature map to a vivid coloured tile."""
    feature_map = feature_map.astype(np.float32)
    low = float(np.percentile(feature_map, 1))
    high = float(np.percentile(feature_map, 99.7))
    if high > low:
        normalised = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
    else:
        normalised = np.zeros_like(feature_map, dtype=np.float32)

    coloured = apply_activation_colour_map(normalised)
    tile = Image.fromarray(coloured, mode="RGB")
    return tile.resize((tile_size, tile_size), resample=Image.Resampling.BILINEAR)


def apply_activation_colour_map(values: np.ndarray) -> np.ndarray:
    """Apply a high-contrast booth-friendly activation colour map.

    The palette keeps low responses close to dark navy so quieter regions do
    not distract, then moves through purple, blue, cyan, yellow, and warm white
    for stronger responses. It is intentionally vivid for a large public
    display while remaining deterministic and dependency-free.
    """
    anchors = np.array(
        [
            [2, 6, 23],
            [24, 18, 77],
            [80, 31, 140],
            [37, 99, 235],
            [14, 165, 233],
            [45, 212, 191],
            [250, 204, 21],
            [255, 247, 237],
        ],
        dtype=np.float32,
    )
    values = np.clip(values, 0.0, 1.0)
    # Slight gamma lift makes meaningful responses pop without washing out the
    # dark background.
    values = values**0.62
    scaled = values * (len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(anchors) - 1)
    blend = (scaled - lower)[..., None]
    rgb = anchors[lower] * (1.0 - blend) + anchors[upper] * blend
    return np.clip(rgb, 0, 255).astype(np.uint8)
