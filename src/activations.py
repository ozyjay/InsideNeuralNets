"""Activation capture helpers for selectable vision-model layers."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

import torch


@dataclass(frozen=True)
class LayerSpec:
    """A public demo layer to capture and explain."""

    key: str
    label: str
    module_path: str
    caption_key: str
    public_note: str


ALEXNET_LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec(
        key="conv1",
        label="Conv 1",
        module_path="features.1",
        caption_key="Conv 1",
        public_note="First convolution response after ReLU, shown as fixed feature-map channels so grid positions stay stable.",
    ),
    LayerSpec(
        key="pool1",
        label="Pool 1",
        module_path="features.2",
        caption_key="Pool 1",
        public_note="First max-pooling output, where nearby strong responses are kept and the map becomes smaller.",
    ),
    LayerSpec(
        key="conv2",
        label="Conv 2",
        module_path="features.4",
        caption_key="Conv 2",
        public_note="Second convolution response after ReLU, where simple patterns are combined into richer local features.",
    ),
    LayerSpec(
        key="pool2",
        label="Pool 2",
        module_path="features.5",
        caption_key="Pool 2",
        public_note="Second max-pooling output, preserving strong responses in a smaller spatial grid.",
    ),
    LayerSpec(
        key="conv3",
        label="Conv 3",
        module_path="features.7",
        caption_key="Conv 3",
        public_note="Third convolution response after ReLU, combining earlier patterns into more complex textures and parts.",
    ),
    LayerSpec(
        key="conv4",
        label="Conv 4",
        module_path="features.9",
        caption_key="Conv 4",
        public_note="Fourth convolution response after ReLU, continuing to combine useful visual patterns.",
    ),
    LayerSpec(
        key="conv5",
        label="Conv 5",
        module_path="features.11",
        caption_key="Conv 5",
        public_note="Final convolution response after ReLU, before the classifier layers.",
    ),
    LayerSpec(
        key="pool5",
        label="Pool 5",
        module_path="features.12",
        caption_key="Pool 5",
        public_note="Final max-pooling output, a compact spatial summary passed toward the classifier.",
    ),
    LayerSpec(
        key="avgpool",
        label="Avg pool",
        module_path="avgpool",
        caption_key="Avg pool",
        public_note="Adaptive average-pooling output, shaped into the fixed grid expected by AlexNet’s classifier.",
    ),
)

RESNET50_LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec(
        key="stem",
        label="Stem",
        module_path="relu",
        caption_key="Early features",
        public_note="The first ResNet-50 stage responds to simple visual patterns after the opening convolution.",
    ),
    LayerSpec(
        key="maxpool",
        label="Max pool",
        module_path="maxpool",
        caption_key="Pooling",
        public_note="Pooling keeps strong nearby responses and reduces the spatial size before the residual stages.",
    ),
    LayerSpec(
        key="layer1",
        label="Residual 1",
        module_path="layer1",
        caption_key="Mid features",
        public_note="The first residual block group combines simple responses into richer local features.",
    ),
    LayerSpec(
        key="layer2",
        label="Residual 2",
        module_path="layer2",
        caption_key="Mid features",
        public_note="This residual block group builds more complex textures and repeated shapes.",
    ),
    LayerSpec(
        key="layer3",
        label="Residual 3",
        module_path="layer3",
        caption_key="Deep features",
        public_note="Deeper residual features combine earlier patterns into more specialised object-part responses.",
    ),
    LayerSpec(
        key="layer4",
        label="Residual 4",
        module_path="layer4",
        caption_key="Deep features",
        public_note="The final residual group produces compact, high-level feature maps for classification.",
    ),
    LayerSpec(
        key="avgpool",
        label="Avg pool",
        module_path="avgpool",
        caption_key="Avg pool",
        public_note="Adaptive average pooling turns the final feature maps into one compact value per channel.",
    ),
)

MOBILENET_V3_LARGE_LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec(
        key="stem",
        label="Stem",
        module_path="features.0",
        caption_key="Early features",
        public_note="The opening MobileNetV3 stage responds to simple visual patterns using efficient convolutions.",
    ),
    LayerSpec(
        key="early",
        label="Early block",
        module_path="features.2",
        caption_key="Early features",
        public_note="Early inverted residual blocks keep useful local patterns while staying fast for live use.",
    ),
    LayerSpec(
        key="middle1",
        label="Middle 1",
        module_path="features.4",
        caption_key="Mid features",
        public_note="Middle MobileNetV3 features combine simple patterns into richer local responses.",
    ),
    LayerSpec(
        key="middle2",
        label="Middle 2",
        module_path="features.7",
        caption_key="Mid features",
        public_note="Later middle blocks build more structured textures and object-part responses.",
    ),
    LayerSpec(
        key="deep",
        label="Deep block",
        module_path="features.13",
        caption_key="Deep features",
        public_note="Deep MobileNetV3 features are compact responses useful for classification.",
    ),
    LayerSpec(
        key="final",
        label="Final conv",
        module_path="features.16",
        caption_key="Deep features",
        public_note="The final convolution prepares high-level features for the classifier.",
    ),
    LayerSpec(
        key="avgpool",
        label="Avg pool",
        module_path="avgpool",
        caption_key="Avg pool",
        public_note="Adaptive average pooling turns the final feature maps into a compact channel summary.",
    ),
)

MODEL_LAYER_SPECS: dict[str, tuple[LayerSpec, ...]] = {
    "alexnet": ALEXNET_LAYER_SPECS,
    "resnet50": RESNET50_LAYER_SPECS,
    "mobilenet_v3_large": MOBILENET_V3_LARGE_LAYER_SPECS,
}

# Backwards-compatible names used by older tests and helpers.
SELECTED_LAYER_SPECS = ALEXNET_LAYER_SPECS
SELECTED_LAYER_NAMES = [spec.label for spec in SELECTED_LAYER_SPECS]


def normalise_model_key(model_key: str | None) -> str:
    """Return a supported model key, defaulting to AlexNet."""
    if model_key in MODEL_LAYER_SPECS:
        return str(model_key)
    return "alexnet"


def layer_specs_for_model(model_key: str | None) -> tuple[LayerSpec, ...]:
    """Return public capture specs for a supported model."""
    return MODEL_LAYER_SPECS[normalise_model_key(model_key)]


def get_module_by_path(model: torch.nn.Module, module_path: str) -> torch.nn.Module:
    """Return a nested module using a dotted path such as `features.4`."""
    module: torch.nn.Module = model
    for part in module_path.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


class ActivationCapture:
    """Context manager that captures selected layer outputs during one forward pass."""

    def __init__(self, model: torch.nn.Module, specs: tuple[LayerSpec, ...] = SELECTED_LAYER_SPECS) -> None:
        self.model = model
        self.specs = specs
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ActivationCapture":
        for spec in self.specs:
            module = get_module_by_path(self.model, spec.module_path)
            self._handles.append(module.register_forward_hook(self._make_hook(spec.key)))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, key: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                self.activations[key] = output.detach().cpu()

        return hook
