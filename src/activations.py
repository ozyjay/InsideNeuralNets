"""Activation capture helpers for selected AlexNet layers."""

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


SELECTED_LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec(
        key="early",
        label="Early layer",
        module_path="features.1",
        caption_key="Early layer",
        public_note="First convolution response after ReLU, shown as fixed feature-map channels so grid positions stay stable.",
    ),
    LayerSpec(
        key="middle",
        label="Middle layer",
        module_path="features.4",
        caption_key="Middle layer",
        public_note="Middle convolution response after ReLU, where simple patterns are combined.",
    ),
    LayerSpec(
        key="deep",
        label="Deep layer",
        module_path="features.11",
        caption_key="Deep layer",
        public_note="Final convolution response after ReLU, before the classifier layers.",
    ),
)

SELECTED_LAYER_NAMES = [spec.label for spec in SELECTED_LAYER_SPECS]


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
