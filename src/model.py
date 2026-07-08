"""Vision model loading, prediction, and activation analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from PIL import Image
import torch

from src.activations import ActivationCapture, layer_specs_for_model, normalise_model_key
from src.visualise import DEFAULT_ACTIVATION_COLOUR_MAP, activation_grid_png_base64


class ModelUnavailableError(RuntimeError):
    """Raised when live model inference is not available locally."""


@dataclass(frozen=True)
class VisionModelOption:
    """A selectable local vision model."""

    key: str
    label: str
    short_label: str
    description: str
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe public model description."""
        return {
            "key": self.key,
            "label": self.label,
            "short_label": self.short_label,
            "description": self.description,
            "recommendation": self.recommendation,
            "layers": [
                {
                    "key": spec.key,
                    "label": spec.label,
                    "caption_key": spec.caption_key,
                    "note": spec.public_note,
                }
                for spec in layer_specs_for_model(self.key)
            ],
        }


MODEL_OPTIONS: tuple[VisionModelOption, ...] = (
    VisionModelOption(
        key="alexnet",
        label="AlexNet",
        short_label="Classic",
        description="Classic 2012 CNN. Great for explaining the original layer-by-layer story, but predictions can be rough.",
        recommendation="Best for history and a simple architecture story.",
    ),
    VisionModelOption(
        key="resnet50",
        label="ResNet-50",
        short_label="Better labels",
        description="A stronger residual CNN that usually gives better ImageNet predictions while still exposing feature maps.",
        recommendation="Recommended for booth predictions.",
    ),
    VisionModelOption(
        key="mobilenet_v3_large",
        label="MobileNetV3 Large",
        short_label="Fast live",
        description="A modern efficient CNN. Often better than AlexNet and a good choice for continuous camera mode.",
        recommendation="Recommended when live camera speed matters.",
    ),
)
MODEL_OPTIONS_BY_KEY = {option.key: option for option in MODEL_OPTIONS}


@dataclass(frozen=True)
class Prediction:
    """A public top-k prediction result."""

    label: str
    probability: float

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-safe representation."""
        return {"label": self.label, "probability": self.probability}


@dataclass(frozen=True)
class ActivationVisualisation:
    """A rendered public visualisation for one selected model layer."""

    key: str
    label: str
    caption_key: str
    note: str
    tensor_shape: tuple[int, ...]
    image_data: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return {
            "key": self.key,
            "label": self.label,
            "caption_key": self.caption_key,
            "note": self.note,
            "tensor_shape": self.tensor_shape,
            "image_data": self.image_data,
        }


@dataclass(frozen=True)
class AlexNetAnalysis:
    """Live predictions and selected layer visualisations."""

    predictions: list[Prediction]
    visualisations: list[ActivationVisualisation]


@dataclass(frozen=True)
class VisionModelBundle:
    """Loaded model, transform, and label metadata."""

    key: str
    label: str
    model: torch.nn.Module
    categories: tuple[str, ...]
    preprocess: Callable[[Image.Image], torch.Tensor]


# Backwards-compatible alias for older imports.
AlexNetBundle = VisionModelBundle


def supported_model_options() -> tuple[dict[str, object], ...]:
    """Return JSON-safe selectable model metadata for the UI."""
    return tuple(option.to_dict() for option in MODEL_OPTIONS)


@lru_cache(maxsize=len(MODEL_OPTIONS))
def load_model(model_key: str = "alexnet") -> VisionModelBundle:
    """Load a pretrained torchvision model in eval mode."""
    model_key = normalise_model_key(model_key)
    option = MODEL_OPTIONS_BY_KEY[model_key]
    try:
        model, weights = _build_torchvision_model(model_key)
        model.eval()
        categories = tuple(weights.meta.get("categories", ()))
        preprocess = weights.transforms()
    except Exception as exc:  # pragma: no cover - exact failures depend on local cache/network state.
        raise ModelUnavailableError(
            f"Pretrained {option.label} weights are unavailable locally. Run setup on a networked machine once, "
            "or choose another model whose weights are already cached."
        ) from exc

    if not categories:
        raise ModelUnavailableError(f"{option.label} loaded, but ImageNet label metadata was unavailable.")

    return VisionModelBundle(
        key=model_key,
        label=option.label,
        model=model,
        categories=categories,
        preprocess=preprocess,
    )


@lru_cache(maxsize=1)
def load_alexnet() -> VisionModelBundle:
    """Load pretrained AlexNet in eval mode."""
    return load_model("alexnet")


def _build_torchvision_model(model_key: str) -> tuple[torch.nn.Module, Any]:
    """Build a supported torchvision model and return its weights metadata."""
    if model_key == "alexnet":
        from torchvision.models import AlexNet_Weights, alexnet

        weights = AlexNet_Weights.DEFAULT
        return alexnet(weights=weights), weights
    if model_key == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT
        return resnet50(weights=weights), weights
    if model_key == "mobilenet_v3_large":
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.DEFAULT
        return mobilenet_v3_large(weights=weights), weights
    raise ModelUnavailableError(f"Unsupported model: {model_key}")


def run_alexnet_top5(image: Image.Image, top_k: int = 5) -> list[Prediction]:
    """Run live AlexNet inference and return top-k ImageNet predictions."""
    analysis = run_model_analysis(image=image, model_key="alexnet", top_k=top_k, include_visualisations=False)
    return analysis.predictions


def run_alexnet_analysis(
    image: Image.Image,
    *,
    top_k: int = 5,
    include_visualisations: bool = True,
    visualisation_keys: set[str] | None = None,
    activation_colour_map: str = DEFAULT_ACTIVATION_COLOUR_MAP,
    model_key: str = "alexnet",
) -> AlexNetAnalysis:
    """Backward-compatible wrapper for model analysis."""
    return run_model_analysis(
        image,
        model_key=model_key,
        top_k=top_k,
        include_visualisations=include_visualisations,
        visualisation_keys=visualisation_keys,
        activation_colour_map=activation_colour_map,
    )


def run_model_analysis(
    image: Image.Image,
    *,
    model_key: str = "alexnet",
    top_k: int = 5,
    include_visualisations: bool = True,
    visualisation_keys: set[str] | None = None,
    activation_colour_map: str = DEFAULT_ACTIVATION_COLOUR_MAP,
) -> AlexNetAnalysis:
    """Run a supported vision model once and return predictions plus activation grids."""
    model_key = normalise_model_key(model_key)
    bundle = load_model(model_key)
    input_tensor = bundle.preprocess(image).unsqueeze(0)
    model_specs = layer_specs_for_model(model_key)
    selected_specs = tuple(spec for spec in model_specs if visualisation_keys is None or spec.key in visualisation_keys)

    capture: ActivationCapture | None = None
    if include_visualisations and selected_specs:
        with torch.inference_mode(), ActivationCapture(bundle.model, specs=selected_specs) as active_capture:
            logits = bundle.model(input_tensor)
            capture = active_capture
    else:
        with torch.inference_mode():
            logits = bundle.model(input_tensor)

    probabilities = torch.nn.functional.softmax(logits[0], dim=0)
    top_probabilities, top_indices = torch.topk(probabilities, k=top_k)
    predictions = _format_predictions(bundle.categories, top_probabilities, top_indices)

    visualisations: list[ActivationVisualisation] = []
    if include_visualisations and capture is not None:
        for spec in selected_specs:
            activation = capture.activations.get(spec.key)
            if activation is None:
                continue
            visualisations.append(
                ActivationVisualisation(
                    key=spec.key,
                    label=spec.label,
                    caption_key=spec.caption_key,
                    note=spec.public_note,
                    tensor_shape=tuple(int(dim) for dim in activation.shape),
                    image_data=activation_grid_png_base64(
                        activation,
                        colour_map=activation_colour_map,
                    ),
                )
            )

    return AlexNetAnalysis(predictions=predictions, visualisations=visualisations)


def _format_predictions(
    categories: tuple[str, ...],
    top_probabilities: torch.Tensor,
    top_indices: torch.Tensor,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for probability, index in zip(top_probabilities.tolist(), top_indices.tolist(), strict=True):
        label = categories[index] if index < len(categories) else f"class {index}"
        predictions.append(Prediction(label=label, probability=float(probability)))
    return predictions
