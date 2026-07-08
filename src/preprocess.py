"""Image preprocessing helpers for ImageNet-style inference."""

from __future__ import annotations

from PIL import Image
import torch
from torchvision import transforms

ALEXNET_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB image without mutating the original object."""
    return image.convert("RGB")


def preprocess_for_alexnet(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalised ImageNet-style batch tensor.

    The output shape is `(1, 3, 224, 224)` and matches ImageNet-style model
    preprocessing used by torchvision.
    """
    pipeline = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(ALEXNET_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return pipeline(ensure_rgb(image)).unsqueeze(0)
