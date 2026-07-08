"""Editable public captions for demo stages and model layers."""

from __future__ import annotations

CAPTIONS: dict[str, str] = {
    "Input": "The image is resized and normalised before entering the network.",
    "Conv 1": "The first convolution often responds to simple patterns such as edges, colour changes, and corners.",
    "Pool 1": "The first pooling layer keeps strong nearby responses while reducing the spatial size.",
    "Conv 2": "The second convolution combines simple patterns into richer local features.",
    "Pool 2": "The second pooling layer keeps strong responses and makes the feature maps smaller again.",
    "Conv 3": "The third convolution combines earlier patterns into textures, curves, and repeated shapes.",
    "Conv 4": "The fourth convolution builds more specialised combinations of visual features.",
    "Conv 5": "The fifth convolution produces deeper feature maps that feed the final classifier pathway.",
    "Pool 5": "The final pooling layer compresses the deepest feature maps into a compact spatial summary.",
    "Avg pool": "Adaptive average pooling shapes the final feature maps into the fixed size expected by the classifier.",
    "Early features": "Early model layers often respond to simple patterns such as edges, colour changes, and corners.",
    "Pooling": "Pooling keeps strong nearby responses while reducing the spatial size.",
    "Mid features": "Middle layers combine simple patterns into richer textures, curves, and repeated shapes.",
    "Deep features": "Deeper layers build more specialised combinations of visual features useful for classification.",
    "Classifier": "The classifier turns the compact feature values into scores for ImageNet training labels.",
    "Prediction": "The final prediction is a likely class from the model’s training labels. It can be wrong.",
}


def get_caption(key: str) -> str:
    """Return a public caption for a stage, falling back to a safe default."""
    return CAPTIONS.get(key, "This shows how a trained vision model responds at this stage.")
