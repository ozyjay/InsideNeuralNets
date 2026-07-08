import numpy as np

from src.visualise import apply_activation_colour_map


def test_activation_colour_map_makes_high_values_brighter() -> None:
    values = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    colours = apply_activation_colour_map(values)
    brightness = colours.astype(np.float32).mean(axis=-1)[0]

    assert brightness[0] < brightness[1] < brightness[2]
    assert colours[0, 2].min() > 200
