import numpy as np

from src.visualise import ACTIVATION_COLOUR_MAP_OPTIONS, apply_activation_colour_map, normalise_activation_colour_map


def test_activation_colour_map_makes_high_values_brighter() -> None:
    values = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    colours = apply_activation_colour_map(values)
    brightness = colours.astype(np.float32).mean(axis=-1)[0]

    assert brightness[0] < brightness[1] < brightness[2]
    assert colours[0, 2].min() > 200


def test_activation_colour_maps_offer_all_booth_palettes() -> None:
    names = {name for name, _label in ACTIVATION_COLOUR_MAP_OPTIONS}

    assert names == {"aurora", "laboratory", "classroom", "neon", "calm", "signal"}


def test_activation_colour_maps_change_rendered_colours() -> None:
    values = np.array([[0.25, 0.75]], dtype=np.float32)

    neon = apply_activation_colour_map(values, colour_map="neon")
    calm = apply_activation_colour_map(values, colour_map="calm")

    assert not np.array_equal(neon, calm)


def test_unknown_activation_colour_map_falls_back_to_default() -> None:
    assert normalise_activation_colour_map("missing") == "aurora"
